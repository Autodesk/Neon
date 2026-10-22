# XLB `ComputeBackend.NEON` → Carbon (in-place backend swap)

**Goal.** Make XLB's existing `ComputeBackend.NEON` run on **Carbon** instead of the
Neon library, *keeping the public enum name and all user-facing scripts unchanged*.
XLB is vendored as a submodule at `XLB/` (branch `carbon`). Carbon's Python layer is
extended to full parity (including the multires LBM builtins) and XLB's internal Neon
paths are rewired to call Carbon.

Approach chosen by the user: **replace-in-place** — rewrite XLB's internal Neon-path
modules to call `carbon.*` directly (dropping the external `neon`/`py_neon`
dependency), while `ComputeBackend.NEON` stays the user-visible name. Kernel *bodies*
(~28 files using `wp.neon_*`) are left untouched and bridged by a `wp.neon_* →
wp.carbon.*` alias shim (see Workstream E).

## Environment (validated)

- PyPI is blocked behind an SSL-intercepting proxy; use the Autodesk Artifactory
  mirror as the pip index (`~/.config/pip/pip.conf`). Needs pip ≥ 24 (Ubuntu 24.04
  base). `jax==0.11.0 + jaxlib==0.11.0` is a cached, working pair.
- The pinned Warp fork (`nvlukasz/warp @ external-source-support-update2`) builds on
  CUDA ≤ 12; the dev host is CUDA 13.1, so all Python/end-to-end validation runs in a
  CUDA-12.8 container. `warp/native/cuda_util.cpp` was patched for CUDA-13
  forward-compat (`decltype` cast + `CUDART_VERSION >= 13000` guard).
- Harnesses:
  - `scripts/py-test-cuda12.sh` — build Warp fork + libcarbonpy + run `tests/test_warp_*.py`.
  - `scripts/xlb-test-cuda12.sh` — full XLB + Carbon + Warp fork import/integration.

## Carbon Python API vs Neon (mapping)

Carbon's Python surface was deliberately shaped to mirror Neon:

| Neon (XLB uses) | Carbon equivalent | Status |
| --- | --- | --- |
| `neon.init()` | `carbon.init(enable_warp=True)` | present |
| `neon.Backend(runtime=..., dev_idx_list=...)` | `carbon.System(gate)` | **gap: no device subset** (all visible GPUs; honour `dev_idx_list` via `CUDA_VISIBLE_DEVICES` for now; C-ABI subset is a follow-up) |
| `neon.Index_3d(x,y,z)` / `py_neon.Index_3d` | `carbon.Index_3d` | **DONE** (`py/carbon/index3d.py`) |
| `neon.dense.dGrid(backend, dim, sparsity, stencil)` | `carbon.Dense(system, ex,ey,ez, halo_radius=1)` | present (halo_radius=1 covers D3Q19/27 Chebyshev radius) |
| `grid.new_field(cardinality=, dtype=)` | `Dense.new_field(components=, dtype=, cardinality=)` | **DONE** (`cardinality=` kwarg alias) |
| `field.zero_run(stream_idx=)` / `field.fill_run(value=, stream_idx=)` | `DenseField.zero_run/fill_run` | **DONE** (`py/carbon/_fieldops.py`, test `tests/test_warp_field_fill.py`) |
| `@neon.Container.factory(name=)` + loading lambda | `@carbon.kernel(name=)` / `build_bound_kernel` | present (capture path) |
| `neon.Loader` (`set_grid`, `get_read_handle`, `declare_kernel`, `set_mres_grid`, ...) | `carbon.Loader` (same methods) | present |
| `neon.Skeleton.sequence(name, [...], occ=)` + `.run()` | `carbon.Skeleton` | present |
| `neon.SkeletonConfig.OCC.none()` | `carbon.SkeletonConfig` / `carbon.OCC` | present |

## Pivotal finding — Neon's `wp.neon_read` is *spanless*

Neon kernels read/write via `wp.neon_read(part, index, comp)` / `wp.neon_write(part,
index, comp, value)` — **no span argument**. The Neon partition embeds its own
addressing metadata. Carbon's dense read is `wp.carbon.read(part, span, idx, comp)`:
all addressing math (pitch, halo_z, z_offset, extents, boundary_mode) lives in the
**span**, not the partition (`libcarbonpy/include/carbonpy/warp_dense.h`,
`carbon_dense_linear_offset`). So `wp.neon_read` **cannot** be a trivial alias of
`wp.carbon.read` — the arities differ and the span is not available at the call site
inside a user `@wp.func`.

**Decision: embed `CarbonDenseSpan` inside the dense partition POD** (Neon does the
same), then add *spanless* builtins that `wp.neon_*` aliases 1:1. This is
backward-compatible: the existing span-carrying builtins stay, so all current tests
keep passing; `get_partition(device, data_view)` additionally fills the embedded span.

### Fat-partition implementation checklist (dense; single-res milestone)

1. `libcarbonpy/include/carbonpy/carbonpy.h` — change C `struct CarbonDensePartition`
   to `{ uint64_t data_ptr; struct CarbonDenseSpan span; }` (drop unused `halo_*`).
2. `libcarbonpy/include/carbonpy/warp_partition_types.h` — relocate the `wp::CarbonDenseSpan`
   definition here (before the partitions) and embed it in each
   `CarbonDensePartition{F32,F64,I32,I64}`; remove the duplicate def from `warp_dense.h`.
3. `libcarbonpy/include/carbonpy/warp_dense.h` — add spanless overloads that delegate to
   the existing span-carrying ones via `part.span`: `read/write/set/global_idx/
   neighbour/read_ngh` + a `neon_ngh_idx`-style offset. Keep all existing functions.
4. `libcarbonpy/src/dense.cpp` — factor `fill_dense_span(dense, device, view, out)`
   (shared with `carbonpy_dense_get_span`); store the owning `carbon::Dense*` in
   `FieldBox` (mirror `MgridFieldBox`); make `carbonpy_dense_field_get_partition`
   honour `data_view` and fill `part.span`.
5. `libcarbonpy/include/carbonpy/detail/field_box.h` — add `void* dense{};` to `FieldBox`.
6. `libcarbonpy/include/carbonpy/pod_layout_checks.h` — update dense-partition field/size checks.
7. `py/carbon/warp_types.py` — `_CarbonDensePartitionLayout = { data_ptr:uint64, span:CarbonDenseSpan }`.
8. `py/carbon/warp_builtins.py` — register spanless `read/write/set/global_idx/read_ngh/ngh_idx`.
9. Tests — extend `tests/test_warp_pod_layout.py` (dense partition size/offset) + a new
   spanless read/write kernel test.

The multires path (Workstream C) needs the analogous spanless surface on the
Block/Mgrid partitions plus the new LBM builtins (`level`, `has_finer_ngh`,
`has_child`, `has_parent`, `lbm_read_coarser_ngh`, `mres_lbm_store_op`, `ngh_idx`,
`read_ngh`, `global_idx`, `get_x/y/z`).

## `wp.neon_*` builtins used by XLB (alias targets)

Map (single-res + shared): `neon_read`, `neon_write`, `neon_read_ngh`,
`neon_ngh_idx`, `neon_global_idx`, `neon_get_x/y/z`.
Multires-only: `neon_level`, `neon_has_child`, `neon_has_parent`,
`neon_has_finer_ngh`, `neon_lbm_read_coarser_ngh`, `neon_mres_lbm_store_op`.

## Workstreams

- **A. Env + Warp fork** — DONE (patched, container harnesses green).
- **B. Single-res Carbon Python gaps** — IN PROGRESS. `zero_run`/`fill_run`,
  `carbon.Index_3d`, `new_field(cardinality=)`, **uint8 mask dtype** DONE. Remaining:
  device-subset (follow-up), host read-back / `update_host` / VTK export.
- **B′. Fat dense partition + spanless builtins** — **DONE (dense)**. `CarbonDenseSpan`
  embedded in the dense partition POD; spanless `set/global_idx/read/write/neighbour/
  read_ngh` registered; `get_partition` + `cuda_driver` fill the embedded span; test
  `tests/test_warp_dense_spanless.py` (suite 29 passed, 1 skipped). Block/Mgrid spanless
  surface still TODO (folds into C).
- **C. Multires LBM builtins** — spanless Block/Mgrid surface + LBM ops. **IN PROGRESS**
  (recon complete 2026-08). Detailed phased plan below.

  ### C recon summary

  XLB's multires path routes through **14 `xlb/` modules** (grid `multires_grid.py` +
  `multires_grid_factory`; stepper `nse_multires_stepper.py`; `simulation_manager.py`
  `_construct_stepper_skeleton`; operators `multires_macroscopic`,
  `multires_quadratic_equilibrium`, `multires_momentum_transfer`; maskers
  `multires_indices_boundary_masker`, `multires_aabb`, `multires_aabb_close`,
  `multires_ray`; `CustomMultiresInitializer`; `MultiresEncodeAuxiliaryData`;
  `utils/mesher.py` `MultiresIO`) plus 3 examples. All bind to Neon's `mGrid`/`mField`,
  `Loader.set_mres_grid` + `get_mres_{read,write}_handle` (with
  `Operation.stencil_up` for cross-level halos), `Container.factory`, and
  `Skeleton.sequence`. Almost every operator loops `for level in range(num_levels)` and
  launches one container per level. Level convention: **0 = finest**; refinement R = 2
  (2×2×2 sub-cells, `StaticBlock<8,8,8, 2,2,2>`).

  Kernel bodies call the standard dense builtins (`neon_read/write`, `neon_read_ngh`,
  `neon_ngh_idx`, `neon_global_idx`, `neon_get_x/y/z` — all already registered for dense)
  **plus five multires-specific device builtins** whose exact Neon semantics are pinned
  (source: `Neon/py/neon/multires/mPartition.{py,h}`,
  `libNeonDomain/.../mGrid/mPartition_imp.h`):

  | `wp.neon_*` | signature | Neon impl / semantics |
  | --- | --- | --- |
  | `neon_has_child(part, idx) -> bool` | — | `hasChildren(idx)`: this cell is a coarse placeholder with a finer child block (always false on level 0). Maps to existing `carbon_mgrid_has_children(part, block, x,y,z)`. |
  | `neon_has_parent(part, idx) -> bool` | — | `hasParent(idx)`: `getParent(idx).isActive()`. Maps to existing `carbon_mgrid_has_parent(part, block)`. |
  | `neon_has_finer_ngh(part, idx, ngh_idx) -> bool` | ngh = `int8_3d` pull dir | `hasChildren(cell, nghDir)`: same-level neighbour at `nghDir` exists, is active, **and is refined**. New: resolve neighbour via `carbon_block_neighbour_resolve(own_span,…)` then `carbon_mgrid_has_children`. Needs own_span. |
  | `neon_lbm_read_coarser_ngh(part, idx, ngh_idx, card, alt, &valid) -> T` | — | **Explosion read.** `uncleOffset(inBlockPos % R, ngh)` (with `off(i,j)=(i+j<=0)?i+j:i+j-1`) → `uncleVal(idx, uncleDir, card, alt)` = **raw** parent-level uncle population, no interpolation/scaling. `valid` = uncle active. New: needs `getUncle` = neighbour-of-parent-in-dir on `parent_span` + parent memory read. |
  | `neon_mres_lbm_store_op(part, idx, card, ngh_idx, value)` | ngh = push dir `getDir(q)` | **Coalescence fan-in.** `uncleDir=uncleOffset(inBlockPos, qDir)`; if same-level ngh at uncleDir inactive but uncle active, `CsDir=uncleDir-qDir`; if that uncle+its child active, `atomic_add(&uncleVal(cell,CsDir,q), value)` into **parent** memory. Pure sum; 1/R and averaging live in separate coalescence/store passes (out of scope for the builtin). New: needs writable parent memory + atomic. |

  `neon_level(part) -> int` is referenced only in comments/TODOs (per-level coord
  scaling), not in live kernel bodies — defer.

  ### C architecture decision — fat mgrid partition

  XLB kernels receive **one** partition handle from `loader.get_mres_{read,write}_handle`
  and use it for own-level `read/write/neighbour/global_idx` **and** the multires builtins.
  Neon's `mPartition` is that single fat handle. Carbon currently splits it
  (`CarbonBlockPartition` own-level vs `CarbonMgridPartition` cross-level, and the cross
  POD takes a *separate* `own_span` arg). So C mirrors the dense fat-partition move:
  **embed `own_span` (`CarbonBlockSpan`) into `CarbonMgridPartition`** and make
  `loader.get_mres_*_handle` return the cross partition, so every mgrid builtin becomes
  spanless `(part, idx, …)`.

  ### C phased increments (each build + ctest + format-check clean before commit)

  1. **C.1 — own_span embedding (foundational).** Add `CarbonBlockSpan own_span` to
     `CarbonMgridPartition` POD across all layers: `warp_mgrid.h` macro, `carbonpy.h`
     ABI struct + `carbonpy_offsetof_mgrid_partition_*` introspection, `warp_types.py`
     ctypes layout, `pod_layout_checks.h` static-asserts, and fill it in
     `mgrid.cpp::fill_cross_partition` (`blit_block_span(grid(level),…)`). Regression:
     `test_warp_mgrid_cross_level` + `test_warp_pod_layout` still green.
  2. **C.2 — spanless mgrid own ops. DONE.** Registered spanless `set/global_idx/read/
     write/neighbour/read_ngh` on `CarbonMgridPartition` (address via embedded
     `own_span`), mirroring the dense spanless overloads; added `carbon_mgrid_own_write`
     in `warp_mgrid.h`. The 3-arg `read(part, span, idx, c)` stays the *parent* read; the
     new 2-arg `read(part, idx, c)` is the *own* read. Validated by
     `tests/test_warp_mgrid_spanless.py` (own fill → read / neighbour / read_ngh
     round-trip through the fat cross partition; suite **34 passed, 1 skipped**).
     **Deferred to C.6:** switching loader `get_mres_*_handle` to return the cross
     partition — doing it now would flip `read(part, span, idx)` to a parent read under
     the existing block-partition mgrid tests, so it lands with the XLB kernel ports that
     use the spanless form.
  3. **C.3 — five multires builtins (C++). DONE.** Implemented in `warp_mgrid.h`:
     `carbon_mgrid_has_child_idx`, `_has_parent_idx`, `_has_finer_ngh`, the `uncle_off`
     (Neon `uncleOffset`) + `parent_voxel` + `get_uncle` helpers, `_lbm_read_coarser_ngh`
     (raw parent-level uncle read), and `_mres_lbm_store_op` (`carbon_mgrid_atomic_add`
     fan-in into parent storage, `__CUDA_ARCH__`-guarded with an int64
     `unsigned long long` cast). Exposed as `wp::carbon::{has_child,has_parent,
     has_finer_ngh,lbm_read_coarser_ngh,mres_lbm_store_op}` in `warp_carbon.h` (+ a mgrid
     `read_ngh` offset overload wiring the trailing `is_valid` out-ref to own-level
     neighbour validity). Registered in `warp_builtins.py`
     (`_register_mgrid_lbm_builtins`) and aliased as
     `wp.neon_{has_child,has_parent,has_finer_ngh,lbm_read_coarser_ngh,mres_lbm_store_op}`
     plus mgrid `neon_{read,write,global_idx,read_ngh}`. Test
     `tests/test_warp_mgrid_multires.py` on a 2-level all-active octree validates level
     semantics (finest: parent, no child; coarsest: child, no parent), the invariant
     `lbm_read_coarser_ngh(ngh=0) == parent read`, and `has_finer_ngh` +
     `mres_lbm_store_op` codegen/run (suite **35 passed, 1 skipped**; format-check clean).
     Full explosion/coalescence *numerics* need a genuinely refined (sparse) grid, which
     lands with C.5's per-level sparsity — the all-active octree makes the store a near
     no-op (same-level neighbours always active).
  4. **C.4 — mgrid field ops + host readback. DONE.**
     `MgridField.{zero_run,fill_run,copy_from_run}` (per-level, block-shaped Warp launches
     over `CarbonBlockPartition`+`CarbonBlockSpan` reusing the validated block read/write),
     `num_levels`/`get_grid`, and host readback `to_numpy(level)`/`update_host(level)`/
     `export_vti(filename, level, …)` (per-device scatter into a full-extent array indexed by
     `global_{x,y,z}`, summed across disjoint device z-slabs; `Mgrid.level_extent`). Validated
     by `tests/test_warp_mgrid_field_ops.py` (suite **36 passed, 1 skipped**). *Deferred to
     C.6:* loader `set_mres_grid` + `Operation.stencil_up` handling.
  5. **C.5 — sparse mGrid construction. Carbon side DONE.** Added
     `carbonpy_mgrid_new_sparse(finest_xyz, depth, ref_factor, masks_flat, mask_shapes,
     mask_origins)` — the Neon pyNeon-parity `mGrid(sparsity_pattern_list,
     sparsity_pattern_origins)` interface. The C++ side rebuilds `MultiResBlock`'s existing
     `is_active(level, voxel)` predicate from the marshalled per-level masks — **no core
     change** (Neon's per-level "origins" are index-space mask placement, not physical world
     origins, so they fold entirely into the predicate; `MultiResBlock`'s single shared origin
     is untouched). Python `Mgrid(..., sparsity_pattern_list=, sparsity_pattern_origins=)`
     accepts per-level level-local int32 masks + origins (all-active default preserved).
     Validated by `tests/test_warp_mgrid_sparse.py` (suite **37 passed, 1 skipped**): full
     masks reproduce the all-active count; a restricted mask yields exactly the masked cells.
     *Remaining for XLB:* port `multires_grid.py` (`NeonMultiresGrid` → `carbon.Mgrid`) +
     `multires_grid_factory`; `create_field` parity; convert XLB's finest-coord physical
     origins to level-local; `new_field(memory_type=)` arg. (folds into C.6 operator work.)
  6. **C.6 — XLB operators + stepper.** Port the 14 modules' `_construct_neon` bodies to
     `@carbon.kernel`/`carbon.Loader`; `nse_multires_stepper` recursion + fused finest
     variants; `MultiresSimulationManager._construct_stepper_skeleton` →
     `carbon.Skeleton.sequence`.
  7. **C.7 — validate `mlups_3d_multires`** end-to-end on 2 GPUs with numerical parity
     vs single-res reference.

  **C.5 sparsity blocker — RESOLVED.** The old `carbonpy_mgrid_new` all-active predicate is
  no longer a wall: `carbonpy_mgrid_new_sparse` marshals per-level masks + origins across the
  C ABI and rebuilds `MultiResBlock`'s `is_active` predicate on the C++ side. The key finding
  (verified against Neon): per-level origins are index-space mask placements that fold into
  the predicate, so **no `MultiResBlock` core change was needed** — the earlier worry about
  intrusive per-level physical origins was unfounded. Remaining follow-up: `multires_aabb_close`
  calls `mGrid.new_field(memory_type=…)` — the Python `Mgrid.new_field` still needs a
  memory-type arg (tracked with the C.6 operator ports).
- **D. XLB config + grids** — **DONE (single-res)**. `default_config.py` NEON branch
  now calls `carbon.init(enable_warp=True)` + `carbon.install_neon_aliases()` and stores
  the gate on `DefaultConfig.carbon_gate`; `neon_grid.py` builds `carbon.System` +
  `carbon.Dense`. Validated end-to-end (`scripts/_xlb_grid_smoke.py`: `grid_factory` →
  `create_field(cardinality=, fill_value=)` → `wp.neon_read` read-back OK on 2 GPUs).
  `multires_grid.py` still TODO (folds into C).
- **E. `wp.neon_*` alias shim** — **DONE (single-res + shared)**.
  `carbon.install_neon_aliases()` registers `wp.neon_{read,write,read_ngh,ngh_idx,
  global_idx,get_x/y/z}` as builtins over the spanless `wp::carbon::` dense ops (added
  C++ `ngh_idx` + an offset+`is_valid` `read_ngh`). Test `tests/test_warp_neon_aliases.py`
  (suite 30 passed, 1 skipped). Multires `neon_*` (`has_child`/`level`/...) land with C.
- **F. Operators + steppers** — **DONE (single-res); multires TODO**. Ported the
  single-res `nse_stepper` (`_construct_neon` + `prepare_skeleton`), `macroscopic`, and
  `quadratic_equilibrium` from `neon.Container.factory`/`neon.Loader`/`neon.Skeleton`/
  `neon.SkeletonConfig` to `@carbon.kernel`/`carbon.Loader`/`carbon.Skeleton`/
  `carbon.SkeletonConfig`; dropped vestigial `import neon` from `operator.py`. `carbon.kernel`
  is the direct analogue of `@neon.Container.factory` (wraps a loading-lambda generator into a
  `BoundKernel` the `Skeleton` consumes), so kernel bodies transferred verbatim (they call
  `wp.neon_*`). Infra gaps closed: LBM component counts `{7,9,15,19,27}` instantiated via an
  X-macro across `dense.cpp` + `cuda_driver.cpp` dense switches; a `capture` flag on
  `DenseContainer` so capture kernels (which bake all fields/span as constants) launch with
  bounds only regardless of `field_count` (>2 fields previously hit `-1`). Validated
  (`scripts/_xlb_stepper_smoke.py`): interior BGK fixed point exact (`min=(1/36)·rho`,
  `max=(1/3)·rho`, `rho=q·f_init`) on 1 GPU and 2 GPUs with `occ=standard` auto-halo.
  `prepare_fields()` now works end-to-end: `DenseField.copy_from_run` (per-dtype Warp
  copy kernel in `_fieldops.py`) + `initialize_eq`'s equilibrium Neon launch
  (`QuadraticEquilibrium` carbon container) + `IndicesBoundaryMasker._construct_neon`
  ported to `@carbon.kernel`/`carbon.Loader` (construction only — its Neon-only
  multi-device loading bodies stay stubbed until BCs are exercised). Validated
  (`scripts/_xlb_stepper_prepare_smoke.py`): `prepare_fields()` → 2 steps reproduces the
  `rho=1` equilibrium interior fixed point (`min=1/36`, `max=1/3`) on 2 GPUs. Remaining:
  BCs (masker loading bodies + BC operators), forced collision, KBC, multires stepper.
- **G. I/O** — `utils/utils.py`, `utils/mesher.py` field export → Carbon host-sync + HDF5/VTK/JAX.
- **H. Install matrix, tests, examples (`mlups_3d`, multires), docs.**

## Status

- Validated end-to-end (CUDA-12 harness, 2 GPUs): fat dense partition + spanless
  builtins (B′), `wp.neon_*` alias shim (E), and the XLB config/grid rewire (D). The
  carbon Python suite is **30 passed, 1 skipped**; `scripts/_xlb_grid_smoke.py` drives
  `xlb.init(ComputeBackend.NEON)` → `grid_factory` → `create_field` → `wp.neon_read`.
- **uint8 dense dtype** (bc/missing/solid masks) landed across every layer (C ABI dtype
  enum + `dispatch_dtype`, warp partition/dense/carbon headers incl. neighbour + spanless
  + neon alias + ngh-offset, POD checks, `dtypes.py`, ctypes mirror + maps, the
  `_DENSE_DTYPES` split that keeps Block/Mgrid on 4 dtypes while dense gets 5, and the u8
  fill kernel). `carbon::Dense::Field<uint8_t,C>` + `Halo<uint8_t,C>` compile clean. Test
  `tests/test_warp_dense_u8.py` (carbon suite **31 passed, 1 skipped**).
- **Single-res operator/stepper port (F) landed and validated.** The fused
  `IncompressibleNavierStokesStepper` (BGK, no BCs) compiles, builds odd/even
  `carbon.Skeleton`s, runs, and reproduces the exact analytic BGK interior fixed point on
  1 GPU and on 2 GPUs (auto-halo via `occ=standard`). carbon regression **31 passed,
  1 skipped**.
- **`prepare_fields()` path landed and validated.** `DenseField.copy_from_run` +
  `initialize_eq`'s equilibrium launch + `IndicesBoundaryMasker` construction all work;
  `scripts/_xlb_stepper_prepare_smoke.py` drives the user-facing entry point end-to-end
  (create_nse_fields → copy_from_run → initialize_eq → 2 steps) to the `rho=1` interior
  fixed point on 2 GPUs. carbon regression **31 passed, 1 skipped** (adds
  `test_copy_from_run`).
- **Boundary conditions + LDC parity landed and validated.** The
  `IndicesBoundaryMasker` (multi-device loading bodies via `loader.get_device_id()` /
  `grid.backend.get_num_devices()` / per-device warp arrays), `EquilibriumBC` (lid) and
  `FullwayBounceBackBC` (walls) now drive a full lid-driven-cavity through the fused
  stepper. Root-caused + fixed a latent bug: `carbon_dense_global_idx` dropped the
  `halo_z` term (`out.z = halo_z + idx.iz + z_offset`), so the top z-plane mis-tagged and
  the sim diverged; after the fix the masker tags exactly the expected cells (484 lid +
  2692 walls on 24³) and the interior stays at `rho≈1` with physical, bounded velocity on
  1 **and** 2 GPUs (`occ=standard`). `neon_get_{x,y,z}` returns `int32` now (matches
  `vec3i`); `wp.set_module_options({"enable_backward": False})` on the container module
  kills spurious `CarbonIndex3` adjoints. Driver: `scripts/_xlb_ldc_smoke.py`.
- **`mlups_3d` example runs end-to-end on Carbon (primary bring-up target).**
  `XLB/examples/performance/mlups_3d.py 24 40 neon fp32/fp32 --gpu_devices=[0,1]`
  completes on 2 GPUs (~10 MLUPs, D3Q19 BGK, `occ=standard`). Ported the single-res
  `EncodeAuxiliaryData` (`helper_functions_bc.py`) and `MomentumTransfer`
  (`momentum_transfer.py`) `_construct_neon`/`neon_implementation` to `carbon.kernel`,
  swapped the example's `neon.SkeletonConfig.OCC` / `neon.Report` to `carbon`, and added a
  pure-Python `carbon.Report` (JSON writer mirroring Neon's `add_member` /
  `add_member_vector` / `write`).
- **Single-res port complete + KBC + export validated.** All remaining single-res `neon`-module
  entry points are on `carbon` (mesh maskers `ray.py`/`aabb.py`, `CustomInitializer`, and the
  `to_warp` readback); the collision operators (`BGK`/`KBC`/`ForcedCollision`/`Smagorinsky`) are
  functional-only and needed no change (`mlups_3d ... --collision_model KBC --velocity_set D3Q27`
  validated on 2 GPUs, ~18.8 MLUPs). Dense host readback + VTI export
  (`DenseField.to_numpy`/`update_host`/`export_vti`) landed and is multi-GPU-correct
  (`tests/test_warp_dense_readback.py`), unblocking `mlups_3d --export_final_velocity`.
- Next gating steps: multires surface + LBM builtins (Workstream C — the multires grid,
  `simulation_manager.py` skeleton, and the multires masker/init/aux-data bodies) to bring up
  `mlups_3d_multires`; then HDF5/nanoVDB export + `mesher.py` (rest of G), and docs/README (H).
