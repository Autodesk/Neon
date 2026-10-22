# carbon — current-status analysis

_Date: 2026-05-12. Branch surveyed: `sycl-cuda`._

_**Status update — 2026-05-12 (end of day): the full P0 foundation pass has
landed.** Every P0 finding from the original audit is resolved; each section
below carries a `— RESOLVED <commit>` callout pointing at the commit that
closed it. The test-coverage matrix is now fully green across `Dense`,
`Block`, and `DenseDisg` — every capability has at least one supporting unit
test. Latest verification: **63/63 CUDA + 52/52 SYCL + 73/73 CUDA+MPI+NCCL
tests pass plus `format-check` is clean** on all three configurations._

_The pre-pass audit text is preserved below for context — each P0 entry's
diagnosis is left intact so the rationale for the fix is still readable._

This is a deep, opinionated audit of carbon as a foundation for the rest of
the Neon port. The TL;DR is: the project is in **good shape** — there is no
architectural rot — but a layer of **foundation cracks** has accumulated and
will get expensive to fix once more is built on top.

Findings are grouped by severity (P0 / P1 / P2). Within each severity the
issues are listed in priority order. Anchors point at the lines I read while
forming each conclusion.

---

## P0 — Foundation cracks worth fixing before adding more grid types or features

### ~~P0.1 — Block kernel factory surface is a strict subset of Dense~~ — RESOLVED 2026-05-12

**Resolution:** `Block` now exposes the same six factories as `Dense`:
`make_kernel`, `make_reduce_kernel`, `make_host_kernel`,
`make_host_reduce_kernel`, `make_universal_kernel`,
`make_universal_reduce_kernel`. Closed in:

- `838c5b5 feat(block): Block::make_host_kernel + make_host_reduce_kernel (OpenMP)`
  — adds `BlockHostKernelImpl` + `BlockReduceHostKernelImpl` and the matching
  factory members.
- `855ef30 feat(block): Block::make_universal_kernel + make_universal_reduce_kernel`
  — adds `BlockUniversalKernelImpl` + `BlockUniversalReduceKernelImpl` and
  wires `if constexpr` dispatch in `Kernel::compute_universal` /
  `Kernel::reduce_universal` so user code calls the same factory regardless
  of grid family.

Verified by `test_block_host_parity` (device ↔ host parity for compute +
reduce + a heterogeneous device + anchor + host_compute pipeline) and
`test_block_universal` (single-target universal compute + reduce). The
original finding kept here for historical reference:

> `Dense` exposes six factories: `make_kernel`, `make_universal_kernel`,
> `make_reduce_kernel`, `make_universal_reduce_kernel`, `make_host_kernel`,
> `make_host_reduce_kernel` (`include/carbon/dense.h:297–437`).
>
> `Block` exposes only **two**: `make_kernel` and `make_reduce_kernel`
> (`include/carbon/block.h:521–552`).
>
> Consequences: `Block` has no OpenMP host launch path; cannot participate in
> heterogeneous host/device pipelines that work on `Dense`; the "Universal"
> (dual-target) story is invisible on `Block`. This is the single biggest
> interchangeability gap.

### ~~P0.2 — Block reductions do not work on SYCL — silent at configure, loud at TU~~ — RESOLVED 2026-05-12

**Resolution:** the `static_assert` in `BlockReduceKernelImpl` is replaced by
a real SYCL backend; `Block::make_reduce_kernel` now works on both backends.
Closed in `19b3384 feat(block): SYCL block reduce + sync H2D in SYCL async_copy`:

- New `include/carbon/detail/backend/sycl/block_reduce_kernel.h` with a
  `block_reduce_fn` concept + `launch_block_reduce` using `sycl::reduction`
  over `sycl::parallel_for` (one work-group per brick, one work-item per
  voxel, identity-elided on inactive voxels — direct analogue of the dense
  SYCL reducer).
- `BlockReduceKernelImpl::launch_on_device` learns a `CARBON_BACKEND_IS_SYCL`
  branch that calls into the new launch helper and shares stages 2 + 3 with
  the CUDA path via `reduce_stages::{run_stage2_combine, run_stage3_cross_rank}`.
- Companion fix: `System::async_copy` in the SYCL backend now drains the
  `q.memcpy` event before returning. AdaptiveCpp's `q.memcpy` does not pin
  the host source for H2D, which was producing a multi-GPU overcount bug
  the reducer surfaced first.

Verified by `test_block_reduce` (Standard + Internal + Boundary + recombine
pin, both single- and multi-GPU paths) and `test_block_dist_reduce` (stage-3
NCCL cross-rank fold) on the SYCL+CUDA matrix. The CUDA-only test gate for
the migrated tests was retired in the same pass; `test_block_reduce` runs on
both backends. The original finding kept here for historical reference:

> `BlockReduceKernelImpl` triggered a `static_assert(sizeof(LoadingFn) == 0, ...)`
> under `CARBON_BACKEND_IS_SYCL`. A SYCL build that did not call
> `Block::make_reduce_kernel` succeeded; anyone who did got a TU error rather
> than a clean runtime exception or a feature-gated typedef. Combined with
> P0.1, carbon under SYCL had 9 Block tests gated to CUDA-only and a feature
> gap that wasn't surfaced in `carbon::info()` or any compile-time
> `CARBON_HAVE_BLOCK_REDUCE_*` macro.

### ~~P0.3 — `Block::Field` holds raw pointers into `Block`-owned device buffers~~ — RESOLVED 2026-05-12

**Resolution:** moved bitmask + 27-way connectivity ownership from
`block::Partition` and `block::Field` onto `block::Span` (the per-GPU topology
descriptor owned by `Block`). `block::Partition` is now a thin
`{ T* ptr_ }` view — symmetric with `dense::Partition` — and
`block::Field` carries no `Block`-borrowed pointers at all. The lifetime
contract collapses to a single rule: **`Block` (and therefore its `Span`s)
must outlive any kernel that captures one**, which is already enforced by
`make_kernel` / `make_halo_kernel` usage. Verified by 54/54 CUDA tests +
45/45 SYCL tests + `format-check`.

The original finding kept here for historical reference:

> `detail::block::Field` stored `std::vector<const BitMask*>` and
> `std::vector<const uint32_t*>` populated from `block.device_bitmask_[g].data()`
> and `block.device_connectivity_[g].data()` — non-owning pointers into
> `Block`-owned device buffers, with a lifetime contract documented only
> around halo construction. Two architectural problems compounded: bitmask was
> duplicated between `Span` (1 ptr) and `Partition` (a second ptr), and
> connectivity (type-independent topology) was stored on every per-component
> `Partition` of every `Field`.

### ~~P0.4 — Dense reduce paths re-implement what `reduce_stages.h` already factored out~~ — RESOLVED 2026-05-12

**Resolution:** the duplicated stage-2 / stage-3 member functions in
`ReduceKernelImpl` (dense) and `UniversalReduceKernelImpl` (dual-target dense)
are deleted; both now route through the same
`reduce_stages::{run_stage2_combine, run_stage3_cross_rank}` helpers that
`BlockReduceKernelImpl` already used. Closed in
`a86e544 refactor(reduce): route dense + universal reducers through reduce_stages`:

- `ReduceKernelImpl`: lost its private `run_stage2_combine_` /
  `run_stage3_allreduce_` members (~70 lines); the launch body now ends with
  the same two `reduce_stages::*` calls + async D→H copy + `mark_fresh` shape
  the Block reducer uses.
- `UniversalReduceKernelImpl`: same cleanup on its device-launch path
  (`launch_on_device`), ~65 lines deleted; the OpenMP host-launch path keeps
  its dedicated single-rank-aware combine because MPI has no fp16/bf16
  datatype and the host pipeline needs the specialised dispatch through
  `dist_host_allreduce_in_place`.

Net effect: ~135 lines deleted, one canonical stage-2 / stage-3 path across
every device reducer, and DenseDisg's reducer (P0.5) plugged straight into
the same helper without inheriting any of the old fork. Verified by the
existing `reduce_sum*` and `dist_reduce` tests plus the new
`dense_disg_reduce` / `dense_disg_dist_reduce` tests. The original finding
kept here for historical reference:

> `detail/reduce_stages.h` exposed `run_stage2_combine` +
> `run_stage3_cross_rank` (~128 lines). `BlockReduceKernelImpl` correctly
> called them. `ReduceKernelImpl` (dense) and `UniversalReduceKernelImpl`
> (dual-target dense) both `#include`d the header but reimplemented the
> logic in private member functions — the cleanest accidental fork in the
> codebase, about 120–160 lines removable across two files with no API
> change.

### ~~P0.5 — `DenseDisg` has no kernel factories and no `Partition::neighbour`~~ — RESOLVED 2026-05-12

**Resolution:** `DenseDisg` is now feature-equivalent to `Dense` at the
kernel-factory + neighbour-read level. The six kernel factories
(`make_kernel`, `make_universal_kernel`, `make_host_kernel`,
`make_host_reduce_kernel`, `make_reduce_kernel`,
`make_universal_reduce_kernel`) all dispatch generic `KernelImpl`s through
a compile-time `if constexpr (requires { typename Grid::DisgPolicy; })`
shim, so the disaggregated path reuses every cross-cutting concern (token
discovery, stage-2 combine, stage-3 NCCL, OpenMP host bodies) without
duplicating code. The `Partition::neighbour` member is implemented via a
new `dense_disg::neighbour_idx` helper that resolves a neighbour's offset,
classifies its buffer-z into the five-region map, and populates the
returned `Idx` with the per-region pitch metadata. Halo plumbing now
includes a `make_halo_kernel` adapter (see below), unblocking
`OverlappedStencil` + `Skeleton` use of DenseDisg. Closed across a
sequence of commits:

- `00cf709 P0.5 DenseDisg kernel factories + neighbour (compute & reduce path)`
  — DisgPolicy marker + backend-specific `dense_disg_kernel.h` /
  `dense_disg_launch.h` / `dense_disg_reduce_kernel.h` + dispatch shims +
  the six factory members on `DenseDisg`.
- `26bc867 test(disg): add DenseDisg reduce + host + PeriodicZ halo tests`
  — single- and multi-GPU reduce, host (OpenMP) compute + reduce, and a
  `BoundaryMode::PeriodicZ` halo regression.
- `daa2312 test(dist): cross-rank reduce/compute for Block + DenseDisg`
  — `mpiexec -n 2` coverage for DenseDisg compute and DenseDisg reduce
  (stage-3 NCCL allreduce on the disg reduce path).
- `8ea47c3 feat(disg): make_halo_kernel + OverlappedStencil/Skeleton coverage`
  — `HaloDisgKernelImpl` + `Kernel::halo_disg` + `DenseDisg::Field::make_halo_kernel`
  unlock `Skeleton::OverlapMode::Standard` auto-halo via the existing
  `KernelLoader::read_stencil` + `TypedHaloFactory` path.
- `ab03250 test(disg): Skeleton OverlapMode::Extended + TwoWayExtended coverage`
  — the last DenseDisg row of the test-coverage matrix.

DenseDisg test coverage: 12 tests across single/multi-GPU compute, single/
multi-GPU reduce, cross-rank compute, cross-rank reduce, OpenMP host compute
+ reduce, PeriodicZ halo, OverlappedStencil, and all three Skeleton
OverlapModes (None/Standard/Extended/TwoWayExtended). All 12 pass on both
the CUDA and SYCL backends (the SYCL row of the matrix is now a clean
"yes"). The original finding kept here for historical reference:

> `DenseDisg` was documented as half-built: kernel factories intentionally not
> exposed yet, `Partition` had only `read`/`write`/`reset` — no `neighbour`,
> no DenseDisg reduction or cross-rank tests. Today it would be "data plane
> + halo only" exposed in `<carbon/carbon.h>` next to `Dense` and `Block`.

### ~~P0.6 — Documentation drift across at least six files~~ — RESOLVED 2026-05-12

**Resolution:** every drift listed below was rewritten to match the
implementation. Closed in `4d180e8 docs: sweep stale "Phase 1 / not yet"
drift across block + boundary docs`. The drift table itself is retained as
a snapshot of what was fixed:

| Where | Was | Now |
|---|---|---|
| `AGENTS.md` status snapshot | "All 28 tests pass" / "migrated unit-test list is currently empty" | Reflects the current test counts (63 CUDA / 52 SYCL / 73 CUDA+MPI+NCCL) and lists the CUDA-only test set |
| `include/carbon/boundary_mode.h` | "Today only `Open` is implemented end-to-end" | `PeriodicZ` documented as honoured by `neighbour()`, halo exchange, and the three `halo_*periodic_z*` tests |
| `include/carbon/block.h:496–503` | `MemorySide::Host` / `HostDevice` "accepted but the host mirror is not allocated" | Documents the eager host-mirror allocation `block.inl:707–734` does |
| `include/carbon/detail/block/field.h:57–60, 222–225` | "Phase 1 returns an anchor (no-op)" | Documents that `Kernel::halo_block` returns a real `HaloBlock` whenever `block_ != nullptr` |
| `include/carbon/detail/dense_disg/span.h:87–88` | mentioned `Partition::neighbour` that did not exist | `Partition::neighbour` exists (P0.5); the cross-reference is now accurate |
| `include/carbon/system.h:173–176` | bounds errors "via `CARBON_LOG_ERROR`" | Documents the `throw_with_log<std::out_of_range>` shape `system.cpp` uses |

The original finding kept here for historical reference:

> Docs are no longer the source of truth. Every drift is an invitation for
> the next contributor to either trust the wrong thing or stop trusting
> docs entirely. These are 5–10 minute fixes individually but they
> undermine the credibility of every other doc statement.

### ~~P0.7 — Inconsistent empty-system / preconditions~~ — RESOLVED 2026-05-12

**Resolution:** picked the "grid ctor enforces non-empty system" policy and
made `Dense`, `Block`, and `DenseDisg` all throw at construction when the
underlying `System` carries zero local GPUs. Introduced a header-only
`CARBON_EXPECT_THROW(expr, ExceptionType)` macro
(`include/carbon/test_support.h`) and migrated the negative-path tests off
manual `try/catch` blocks. Closed in
`552fb0e feat(tests): empty-system policy + CARBON_EXPECT_THROW helper`:

- `Dense::Dense` and `DenseDisg::DenseDisg` now mirror `Block`'s contract:
  empty `GpuGroup` → `std::invalid_argument` via `throw_with_log`. The old
  "constructible empty + `newField` throws" footgun is gone.
- `CARBON_EXPECT_THROW` is a 5-line `do { ... } while (0)` macro that runs
  the expression, asserts it throws `ExceptionType`, returns `EXIT_FAILURE`
  with a diagnostic on miss / wrong type. No GoogleTest dependency.
- `test_dense.cpp`, `test_block_static.cpp`, `test_dense_disg.cpp` updated
  to use the macro for their empty-system assertions.

The original finding kept here for historical reference:

> `Dense(System with 0 GPUs)` was constructible; `Dense::newField` then
> threw. `Block(System with 0 GPUs)` throws in the constructor.
> Negative-path testing was manual `try/catch` everywhere — no
> `EXPECT_THROW` / `ASSERT_THROW` matcher in the suite.

---

## P1 — Should-fix soon, but won't block the next port pass

### P1.1 — `compute_dense_grid` diverges from the `throw_with_log` convention

`include/carbon/detail/dense/launch_shape.h:64–76` does `CARBON_LOG_WARN` plus
raw `throw std::runtime_error(...)`. Every other validation site uses
`carbon::log::throw_with_log<E>(...)`. This is the only divergence I found and
worth normalizing.

### P1.2 — `Span` and `Partition` are duck-typed across three grids; no concept

There is no `SpanConcept` or `PartitionConcept` anywhere. Every `KernelImpl`
calls `span.iter_extent(view)`, `span.set_and_validate(...)`,
`partition.read(...)` etc. by template duck typing. As soon as a fourth grid
lands (e.g. `bGrid`+adaptivity, `mGrid`), divergent shapes will be discovered
at TU instantiation, not at the concept site.

C++23 `concept`s plus a couple of `static_assert`s near the kernel impls would
cost ~40 lines and catch these as immediate compiler errors.

### P1.3 — `KernelLoader` lambda capture is a footgun

`DeviceKernelImpl` stores `loading_fn_` by value
(`include/carbon/detail/dense_kernel_impl.h:87–94`). A user that captures
temporaries by reference (`auto k = grid.make_kernel([&](KernelLoader& l) { l.read(field_temporary); ... });`)
gets a silent UAF on first `launch()`.

Mitigations to consider:

- Static check that the loading lambda is a "structural lambda" (only by-value
  or `mutable` captures) — at least for the common arity.
- Detect captures-by-reference in debug via address tracking, or
- Loud documentation in `KernelLoader` doc.

### P1.4 — `OverlappedStencil` constructor that takes user streams/events does not validate provenance

`include/carbon/overlapped_stencil.h:107–112` explicitly says mismatched
provenance is unchecked. This is an obscure failure mode in user code that
you will be asked to debug.

A cheap fix: stash the `System` shallow handle and assert on `gpu_count()` and
uid match in debug builds.

### P1.5 — `tests/CMakeLists.txt` has no `TIMEOUT` / `RUN_SERIAL` outside the GPU lock

A misconfigured test that hangs will hang `ctest` indefinitely. A 5-minute
default `TIMEOUT` (overridable per test) is cheap insurance.

### P1.6 — CI is CPU-only — every GPU-bound regression is found locally or by Docker matrix

`.github/workflows/ci.yml` runs only the host subset on GitHub-hosted runners.
The Docker matrix (`scripts/docker-build-all.sh`) is solid but local-only.
There is no CI-side gate against GPU-bound regressions; that is why flakes
like the `OverlappedStencil` peer-copy race were caught only after manual
stress runs.

If this repo is going to be a foundation, the next investment is a
self-hosted GPU runner (or a third-party CI with GPU support) that runs at
least a CUDA + 1×NCCL config on every PR.

### ~~P1.8 — `Block` cross-rank stencil produces divergent numerics — surfaced by `jacobi_unified` (2026-05-12)~~ — RESOLVED 2026-05-12

**Resolution:** root cause was in the connectivity rewrite in
`include/carbon/block.inl::reserve_row`. The lookup
`candidate_to_ghost_local[g][cand_idx]` was gated by
`else if (neigh.gpu_id != kInvalidBlockIdx)`. Under MPI (one rank per GPU,
`gpu_count == 1`), the local sweep over `candidate_to_local` only filled
entries for this rank's owned z-rows, so remote-rank bricks had
`neigh.gpu_id == kInvalidBlockIdx` and the ghost-id branch never fired —
halo bytes arrived in the ghost slots but `Partition::neighbour` could not
see them. Fix: replace the inner `else if` with an unconditional `else`
that always consults `candidate_to_ghost_local` when the neighbour is not
locally owned by GPU `g`. New regression test
`tests/test_block_neighbour_dist.cu` runs a halo exchange + a single
`Partition::neighbour(+z, ...)` probe under `mpiexec -n 2` and asserts
every cross-rank `valid == true` with the expected encoded payload.
Verification: `jacobi_unified --grid block --size 32` now gives
`sum|u|^2 = 3.2778e+03` under both single-process-2-GPU and
`mpiexec -n 2 -1-GPU-each` transports (was 3.278e+03 vs 4.206e+03).
The original finding kept here for historical reference:



The new `examples/jacobi_unified.cu` example exercises the same per-cell
Jacobi body + `Skeleton` pipeline against all three grid families across
single-GPU / multi-GPU same-node / MPI cross-rank transports. Dense and
`DenseDisg` produce **bit-for-bit equal** final L2 norms across all three
transports (1.672797e+02 at `N=64`, 200 sweeps, 7-point Laplace with
hot/cold/insulated faces). `Block` is **consistent** across single-GPU and
single-process multi-GPU but **diverges** under MPI (one rank per GPU): the
same input + halo + step + reduce pipeline gives sum|u|^2 ~= 1.732e+04 on
1 process / 2 GPUs vs ~= 2.798e+04 on 2 MPI ranks / 1 GPU each, both with
`N=64`. The discrepancy is **independent of the auto-halo path** — a
manually-built `HaloBlock` + manual launch loop reproduces it identically.

This means `test_halo_block_dist` is genuinely orthogonal: it verifies that
the byte payload reaches the ghost slots, but never asks "can the user
kernel then read those ghost voxels through `Partition::neighbour` and get
the same answer as the single-process multi-GPU layout?". The unified
example is the first regression that asks that question, and it fails.
Likely suspects (in decreasing order of suspicion):

1. The cross-rank ghost-brick **connectivity** built in
   `block.inl::reserve_row` may not point at the same ghost slots that the
   `HaloBlock` cross-rank `DistTransfer` populates, so neighbour reads hit
   either `kInvalidBlockIdx` (BC fallback, extra ±1 cells at the rank
   boundary) or stale slots.
2. The cross-rank **ghost bitmask** path re-evaluates the predicate locally
   (`block.inl:407–414`) on the assumption it's deterministic; under MPI
   that *should* match the remote rank's owned bitmask, but the candidate-
   to-ghost-local mapping might not survive the brick-row layout that
   `world_size`-driven slabbing produces.
3. The `Block` reduce stage 3 (NCCL allreduce) might double-count voxels
   that sit on the slab boundary — Block stage 1 already excludes inactive
   voxels via bitmask, so this is the *least* likely path, but worth a
   verify pass.

Repro (pre-fix):

```bash
# Both transports, same example, same flags:
mpiexec -n 2 ./build-mpi/examples/carbon_example_jacobi_unified --grid block --size 32
./build-mpi/examples/carbon_example_jacobi_unified           --grid block --size 32
# Look at "sum|u|^2" in the final log line.
```

### P1.7 — No `weak_ptr` / no `Distributed` weak handle

Every shared state is `shared_ptr` with no `weak_ptr` anywhere in the public
API. That is defensible (cycle risk is low), but `Skeleton` holding kernels
that hold `Field`s that reference `Dense`/`Block` could grow a cycle if a
kernel ever captures the grid by `shared_ptr`. Worth thinking about now,
before composite types appear.

---

## P2 — Nice-to-have / cleanup

- **`UniversalKernelImpl` inlines copies** of the device + host launch bodies
  (deliberate, but ~50 lines of duplicated launch-validation scaffolding could
  be extracted into a tiny helper used by `DeviceKernelImpl`,
  `DenseHostKernelImpl`, and `UniversalKernelImpl`).
- **`Dense::newField` / `Block::newField` / `DenseDisg::newField` policies for
  `MemorySide`** drift. Dense uses lazy host mirroring; Block uses eager host
  mirroring; DenseDisg follows Dense. Consistent semantics (and a comment in
  `memory_side.h` saying "policy is per-grid; check the grid header") would
  help.
- **`block.inl` is 774 lines** and contains both the partitioning algorithm
  and `newField`. Splitting partitioning into a free function or
  `Block::Builder` would unlock unit tests for the partitioner without
  spinning up a `System`.
- **`io.h` is 1022 lines**, mixing four formats (native, VTK, HDF5, nanoVDB).
  The optional gates make this manageable, but split-by-format headers
  (e.g. `io/native.h`, `io/vtk.h`) would let consumers include only what they
  pay for.
- **`AGENTS.md` is 699 lines**; it now mixes operational guide and historical
  changelog. The `## Status snapshot` section is overdue for a trim — the
  source of truth for "what landed" is `docs/CHANGELOG.md`. AGENTS.md should
  be the 2-page onboarding doc.

---

## Test coverage matrix at a glance

_All rows are now green as of 2026-05-12; the supporting test files are listed
in each cell so a regression flips the file's exit code to red._

| Capability | Dense | Block | DenseDisg |
|---|---|---|---|
| Single-GPU compute | yes (`kernel_laplacian_single_gpu`) | yes (`block_kernel`) | yes (`dense_disg_kernel`) |
| Multi-GPU compute | yes (`partition_read`) | yes (`block_multi_gpu`) | yes (`dense_disg_multi_gpu`) |
| Cross-rank compute | yes (`dense_dist`) | yes (`halo_block_dist`) | yes (`dense_disg_dist`) |
| Single-GPU reduce | yes (`reduce_sum`) | yes (`block_reduce`) | yes (`dense_disg_reduce`) |
| Multi-GPU reduce | yes (`reduce_sum_multi_gpu`) | yes (`block_reduce` ≥2 GPUs) | yes (`dense_disg_reduce` ≥2 GPUs) |
| Cross-rank reduce | yes (`dist_reduce`) | yes (`block_dist_reduce`) | yes (`dense_disg_dist_reduce`) |
| OpenMP host kernel | yes (`kernel_host_parity`) | yes (`block_host_parity`) | yes (`dense_disg_host`) |
| `BoundaryMode::PeriodicZ` halo | yes (3 tests: `halo_periodic_z`, `halo_dist_periodic_z`, `halo_host`) | n/a (no `BoundaryMode` on Block) | yes (`halo_disg_periodic_z`) |
| `OverlappedStencil` | yes (`overlapped_stencil`) | yes (`block_overlapped_stencil`) | yes (`dense_disg_overlapped`) |
| `Skeleton::OverlapMode::Standard` | yes (`skeleton` D/E) | yes (`skeleton_block`) | yes (`dense_disg_overlapped`) |
| `Skeleton::OverlapMode::Extended` / `TwoWayExtended` | yes (`skeleton` F/G) | yes (`skeleton_block_extended`) | yes (`dense_disg_skeleton_extended`) |
| SYCL backend | yes | yes (compute + reduce after P0.2) | yes (full Disg row passes under SYCL) |

---

## Architectural duplication summary

From a duplication scan of `detail/` (full report kept in the audit notes).
P0.4 + P0.5 closed two of the worst offenders; the surviving items are
structural rather than accidental:

- **A. Reduce impls** (`reduce_kernel_impl.h` vs `block_reduce_kernel_impl.h`
  vs `universal_reduce_kernel_impl.h`): ~~stage 2/3 factored into
  `reduce_stages.h`; Block uses it, Dense + Universal accidentally do
  not~~ — **resolved by P0.4**. All three (and DenseDisg's reducer) now
  route through the same `reduce_stages::{run_stage2_combine,
  run_stage3_cross_rank}` helpers; stage 1 stays layout-specific because
  the kernel shape itself is layout-specific.
- **B. Halos** (`halo.h` vs `halo_block.h` vs `halo_disg.h`): three storage
  models → three transfer builders. **`HaloDisg` still lacks NCCL** — same
  caveat as the original audit (and it shows up in
  `HaloDisg::launch_on_host` throwing on cross-rank pairs). Adding NCCL to
  `HaloDisg` is a contained follow-up; the surface is already in place via
  `Kernel::halo_disg` + `DenseDisg::Field::make_halo_kernel`.
- **C. Spans** (`detail/dense/span.h` vs `detail/block/span.h` vs
  `detail/dense_disg/span.h`): three genuinely different indexing models;
  no `SpanConcept` (P1.2). Structural duplication, not accidental.
- **D. Partitions**: Dense vs Disg share thin "read/write + DataView";
  Block adds connectivity / `NghData`. ~~Disg is missing `neighbour`~~ —
  **resolved by P0.5**: `dense_disg::Partition::neighbour` is wired through
  the `neighbour_idx` helper that owns the five-region pitch reconstruction.
- **E. Device vs host dense kernel impls**: parallel scaffolding (ctor,
  tokens), orthogonal launch engines (GPU grid vs OMP). Universal **copies
  both bodies** rather than delegating. Still a P2 cleanup candidate;
  factoring out the ctor/token scaffolding would save ~50 lines.
- **F. Universal compute vs Universal reduce**: both are dual-path
  reimplementations in one owning class **by design** (one lambda copy).
  Reduce path now adopts `reduce_stages` (P0.4).

---

## Robustness summary

For a foundation GPU library the **baseline is respectable**: precondition
checks in ctors / planners are common; backend failures funnel to
`runtime_error`; `throw_with_log` gives discoverable diagnostics; there is
deliberate attention to unreachable distributed paths
(e.g. `Halo` host + cross-rank — `halo.h:218–235, 504–561`).

What the P0 pass closed:

- ~~documentation drift~~ — P0.6.
- ~~`Dense` / `Block` empty-GPU inconsistency~~ — P0.7 (`Dense` and
  `DenseDisg` now throw at the ctor like `Block` does).
- ~~raw pointers into `Block` device metadata + narrow lifetime wording~~ —
  P0.3 (bitmask + connectivity own-by-`Span`; `Partition` is `{ T* }`).
- ~~manual negative tests~~ — P0.7 (`CARBON_EXPECT_THROW` migration).

Surviving weak spots: **`Kernel` shared-mutable impl** plus by-reference
captures in lambdas without lint-level guardrails (P1.3), and
**`launch_shape` exception/logging inconsistency** (P1.1) — both still
open as P1.

Concrete dangling / use-after-free risks:

1. ~~`Block` destroyed before `Block::Field` (or kernels using its
   `Partition`s): bitmask / connectivity / device-origin pointers baked
   into `Partition` can outlive `Block`'s GPU allocations.~~ Closed by P0.3
   — `Partition` is now `{ T* }` and the only lifetime contract is
   "kernels that captured a `Span` outlive the `Block`", which is
   already what every factory enforces.
2. User kernels / `Skeleton` pipelines that clone `Kernel`s then mutate
   shared state reachable from the duplicated handle: shared impl semantics
   (`kernel.h:314–316`). **Still open** (P1.3-adjacent).
3. Lambdas capturing references to `Field`s, `ReductionResult`s, scratch
   buffers, etc.: silent UAF at first `launch()`
   (`detail/dense_kernel_impl.h:87–95`). **Still open** — tracked as P1.3.

---

## Ranked recommendation: what I would fix first

The original concrete "solid foundation" pass, with status callouts. Items
1–6 below are all P0 and all closed — the full P0 sweep has landed.

1. ~~**P0.6 doc-drift sweep**~~ — done (`4d180e8`). `boundary_mode.h`,
   `block.h`, `detail/block/field.h`, `dense_disg/span.h`, `system.h`, and
   `AGENTS.md` all rewritten to match the implementation.
2. ~~**P0.4 reduce-stages adoption in dense**~~ — done (`a86e544`).
   ~135 duplicated lines removed; `reduce_stages` is now **the** stage 2/3
   path for every device reducer (Dense, Block, Universal, DenseDisg).
3. ~~**P0.3 `Block` / `Field` lifetime tightening**~~ — done (`9f9bf03`).
   Bitmask + connectivity moved to `Span`; `Partition` is now `{ T* }`;
   `Field` carries no `Block`-borrowed pointers.
4. ~~**P0.7 empty-system policy + `EXPECT_THROW` migration**~~ — done
   (`552fb0e`). `Dense` and `DenseDisg` now mirror `Block`'s ctor-time
   non-empty check; `CARBON_EXPECT_THROW` retired the manual try/catch
   negative-path tests.
5. ~~**P0.1 + P0.2 Block factory parity**~~ — done (`838c5b5`, `855ef30`
   for the factories; `19b3384` for the SYCL block reduce). All six
   factories on `Block`; `make_reduce_kernel` works on both CUDA and SYCL.
6. ~~**P0.5 close Disg's status**~~ — done (`00cf709`, `26bc867`,
   `daa2312`, `8ea47c3`, `ab03250`). DenseDisg now has all six kernel
   factories, `Partition::neighbour`, `make_halo_kernel`, cross-rank
   reduce, OverlappedStencil, and Skeleton OCC coverage. SYCL row of the
   matrix is fully green.

Still open (P1 work, tracked separately):

7. **P1.2 `SpanConcept` / `PartitionConcept`** (½ day). Cheap insurance for
   the next grid; concept now has three implementations to anchor against.
8. **P1.6 GPU CI** (1 week, mostly infra). The single biggest robustness
   multiplier; everything else lands faster once GPU regressions get caught
   at PR time.
9. **P1.3 `KernelLoader` lambda capture footgun**. The remaining
   non-resolved dangling-pointer risk; the P0.3 lifetime tightening
   eliminated the `Block` family of UAFs but lambdas-by-reference are still
   user-visible footguns.

If you only have one afternoon, do (7). If you have one week, do (7)+(9).
If you have one quarter, do all of P1 and you have a foundation that can
comfortably absorb a second sparse grid (mGrid) and a third backend.
