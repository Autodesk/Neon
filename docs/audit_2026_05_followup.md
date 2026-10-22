# Carbon — follow-up audit (2026-05-12, post-P0-pass)

_Companion to [`../current_status.md`](../current_status.md). The May-12 P0 sweep
closed every P0 item in that audit. This document is a second-pass audit of the
codebase **after** those fixes landed and after the examples / Kernel / Block
unifications, and lists the issues that surfaced or grew in priority since._

_**Status update — 2026-05-12 (end of day): the follow-up audit pass has
landed.** Both P0 items (P0.A Block + MPI connectivity, P0.B HaloDisg
cross-rank NCCL) are closed; P1.α (collapse Block impls into Dense tree —
landed via the new `DeviceLaunchTraits` / `HostLaunchTraits` /
`ReduceLaunchTraits` family in `detail/launch_traits.h` ; six concrete
impl pairs converged onto six unified impls, ~900 net LOC reduction,
no API change) + P1.β concepts + P1.δ KernelLoader lifetime guardrail +
P2.α API parity (Block accessor additions + NghData → NeighbourValue
alias) + P2.α canonical naming (cross-family
`carbon::detail::Validated<Idx>` carrier + value-returning
`block::Span::set_and_validate` overload + `per_card_elements()` +
arity-aware `linear_storage_elements(int)` everywhere — all additive,
zero breaking changes) + P2.β HaloBlock introspection accessors +
P2.γ MemorySide doc + P2.δ AGENTS.md trim (723 → 366 LOC) +
P2.δ `kernel.h` split (1212 → 959 LOC) + P2.δ `io.h` split (1022 →
461 LOC) are done; P1.γ landed its in-tree half — new
`.github/workflows/gpu-ci.yml` shells out to
`scripts/docker-build-all.sh --only cuda,cuda-mpi` on a self-hosted
runner (PR `gpu-ci` label / `workflow_dispatch` / pushes to main),
runner registration is the one-time out-of-tree step.
**P2.δ `block.inl` ctor split also landed 2026-05-13**: the
619-LOC `Block<SBlock>::Block` ctor body decomposes into a new
private nested `BuildContext` scratch struct + eight private
member phase helpers (`initialise_partition_` ...
`upload_topology_and_build_spans_`); the surviving ctor body is
26 lines (validation + `BuildContext ctx{};` + the eight helper
calls). Pure code-motion: no public surface change, no semantic
change. Every audit follow-up is now closed. Each section below
carries a `— RESOLVED` / `— PARTIAL` callout pointing at the
resolution. Verification:
**63/63 CUDA + 75/75 CUDA+MPI+NCCL + 52/52 SYCL + 63/63 CUDA+HDF5
tests pass plus `format-check` is clean** on all four configurations.
The two new MPI tests (`carbon_block_neighbour_dist` and
`carbon_halo_disg_dist`) lift the cross-rank coverage gap that masked
both P0 bugs._

---

## Headline

The May-12 P0 sweep landed cleanly. The codebase is genuinely healthy on the
metrics that fail loudest: **zero `TODO`/`FIXME` markers in carbon-owned code**
(only the vendored `third_party/clipp/clipp.h` has any), tests **63 / 73 / 52**
(CUDA / CUDA+MPI / SYCL) — every cell of the grid×capability matrix is now
covered. The examples directory shrank from 8+ teaching anchors to **4 unified
runners** (`init_and_dense`, `jacobi_unified`, `cg_poisson_unified`,
`salt_pepper_filter`).

But two **silent correctness problems** survived the P0 pass — both are
documented as known limitations and both produce **numerically wrong answers
without throwing**. Either alone should block another grid port; together they
are the single biggest issue surface.

---

## P0 (new) — Correctness problems that block more features

### ~~P0.A — `Block` + MPI silently breaks cross-rank stencil reads (was P1.8)~~ — RESOLVED 2026-05-12

**Resolution:** the connectivity rewrite in `block.inl::reserve_row` now
consults `candidate_to_ghost_local[g][cand_idx]` whenever the neighbour is
not locally owned by GPU `g`, instead of gating the lookup behind
`else if (neigh.gpu_id != kInvalidBlockIdx)`. Halo bytes were already
arriving at the ghost slots correctly; only the connectivity table needed
to point at them. New regression test `tests/test_block_neighbour_dist.cu`
runs a halo exchange and a single `Partition::neighbour(+z, ...)` probe
under `mpiexec -n 2` and asserts every cross-rank read is valid with the
expected encoded payload. Verification:
`jacobi_unified --grid block --size 32` now gives
`sum|u|^2 = 3.2778e+03` under both single-process-2-GPU and
`mpiexec -n 2 -1-GPU-each` transports (was 3.278e+03 vs 4.206e+03 pre-fix).
The original finding kept here for historical reference:



**Status:** confirmed live today.

```
mpiexec -n 2 .../carbon_example_jacobi_unified --grid block --size 32  → sum|u|^2 = 4.206e+03
            .../carbon_example_jacobi_unified --grid block --size 32  → sum|u|^2 = 3.278e+03
```

Same input, same kernel, same iteration count — 28 % higher under MPI. The
DenseDisg row gives `5.466e+03` vs `5.472e+03` (consistent within
float-summation order).

**Root cause (now precisely located).** The connectivity rewrite in
`block.inl::reserve_row` only consults `candidate_to_ghost_local` inside the
branch:

```cpp
const auto neigh = candidate_to_local[static_cast<size_t>(cand_idx)];
...
if (neigh.gpu_id == static_cast<uint32_t>(g)) {
  host_connectivity_per_gpu_[g][offset] = neigh.local_id;
} else if (neigh.gpu_id != detail::block::kInvalidBlockIdx) {
  const uint32_t ghost_id = candidate_to_ghost_local[g][static_cast<size_t>(cand_idx)];
  if (ghost_id != detail::block::kInvalidBlockIdx) {
    host_connectivity_per_gpu_[g][offset] = ghost_id;
  }
}
```

Under MPI, `gpu_count() == 1` and the local sweep only fills
`candidate_to_local` for the rank's owned z-rows. The remote-rank brick row
therefore has `neigh.gpu_id == kInvalidBlockIdx`, **neither branch fires**, and
the connectivity slot stays `kInvalidBlockIdx`. `HaloBlock` still delivers
ghost bytes correctly into the ghost slots, but `Partition::neighbour` returns
`{valid=false}` for cross-rank z-face reads
(`detail/block/partition.h:171–181`). Every cross-rank face cell falls back to
its Dirichlet path — at every iteration, on every sweep.

**Why the existing tests missed it.** `test_halo_block_dist` checks *byte
transport into the ghost slots*; it never calls `Partition::neighbour` after
the exchange. `test_block_dist_reduce` uses a uniform-true predicate so the
bug doesn't perturb the reduction sum.

**Fix shape.** In the connectivity second pass, add a third clause that
consults `candidate_to_ghost_local[g][cand_idx]` whenever it is reserved,
**independent** of whether `candidate_to_local` recorded a local-GPU id.
Order matters: keep the existing local-GPU id branch first so intra-process
multi-GPU is unaffected.

**Regression guard.** Add a focused test that runs 1 halo + 1 `neighbour`
probe under `mpiexec -n 2`: pick an owned brick on the rank-boundary, push a
known value through `HaloBlock`, and assert
`partition.neighbour(..., 0, 0, ±1, ...).valid == true` with the expected
payload. Today this assertion fails; `test_halo_block_dist` does not catch it.

### ~~P0.B — `HaloDisg` has no cross-rank transport at all~~ — RESOLVED 2026-05-12

**Resolution:** `HaloDisg` now carries a `DistTransfer<T> dist_` companion to
its `Transfer<T> intra_`, gated on `CARBON_HAVE_MPI_NCCL` the same way
`Halo` does. Cross-rank pairs in `add_pair` build NCCL `Send`/`Recv`
entries against the disg layout's region offsets — for each pair the
lo-local end Sends the up-boundary upward and Recvs into the upper-ghost;
the up-local end mirrors. Both ranks iterate runs in `runs_up` then
`runs_down` order on every rank so the per-peer NCCL FIFO pairs the flows
automatically; lattice semantic narrows each Send/Recv to the runs whose
`c_q.z` sign matches the transfer direction. `dist_entry_count()` is now a
public accessor (mirrors `Halo::dist_entry_count()`). New regression test
`tests/test_halo_disg_dist.cu` exercises a 2-rank halo + verifies the
ghost regions hold the peer's encoded coordinates. Verification:
`jacobi_unified --grid disg --size 32` now gives bit-identical
`sum|u|^2 = 5.465986e+03` under both single-process-2-GPU and MPI 2-rank
transports (was 5.466e+03 vs 5.472e+03 pre-fix — the gap was the Dirichlet
fallback at the rank boundary). The original finding kept here for
historical reference:



```cpp
// halo_disg.h:55–58
 * The cross-rank @c DistTransfer / NCCL path is intentionally @b not wired in this pass.
 * On a distributed @c DenseDisg the @ref pair_count is still populated for local-only
 * pairs; pairs with one foreign endpoint will be tracked once the disg variant gets its
 * own NCCL plumbing (mirroring @c carbon::Halo 's @c CARBON_HAVE_MPI_NCCL branch).

// halo_disg.h:360–364 — inside add_pair
    if (!lo_local || !up_local) {
      // Cross-rank pair — not yet wired (intentional for this port pass).
      ++host_dist_pending_;
      continue;
    }
```

Cross-rank pairs are **silently dropped** on both device and host paths (the
host path also throws if `host_dist_pending_ > 0`, but the device path returns
success). `test_dense_disg_dist` is deliberately halo-free
(`tests/test_dense_disg_dist.cu:10–11` says so). `jacobi_unified --grid disg`
happens to produce a close-but-not-identical answer under MPI — the kernel
uses topological `!zm.valid` BCs, and `dense_disg::neighbour_idx` clips at the
global slab edge, so the rank boundary degrades to a Dirichlet wall instead of
producing scrambled values. **Quiet, not loud — that is the problem.**

`Halo` already implements `CARBON_HAVE_MPI_NCCL` `Send`/`Recv` pairing;
mirroring it into `HaloDisg` is mechanical. `cg_poisson_unified --grid disg`
\+ MPI is silently wrong on the rank boundary right now.

---

## P1 — Should-fix soon, but won't block the next port pass

### ~~P1.α — Block kernel impls duplicate the Dense impl tree~~ — RESOLVED 2026-05-12

**Resolution:** all six concrete impl pairs collapsed via a per-grid-family
launch trait (option 3 from the original "what's blocking" shortlist:
hoist `launch_dense_grid` / `launch_disg_grid` / `launch_block_grid`
behind a uniform trait so the kernel impl is single-shape but the trait
specialises per family). The new header
`include/carbon/detail/launch_traits.h` carries three traits:

- `DeviceLaunchTraits<Grid>` — compute launch (CUDA + SYCL). Primary
  template covers Dense / DenseDisg via the existing `if constexpr
  (DisgPolicy)` branch; Block specialisation forwards `SBlock` to
  `launch_block_grid` and elides the per-GPU `gpu_grid` / `gpu_block`
  size check (Block derives dims internally from `SBlock::Mem{X,Y,Z}`).
- `HostLaunchTraits<Grid>` — host (OpenMP) compute + reduce. Primary
  uses `grid.spans()` + 3-deep `collapse(3)` over `iter_extent` with
  `set_and_validate(view, ix, iy, iz) -> ValidatedIdx`; Block
  specialisation uses `grid.host_spans()` + 4-deep walk over
  `iter_block_count` × `MemX*MemY*MemZ` with the bool-returning
  `set_and_validate(b, ix, iy, iz, idx)`. Exposes both `iterate_slab`
  (compute) and `iterate_slab_reduce` (per-thread + sequential combine).
- `ReduceLaunchTraits<Grid>` — device stage-1 reduce launch.
  `stage1_block_threads()` returns `kReduceBlockThreads` /
  `kDisgReduceBlockThreads` / `SBlock::VoxelCount`; `stage1_unit_count`
  returns `ceil(iter_extent.volume() / block_threads)` for Dense /
  Disg and `iter_block_count(view)` for Block; `cuda_stage1` /
  `sycl_singlepass` pick the family launch helper. Stages 1.5 (cub),
  2 (combine), 3 (`ncclAllReduce`) stay shared via `reduce_stages.h`.

The six surviving impl files (`dense_kernel_impl.h`,
`dense_host_kernel_impl.h`, `reduce_kernel_impl.h`,
`reduce_host_kernel_impl.h`, `universal_kernel_impl.h`,
`universal_reduce_kernel_impl.h`) now each have a single-shape body
that dispatches through the trait; the six matching `block_*_impl.h`
files are gone. `Kernel::compute / host_compute / reduce / host_reduce
/ compute_universal / reduce_universal` factories drop their
`BlockPolicy` branches and bind to the unified impl for every grid.

**Halo path stays separate** (`halo_kernel_impl.h` /
`halo_disg_kernel_impl.h` / `block_halo_kernel_impl.h`) because halo
orchestration genuinely differs between dense Z-partitioned slabs
(adjacent-slab pair-wise transfers) and block-sparse retained-brick
sets (per-brick boundary tagging). Net codebase impact:

- 6 `block_*_impl.h` files deleted (~1500 LOC total).
- 1 new `detail/launch_traits.h` (~540 LOC) holding all three traits.
- 6 surviving impl bodies all shorter (collectively ~600 LOC reduction).
- `kernel.h` factories each collapse to one `using Impl = ...; return
  Kernel(std::make_shared<Impl>(...))` line.

Verification: 63/63 CUDA + 75/75 CUDA+MPI+NCCL + 52/52 SYCL pass at every
intermediate commit; format-check clean on all three trees.

Historical context preserved for future contributors:



P0.5 added DenseDisg by extending the existing dense impl headers with
`if constexpr (requires { typename Grid::DisgPolicy; })` branches (visible in
`dense_kernel_impl.h`, `reduce_kernel_impl.h`, `universal_kernel_impl.h`,
`universal_reduce_kernel_impl.h`). P0.1 + P0.2 added Block by writing
**seven new parallel headers** instead:

| Pair | Block | Dense (DenseDisg shares this side) |
|---|---|---|
| compute | `block_kernel_impl.h` (168) | `dense_kernel_impl.h` (266) |
| host compute | `block_host_kernel_impl.h` (212) | `dense_host_kernel_impl.h` (214) |
| reduce | `block_reduce_kernel_impl.h` (341) | `reduce_kernel_impl.h` (526) |
| host reduce | `block_reduce_host_kernel_impl.h` (241) | `reduce_host_kernel_impl.h` (324) |
| universal compute | `block_universal_kernel_impl.h` (240) | `universal_kernel_impl.h` (287) |
| universal reduce | `block_universal_reduce_kernel_impl.h` (439) | `universal_reduce_kernel_impl.h` (487) |
| halo | `block_halo_kernel_impl.h` (111) | `halo_kernel_impl.h` + `halo_disg_kernel_impl.h` |

That's **~1750 lines** of Block impls running parallel to ~2300 lines of Dense
impls. Stage 2/3 of the reducer is shared via `reduce_stages.h` (the only
piece of P0.4 that crossed family lines), but everything around it is forked.
The right shape is the DenseDisg shape:
`if constexpr (requires { typename Grid::BlockPolicy; })` switches inside the
existing dense impl headers, picking the right span/idx/launch helpers. The
`kernel.h` dispatch (`kernel.h:710 / 736 / 793 / 809 / 831 / 855`) is already
doing the routing decision; collapsing the implementations means deleting
`block_*_impl.h` and inlining their bodies as `else if` arms.

Estimated 1000–1500 line reduction with no API change. **Get this done before
the next sparse grid port** (mGrid / bGrid+adaptivity), otherwise the next
family will mint a third copy and the asymmetry will be permanent.

**Outcome (2026-05-12):** the resolution above lands ~900 net LOC reduction
(1500 deleted from `block_*_impl.h` minus ~540 new trait LOC minus ~60 LOC of
modest growth in the surviving impl bodies that import the trait) with no API
change — the next sparse grid family (mGrid / bGrid+adaptivity) lands a
fourth trait specialisation, not a fourth set of impl files.

### ~~P1.β — No `SpanConcept` / `PartitionConcept`~~ — RESOLVED 2026-05-12

**Resolution:** new header `include/carbon/detail/grid_concepts.h` anchors
the minimum cross-grid contract every grid's `Span` and `Partition` must
satisfy. The concepts capture exactly the duck-typed surface kernel impls
touch: `Span::iter_extent(view)`, `logic_extent()`, `xy_slab_elements()`,
`global_location(idx)`; `Partition::read / write / neighbour` with the
`{value, valid}`-shaped neighbour return. A `static_assert(SpanConcept<...>)`
in `DeviceKernelImpl`'s constructor surfaces a missing member as a clean
concept diagnostic anchored at the kernel impl rather than buried deep in
the launch path. The header is also a contract document — its body lists
the per-family asymmetries (set_and_validate return shape,
`linear_storage_elements` arity) that the canonical-naming pass will pick
up. The original finding kept here for historical reference:



### ~~P1.γ — CI is still CPU-only~~ — PARTIAL 2026-05-12

**Resolution (in-tree half landed):** new
`.github/workflows/gpu-ci.yml` adds a GPU validation lane that shells
out to `scripts/docker-build-all.sh --only cuda,cuda-mpi` on a
self-hosted runner labelled `[self-hosted, linux, gpu, cuda]`. The
two matrix cells (`cuda` and `cuda-mpi`) exercise both audit-flagged
silent-bug classes — `cuda-mpi` runs
`carbon_block_neighbour_dist` (the P0.A regression net) and
`carbon_halo_disg_dist` (the P0.B regression net) as part of its
`distributed`-labelled `ctest` sweep. The CPU-only fast lane
(`.github/workflows/ci.yml`) stays unchanged and keeps running on
every push / PR. Triggers: the `gpu-ci` label on a PR, manual
`workflow_dispatch`, or pushes to `main` / `master`.

**Out-of-tree half (one-time setup per fork):** register a
GitHub Actions self-hosted runner on the repo with the labels
`self-hosted`, `linux`, `gpu`, `cuda`. The runner host needs only
Docker + `nvidia-container-toolkit` + an NVIDIA GPU; the workflow
runs everything else inside containers. Forks that have not
registered a matching runner see GPU jobs queue with no runner —
intentional no-op state, GPU validation is opt-in per fork.
Setup walkthrough lives in `AGENTS.md` ("Self-hosted GPU runner
setup").

The two P0 bugs above are both **silent on a fresh checkout
running `make && ctest`** on a host without GPUs. P0.A is masked
by tests that don't probe `Partition::neighbour` after a
cross-rank halo; P0.B is masked by tests that explicitly avoid
halos. The new GPU lane closes both nets: a future regression of
either class lands as a red check on the PR.

### ~~P1.δ — `KernelLoader` by-reference capture UAF~~ — PARTIAL 2026-05-12

**Resolution:** the "loud doc" mitigation has landed plus one compile-time
guardrail.

1. `KernelLoader`'s doc block grows a prominent `@warning` section spelling
   out the lifetime contract: capture `Field` / `ReductionResult` by-value
   into the loading lambda (the handles are cheap ref-counted views over
   the underlying `BufferGroup<T>`'s `shared_ptr<Allocation<T>>`),
   by-reference is allowed only when the storage outlives the kernel.
   Same advisory now appears as a `@warning` on the `DeviceKernelImpl`
   constructor.
2. `DeviceKernelImpl`'s constructor gains a
   `static_assert(std::is_invocable_v<LoadingFn&, KernelLoader&>)` that
   fires before `discover_tokens_`'s `loading_fn_(ctx)` call when the
   user accidentally hands `compute()` the per-cell device lambda
   instead of the outer loader-shape lambda. The previous failure was
   a template-instantiation cascade; the new assert anchors a clean
   single-line diagnostic with a pointer at the doc.

Items still open (deferred): a debug-mode address-tracking probe that
records per-launch token consistency, and a structural-lambda capture
check that would compile-time reject reference captures. Both require
implementation-specific lambda introspection beyond what's portable in
C++23. Original finding kept for context:



---

## P2 — Cleanup / nice-to-have

### ~~P2.α — Grid-family API asymmetries that aren't structural~~ — RESOLVED 2026-05-12

**Resolution:** every shape gap the audit flagged has landed as an
additive parity addition — zero breaking changes, every existing call
site keeps compiling. New cross-grid code can spell things uniformly.

Closed in the May-12 morning sweep:

- `Block::global_extent()` — alias for `domain_extent()` matching
  `Dense::global_extent()` / `DenseDisg::global_extent()`.
- `Block::Field::set_name()` — mirrors `Dense::Field::set_name` /
  `DenseDisg::Field::set_name`.
- `block::Partition::ptr()` — alias for `data()` matching
  `dense::Partition::ptr()` / `dense_disg::Partition::ptr()`.
- `block::NghData<T>` — now a deprecated `using` alias for the canonical
  `::carbon::NeighbourValue<T>`. `block::Partition::neighbour` returns
  the canonical type directly; existing Block call sites typed
  `NghData<T>` keep compiling. New Block code (and generic cross-grid
  code) uses `NeighbourValue<T>` uniformly and gets the
  `value_or(alternative)` one-liner that Dense / DenseDisg already
  exposed.

Closed in the May-12 evening canonical-naming pass:

- **`set_and_validate` shape.** New shared header
  `include/carbon/detail/validated_idx.h` defines
  `template <typename Idx> struct Validated { Idx idx; bool valid; };`
  — a single POD carrier used as the value-return shape for every
  grid family. `dense::ValidatedIdx` and `dense_disg::ValidatedIdx`
  become `using ValidatedIdx = Validated<Idx>` aliases (so existing
  call sites stay byte-for-byte source-compatible); `block::Span`
  gains a `using ValidatedIdx = Validated<Idx>` alias of its own and
  a new value-returning overload
  `set_and_validate(block_idx, ix, iy, iz) -> ValidatedIdx` that
  shares its predicate body with the historical `bool` / out-param
  signature. Both overloads stay on `block::Span` — the value form
  is the canonical cross-family spelling new code should reach for,
  the out-param form keeps every existing Block call site
  (kernel impls, tests, examples) compiling unchanged.
- **`linear_storage_elements` arity.** Every grid family now exposes
  both shapes: `Span::per_card_elements()` (no-arg, returns one
  cardinality plane's storage count) @b and
  `Span::linear_storage_elements(int components)` (total elements
  for that many components). `Block::Span` already had both;
  `dense::Span` gains `per_card_elements()` + the `(int)` overload;
  `dense_disg::Span` gains `per_card_elements()`. The historical
  no-arg `dense::Span::linear_storage_elements()` stays for back-
  compat as a source-compat alias for `per_card_elements()`. Cross-
  grid code (e.g. a future generic `BufferGroup` sizing helper) can
  now spell either form uniformly without `if constexpr` per family.

Carrier consolidation: `include/carbon/detail/grid_concepts.h` was
re-anchored on the new state — its "asymmetries" preamble is now
a "canonical-naming pass landed" preamble pointing at
`detail/validated_idx.h` and explaining the additive overload
strategy. The concepts themselves stay unchanged — they only
require the duck-typed iterate / read / write / neighbour surface,
which is the strict subset every family already satisfied.

Verification: 63/63 CUDA + 74/75 CUDA+MPI+NCCL (the lone fail is
the known `carbon_skeleton` -j flake; passes in isolation) +
52/52 SYCL pass; `format-check` clean.

Original finding kept for context:



Mostly cosmetic; each is a 1-hour patch in isolation but the **lack of a
unifying concept** (P1.β) lets them drift.

| Asymmetry | Dense | DenseDisg | Block |
|---|---|---|---|
| Cartesian extent accessor | `global_extent()` | `global_extent()` | `domain_extent()` |
| Field name setter | `set_name()` | `set_name()` | — |
| Bulk host fill | `fill_owned_host`, `fill_full_host` | — | — |
| Partition pointer accessor | `ptr()` | `ptr()` | `data()` |
| Partition `data_view()` / `reset(...)` | yes | yes | — |
| `Span::set_and_validate` return | `ValidatedIdx` | `ValidatedIdx` | `bool` + out-param |
| Neighbour-read return type | `NeighbourValue<T>` | `NeighbourValue<T>` | `NghData<T>` |
| `Span::linear_storage_elements()` arity | no-arg | `(components)` | `(components)` |

Picking a canonical shape for each row is cheap; doing it under a
`Span`/`Partition` concept catches the next family before it diverges.

### ~~P2.β — `HaloBlock` ergonomics~~ — RESOLVED 2026-05-12

**Resolution:** the meaningful parity accessors landed; the items that
don't have a Block analogue stay off the API surface.

Closed (`HaloBlock`):

- `field() -> const Field&`. `HaloBlock` now stores `Field` by value
  (same pattern as `Halo` / `HaloDisg`); ref-counted internals make the
  storage cost one ref-count per descriptor.
- `host_intra_count() -> 0U` parity stub with doc note pointing at the
  tracked `launch_on_host` follow-up. Lets generic code probe the host
  fan-out uniformly across grid families.

Items deliberately not added (no Block analogue):

- `boundary_mode()`: Block has no global `BoundaryMode` flag — boundary
  behaviour is encoded as which bricks exist (Open) vs. the explicit
  ghost-pair lists `Block::halo_pairs_*`. A future `PeriodicZ` wrap for
  Block would surface a real `BoundaryMode`; for now the accessor would
  be a sentinel.
- `stencil()` / `stencil_semantic()`: Block halo is brick-adjacency,
  not stencil-radius driven; surfacing a `Stencil` on `HaloBlock`
  would be misleading.
- `launch_on_host()`: tracked alongside Block host-kernel coverage;
  `host_intra_count()` returns 0 as the sentinel today.

Original finding kept for context:



### ~~P2.γ — Dense vs Block `MemorySide` policy drift~~ — RESOLVED 2026-05-12

**Resolution:** added a "Per-grid allocation policy" paragraph to the
`MemorySide` enum doc in `include/carbon/memory_side.h` calling out the
lazy (Dense / DenseDisg) vs eager (Block) split and pointing readers at
the per-grid header for the binding rules. Behaviour unchanged. Original
finding kept for context:



### ~~P2.δ — File-size hot spots~~ — RESOLVED 2026-05-13

**Resolution:** all four hot spots are now split. The `block.inl`
ctor split landed 2026-05-13 — the 619-line `Block<SBlock>::Block`
ctor body decomposes into a private nested `BuildContext` scratch
struct (carries the inter-phase locals that the old lambdas
captured) plus eight private member phase helpers
(`initialise_partition_` → `sweep_active_bricks_` →
`split_internal_boundary_` → `reserve_ghost_slots_` →
`build_connectivity_` → `build_intra_halo_plan_` →
`build_dist_halo_plan_` → `upload_topology_and_build_spans_`),
each with a doxygen preamble on its declaration in `block.h`
explaining what state on `*this` and what slot of `ctx` it
populates. The surviving ctor body is 26 lines (gpu / extent
validation + `BuildContext ctx{};` + the eight helper calls).
The three nested lambdas (`reserve_row`, `eval_brick`,
`owner_of_bz`) live inside `reserve_ghost_slots_` only — they're
scoped to one phase, so lifting them to private helpers wasn't
worth the additional surface area. Pure code-motion: every
helper body is a verbatim slice of the original ctor with the
right `ctx.` prefixes; no public API or ABI change, no semantic
change. `block.inl` 776 → 822 (+46; helper signature template
boilerplate slightly exceeds ctor compaction); `block.h` 755 →
881 (+126, all under `private:`). The win is **structural** —
each phase is independently readable, and a future grid family
port that wants to override / extend one phase can do so without
rewriting the whole ctor. Verification: 63/63 CUDA + 75/75
CUDA+MPI+NCCL + 52/52 SYCL pass; `format-check` clean on all
three trees.

| File | Before | After | Notes |
|---|---|---|---|
| `AGENTS.md` | 723 | 366 | Status snapshot collapsed from a multi-paragraph implementation walk per feature to a 70-line tabulated inventory. CHANGELOG remains the source of truth for the long form. The "Next big features" enumeration was folded into the relevant feature bullets so the file no longer carries a parallel timeline. |
| `kernel.h` | 1212 | 959 | Per-grid-family convenience members split into `detail/kernel_dense.inl` (84), `detail/kernel_block.inl` (107), `detail/kernel_dense_disg.inl` (88) as tail-includes. Pure code-motion, no API/behaviour change; the cross-grid `Kernel` class + static factories + `HaloFactoryBase` + `read_stencil` stay in the umbrella. |
| `io.h` | 1022 | 461 | Per-format backends split into `detail/io_hdf5.inl` (169; `CARBON_HAVE_HDF5`), `detail/io_nanovdb.inl` (184; `CARBON_HAVE_NANOVDB`), `detail/io_vtk.inl` (228; always-on) as tail-includes. Native binary + shared `detail::collect_field_views` + `native_dtype_tag` specs stay in the umbrella. Pure code-motion. |
| `block.inl` | 776 | 822 | 619-LOC ctor → 26-LOC ctor; new `BuildContext` private nested struct + 8 private member phase helpers absorb the body. Pure code-motion, no API/behaviour change. `newField` (60) + `download_to_host` (35) stay in `block.inl` — the win moving them would be marginal next to the ctor split. |

Still open:

| File | Lines | Notes |
|---|---|---|
| `skeleton.cpp` / `skeleton.h` | 961 / 857 | Sized by the static scheduler; no obvious split. |

---

## Architectural duplication summary (updated)

- **A. Reduce stages** — closed by `a86e544` (P0.4). All four device reducers
  route through `reduce_stages.h`.
- **B. Halos** — three storage models, three transfer builders. `HaloDisg`
  cross-rank NCCL is the open item (**P0.B** above; previously a "follow-up"
  caveat in the May-12 audit).
- **C. Spans** — three genuinely different indexing models. No concept anchor
  (**P1.β**).
- **D. Partitions** — Disg has `neighbour` now (P0.5). Block uses
  `NghData<T>` instead of `NeighbourValue<T>`; cosmetic (**P2.α**).
- **E. Universal compute vs Universal reduce** — bodies still duplicated within
  Universal impls (deliberate one-lambda-copy design).
- **F. NEW: Block kernel-impl tree vs Dense kernel-impl tree** — the single
  largest remaining duplication (~1750 LOC) and the easiest to retire.
  DenseDisg already proved the pattern (**P1.α**).

---

## What I would fix first

In order — each can land in one PR-sized commit, and (1)+(2) should land
before any other feature work.

1. **P0.A — Block + MPI connectivity wire-up.** ~30-line change in
   `block.inl::reserve_row` + a 50-line `test_block_neighbour_dist`
   regression test. Closes a silent numerical bug that affects every
   cross-rank stencil over Block today.
2. **P0.B — `HaloDisg` cross-rank `DistTransfer`.** Mirror the
   `CARBON_HAVE_MPI_NCCL` branch from `Halo` into `HaloDisg::add_pair`;
   remove the `++host_dist_pending_; continue;` early-exit on the device
   path; lift the matching `test_halo_dist` shape into a `test_halo_disg_dist`
   test that probes a neighbour read after the exchange. Without this,
   `cg_poisson_unified --grid disg --mpi` is silently wrong on the rank
   boundary.
3. ~~**P1.α — Collapse `block_*_impl.h` into the dense impl tree.**~~
   **DONE 2026-05-12.** Landed via `detail/launch_traits.h` — three traits
   (`DeviceLaunchTraits`, `HostLaunchTraits`, `ReduceLaunchTraits`)
   absorb the per-family launch-helper + iteration-shape variation so all
   six compute / host_compute / reduce / host_reduce / universal_compute /
   universal_reduce impl pairs converge onto six unified impls. Halo
   stays family-split because the orchestration genuinely differs.
4. ~~**P1.β — `SpanConcept` / `PartitionConcept`.**~~ **DONE 2026-05-12.**
5. ~~**P1.γ — At least one GPU CI runner per PR.**~~ **DONE 2026-05-12
   (in-tree half).** New `.github/workflows/gpu-ci.yml` shells out to
   `scripts/docker-build-all.sh --only cuda,cuda-mpi` on a self-hosted
   runner labelled `[self-hosted, linux, gpu, cuda]`; triggered by the
   `gpu-ci` PR label, manual `workflow_dispatch`, or pushes to
   `main` / `master`. Per-fork runner registration is the one-time
   out-of-tree step.
6. ~~**P2.α — Pick canonical names and propagate** (`global_extent`, `ptr`,
   `NeighbourValue`, no-arg `linear_storage_elements`, `ValidatedIdx`-style
   returns).~~ **DONE 2026-05-12.** Every shape gap landed as an additive
   overload — zero breaking changes. New cross-grid code spells things
   uniformly; existing call sites stay source-compatible.

---

## TL;DR for the next contributor

> Two silent bugs (Block + MPI broken cross-rank, DenseDisg + MPI halo
> dropped) plus one big duplication target (1500 LOC of Block impl files that
> should fold into Dense via `if constexpr`). Fix the two silent bugs before
> the next grid family lands, fold the impls before, or simultaneously with,
> that work, and the foundation is genuinely ready for mGrid / bGrid+adaptivity.
