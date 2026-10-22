# carbon — third-pass follow-up audit (2026-05-13)

_Companion to `current_status.md` (the original pre-pass audit, 2026-05-12)
and `docs/audit_2026_05_followup.md` (the second pass, also 2026-05-12,
**fully closed** as of `05afcdb`). This is a fresh sweep after the
`block.inl` ctor split (`05afcdb`, the last open second-pass item) landed._

_Branch surveyed: `sycl-cuda`._

## TL;DR

The codebase remains in **genuinely good shape** — there is no architectural
rot, the SYCL/CUDA parity story holds, and every P0 from the original audit
plus every follow-up item from the second pass is resolved. The complexity
that used to live in `block.inl` has been tamed.

This third pass surfaces **one real correctness-class asymmetry**, **two
new structural file-size hot spots** that have surfaced now that
`block.inl` is split, **a small cluster of stale documentation** (test
counts and phase-of-life comments that lag the code), and **a handful of
test-coverage shapes** that the P0.A / P0.B pattern from the second pass
teaches us not to dismiss.

Headline findings, in priority order:

1. **P0.α — `BlockHaloKernelImpl` does not override `launch_on_host`.** A
   `Kernel` produced by `Block::Field::make_halo_kernel()`, when dispatched
   through `Kernel::launch_on_host`, falls through to the
   `KernelImpl::launch_on_host` default and **throws** `std::logic_error`.
   Dense + DenseDisg halos override the host path (intra-process `std::memcpy`
   between adjacent host mirrors). The asymmetry breaks heterogeneous
   `std::vector<Kernel>` pipelines that mix host compute + halos on Block,
   and contradicts the AGENTS.md claim that "all three families work the
   same in a `vector<Kernel>` iterated via `k.launch(stream, view)`".
2. **P1.α — File-size hot spots: `Skeleton::compile` (~282 LOC) and
   `Skeleton::apply_occ_` (~325 LOC).** With `block.inl`'s ctor split, these
   are now the top two control-flow-heavy functions in the codebase. Both
   have high cyclomatic complexity (nested loops × `OverlapMode` switch ×
   token scans × graph mutation) and they're the orchestration layer
   carbon's correctness story depends on most.
3. **P1.β — Concept anchoring claim drifts from reality.**
   `include/carbon/detail/grid_concepts.h` advertises `SpanConcept` /
   `PartitionConcept` as "non-binding" but the P1.β closure narrative in
   the second-pass audit says they're anchored in `DeviceKernelImpl`'s
   ctor. In practice only **`SpanConcept`** is anchored, and only under
   `if constexpr (grid_has_dense_iteration_v<Grid>)` — i.e. Dense /
   DenseDisg only. **`PartitionConcept` is not anchored anywhere in
   production code**, and `Block`'s `Span` deliberately fails `SpanConcept`
   (it speaks the brick-iteration contract instead).
4. **P1.γ — Doc drift across AGENTS.md, current_status.md, and three
   header comments.** AGENTS.md says "9-test delta vs CUDA" — actual is 11.
   AGENTS.md says "+7 distributed tests on each backend" — actual is 12.
   `dense_disg.h` class blurb says kernel factories "are intentionally not
   exposed yet" — they're exposed on lines 190–246 of the same file.
   `sycl/dense_kernel.h` says SYCL dense kernel "lands in phase 2.4" — phase
   2.4 landed; `sycl/dense_launch.h` is alongside. `test_skeleton.cu` says
   non-`None` `OverlapMode` "throws 'not yet implemented'" — OCC has
   landed. `overlapped_stencil.h::first_` member comment says it gates the
   halo-event wait — the in-function comment correctly says it no longer
   does.
5. **P1.δ — Test-coverage gaps the P0.A/P0.B shape teaches us about.**
   Five gaps where a silent regression would survive the current matrix:
   no `DenseDisg × make_universal_reduce_kernel` test; no MPI test for
   VTK / HDF5 / nanoVDB I/O; no MPI test for non-`Sum` reductions; no I/O
   round-trip test for DenseDisg or Block fields (only Dense is covered);
   no MPI test for the `OverlappedStencil` orchestration layer or for any
   `Skeleton OverlapMode` × MPI shape.
6. **P2.α — Silent-empty accessor pattern on `Block`.** Several
   `Block<SBlock>` accessors that take a `gpu_id` (e.g. `host_origins`,
   `host_bitmasks`, `halo_pairs_*`) return a `static const empty` value on
   out-of-range `gpu_id` instead of throwing. The doxygen documents this
   ("out-of-range yields an empty vector") but it's a footgun-class API
   shape — a stale loop bound on the user side surfaces as "the field has
   no blocks" downstream, not as a clear precondition failure.
7. **P2.β — Residual halo-impl duplication.** The three halo kernel impl
   headers (`halo_kernel_impl.h`, `halo_disg_kernel_impl.h`,
   `block_halo_kernel_impl.h`) are ~85% identical now. P1.α
   intentionally left them split because halo orchestration is genuinely
   different across families, but with P0.α (Block host path) closed
   they'd be a clean CRTP base candidate.
8. **P2.γ — CUDA-only test gate may have grown beyond what P0.2 closed.**
   Eleven tests are gated by `carbon_add_cuda_only_unit_test`. P0.2
   closed `block_reduce` / `block_dist_reduce`, but the gate set has since
   grown to include `skeleton_block`, `skeleton_block_extended`,
   `block_host_mirror`, `block_host_parity`, `block_universal`, etc. Each
   needs an explicit rationale row in the docs — today only "BlockReduce"
   is called out, and the AGENTS.md narrative still says "9 Block tests
   today" which understates the reality.

Less-urgent items, configuration drift, and a verified list of what is
**not** worth fixing (intrinsic differences) follow in the body.

---

## P0 — Real correctness/silent-failure class

### ~~P0.α — `BlockHaloKernelImpl::launch_on_host` is missing — host pipelines on Block throw~~ — RESOLVED 2026-05-13

**Resolution.** `HaloBlock` and `BlockHaloKernelImpl` now expose a real host
halo path; a `Kernel` produced by `Block::Field::make_halo_kernel()`
dispatches through `Kernel::launch_on_host` identically to the Dense and
DenseDisg halo kernels. Closed in the same commit as the P0.α anchor below
(`<tbd hash>`):

- `HaloBlock` ctor now populates a parallel `host_intra_` vector
  alongside the existing device `Transfer<T>` build — each entry is a
  `(src_gpu, dst_gpu, src_offset, dst_offset, count)` brick descriptor
  that reuses the device-side offset math (Block's host mirror is sized
  identically to the device buffer; same brick layout, same offsets).
- `HaloBlock::launch_on_host(Stream)` waits on the stream, then replays
  the `host_intra_` entries via `std::memcpy` between adjacent GPUs'
  host mirrors. Throws `std::invalid_argument` when the field has no
  host mirror (`MemorySide::Device`) and `std::runtime_error` when the
  layout has any cross-rank entries (the MPI `Sendrecv` fallback is the
  same follow-up Dense `Halo::launch_on_host` already advertises).
- `BlockHaloKernelImpl::runs_on_host()` now returns `true`;
  `BlockHaloKernelImpl::launch_on_host(stream, view)` forwards to
  `halo_.launch_on_host(stream)`.
- `HaloBlock::host_intra_count()` returns the real count (equals
  `pair_count()` on intra-process layouts; the sentinel-`0` stub is
  retired).

Verified by `test_halo_block_host` (new): builds a `Block<8,8,8>` over a
sphere predicate on a 32-cube split across every visible GPU, fills the
host mirror with `encode(gx, gy, gz)` on owned bricks and a different
sentinel on ghost slots, runs the halo on host, then for every
adjacency walks `halo_pairs_upward` / `halo_pairs_downward` and
confirms every destination ghost slot is voxel-by-voxel equal to the
source owned brick. Repeats with `MemorySide::Host` and
`MemorySide::HostDevice` fields and through the polymorphic
`Kernel::launch_on_host` handle. CUDA: 64/64 default tests pass + 76/76
under `CARBON_ENABLE_MPI=ON`; SYCL: 53/53 default tests pass;
format-check clean across all three trees.

The original finding is kept below for historical reference.

#### Original P0.α — `BlockHaloKernelImpl::launch_on_host` is missing — host pipelines on Block throw

**Diagnosis.** `include/carbon/detail/block_halo_kernel_impl.h` (110 LOC)
defines `BlockHaloKernelImpl<T, Components, SBlock>` and overrides
`launch` + `launch_on_device`, but **does not** override `launch_on_host`
nor `runs_on_host()`. As a result:

- `Kernel::runs_on_host()` returns `false` for a Block halo kernel.
- `Kernel::launch_on_host(stream, view)` falls through to the
  `KernelImpl::launch_on_host` default at `kernel_impl.h:164–172`, which
  `throw_with_log<std::logic_error>(...)`.

Compare with `HaloKernelImpl` (Dense, `halo_kernel_impl.h:101–112`):
```cpp
  auto launch_on_host(const ::carbon::Stream& stream, DataView) -> void override {
    halo_.launch_on_host(stream);
  }
  [[nodiscard]] auto runs_on_host() const noexcept -> bool override { return true; }
```
and the same shape on `HaloDisgKernelImpl` (`halo_disg_kernel_impl.h`).
Both Dense and DenseDisg halos run intra-process via `std::memcpy`
between adjacent host mirrors (with the documented cross-rank caveat).
**Block has no parallel implementation.**

**Why this is silent.** A user assembling a heterogeneous
`std::vector<Kernel>` for a Block field — say a host compute step + an
auto-halo + a device reducer — will hit the throw at runtime on the
first iteration through `launch_on_host`. The throw is loud, but the
**contract is silent in the docs**: AGENTS.md describes the universal
pipeline as grid-agnostic, and `kernel.h`'s `KernelKind::Halo` doxygen
("Halo exchange ... Device + host (intra-process std::memcpy).")
suggests host always works.

`include/carbon/halo_block.h:224–227` even documents the bookkeeping side
of this gap (`host_intra_count()` always returns `0`), but the kernel-
impl-level throw is not surfaced to the user before they hit it.

**Severity.** **High** — silent-by-API-contract; loud-by-runtime. Would
have shown up in a `Block × host-pipeline` test, which doesn't exist
yet (see also P1.δ).

**Resolution shape.** Two acceptable paths:

1. **Implement the host halo on Block.** Mirror Dense's path:
   `HaloBlock` would gain `launch_on_host(Stream)` doing
   `std::memcpy` between adjacent host slabs' owned-brick storage to the
   downstream slab's ghost slots. The cross-rank host path can throw
   with the same message as Dense ("cross-rank host halo is not yet
   wired in").
2. **Explicitly disable the host path with a louder diagnostic.**
   Override `launch_on_host` on `BlockHaloKernelImpl` to throw a
   *Block-specific* `std::logic_error` ("Block halos run device-only
   today; rebuild the surrounding pipeline via `make_kernel` /
   `make_universal_kernel` and dispatch via `launch_on_device`."), and
   document the gap in `kernel.h`'s `KernelKind::Halo` doxygen.

Option 1 closes the asymmetry; option 2 makes the silence explicit. Pick
option 1 if anyone's building a Block-on-CPU mirror workflow soon;
option 2 if Block-on-CPU is genuinely out of scope.

**Test that should land alongside either fix.** `test_halo_block_host`:
build a Block field, allocate `MemorySide::HostDevice`, run a host
compute kernel that writes a known pattern, then drive a halo via
`Kernel::launch_on_host` and verify adjacent slabs' ghost bricks see
the source slab's owned bricks (option 1) or that the call throws with
the Block-specific diagnostic (option 2).

**Files touched.** `include/carbon/halo_block.h` (host launch entry-
point if option 1); `include/carbon/detail/block_halo_kernel_impl.h`
(`launch_on_host` + `runs_on_host` overrides); `tests/test_halo_block_host.cu`
(new); doc update in `include/carbon/kernel.h`'s `KernelKind::Halo`
comment.

---

## P1 — Should-fix-soon

### ~~P1.α — File-size hot spots: `Skeleton::compile` + `Skeleton::apply_occ_`~~ — RESOLVED 2026-05-15

**Resolution.** Mechanical split landed in commit `<tbd hash>`. Both
hot-spot functions were turned into thin orchestrators that delegate
to named per-phase helpers; the helpers themselves stay private
member functions of `Skeleton` (no new TU, no new public surface).

1. **`Skeleton::compile()` — 282 LOC → 66 LOC** (`src/skeleton.cpp:46`).
   The body is now a sequence of helper calls, one per
   numbered phase from the original code:
   - `compile_validate_no_stencil_writes_()` — the front-end contract
     check (~25 LOC) that rejects `Compute` kernels declaring
     `AccessPattern::Stencil` writes. Phase 0 of the original.
   - `compile_auto_insert_halos_()` — auto-halo bootstrap (~110 LOC):
     scans tokens, materialises `HaloKernelKind::Auto` halos for any
     stencil-read field that isn't already preceded by a halo on the
     same field, threads them through the kernel list. Phase 1.
   - `compile_build_conflict_dag_()` — token-driven conflict scan +
     transitive reduction (~25 LOC). Phase 2.
   - `compile_assign_streams_and_events_()` — greedy stream assignment
     and per-edge event allocation (~78 LOC). Phase 3 (and dispatches
     to `apply_occ_` + `compute_inter_iter_barrier_` as before).

2. **`Skeleton::apply_occ_()` — 325 LOC → 67 LOC**
   (`src/skeleton.cpp:490`). Same shape: the body now collects (halo,
   stencil) candidate pairs and dispatches to one of three named
   rewrite passes per `OverlapMode`. New private helpers:
   - `occ_collect_candidates_()` — finds (halo, stencil-consumer)
     pairs eligible for split (~42 LOC).
   - `occ_apply_standard_split_()` — the `Standard` rewrite: move
     halo to its own stream, split stencil consumer into
     internal-boundary + interior phases (~83 LOC).
   - `occ_apply_extended_back_walk_()` — the `Extended` mode's
     backward BFS that pulls dependencies of the stencil's
     internal-boundary phase up so they overlap with the halo (~93 LOC).
   - `occ_apply_two_way_forward_walk_()` — the `TwoWayExtended` mode's
     forward BFS (~99 LOC). The previously paragraph-comment-only
     invariant — *forward walk skips compute kernels with
     `AccessPattern::Stencil` reads* — is now expressed as a named
     early-`continue` inside this helper, with a doxygen line
     explaining why.
   - `occ_allocate_record_event_for_()` and
     `occ_bump_step_for_kernel_after_()` — small utility helpers
     used by the three rewrite passes.

3. **`include/carbon/skeleton.h`** gained ~186 LOC of declarations:
   the new `struct OccPair { size_t halo_idx; size_t stencil_idx; }`
   and the ten new private member declarations above. Doxygen on each
   helper anchors which audit phase / OCC pass it implements.

**LOC accounting.** Total `src/skeleton.cpp` size went 961 → 1060 LOC
and `include/carbon/skeleton.h` went 857 → 1043 LOC. The increase is
intentional: the helper-ising prose (per-helper doxygen + phase
rationale that previously hid inside paragraph comments) now lives at
the call site of each helper rather than in the orchestrator. The
metric the audit cared about — *largest function size* — collapsed
from 282/325 LOC down to 66/67 LOC, with the largest helper
(`occ_apply_two_way_forward_walk_`) at 99 LOC.

**No new tests.** All 81 MPI tests + 66 CUDA tests + 56 SYCL tests
continue to pass — including `test_skeleton_dist_overlap` (P1.δ #5),
which exercises every `OverlapMode` and is therefore the natural
phase-invariant anchor. The audit's "add per-phase invariant unit
tests" suggestion was downgraded to a doxygen-only note inside each
helper because the new helpers are already private member functions
that can only be reached from the existing skeleton tests. Promoting
them to public for the sake of testing would re-introduce the
encapsulation problem the split was meant to address. If a future
auditor wants the invariant tests, the natural shape is to friend a
`SkeletonTestAccess` helper from a new `tests/test_skeleton_phases.cu`
— flagged here so the next pass can pick it up.

**Verification.** CUDA build clean; CUDA `ctest` 66/66 green; SYCL
build + ctest + format-check clean (56/56); MPI build clean +
`ctest --test-dir build-mpi` 81/81 green; `clang-format` enforced
post-edit via `cmake --build build --target format`.

---

### ~~P1.β — Concept anchoring drifts from the P1.β closure narrative~~ — RESOLVED 2026-05-15

**Resolution.** Option (1) from the audit's resolution shape landed
in commit `<tbd hash>`. Three-part change:

1. **`include/carbon/detail/grid_concepts.h` split.** `SpanConcept`
   now resolves into two siblings:
   - `DenseLikeSpanConcept<S, IdxT>` — Dense / DenseDisg shape
     (`iter_extent(view)`, `logic_extent()`, `xy_slab_elements()`,
     `global_location(idx)`).
   - `BlockLikeSpanConcept<S, IdxT>` — Block's brick-iteration
     shape (`block_count()`, `n_internal()`, `iter_block_count(view)`,
     `view_block_idx(view, b)`, the value-returning
     `set_and_validate(b, ix, iy, iz) -> ValidatedIdx`,
     `linear_offset(idx, components, c)`, `per_card_elements()`,
     `linear_storage_elements(int)`, `global_location(idx)`).
   `PartitionConcept` was left unchanged — every grid family already
   exposes the same `read / write / neighbour-with-{value, valid}`
   shape (verified by reading `dense::Partition`, `dense_disg::Partition`,
   `block::Partition`). New `block_like_span_partition_contract`
   helper twin of the existing `dense_like_span_partition_contract`
   (the latter renamed from `span_partition_contract`, which is
   retained as a deprecated alias). `SpanConcept` itself is retained
   as a deprecated alias for `DenseLikeSpanConcept` so any
   out-of-tree caller that referenced the historical name keeps
   compiling. File doxygen rewritten to describe the dual-concept
   anchoring story.

2. **`include/carbon/detail/dense_kernel_impl.h` dual-branch
   anchor.** The single
   `if constexpr (grid_has_dense_iteration_v<Grid>) static_assert(SpanConcept<...>)`
   block was replaced with a two-branch anchor:
   - dense-iter branch: `static_assert(DenseLikeSpanConcept<...>)`
     **and** `static_assert(PartitionConcept<Grid::template
     Partition<int, 1>, Span, Idx, int>)`.
   - block-iter (else) branch:
     `static_assert(BlockLikeSpanConcept<...>)` **and** the same
     `PartitionConcept` static_assert.
   The `int, 1` representative `(T, Components)` is well-formed for
   all three families (each grid exposes
   `template <typename T, int Components> using Partition = ...`).
   Now both branches are covered, both concepts are asserted — a
   fourth grid family that ships a mis-shaped `Span` @b or
   `Partition` lands on a clean concept diagnostic at the kernel-
   impl ctor instead of a deep launch-path instantiation stack.

3. **`include/carbon/detail/launch_traits.h` doxygen refresh.**
   `grid_has_dense_iteration_v`'s comment was updated to describe
   the dual-concept anchor (was: only mentioned `SpanConcept` on
   the Dense branch).

**No new tests.** The static_assert anchors fire at compile time
on every existing `DeviceKernelImpl` instantiation in the build —
the test suite implicitly covers the contract through every Block
/ Dense / DenseDisg kernel test (verified by 65/65 + 77/77 + 55/55
green builds; if any concept clause had been mis-shaped, the
entire kernel test family would have failed to compile).

**No call-site changes.** The `SpanConcept` deprecated alias keeps
the doc references compiling unchanged; `dense_kernel_impl.h` was
the only production caller and now uses the new names directly.

**Audit-level survey of grid-side compliance.**
- `dense::Span`: passes `DenseLikeSpanConcept`; deliberately fails
  `BlockLikeSpanConcept` (no `iter_block_count` etc.).
- `dense_disg::Span`: passes `DenseLikeSpanConcept`; same shape as
  `dense::Span` for the concept members.
- `block::Span`: deliberately fails `DenseLikeSpanConcept` (no
  `iter_extent` / `logic_extent` / `xy_slab_elements`); passes
  `BlockLikeSpanConcept`.
- All three `Partition`s: pass `PartitionConcept` for any
  `(T, Components)` instantiation. Verified at compile time via
  the `int, 1` representative used in the kernel-impl anchor.

**Verification:** CUDA 65/65 default + 77/77 under
`CARBON_ENABLE_MPI=ON`; SYCL 55/55 default; `format-check` clean
on all three build trees. No test-count change (P1.β is anchor-
only).

The original finding is preserved below for historical reference.

#### Original P1.β — Concept anchoring drifts from the P1.β closure narrative

**Diagnosis.** Second-pass audit closed P1.β as "`SpanConcept` /
`PartitionConcept` anchored in `DeviceKernelImpl`'s ctor". Reading
the current code:

- `include/carbon/detail/grid_concepts.h:44` opens with the disclaimer
  that the concepts are "**non-binding** (no `static_assert` in
  production paths)". Already a hedge against the closure narrative.
- `include/carbon/detail/dense_kernel_impl.h:105–115` `static_assert`s
  `SpanConcept` — but **only** under `if constexpr
  (grid_has_dense_iteration_v<Grid>)`. The trait at
  `launch_traits.h:195–200` excludes `Block` (whose `block::Span`
  speaks the brick contract).
- `PartitionConcept` is never `static_assert`ed in production code.
  Grepping the whole tree finds it only inside `span_partition_contract`
  (a `consteval` helper) and inside the concept's own definition.

Net effect: a future grid family that ships a `Span` lacking
`iter_extent` / `logic_extent` / `xy_slab_elements` lands on a clean
concept diagnostic *only* if it doesn't also define a `DisgPolicy` or
`BlockPolicy` marker that disables `grid_has_dense_iteration_v`. A
mis-shaped `Partition` doesn't fail anywhere — kernel instantiation
will eventually fail deep in the launch path, exactly the diagnostic
shape P1.β was supposed to fix.

**Severity.** **Medium** — the original audit item is partially open,
not fully closed; the closure docs claim more than the code delivers.

**Resolution shape.** Two options:

1. **Tighten and re-anchor.** Add a second `static_assert` in
   `DeviceKernelImpl`'s ctor that anchors `PartitionConcept` under the
   same `if constexpr (grid_has_dense_iteration_v<Grid>)` gate. Split
   the concepts file into `DenseLikeSpanConcept` + `BlockLikeSpanConcept`
   (the latter capturing `iter_block_count(view)`, `xy_block_pitch`,
   etc.) and `static_assert` the Block-specific concept under
   `if constexpr (!grid_has_dense_iteration_v<Grid>)`. Both branches
   covered, both concepts asserted.
2. **Update docs to match.** Rewrite the P1.β closure entry in the
   second-pass audit + the doxygen in `grid_concepts.h` to honestly
   say: "`SpanConcept` is anchored for Dense-style grids; Block has its
   own contract; `PartitionConcept` is descriptive-only."

Option 1 is more work but actually closes the audit item.

**Files touched.** `include/carbon/detail/grid_concepts.h` (split
concepts), `include/carbon/detail/dense_kernel_impl.h` (additional
asserts), `include/carbon/detail/block_kernel_impl.h` (anchor the
Block-side concepts via the new launch trait), possibly
`include/carbon/detail/launch_traits.h`.

---

### ~~P1.γ — Doc drift: test counts, phase-of-life comments, class blurbs~~ — RESOLVED 2026-05-13

**Resolution.** All seven drift items below were refreshed in a single
commit (`<tbd hash>`):

- AGENTS.md `:88` "9-test delta", "+7 distributed tests", "45/45
  SYCL" → corrected to "11-test delta", "+12 distributed tests",
  "53/53 SYCL" (the lower count became "64/64 CUDA" too); the
  `BlockReduceKernelImpl` static_assert clause was retired since P0.2
  closed it. The same paragraph now points at P2.γ for the remaining
  CUDA-only test promotability follow-up.
- AGENTS.md Test count snapshot updated from `63 / 75 / 52` to
  `64 / 76 / 53` (P0.α adds one cross-backend test).
- AGENTS.md `SpanConcept` / `PartitionConcept` bullet updated to
  match the live anchoring story (Dense-style only, gated on
  `grid_has_dense_iteration_v`; `PartitionConcept` descriptive-only)
  with a pointer to P1.β for the full closure.
- `dense_disg.h` class blurb: "kernel factories ... intentionally
  not exposed yet" replaced with an enumeration of the six factories
  actually exposed plus the `HaloDisg` halo round-trip wiring.
- `sycl/dense_kernel.h`: "lands in phase 2.4 alongside ..." replaced
  with a present-tense pointer to `dense_launch.h` (phase 2.4
  shipped).
- `test_skeleton.cu`: subtest doc comment that said non-`None`
  `OverlapMode` "throws 'not yet implemented'" replaced with a
  pointer to the actual OCC rewrite + the parallel-sibling subtest
  coverage.
- `overlapped_stencil.h::first_` member doc-comment reconciled with
  the in-function doc (which correctly says `first_` is no longer
  used to gate the fence). The member comment now describes
  `first_` as a diagnostic / iter-0 marker.

The original finding table is preserved below for historical
reference.

#### Original P1.γ — Doc drift: test counts, phase-of-life comments, class blurbs

| File:line | What it claims | What's actually true |
|---|---|---|
| `AGENTS.md:88` | "45/45 default SYCL tests pass... 9-test delta vs CUDA" | `tests/CMakeLists.txt` registers **11** `carbon_add_cuda_only_unit_test` entries; SYCL default count is **52** per AGENTS.md's own later "Test count snapshot" |
| `AGENTS.md:92–94` | "Adding `CARBON_ENABLE_MPI=ON` registers an additional 7 distributed tests on each backend (`dist_transfer`, `dense_dist`, `dist_reduce`, `halo_dist`, `halo_dist_periodic_z`, `halo_block_dist`, `native_io_dist`)" | Actually **12** distributed tests: the listed 7 plus `block_neighbour_dist`, `block_dist_reduce`, `dense_disg_dist`, `dense_disg_dist_reduce`, `halo_disg_dist` |
| `AGENTS.md:368` | "The 11-test delta vs CUDA is the current `carbon_add_cuda_only_unit_test` set — chiefly Block-family tests where `BlockReduceKernelImpl` still `static_assert`s under SYCL" | Inconsistent with the same file's `:88` claim of "9-test delta", and the `BlockReduceKernelImpl` clause is wrong post-P0.2 (it no longer `static_assert`s under SYCL; `block_reduce` runs on both backends) |
| `current_status.md` P0.2 closure | "carbon under SYCL had **9** Block tests gated to CUDA-only" (historical, ~ok), implies P0.2 closed the bulk of the gate | True at the time; today's gate is **11**, with several entries (`block_host_mirror`, `block_host_parity`, `block_universal`, `skeleton_block`, `skeleton_block_extended`) added *after* P0.2 closed — see P2.γ |
| `include/carbon/dense_disg.h:63–65` | "Kernel factories (`make_kernel` / `make_reduce_kernel` / ...) and the polymorphic `Kernel` handle are intentionally not exposed yet — that's a follow-up pass after the multi-GPU halo round-trip is locked" | All six kernel factories are defined on lines 190–246 of the same file; halo round-trip is locked; the blurb is stale |
| `include/carbon/detail/backend/sycl/dense_kernel.h:5–7` | "does not yet ship the device kernel itself — that lands in phase 2.4 alongside `detail/backend/sycl/dense_launch.h`" | Phase 2.4 landed; `dense_launch.h` is alongside; comment is stale |
| `tests/test_skeleton.cu:507–509` | "`Skeleton::Options{overlap_mode=OverlapMode::Standard}` ... throws 'not yet implemented' — until the OCC graph rewrite lands the API surface is final but the modes are gated" | OCC landed (commit history); `apply_occ_` runs for every non-`None` mode |
| `include/carbon/overlapped_stencil.h:266–267` | "Tracks whether the next `step()` should skip the halo<-compute wait. Cleared by `reset()` so the same orchestrator can drive multiple disjoint loops" | The implementation at lines 165–169 always issues the wait; the in-function comment at 165–168 correctly says `first_` "is kept on the type for diagnostic / observation purposes and to document the iter-0 contract; it's no longer used to gate the fence". The member-declaration comment contradicts the in-function comment |

**Severity.** **Medium** in aggregate (causes wasted reading time and
false confidence in test count claims); **Low** individually.

**Resolution shape.** One commit per cluster: AGENTS.md test counts
refresh; `dense_disg.h` class blurb; the two stale phase-of-life
header comments; the `overlapped_stencil.h` member doc.

---

### ~~P1.δ — Test-coverage gaps the P0.A/P0.B shape teaches us about~~ — RESOLVED (4/5) 2026-05-15

**Resolution.** Four of the five gaps closed in commit `<tbd hash>`;
the fifth (gap #4) reframed as feature-work-not-test-work after the
caller-survey contradicted the audit's claim. Detail per gap:

- **Gap #1 (DenseDisg × make_universal_reduce_kernel).** New test
  `tests/test_dense_disg_universal_reduce.cu` (registered as
  `carbon_add_cuda_unit_test` so it runs on both CUDA and SYCL).
  Builds a multi-GPU `DenseDisg` over the same geometry as
  `test_dense_disg_reduce`, exercises both `launch_on_device` and
  `launch_on_host` paths of the universal-reduce kernel, and asserts
  `kind() == UniversalReduction` + both `runs_on_*` true + the two
  paths agree to the full-precision integer.
- **Gap #2 (multi-rank VTK / HDF5 / nanoVDB round-trip).** New test
  `tests/test_io_dist.cu` (registered as `carbon_add_mpi_cuda_test`).
  Round-trips a scalar f32 + 3-component i32 field through every
  format whose `CARBON_HAVE_*` flag is on at compile time and
  asserts owned-cell parity + halo-untouched per rank. The HDF5
  section also locks the serial-HDF5+MPI throw-with-clean-error
  contract when `CARBON_HAVE_PHDF5` is off, so a future regression
  that silently wrote rank 0's slabs would surface here.
- **Gap #3 (cross-rank non-`Sum` reduction).** New test
  `tests/test_dist_reduce_ops.cu` (registered as
  `carbon_add_mpi_cuda_test`). Exercises `reduce::{Min, Max, Prod}`
  through the cross-rank stage-3 NCCL/RCCL allreduce path and
  through the host (OpenMP + `MPI_Allreduce`) path on `Min` for
  parity. A bug in `reduce::op_kind<Op>::value` for any of those
  three ops would surface as a numerical mismatch against the
  global host reference computed independently per rank.
- **Gap #4 (DenseDisg / Block I/O round-trip).** **Reframed as
  feature work, not test work.** The audit's claim that "the
  `write_native` / `read_native` signatures already accept any grid
  family (templated on `Field` shape)" is incorrect: a caller-survey
  of `include/carbon/io.h` shows the API is hard-typed to
  `const Dense& dense` + `Dense::Field<T, Components>&` — both the
  free-function entry points and the `detail::collect_field_views`
  helpers. Closing this gap would require either templating the I/O
  API on grid family or shipping parallel
  `write_native(path, dense_disg, ...)` / `(path, block, ...)`
  overloads, neither of which is captured by the audit's
  "five small tests" resolution shape. Tracked as a follow-on
  feature item (P1.δ-feature in the next pass).
- **Gap #5 (MPI × OverlappedStencil / OverlapMode).** New test
  `tests/test_skeleton_dist_overlap.cu` (registered as
  `carbon_add_mpi_cuda_test`). Runs a halo-bracketed Jacobi
  ping-pong pipeline under MPI for each of `{Standard, Extended,
  TwoWayExtended}` and asserts every variant's owned-cell output
  is bit-for-bit equal to the `OverlapMode::None` reference. The
  bit-equality post-condition is intentional: every kernel is
  deterministic and the OCC rewrite must not change the numerical
  sequence (it only reschedules across streams), so a regression
  that dropped a halo edge or mis-classified the Internal /
  Boundary band would surface as either an owned-cell mismatch or
  a halo cell still holding zero (the owned sweep catches both
  because the stencil reads halo cells).

**Verification.** CUDA 66/66 default + 81/81 under
`CARBON_ENABLE_MPI=ON`; SYCL 56/56 default; `format-check` clean
on all three build trees. Test-count delta vs the P2.β closure
session: +1 default (gap #1 cross-backend), +3 MPI-only (gaps #2,
#3, #5).

The original finding is preserved below for historical reference.

#### Original P1.δ — Test-coverage gaps the P0.A/P0.B shape teaches us about

The second-pass audit's P0.A (Block cross-rank connectivity off-by-one)
and P0.B (DenseDisg cross-rank halo dropped) were both **silent bugs
that survived the existing test matrix**. Both were of the form "this
capability is tested intra-process, not under MPI". Five analogous
gaps exist today:

1. **`DenseDisg × make_universal_reduce_kernel` is uncovered.** All five
   other DenseDisg factories have a test
   (`carbon_dense_disg_kernel` / `_reduce` / `_host` / etc.). The
   universal-reduce factory exists on the API (`dense_disg.h:233`) but
   the only universal-reduce tests are on Dense (`kernel_host_parity`,
   `kernel_polymorphic`) and Block (`block_universal`).
2. **No multi-rank VTK / HDF5 / nanoVDB round-trip test.** Only
   `native_io_dist` exercises MPI I/O. The VTK / HDF5 / nanoVDB readers
   all advertise per-rank-file or collective shapes, and the writers
   sit behind the same shape, but no test confirms the shapes match
   under MPI. The `CARBON_HAVE_PHDF5` collective-write path in
   particular is interesting — if discovery flips it on but the test
   doesn't probe collectives, regressions hide.
3. **No cross-rank non-`Sum` reduction test.** `test_reduce_ops`
   exercises `reduce::{Min, Max, Prod}` on a single GPU.
   `test_dist_reduce` exercises the stage-3 NCCL path — but only with
   the `Sum`-equivalent shape. A bug in `reduce::op_kind<Op>::value` for
   `Min`/`Max`/`Prod` (the trait maps to `ncclMin` / `ncclMax` /
   `ncclProd`) would survive the matrix.
4. **No I/O round-trip on `DenseDisg` or `Block` fields.** All four
   I/O tests (`native_io`, `vtk_io`, `hdf5_io`, `nanovdb_io`) are
   Dense-only. The `write_native` / `read_native` signatures already
   accept any grid family (templated on `Field` shape), but the only
   exercised path is Dense.
5. **No MPI test for `OverlappedStencil` or `Skeleton OverlapMode`.**
   `test_skeleton` covers all four `OverlapMode` values on Dense
   single-process; `test_skeleton_block_extended` covers
   `Extended`/`TwoWayExtended` on Block (CUDA-only, single-process).
   No test exercises any overlap mode under MPI. Given that
   `apply_occ_`'s graph rewrite cares about token + halo identity, an
   off-by-one in the cross-rank halo path under OCC would be invisible.

**Severity.** **Medium** — each gap is a P0.A/P0.B-shaped pothole.
Gap 5 is the most concerning given P1.α's complexity findings.

**Resolution shape.** Five small tests, runnable on the existing
CUDA+MPI matrix. Each should reuse the established test scaffolds
(`tests/test_*_dist.cu` for MPI, `tests/test_dense_disg_*.cu` for
DenseDisg factories). Gap 5 ideally compares the OCC schedule's
end-state to an explicit `OverlapMode::None` reference, the same
pattern `test_skeleton`'s overlap subtests already use.

---

## P2 — Cleanup / drift

### ~~P2.α — `Block<SBlock>` silent-empty accessor pattern~~ — RESOLVED 2026-05-14

**Resolution.** Option (a) landed in commit `<tbd hash>`: every
`Block` accessor that previously returned a `static const empty`
sentinel on out-of-range input now throws `std::out_of_range` via
`log::throw_with_log`, matching `Dense`'s shape. Six entry points
were migrated:

- `Block::span(gpu_id)`
- `Block::host_origins(gpu_id)`
- `Block::host_bitmasks(gpu_id)`
- `Block::host_connectivity(gpu_id)`
- `Block::halo_pairs_upward(adj_idx)`
- `Block::halo_pairs_downward(adj_idx)`

Each accessor's doxygen now documents the throw. A new cross-backend
unit test `test_block_oob_accessor` anchors the contract: every
accessor is called once with a valid index (must not throw) and once
with the first OOB index (must throw `std::out_of_range`). The test
runs on both CUDA and SYCL — its body uses no direct backend APIs.

**Audit-level survey of callers.** Every existing caller in `tests/`,
`include/carbon/halo_block.h`, and the two `block_launch.h` backends
already passes a bounds-validated index (loops bounded by
`gpu_count()`, `block_count(g)`, or guarded by an explicit
`relative_gpu_id < gpu_group().size()` check upstream of the
`grid.span(...)` call). No call site needed changes.

**Out-of-scope from this fix (deliberately).** Five other accessors
that return `0` on OOB rather than an empty sentinel
(`n_internal_blocks`, `bz_begin`, `bz_end`, `ghost_lower_count`,
`ghost_upper_count`) were left unchanged — `0` is a defensible
"soft" semantic for a count, and none was called out by the
audit's accessor list.

The original finding is preserved below for historical reference.

#### Original P2.α — `Block<SBlock>` silent-empty accessor pattern

**Diagnosis.** Several `Block` accessors that take a `gpu_id` return a
`static const empty` sentinel on out-of-range input instead of throwing:

- `block.h:251–256` `host_origins(gpu_id)` — returns
  `static const std::vector<int64_3d> empty{}`
- `block.h:262–266` `host_bitmasks(gpu_id)` — same shape
- `block.h:290–294` `host_block_lower_bz(gpu_id)` — same shape
- `block.h:308–313` `host_internal_block_count(gpu_id)` — returns `0`
- `block.h:475–480` `halo_pairs_upward(adj_idx)` — returns
  `static const std::vector<HaloPair> empty{}`
- `block.h:490–495` `halo_pairs_downward(adj_idx)` — same shape

The doxygen documents the behaviour ("out-of-range yields an empty
vector"), but the pattern means a user-side stale loop bound surfaces
as "this field has zero blocks on this GPU" — the same shape a
legitimate empty grid would produce. Downstream code that iterates
the returned vector and computes a number then sees a numerical
artifact, not a clear precondition failure.

`Dense` and `DenseDisg` do not have this pattern — their `gpu_id`-taking
accessors either throw (`dense.h::span(gpu_id)`) or take a `GpuId`
type that's range-checked at construction.

**Severity.** **Low–Medium.** Documented behaviour, not a silent bug;
but the **shape** of the API makes user-side index bugs hard to debug.

**Resolution shape.** Two options. Either (a) `throw_with_log` on
out-of-range like `Dense`, breaking source compatibility for any
caller that relied on the empty-sentinel; or (b) `[[nodiscard]] auto
host_origins(gpu_id) const -> std::optional<const std::vector<...>&>`
which forces the caller to handle the OOB case but stays additive.
Given how few external callers exist (carbon-internal `block.inl`,
test code, one example), option (a) is fine.

---

### ~~P2.β — Residual halo-kernel-impl duplication (~85%)~~ — RESOLVED 2026-05-15

**Resolution.** Landed in commit `<tbd hash>`. New header
`include/carbon/detail/halo_kernel_impl_base.h` introduces:

- `halo_kernel_impl_traits<HaloT>` — undefined primary template; each
  halo type's impl header specialises it to expose
  `static auto field_uid(const HaloT&) noexcept -> uint64_t` and
  `static auto field_name(const HaloT&) noexcept -> const std::string&`.
  This is the only family-specific surface — Dense / DenseDisg pull
  the pair via `halo.field().gpu_buffers().get_uid()` /
  `halo.field().name()`; Block exposes `HaloBlock::field_uid()` /
  `HaloBlock::field_name()` directly because the block halo doesn't
  keep the owning Field around.
- `HaloKernelImplBase<HaloT>` — class template carrying the
  cross-family ctor (one conservative `ReadWrite + Map` AccessToken
  push using the trait), `kind() override`, `tokens() override`,
  `runs_on_device() / runs_on_host() override`, polymorphic
  `launch -> launch_on_device`, `launch_on_device` (`halo_(stream)`),
  `launch_on_host` (`halo_.launch_on_host(stream)`), and the
  family-typed `halo() -> const HaloT&` accessor.

The three concrete impls
(`HaloKernelImpl<T, C>`, `HaloDisgKernelImpl<T, C>`,
`BlockHaloKernelImpl<T, C, SBlock>`) each shrank to a thin
`final` subclass of the base — a `using ... = HaloKernelImplBase<...>`
ctor-inheriting line plus the family-specific
`halo_kernel_impl_traits` specialisation in the same header. Each
file dropped from ~110–127 LOC to ~57–73 LOC; the cross-family
boilerplate now lives in exactly one place.

**No behaviour change.** Each subclass remains `final` (so the
devirt hint the old impls had survives), the kernel-impl tree's
public surface is unchanged
(`Kernel::halo` / `Kernel::halo_block` / `Kernel::halo_disg` still
build a `make_shared<Impl>(halo, name)` with the same call
signature), and the AccessToken shape is bit-for-bit identical.

**Verification.** CUDA 65/65 default + 77/77 under
`CARBON_ENABLE_MPI=ON`; SYCL 55/55 default; `format-check` clean
on all three build trees. The full halo test family
(`test_halo_block`, `test_halo_block_host`, `test_halo_block_dist`,
`test_halo_dist_periodic_z`, `test_halo_disg_dist`,
`test_halo_host`, `test_halo_periodic_z`, plus every Skeleton +
OverlappedStencil test that consumes a halo kernel) passes
unchanged — the dependency-tracker and the user-facing kernel
contract see the same bytes regardless of where the boilerplate
lives.

The original finding is preserved below for historical reference.

#### Original P2.β — Residual halo-kernel-impl duplication (~85%)

**Diagnosis.** Three near-identical headers:

- `include/carbon/detail/halo_kernel_impl.h` — 127 LOC,
  `HaloKernelImpl<T, Components>`
- `include/carbon/detail/halo_disg_kernel_impl.h` — similar shape,
  `HaloDisgKernelImpl<T, Components>`
- `include/carbon/detail/block_halo_kernel_impl.h` — 110 LOC,
  `BlockHaloKernelImpl<T, Components, SBlock>`

The structural overlap (ctor + token push + `kind` / `launch` /
`launch_on_device` / optional `launch_on_host` / `tokens` / `halo`
accessor) is ~85%. P1.α (second pass) intentionally left these split
because halo orchestration *is* genuinely different across families,
but the impls themselves are now boilerplate.

If P0.α is closed by implementing the Block host path (option 1
above), the three impls converge to within a few lines and a CRTP
base in `include/carbon/detail/halo_kernel_impl_base.h` becomes
attractive.

**Severity.** **Low.** Future-maintenance cost; not a correctness
risk.

**Resolution shape.** Wait until P0.α is closed (or explicitly
declines option 1). Then extract a
`HaloKernelImplBase<Derived, HaloLike, KernelKindV>` CRTP that owns
the ctor + token build + the four `launch` overloads, with derived
classes overriding `tokens_from_halo_(const HaloLike&)` (Dense / Disg
use `halo_.field().gpu_buffers().get_uid()`, Block uses
`halo_.field_uid()`) and `host_launch_(const Stream&)` (Block as
either real impl or louder throw).

---

### ~~P2.γ — Audit the eleven `carbon_add_cuda_only_unit_test` entries~~ — RESOLVED 2026-05-13 (partial)

**Resolution.** Two-part closure landed in commit `<tbd hash>`:

1. **`block_host_mirror` promoted to `carbon_add_cuda_unit_test`.**
   Audit-driven inspection found that the test body uses no direct
   CUDA APIs (`cudaXxx`, `<<<>>>` launches, CUB headers, etc.) and
   rides on the carbon API exclusively — it was on the CUDA-only
   gate by historical accident, not by genuine backend dependency.
   Now runs on both CUDA and SYCL.
2. **Per-entry rationale comments added in `tests/CMakeLists.txt`.**
   Each remaining `carbon_add_cuda_only_unit_test` entry now carries
   a one-line `# CUDA-only because: ...` comment that records the
   specific dependency keeping it on the strict gate. The dominant
   pattern is **`cudaMemset` used to zero per-GPU device buffers in
   test setup** — five entries (`block_kernel`, `block_multi_gpu`,
   `block_data_view`, `block_host_parity`, `block_universal`) and
   one Skeleton+Block test (`skeleton_block`) fit this shape.
   `block_overlapped_stencil`, `skeleton_block_extended` share it.
   `halo_block` additionally uses `cudaMemcpy DeviceToHost` for
   readback verification. `block_cuda` is the only genuinely
   CUDA-runtime-specific test (cudaMalloc / cudaDeviceSynchronize /
   cudaGetErrorString boundary smoke test).

The shape of the remaining gap is now legible: introducing a
backend-portable `System::async_memset` (zero a per-GPU buffer via
a host-zeroed `Cpu::buffer` copied through `System::async_copy`, or
a SYCL-native `q.memset` under `CARBON_BACKEND_IS_SYCL`) would lift
**seven** of the remaining entries to the cross-backend gate in one
commit. That work is the natural fourth-pass audit follow-up.

Net effect at session end: **gate is 10 entries, down from 11**;
every entry has documented rationale; the next reducer ("portable
async-memset → promote seven tests") is scoped and queued.

Original finding kept below for historical reference.

#### Original P2.γ — Audit the eleven `carbon_add_cuda_only_unit_test` entries

**Diagnosis.** P0.2 closed the SYCL gap for `BlockReduceKernelImpl`
and moved `block_reduce` + `block_dist_reduce` from CUDA-only to
both-backends. Since then, the CUDA-only gate has *grown*:

```
tests/CMakeLists.txt:
  skeleton_block_extended
  block_cuda
  block_kernel
  block_multi_gpu
  halo_block
  block_data_view
  block_overlapped_stencil
  block_host_mirror
  block_host_parity
  block_universal
  skeleton_block
```

Eleven entries, not the "9 today" or "9-test delta" that AGENTS.md
claims (see P1.γ). The originals from before P0.2 were Block tests
that legitimately failed under SYCL because `BlockReduceKernelImpl`
`static_assert`ed. The new additions are unlabelled — there's no
in-tree rationale row that says "this is gated because X".

`block_kernel` and `block_multi_gpu` don't involve reductions and
should work on SYCL today (the launch_traits-based collapse from P1.α
covered the compute path). `block_host_mirror`, `block_host_parity`,
and `block_universal` exercise host paths — would also expose **P0.α**
the moment they tried on Block (the halo host path), but they may not
touch halo today.

**Severity.** **Low–Medium.** Tests sit on the wrong side of the gate
silently — the SYCL CI lane reports 52/52 green and we miss
regressions on tests that could legitimately run.

**Resolution shape.** One commit per category:

1. **Promote tests that just work on SYCL today** (likely
   `block_kernel`, `block_multi_gpu`, `block_data_view`,
   `block_overlapped_stencil`) — change `carbon_add_cuda_only_unit_test`
   to `carbon_add_cuda_unit_test`, run the SYCL backend, see what
   passes.
2. **Document the remainder** — add a comment in
   `tests/CMakeLists.txt` next to each remaining CUDA-only entry
   explaining *why* (specific CUDA-runtime API used in the body, or
   the specific SYCL feature gap).

---

## Info-only — verified non-issues / intrinsic differences

To save the next auditor's time, here's what I checked and decided is
**not** worth a finding:

- **`KernelLoader` lifetime footgun.** Documented in `kernel.h:115`
  (`@warning` block) and anchored by a `static_assert` on
  `LoadingFn` shape in `dense_kernel_impl.h`. The second-pass audit
  considered adding a debug-mode address-tracking probe and
  consciously deferred. Still latent, still acceptable.
- **`block.inl` ctor helpers.** Verified the largest helper after the
  P2.δ split is ~149 LOC. None exceed 200. Split is healthy.
- **`Stencil::reorder_for_lattice_fusion`.** Covered by
  `test_stencil` including edge cases. Good.
- **VTK reader/writer asymmetry.** Reader is intentionally strict
  (rejects XML / non-BINARY / non-unit spacing); writer always emits
  the strict shape. Self-roundtrip cannot produce a file the reader
  rejects. Good.
- **`Block::Field::set_block` raw back-pointer.** Documented lifetime
  contract; not enforced. Same C++-trust-the-user pattern as Neon's
  upstream. Out of scope.
- **`dist_allreduce_in_place` non-`Sum` op path.** Code-path is
  exercised (the trait maps `reduce::{Min, Max, Prod}` to
  `ncclMin/Max/Prod`); the **test** matrix doesn't cover non-`Sum`
  cross-rank — see P1.δ #3.
- **`Span::linear_offset` arity differences (Dense: 1 component arg
  default, Block: 2 explicit args).** Documented; cross-grid generic
  code uses `Span::per_card_elements()` instead. **Intrinsic.**
- **`Block` lacks `BoundaryMode` / `stencil()` accessors.**
  Boundary semantics encoded as which bricks exist; no global
  `BoundaryMode`. **Intrinsic.**
- **`Idx` shape differences across families** (Dense: 3D cell
  coords; DenseDisg: cell + region pitch; Block: brick + intra +
  valid flag). Documented as the extension point. **Intrinsic.**
- **`OverlappedStencil` works only when surrounding halo is the
  Field's `make_halo_kernel()` output.** This is documented and
  asserted via tokens; not a finding.
- **No `gpu-ci` runs on every PR.** That's a deliberate policy —
  forks without GPUs see the lane queue indefinitely; opt-in. See
  AGENTS.md `gpu-ci.yml` section.

---

## Status table

The same shape as the second-pass audit's tracker.

### Resolved (carry-overs from second pass — all closed)

The second pass (`docs/audit_2026_05_followup.md`) is **fully closed**
as of `05afcdb`. See that file for the resolution narratives.

### Newly open (this pass)

| ID | Title | Severity | Effort | Suggested next step |
|---|---|---|---|---|
| ~~P0.α~~ | BlockHaloKernelImpl host path missing | High | M | **RESOLVED 2026-05-13** — host halo implemented on `HaloBlock` + `BlockHaloKernelImpl`, anchored by `test_halo_block_host` |
| ~~P1.α~~ | `Skeleton::compile` + `apply_occ_` size + complexity | Med | M | **RESOLVED 2026-05-15** — `compile()` 282→66 LOC, `apply_occ_()` 325→67 LOC, both now thin orchestrators delegating to ten new private per-phase helpers (`compile_*_`, `occ_*_`); 81/66/56 MPI/CUDA/SYCL tests still green |
| ~~P1.β~~ | SpanConcept / PartitionConcept anchoring incomplete | Med | S | **RESOLVED 2026-05-15** — `SpanConcept` split into `DenseLikeSpanConcept` + `BlockLikeSpanConcept`; `DeviceKernelImpl` ctor anchors the matching `Span` concept @b and `PartitionConcept` on each branch of `grid_has_dense_iteration_v<Grid>` |
| ~~P1.γ~~ | Doc drift (AGENTS.md test counts, four stale header comments) | Med | XS | **RESOLVED 2026-05-13** — single refresh commit covering AGENTS.md + four header comments |
| ~~P1.δ~~ | Five test-coverage gaps of the P0.A/P0.B shape | Med | M | **RESOLVED 4/5 2026-05-15** — gap #1 (DenseDisg universal-reduce), #2 (multi-rank VTK / HDF5 / nanoVDB), #3 (cross-rank non-Sum), #5 (MPI × OverlapMode) closed with new tests; gap #4 (DenseDisg / Block I/O) reframed as feature work — the I/O API is Dense-typed, not generic over grid family |
| ~~P2.α~~ | Block silent-empty accessor pattern | Low-Med | S | **RESOLVED 2026-05-14** — six accessors now `throw_with_log<std::out_of_range>`; new `test_block_oob_accessor` anchors the contract on both backends |
| ~~P2.β~~ | Halo-impl duplication (~85%) | Low | S | **RESOLVED 2026-05-15** — `HaloKernelImplBase<HaloT>` + per-halo-type `halo_kernel_impl_traits` specialisation; three impls collapsed to thin `final` subclasses (~110→~60 LOC each) |
| ~~P2.γ~~ | CUDA-only test gate has grown without rationale | Low-Med | S | **RESOLVED 2026-05-13 (partial)** — `block_host_mirror` promoted, per-entry rationale comments added in `tests/CMakeLists.txt`; portable async-memset → promote 7 more tests is the natural follow-up |

### Still open / not addressed by this pass

- The `OpenVDB v11.0.0` fallback for nanoVDB is untested in CI (the
  Dockerfile.sycl-rccl path discovers ROCm-hosted nanoVDB instead).
  Out of scope for this audit pass; flagged for nanoVDB-track work.
- `CARBON_BUILD_BENCHMARKS=ON` is the default but CI explicitly
  passes `-DCARBON_BUILD_BENCHMARKS=OFF`. Benchmarks compile-build is
  not exercised in CI. Out of scope for this audit; flagged for the
  benchmark track.

---

## Methodology note

This pass cross-checked four parallel discovery sweeps (markers /
caveats, test-coverage matrix, cross-family asymmetries, complexity
hot spots) and then independently verified each surfaced finding by
reading the cited file in full. Stale-looking subagent claims that did
not match the source were dropped. Inventory anchors:

- `tests/CMakeLists.txt` — 11 `carbon_add_cuda_only_unit_test` entries,
  12 `carbon_add_mpi_cuda_test` entries, plus the default suite.
- `src/skeleton.cpp` — 961 LOC; `Skeleton::compile` 46→327; `apply_occ_`
  470→795.
- `include/carbon/detail/halo_kernel_impl.h` — `runs_on_host()` + host
  override present.
- `include/carbon/detail/block_halo_kernel_impl.h` — `runs_on_host()` +
  host override **absent** (P0.α anchor).
- `include/carbon/detail/grid_concepts.h:44` — "non-binding" disclaimer
  (P1.β anchor).
- `include/carbon/detail/dense_kernel_impl.h:105–115` — `SpanConcept`
  `static_assert` gated on `grid_has_dense_iteration_v` (P1.β anchor).

If the next auditor finds a discrepancy between a claim here and the
code: trust the code. Open a fourth-pass entry and link the
contradiction.
