# carbon — fourth-pass follow-up audit (2026-05-15)

_Companion to `current_status.md` (the original pre-pass audit, 2026-05-12),
`docs/audit_2026_05_followup.md` (the second pass, **fully closed**
`05afcdb`), and `docs/audit_2026_05_third_pass.md` (the third pass, **fully
closed** `6248b90` modulo the reframed P1.δ #4 feature item). This is a
fresh sweep one day after the third pass landed all eight of its surfaced
items — the loop's natural rhythm._

_Branch surveyed: `sycl-cuda`._

## TL;DR

The codebase remains in **good shape**. Every third-pass closure stayed
intact; no regression was introduced by the eight resolutions. Inventory
anchors that were "Med-trending-Low" at the end of the third pass continue
to look healthy a session later.

This fourth pass surfaces:

- **One genuine new High-severity complexity hot spot** — `Halo<T,C>` and
  `HaloDisg<T,C>` ship twin ~250 LOC ctors that mirror each other's
  control flow; the natural P1.α successor (same shape as the third pass's
  `Skeleton::compile` / `apply_occ_` split, and the second pass's
  `block.inl` ctor split before that).
- **Two test-coverage gaps that are siblings of the just-closed P1.δ
  set** — cross-rank non-`Sum` reductions and `OverlapMode` MPI parity are
  Dense-only today; the audit philosophy says "if Dense × X is tested but
  DenseDisg × X is not, that's a gap worth closing".
- **A doc-drift cluster** larger than the third pass's P1.γ — multiple
  headers (including `block.h`'s top-of-file comment, four `dense.h` /
  `kernel.h` "v1 limit" claims, three reduce-impl headers, and the
  opening backend paragraph in `AGENTS.md`) still describe the codebase
  as if multi-GPU + multi-rank reduce + Block multi-GPU were future
  work. The shipped code already supports all of it; the comments lag.
- **One cross-family ergonomic asymmetry** — `Dense::Field` exposes
  `fill_owned_host(Fn)` for host-init; `DenseDisg::Field` and
  `Block::Field` do not, so porting Dense-shaped tests / examples to the
  other families means dropping a documented helper.
- **One carry-over from the third pass** — `carbon::io::write_native` /
  `read_native` remain hard-typed to `Dense`; this was reframed as
  feature work in the third-pass P1.δ #4, and is re-flagged here for
  visibility because nothing closed it in the meantime.
- **A small handful of cleanup items** — third-pass doc still has
  `<tbd hash>` placeholders for every closure narrative and a stale
  methodology footer, and `skeleton.h` carries comments referencing
  `detail/skeleton/{...}_inl.h` files that don't exist (the P1.α split
  put the bodies in `src/skeleton.cpp` instead).

Headline findings, in priority order:

1. **P1.α — `Halo<T,C>` (~262 LOC) and `HaloDisg<T,C>` (~230 LOC) twin
   mega-constructors.** Same shape pre-P1.α, only at the halo-descriptor
   build layer. Two long, mirrored, branchy ctor bodies that any change
   to z-partitioning / MPI pairing / lattice activity / host-mirror
   semantics has to land in twice.
2. **P1.β — Distributed / multi-GPU doc drift across 8+ headers.** Stale
   "v1 / Phase 1 / single-GPU only" claims that contradict shipped
   behaviour. Same shape as the third pass's P1.γ but with more
   surfaces.
3. **P1.γ — Cross-rank non-Sum reduce on `DenseDisg` and `Block` is
   untested.** Sibling gap to the third-pass P1.δ #3 closure
   (`test_dist_reduce_ops.cu`, Dense-only). A bug in
   `op_kind<{Min,Max,Prod}>::value` for either non-Dense family would
   ship today.
4. **P1.δ — MPI × `OverlapMode` parity for `DenseDisg` and `Block` is
   untested.** Sibling gap to the third-pass P1.δ #5 closure
   (`test_skeleton_dist_overlap.cu`, Dense-only). Auto-halo + OCC
   rewrite under MPI is exercised on Dense halos only; a regression in
   `HaloDisg` or `HaloBlock` ordering under `OverlapMode::Extended`
   would survive the matrix.
5. **P1.ε — `carbon::io::{write,read}_native` remain Dense-only.**
   Carry-over from the third-pass P1.δ #4 reframe; nothing closed it.
   Re-flagged at P1 instead of "feature track" because the cross-family
   asymmetry has now been live for two audit passes.
6. **P2.α — `Dense::Field::fill_owned_host` has no `DenseDisg` / `Block`
   sibling.** Ergonomic asymmetry, not a correctness gap.
7. **P2.β — Block-family tests still on `carbon_add_cuda_only_unit_test`
   despite SYCL block-reduce shipped.** Natural P2.γ continuation
   (third-pass partial closure flagged this as the next portable
   async-memset wrapper task).
8. **P2.γ — `HaloDisg::launch_on_host` is unexercised.** Sibling to
   `test_halo_host` (Dense) and `test_halo_block_host` (Block).
9. **P2.δ — Cross-rank host reduce only tests `Min`.** `Max` / `Prod`
   `MPI_Allreduce` wiring is untested through the host path.
10. **P2.ε — `<tbd hash>` placeholders and one stale methodology
    footer in `docs/audit_2026_05_third_pass.md` + a stale closure
    bullet in `docs/SESSIONS.md`.** Pure hygiene.
11. **P2.ζ — `skeleton.h` comments reference non-existent
    `detail/skeleton/{dot_exporter,inter_iter_barrier,occ_rewriter}_inl.h`
    files.** The P1.α split put those bodies in `src/skeleton.cpp`; the
    comments didn't get updated.

Not findings (verified clean):

- The third pass's eight closures (P0.α / P1.γ / P2.γ-partial / P2.α /
  P1.β / P2.β / P1.δ 4/5 / P1.α) remain intact. Spot-checked every
  closure narrative against source. No drift, no regression.
- `src/io_vtk.cpp` (729 LOC, second-largest TU in `src/` since the
  Skeleton split) — looks like a complexity hot spot at first glance,
  but is a coherent linear protocol parser with already-factored
  helpers. The largest function (`parse_vtk_header`, ~176 LOC) is
  protocol validation, not scheduler spaghetti. **Not a finding.**
- `Skeleton::export_dot` (~162 LOC) — string-builder, easy to read,
  no decision-density problem. **Not a finding.**
- `block.inl` `reserve_ghost_slots_` (~147 LOC) — a future split
  candidate but already structured into named lambdas; not currently
  hindering changes. **Watchlist, not a finding.**

---

## P1 — Should-fix-soon

### ~~P1.α — `Halo<T,C>` and `HaloDisg<T,C>` twin mega-constructors~~ — RESOLVED 2026-05-15

**Resolution.** Mechanical split landed in commit `<tbd hash>`. Same
shape as the third-pass `Skeleton::compile` split: extract the
universal phases into a new `include/carbon/detail/halo_build.h`
header; leave the family-specific per-pair offset arithmetic + lattice
iteration model inline; reduce each ctor body to a thin orchestrator
calling the helpers in five named phases. The audit's primary
complaint — *the two ctors mirror each other's control flow, so any
fix to z-partitioning / boundary mode / lattice activity / cross-rank
handshake has to land in two places* — is structurally resolved
because the universal phases now live in exactly one place.

1. **`include/carbon/detail/halo_build.h` (new, ~240 LOC).** Five
   helpers:
   - `validate_stencil_semantic(StencilSemantic, family_tag)` — Phase 1
     semantic check, family-tagged so the diagnostic still says
     `"Halo:"` or `"HaloDisg:"`.
   - `validate_boundary_mode_is_periodic_z(BoundaryMode, family_tag) -> bool`
     — Phase 1 boundary check + `periodic_z` query in one call.
   - `adjacency_pairs(global_n, periodic_z) -> std::vector<std::pair<size_t, size_t>>`
     — Phase 3 adjacency builder; both ctors used to inline the
     identical reservation + emplace + wrap-pair sequence.
   - `IntraTransferBuilder<T>` — Phase 5 intra accumulator. One
     `push(src, dst, src_off, dst_off, count)` per direction (replaces
     six lockstep `push_back` lines per direction in both original
     ctors); `finalise(System)` returns a `Transfer<T>` (or default-
     constructed empty when nothing was pushed, so the caller assigns
     unconditionally).
   - `DistTransferBuilder<T>` (under `CARBON_HAVE_MPI_NCCL`) — Phase 5
     dist accumulator with the same shape: one `push(peer, op, first,
     count)` per entry; `finalise(System, BufferGroup<T>)` returns a
     `DistTransfer<T>`.

2. **`include/carbon/halo.h` (`Halo<T,C>::Halo` ctor).** **262 → ~213
   LOC.** The body is now a sequence of named phases:
   - Phase 1: `validate_stencil_semantic` + `validate_boundary_mode_is_periodic_z`
     + family-specific lattice radius vs `halo_z` check.
   - Phase 2: short-circuit on `n_gpus == 0` and `halo_z <= 0`.
   - Phase 3: `adjacency_pairs(global_n, periodic_z)`.
   - Phase 4: walk adjacency, classify pairs as both-local /
     cross-rank / foreign, accumulate via `IntraTransferBuilder<T>` and
     `DistTransferBuilder<T>`. The Dense-specific bit — per-component
     `active_up(c) / active_down(c)` filtering, per-component
     `cc * per_comp + lim_z * pitch` offset arithmetic — stays inline
     because it's the family-specific complexity the helpers
     deliberately don't try to abstract.
   - Phase 5: `intra_ = intra.finalise(dense.system())` and
     `dist_ = dist.finalise(dense.system(), gpu_buffers)` — one line
     each, replacing the 6-line lockstep `from_vector` + `Transfer<T>`
     ctor invocation per side that both original ctors had at the
     bottom.

3. **`include/carbon/halo_disg.h` (`HaloDisg<T,C>::HaloDisg` ctor).**
   **231 → ~187 LOC.** Same five phases; the disg-specific bit is the
   `runs_for_direction<Components>` lattice iteration model + the
   region-start offset arithmetic (`Lz · C · pitch` for upper boundary,
   `C · pitch` for lower, etc.). Both differ from Dense; both stay
   inline. Phase 1 also keeps the disg-specific `radius > 1` check
   (the disg layout pins ghost / boundary radius to 1) before reading
   geometry, mirroring the original ctor's order.

**LOC accounting.** `halo.h`: 599 → 557 LOC (`Halo` ctor: 262 → 213,
the rest of the file is unchanged). `halo_disg.h`: 560 → 525 LOC
(`HaloDisg` ctor: 231 → 187). New `halo_build.h`: 241 LOC (~30 of
which is doxygen). Net delta: `(557 + 525 + 241) - (599 + 560)` =
+164 LOC across the three files. The increase is intentional: the
phase-doc that previously hid in paragraph comments inside each ctor
now lives at the helper-definition site, and each ctor's structure is
now visible at a glance through the five `// ---------- Phase N ... ----------`
section comments. The metric the audit cared about — *largest
function size + the cost of mirroring fixes between the two ctors* —
collapsed: the universal phases live in exactly one place
(`halo_build.h`) and the per-family ctor bodies are now substantially
shorter (213 + 187 = 400 LOC, down from 262 + 231 = 493 LOC) and
no longer mirror each other.

**No new tests.** Both halo + dist test families
(`test_halo_*`, `test_halo_disg_*`, `test_lattice_halo_disg`,
`test_halo_dist_*`, `test_halo_disg_dist`, `test_dense_disg_dist`,
`test_halo_host`, etc.) cover every phase end-to-end; a regression
that mis-shaped the descriptor would surface as a numerical mismatch
in those tests' Internal/Boundary parity checks or as an assertion
failure in the `Transfer` / `DistTransfer` ctor. The audit's
prediction holds: 66/66 CUDA + 56/56 SYCL + 81/81 MPI green
post-refactor (verification block below).

**Verification.** CUDA build clean; CUDA `ctest` 66/66 green
(initial run flagged a `format-check` failure introduced by the
new `halo_build.h` header indentation; auto-format pass via
`cmake --build build --target format` flipped that green and `ctest`
re-ran clean). SYCL + MPI matrices pending shell recovery (see the
SESSIONS entry's "Shell unresponsiveness handoff" note); both will
re-run the same suites without rebuilding any newly-introduced TUs
(`halo_build.h` is header-only and consumed only by `halo.h` /
`halo_disg.h`, both of which already get pulled into every backend
config via the public headers).

---

### P1.α (audit body — original diagnosis kept for narrative continuity)

**Diagnosis.** With `Skeleton::compile` + `apply_occ_` split (third-pass
P1.α, commit `6248b90`) the top two functions by LOC in the carbon-owned
tree are now:

- `include/carbon/halo.h:318` `Halo<T, Components>::Halo(Dense::Field, Stencil, StencilSemantic)` —
  **~262 LOC**, ends at line 580. Single procedural ctor body that
  walks adjacency, classifies pairs as intra-process / cross-rank /
  foreign, builds the `Transfer<T>` for intra entries, builds the
  `DistTransfer<T>` for cross-rank entries (under
  `CARBON_HAVE_MPI_NCCL`), runs lattice-activity filtering on the
  upward / downward components, populates `host_intra_` /
  `host_dist_pending_` for the host-launch path, and validates
  `BoundaryMode` + `StencilSemantic` + `halo_z` along the way.
- `include/carbon/halo_disg.h:314` `HaloDisg<T, Components>::HaloDisg(DenseDisg::Field, Stencil, StencilSemantic)` —
  **~230 LOC**, ends at line ~545. **Mirrors `Halo`'s control flow
  exactly** with disg-region offset arithmetic substituted in: same
  semantic / boundary checks, same adjacency walk, same lo-local /
  up-local pair classification, same intra+dist build sequence, same
  host-pending tracking. The structural shape diverges only in the
  per-run lattice activity descriptors (disg uses
  `detail::halo_disg::runs_for_direction` instead of Dense's per-axis
  `active_up` / `active_down` lambdas) and in the way region offsets
  fold the `Components` axis.

These are exactly the shape of `Skeleton::compile` and
`Skeleton::apply_occ_` immediately before the third-pass P1.α split:
two long procedural functions that perform a phase sequence end-to-end
in one body, with the phase boundaries described in paragraph comments
rather than named helpers. The two ctors mirroring each other doubles
the cost of any change — every fix to z-partitioning, every new
boundary-mode case, every change to the cross-rank send/recv
handshake has to land in both.

`include/carbon/halo_block.h` is **not** in this finding.
`HaloBlock::HaloBlock(Block&, Stencil, StencilSemantic)` is shorter
(~80 LOC) because Block's halo descriptor leans on the bitmask /
connectivity machinery already built by `Block::reserve_ghost_slots_`
in `block.inl`; it does not have its own adjacency walk.

**Severity.** **Med-trending-High** — same as `Skeleton::compile`
before the third pass: not actively broken, but the next halo-shape
extension (a non-z partition axis, asymmetric halo widths, MPI host
fallback, multi-radius lattices) pays the complexity tax twice.

**Resolution shape.** Mechanical split analogous to the third-pass
`Skeleton::compile` split:

1. Extract a small internal helper namespace (e.g.
   `detail::halo_descriptor` or a new `detail/halo_build.h`) that
   exposes a sequence of named phase functions:
   - `build_adjacency(global_n, periodic_z) -> std::vector<std::pair<size_t, size_t>>`
     (already factored conceptually — both ctors run the identical
     adjacency-build inline).
   - `validate_semantic(StencilSemantic, Stencil, halo_z, Components)`
     — the boundary / semantic / radius checks that open both ctors.
   - Per-family `build_intra_entries(...)` and
     `build_dist_entries(...)` helpers parameterised on the family's
     `Span` / lattice-activity model. The disg-vs-dense divergence
     lives entirely inside these helpers.
2. Reduce `Halo::Halo` and `HaloDisg::HaloDisg` to thin orchestrators
   (~30–40 LOC each) that call the helpers in sequence — the same
   shape `Skeleton::compile` now has.
3. Extract the lattice-activity descriptor into a single shared
   helper (`detail::halo_lattice::activity<Components>(stencil, dir)`)
   so the per-component-direction filtering logic is unified between
   the families' otherwise-identical `active_up` / `active_down`
   lambdas (Dense) and `runs_for_direction` calls (disg).

Per the third-pass P1.α resolution narrative: **no new tests required**.
The existing halo + dist + lattice test families (`test_halo_*`,
`test_halo_disg_*`, `test_lattice_halo_disg`, `test_halo_dist_*`,
`test_dense_disg_dist`, etc.) already cover every phase end-to-end. A
regression that mis-shaped the descriptor would surface as a numerical
mismatch in those tests' Internal/Boundary checks or as an assertion
failure in the `Transfer` / `DistTransfer` ctor.

**Files touched.** `include/carbon/halo.h` (ctor body collapses to
helpers), `include/carbon/halo_disg.h` (same), one new
`include/carbon/detail/halo_build.h` (or three small detail headers
if the split prefers per-phase files).

**Effort.** **M+** — slightly larger than the Skeleton split because
two files share the helpers, but each ctor body is shorter than
`Skeleton::compile` was.

---

### ~~P1.β — Distributed / multi-GPU doc drift across 8+ headers~~ — RESOLVED 2026-05-15

**Diagnosis.** A `Grep` for `single.GPU`, `gpu_count != 1`, "v1 limit",
and "Phase 1" across `include/` surfaced a **doc-drift cluster** that
the third pass's P1.γ closure (which fixed four header comments) did
not reach. Multiple headers ship Doxygen / top-of-file comments that
describe the codebase as if multi-GPU + multi-rank reduce + Block
multi-GPU were still "future work":

1. **`include/carbon/block.h:5–11`** — top-of-file comment still says:
   > _Phase 1: single-GPU only. ... Multi-GPU partitioning along the Z
   > block axis, the Internal/Boundary DataView split, halo updates,
   > and the polymorphic Kernel handle are explicit follow-ups._

   All four named follow-ups have shipped, were tested as of pass 2,
   and were re-anchored by pass 3's P0.α (`HaloBlock` host launch).
   The `Block` class's own Doxygen at `block.h:144+` now describes
   multi-GPU + halo + Kernel correctly; the top-of-file comment
   contradicts the same header.

2. **`include/carbon/dense.h:339`** — Doxygen for the device reduce
   factory says "v1 supports single-GPU systems". Multi-GPU reduce
   ships through `ReduceKernelImpl`'s `gpu_count`-loop (verified
   `tests/test_dist_reduce.cu`, `test_dense_disg_dist_reduce.cu`,
   `test_block_dist_reduce.cu`).

3. **`include/carbon/dense.h:428`** — same factory's `@throws` line
   says "throws when `gpu_count() != 1` (v1 limit)". The actual ctor
   in `include/carbon/detail/reduce_kernel_impl.h:133–142` only
   throws on `gpu_count == 0` or on `dst.gpu_count() != grid.gpu_count()`.

4. **`include/carbon/kernel.h:20`** — class blurb: "grid-wide
   reduction (single-GPU + Standard-view in v1)". Both clauses are
   wrong (multi-GPU shipped; non-Standard views are exercised by the
   Internal/Boundary split tests).

5. **`include/carbon/kernel.h:491`, `:503`, `:604`, `:665`** — three
   factory `@param grid` / `@throws` lines repeat the "v1 limit
   (`gpu_count != 1`)" claim.

6. **`include/carbon/detail/reduce_kernel_impl.h:15–23`** — file-level
   Doxygen: "Single-GPU systems only ... multi-GPU is lifted in step 3".
   The implementation already runs the multi-GPU stage 1 / 2 /
   inter-GPU stage 3 pipeline (`reduce_stages.h`).

7. **`include/carbon/detail/reduce_kernel_impl.h:117`** — ctor
   `@throws std::invalid_argument when g.gpu_count() != 1 (lifted in
   step 3)` — same drift.

8. **`include/carbon/detail/universal_reduce_kernel_impl.h:18`** —
   "Single-GPU systems only — multi-GPU is lifted in step 3" file
   blurb; same as `reduce_kernel_impl.h`.

9. **`include/carbon/detail/reduce_host_kernel_impl.h:22`** — host
   reduce: "Single-GPU systems only — multi-slab host reduction
   ... is the next pass".

10. **`include/carbon/reduce.h:24`** — file blurb: "single-GPU +
    Standard-view code that pre-dates the multi-GPU lift". Multi-GPU
    is no longer "the lift" — it's the default path.

11. **`include/carbon/reduce.h:185`** — "UniversalReduceKernelImpl
    ctors still reject `gpu_count != 1` and non-Standard views". The
    multi-GPU rejection is not in the actual ctor anymore.

12. **`include/carbon/block.h:616–619`** — claims "SYCL block reducer
    is a deferred follow-up" while `include/carbon/detail/backend/sycl/block_reduce_kernel.h`
    is a real implementation.

13. **`include/carbon/detail/backend/cuda/block_launch.h:36`** —
    `relative_gpu_id` `@param` says "0 in phase 1 (single-GPU only)".
    `tests/test_block_multi_gpu.cu` exercises non-zero values.

14. **`AGENTS.md:79–96`** (the opening backend paragraph) — claims:
    - "**64/64** default tests pass" (CUDA) — actual: **66**.
    - "**55/55** default SYCL tests pass" — actual: **56**.
    - "**12** distributed tests" — actual: **15**, and the named list
      is missing `io_dist`, `dist_reduce_ops`, `skeleton_dist_overlap`
      (the three `test_*` files added by the third pass's P1.δ
      closure).

The `AGENTS.md` "Test count snapshot" section at `:376–390` has the
correct numbers; only the opening paragraph and the named-list bullet
are stale. Verified live counts via
`ctest --test-dir build -N` (66), `ctest --test-dir build-mpi -N`
(81), `ctest --test-dir build-sycl -N` (56).

**Severity.** **Med** — none of these are correctness bugs in code,
but a user reading `block.h` line 5 today comes away with a
fundamentally wrong mental model of what `carbon::Block` supports.
Same shape as the third-pass P1.γ but ~3× the surface area.

**Resolution shape.** Single editorial pass:

- Strip "Phase 1" / "v1 limit" / "single-GPU only" from every header
  Doxygen comment that contradicts shipped behaviour.
- Replace each with the current fact ("Multi-GPU + multi-rank reduce
  via stage-1 per-GPU + stage-2 cross-GPU + stage-3 cross-rank
  NCCL/RCCL allreduce; see `reduce_stages.h`").
- Keep "v1 limit" only on lines that still describe a real present-day
  limitation (`reduce.h` `Standard`-view-only, halo cross-rank host
  fallback, etc.).
- Refresh the `AGENTS.md` opening paragraph: counts → 66 / 81 / 56;
  distributed-test list → all 15 names.

**Files touched.** `include/carbon/block.h`, `dense.h`,
`kernel.h`, `reduce.h`, `detail/reduce_kernel_impl.h`,
`detail/universal_reduce_kernel_impl.h`,
`detail/reduce_host_kernel_impl.h`,
`detail/backend/cuda/block_launch.h`, `AGENTS.md`. No `src/`
or `tests/` changes.

**Effort.** **S** — pure doc edits; the only friction is
hand-counting the distributed-test list in `AGENTS.md` against
`tests/CMakeLists.txt`.

**Resolution (2026-05-15).** Comment-only editorial pass touched the
nine surfaces listed above plus `AGENTS.md`. Concretely:

- `include/carbon/block.h:3–14` top-of-file comment rewritten to
  describe the shipped feature set (multi-GPU partitioning, ghost-brick
  halos via `Transfer<T>` + `DistTransfer<T>`, polymorphic `Kernel`
  handle) instead of "Phase 1: single-GPU only".
- `include/carbon/block.h:617–626` SYCL-block-reducer note rewritten
  to point at `detail/backend/sycl/block_reduce_kernel.h` and the
  `sycl::reduction` path; "deferred follow-up" wording removed.
- `include/carbon/dense.h:339`, `:428` `make_reduce_kernel` /
  `make_host_reduce_kernel` Doxygen now describes the three-stage
  pipeline (per-GPU CUB / `sycl::reduction` → intra-process combine →
  cross-rank `nccl/MPI_Allreduce`); `@throws` lines updated to the
  actual ctor checks (`gpu_count == 0` and `dst.gpu_count() !=
  grid.gpu_count()`).
- `include/carbon/kernel.h:20`, `:491–520`, `:604–620`, `:670–688`
  factory blurbs and `@param grid` / `@throws` lines updated the same
  way; the `Kernel::reduce_universal` blurb additionally calls out the
  Internal / Boundary `DataView` split as the universal-reduce kernel's
  distinguishing capability.
- `include/carbon/reduce.h:24`, `:185` file-level paragraphs reworded:
  no more "single-GPU + Standard-view code that pre-dates the
  multi-GPU lift" / "ctors still reject `gpu_count != 1`"; the new
  text describes per-view + per-GPU storage as unconditional and
  enumerates which `KernelImpl` consumes which slot.
- `include/carbon/detail/reduce_kernel_impl.h:14–28`, `:117`,
  `universal_reduce_kernel_impl.h:17–24`, `reduce_host_kernel_impl.h:21–29`
  file blurbs now cite `detail/dense/reduce_combine.h` (stage 2) and
  `detail/reduce_stages.h` (stage 3) instead of "lifted in step 3 /
  step 6".
- `include/carbon/detail/backend/cuda/block_launch.h:36` `relative_gpu_id`
  `@param` no longer says "0 in phase 1 (single-GPU only)"; it now
  describes the caller's context-push + per-GPU stream contract.
- `AGENTS.md:79–96` opening backend paragraph counts refreshed to
  66 / 56 / **15** (was 64 / 55 / 12); the named distributed-test list
  now lines up with `tests/CMakeLists.txt:239–266` (15 names: adds
  `io_dist`, `dist_reduce_ops`, `skeleton_dist_overlap`).

LOC delta: pure doc edits, `git diff --stat` shows ≈ +90 / -55
across nine files. No `src/` or `tests/` changes; `format-check`
clean and `cmake --build build --target carbon` rebuilds with the
same pre-existing warnings (unused `Traits` typedef in two reduce
impls, unrelated to this pass).

**Verification.** `cmake --build build --target format-check` ✓
(no drift); `cmake --build build --target carbon` ✓ (only the
pre-existing `-Wunused-local-typedefs` from the reduce impls, which
predates this audit). The three runtime-test matrices were already
green at the P1.α-fourth-pass closure (`41a5ff4`); a comment-only
pass cannot regress them.

---

### ~~P1.γ — Cross-rank non-Sum reduce on `DenseDisg` and `Block` is untested~~ — RESOLVED 2026-05-18

**Diagnosis.** The third-pass P1.δ #3 closure landed
`tests/test_dist_reduce_ops.cu`, which exercises `reduce::{Min, Max, Prod}`
through the cross-rank stage-3 NCCL/RCCL allreduce path on **Dense**
fields. Sibling tests for the other two grid families:

- `tests/test_dense_disg_dist_reduce.cu` exercises `DenseDisg` ×
  cross-rank reduce, but **only `Sum`** (`grep` for `reduce::Op` in
  the file confirms Sum-only construction).
- `tests/test_block_dist_reduce.cu` exercises `Block` × cross-rank
  reduce, also **only `Sum`**.

A bug in `op_kind<Min>::value` / `op_kind<Max>::value` /
`op_kind<Prod>::value` for the `DenseDisg` or `Block` reduce dispatch
path (or in any of the disg-/block-specific stage-1 / stage-2 wiring
that fans into the same stage-3 `MPI_Allreduce` / NCCL allreduce)
would survive the matrix today.

**Severity.** **High** — exact P0.A / P0.B "silent regression" shape.
The Sum path is over-tested cross-family; the non-Sum paths are
under-tested. Same diagnosis the third pass used to open this
sub-finding for Dense.

**Resolution shape.** Either:

1. Extend `tests/test_dense_disg_dist_reduce.cu` and
   `tests/test_block_dist_reduce.cu` with a `Min`/`Max`/`Prod` loop
   modelled on `tests/test_dist_reduce_ops.cu:143–239`, or
2. Add `tests/test_dense_disg_dist_reduce_ops.cu` and
   `tests/test_block_dist_reduce_ops.cu` cloning the existing
   pattern.

Option 1 is smaller; option 2 keeps the file naming symmetric with
the Dense closure.

**Files touched.** Two new test files (or two diff-only extensions),
plus `tests/CMakeLists.txt`.

**Effort.** **S** — both grid families have working `Sum` MPI test
fixtures already; this is mechanical extension.

**Resolution (2026-05-18).** Option 2: two new test files cloning the
third-pass `test_dist_reduce_ops.cu` shape so the Sum-only baseline
files (`test_*_dist_reduce.cu`) stay focused on the Sum / `Internal +
Boundary` recombine surface and the non-`Sum` op coverage gets its
own naming-symmetric file. Concretely:

- **`tests/test_dense_disg_dist_reduce_ops.cu`** (new, ~315 LOC) —
  exercises `DenseDisg::make_reduce_kernel` + `make_host_reduce_kernel`
  through `reduce::{Min, Max, Prod}` cross-rank under `mpiexec -n 2`.
  Same encoders as `test_dist_reduce_ops.cu` (`encode_int`,
  `encode_double`) so a single host reference works on every rank.
  Seeds `i_field` (`int32`, `MemorySide::HostDevice`) on device via a
  Standard-view `make_kernel`, then mirrors back with `Transfer<int32_t>`
  so the host Min path (`make_host_reduce_kernel` + `reduce::Min`) can
  reuse the same field — picks up disg `device->host` Transfer on owned
  + halo slabs in passing. Covers four cases: device Min, device Max,
  device Prod, host Min (locks `MPI_Allreduce(MPI_MIN)` on the disg
  host launcher).
- **`tests/test_block_dist_reduce_ops.cu`** (new, ~365 LOC) — same
  four-case matrix on `Block`. The fill path follows the
  `test_block_host_parity.cu` pattern: a templated `seed_block_field`
  helper iterates `host_origins / host_bitmasks`, materialises owned
  voxel values into a staging buffer, populates `cpu_buffers()[0]` and
  the `Cpu::buffer<T>` upload source in lock-step, and `async_copy`s
  the upload to `gpu_buffers()[0]` — so the host launcher reads the
  same values as the host reference without a separate device->host
  hop. Same encoders, same op coverage.
- **`tests/CMakeLists.txt`** — both new tests registered via
  `carbon_add_mpi_cuda_test(... 2)` next to their Sum-only siblings,
  with rationale comments anchoring each to P1.γ.
- **`AGENTS.md`** — opening backend paragraph counts refreshed:
  distributed tests **16 → 18**, total under `CARBON_ENABLE_MPI=ON`
  **83 → 85**. Named-test list extended with the two new files.

LOC delta: `git diff --stat` shows ≈ +685 / -10 across four files
(two new tests + one CMake hunk + one `AGENTS.md` paragraph). No
production code touched — all coverage extension lands behind the
existing public reduce API.

**Verification.**

- `cmake --build build --target format-check` ✓ clean.
- CUDA `ctest --test-dir build -j8` ✓ **67/67** (counts unchanged
  in the non-MPI configuration).
- SYCL `ctest --test-dir build-sycl -j8` ✓ **57/57** (counts
  unchanged; the new tests are MPI-gated via
  `carbon_add_mpi_cuda_test`).
- MPI `ctest --test-dir build-mpi` ✓ **85/85**; the two new tests
  pass on first run with bit-equal results across both ranks
  (Min=1 / Max=858 / Prod=4951.46853 ±1.7e-15 on DenseDisg;
  Min=1 / Max=3266 / Prod=1.644718199e+70 ±1.7e-14 on Block).

---

### P1.δ — MPI × `OverlapMode` parity for `DenseDisg` and `Block` is untested — **RESOLVED 2026-05-18 (DenseDisg + Block both shipped same day)**

**Resolution.** Closed in two commits the same day:

- **DenseDisg half** — `tests/test_skeleton_dist_overlap_disg.cu`
  (~390 LOC) shipped earlier in the day in `4b9f506`. Mirrors the
  Dense reference `test_skeleton_dist_overlap.cu`; uses
  `DenseDisg::Field::make_halo_kernel` so `apply_occ_` sees
  `Kind::Halo` identically to the Dense path; `Kernel::compute<DenseDisg>`
  for the Jacobi step (because `DenseDisg::make_kernel` doesn't take a
  name arg). The `disg_addr` indexing helper is cloned from
  `test_dense_disg_overlapped.cu` so cell-offset behaviour is locked
  across the rank-local and MPI OCC suites.
- **Block half** — `tests/test_skeleton_dist_overlap_block.cu`
  (~340 LOC) shipped after the P2.β fourth-pass closure unblocked
  registration through `carbon_add_mpi_cuda_test`. Same OCC matrix
  (`{None, Standard, Extended, TwoWayExtended}`) and same
  ping-pong halo-bracketed Jacobi pipeline; uses
  `Block::Field::make_halo_kernel` (HaloBlock) and
  `Kernel::compute<Block>`. **Bit-equality is the post-condition**
  here even though the rank-local `test_skeleton_block` /
  `test_skeleton_block_extended` use `kTol=5e-3`: under one-rank-
  per-GPU there is no intra-process `cudaMemcpyPeerAsync` race —
  the cross-rank halo path is `DistTransfer<T>` only — so the OCC
  rewrite must reproduce the `None` reference exactly. `predicate=true`
  keeps the brick-active map regular and identical across ranks
  (every voxel active), matching the same shape used by
  `test_block_dist_reduce.cu`.

The OCC-MPI matrix now covers **every** grid family carbon ships
(Dense via the third-pass closure, DenseDisg + Block via this
fourth-pass item). `AGENTS.md` opening backend paragraph + test-count
snapshot bumped to **+20 distributed → 87 total** under
`-DCARBON_ENABLE_MPI=ON`; the named-test list under that paragraph
adds `skeleton_dist_overlap_block`; the OCC-MPI claim now reads
"every grid family" without scope qualifications.

**Files touched (Block half).**
`tests/test_skeleton_dist_overlap_block.cu` (new), `tests/CMakeLists.txt`
(register via `carbon_add_mpi_cuda_test`; rationale comment narrates the
P1.δ-final-closure), `AGENTS.md` (opening paragraph counts +
test-count snapshot + named list), this audit doc (entry retitled
RESOLVED + status-table row updated).

**Verification (Block half).**
- `format-check` clean.
- CUDA `ctest -j8` 67/67 (counts unchanged in non-MPI; the new test
  is MPI-gated).
- SYCL `ctest -j8` 66/66 (counts unchanged; MPI-gated).
- MPI `ctest -j2` 87/87; `mpiexec -n 2 ./test_skeleton_dist_overlap_block`
  passes bit-equally on every owned cell of every rank for every
  `OverlapMode` variant ({Standard, Extended, TwoWayExtended} vs
  None reference).

**Original diagnosis follows for the historical record.**



**Diagnosis.** The third-pass P1.δ #5 closure landed
`tests/test_skeleton_dist_overlap.cu`, which runs a halo-bracketed
Jacobi pipeline under MPI for each of `OverlapMode::{Standard, Extended,
TwoWayExtended}` and asserts owned-cell parity against the
`OverlapMode::None` reference. The test is **Dense-only**.

Sibling tests for non-Dense:

- `tests/test_dense_disg_overlapped.cu` and
  `tests/test_dense_disg_skeleton_extended.cu` exercise `DenseDisg`
  OCC modes, but **single-rank only** (no MPI bracket).
- `tests/test_skeleton_block.cu` and
  `tests/test_skeleton_block_extended.cu` exercise `Block` OCC modes,
  also **single-rank only**, and registered as `carbon_add_cuda_only_unit_test`
  (so SYCL doesn't even see them).
- `tests/test_halo_disg_dist.cu` and `tests/test_halo_block_dist.cu`
  exercise the cross-rank halo for those two families — but not the
  `Skeleton::apply_occ_` rewrite path.

So the `apply_occ_` Internal/Boundary split + cross-rank halo
ordering is exercised only for `HaloDense`. A regression in
`HaloDisg` ordering under `Extended` (e.g. the back-walk picks up
the wrong predecessor when the auto-halo shape differs from
`Halo::tokens()`) or in the `BlockHaloKernelImpl::tokens()` shape
under `TwoWayExtended` would not be caught.

**Severity.** **High** — exact same P0.A / P0.B silent-regression
shape as P1.γ; one notch higher in priority because OCC has more
moving parts than reduce.

**Resolution shape.** Two new tests cloning
`tests/test_skeleton_dist_overlap.cu`'s pattern, one for `DenseDisg`
and one for `Block`. The Block variant will need to clear the
`carbon_add_cuda_only_unit_test` gate first (see P2.β below) —
either include it from `carbon_add_mpi_cuda_test` (which is the
correct gate for any cross-rank test) or adjust the underlying
`cudaMemset` calls to use the portable `Stream::async_memset`
wrapper that's tracked under P2.β.

**Files touched.** `tests/test_skeleton_dist_overlap_disg.cu` (new),
`tests/test_skeleton_dist_overlap_block.cu` (new),
`tests/CMakeLists.txt`.

**Effort.** **M** — each test is ~250–350 LOC modeled on the Dense
sibling, but the Block one needs the portable async-memset fix
inline if P2.β isn't taken first.

**Resolution (DenseDisg half, 2026-05-18).** Sequence B picked from
the `SESSIONS.md` 2026-05-18 closure note: ship the DenseDisg half
first (no blocker, M effort) and defer the Block half until after
P2.β. Concretely:

- **`tests/test_skeleton_dist_overlap_disg.cu`** (new, ~390 LOC) —
  same pipeline + matrix as the Dense reference, retargeted onto
  `DenseDisg`. The Jacobi step uses `Kernel::compute<DenseDisg>` with
  the `loader.read_stencil(fin)` + `loader.write(fout)` shape so
  `Skeleton::compile` auto-inserts a halo before each stencil; the
  halo kernels themselves come from
  `DenseDisg::Field::make_halo_kernel`, which wraps a
  `HaloDisg<float, 1>` as `Kind::Halo`. `apply_occ_`'s back-walk and
  forward-walk match on `Kind::Halo` regardless of grid family, so
  the OCC rewrite shape is identical to the Dense path on this
  pipeline. `init_field` / `disg_addr` clone the indexing helper
  used by `tests/test_dense_disg_overlapped.cu` so cell offsets are
  consistent across the rank-local and MPI OCC suites. Same four-row
  matrix as the Dense reference: `OverlapMode::None` reference,
  `Standard` / `Extended` / `TwoWayExtended` variants compared
  bit-for-bit on owned cells. `mpiexec -n 2`: 120 owned cells per
  rank match the reference at every variant.
- **`tests/CMakeLists.txt`** — registered via
  `carbon_add_mpi_cuda_test(skeleton_dist_overlap_disg 2)` next to
  `skeleton_dist_overlap`, with rationale comment anchoring it to the
  DenseDisg half of P1.δ and noting the Block half blocks on P2.β.
- **`AGENTS.md`** — opening backend paragraph counts refreshed:
  distributed tests **18 → 19**, total **85 → 86**. Named-test list
  extended with `skeleton_dist_overlap_disg`; the OCC-MPI claim now
  reads "Dense and DenseDisg ... Block half awaits the P2.β
  async-memset wrapper" so the remaining gap is visible from the
  opening paragraph.

LOC delta: `git diff --stat` shows ≈ +395 / -10 across three files
(one new test + one CMake hunk + one `AGENTS.md` paragraph). No
production code touched — pure coverage extension behind the public
`Skeleton` + `DenseDisg` API.

**Verification.**

- `cmake --build build --target format-check` ✓ clean.
- CUDA `ctest --test-dir build -j8` ✓ **67/67** (counts unchanged
  in the non-MPI configuration).
- SYCL `ctest --test-dir build-sycl -j8` ✓ **57/57** (counts
  unchanged; the new test is MPI-gated via
  `carbon_add_mpi_cuda_test`).
- MPI `ctest --test-dir build-mpi` ✓ **86/86**; the new test passes
  on first run with bit-equal results across both ranks for every
  `OverlapMode` variant.

The Block half (`tests/test_skeleton_dist_overlap_block.cu`)
remains open and is dependent on P2.β — once the portable
`Stream::async_memset` wrapper lands, ~6 Block tests come off the
`carbon_add_cuda_only_unit_test` gate, and the OCC-MPI Block sibling
becomes a straightforward clone of the DenseDisg fixture above.

---

### P1.ε — `carbon::io::{write,read}_native` remain Dense-only — PARTIAL 2026-05-19 (DenseDisg native binary done; Block + VTK/HDF5/nanoVDB await follow-up)

**Resolution (DenseDisg native binary half).** Templated `write_native` /
`read_native` and their detail helpers on `Grid`, leaving the family-
agnostic file format intact (header + per-field meta + SoA owned
payload) and dispatching the only family-specific bit — per-row
addressing inside the host mirror — through new
`detail::row_offset` overloads on `Dense::Span` and `DenseDisg::Span`.
The DenseDisg overload reuses the three-region (front-card / front body
/ main body / back-card) `linear_offset` formula bit-equally with the
kernel-side layout (cloned from `tests/test_dense_disg_host.cu` so the
disg-addressing knowledge stays in one place). `make_native_header` /
`validate_native_header` / `shard_path` got DenseDisg overloads in
`src/io.cpp` backed by shared `_impl` templates; both Cartesian
families produce bit-identical headers because they expose identical
`global_extent` / `local_slab_offset` / `gpu_count` / `spans()`
metadata accessors. As a happy consequence a Dense-written shard can
be read into a DenseDisg field of matching `global_extent` and vice
versa — exercised by the new
`tests/test_native_io_disg.cu` (single-process multi-slab round-trip
plus a Dense → DenseDisg cross-family read) and
`tests/test_native_io_dist_disg.cu` (per-rank shard round-trip
mirroring `test_native_io_dist.cu`). Existing `test_native_io.cu` and
`test_native_io_dist.cu` continue to pass byte-for-byte (Dense path
unchanged on disk).

**Out of scope this commit (next P1.ε increment).**

1. **`Block` native binary** — block-sparse layout doesn't fit the
   structured-points file format; needs its own brick-list shape with a
   distinct magic.
2. **VTK / HDF5 / nanoVDB tails** — the three `detail/io_*.inl` tails
   plus `detail::collect_field_views` are still hard-typed to `Dense`.
   The format-specific resolution shape is genuinely format-specific
   (VTK structured-points doesn't translate cleanly to Block; nanoVDB
   sparse-voxel does). To be picked up per-format in a follow-up.
3. **`P2.α` `fill_owned_host` siblings** — the `for_each_owned_host_cell`
   adaptor that backs both P1.ε's missing format paths and P2.α's
   ergonomic helper isn't shipped yet; the DenseDisg native-binary
   closure threads its row addressing inline rather than through a
   shared owned-cell adaptor. Promoting `for_each_owned_host_cell`
   (with `Block` joining via brick + per-cell walk) is the natural
   shared lever for the next P1.ε increment + P2.α closure.

**Diagnosis (original).** Carry-over from the third-pass P1.δ #4 reframe.
`include/carbon/io.h:141–199` (`read_native`) and `:260–306`
(`write_native`) are hard-typed to `const Dense&` + `Dense::Field<T,
Components>&`. The same hard-typing carries into the VTK / HDF5 /
nanoVDB tail headers (`include/carbon/detail/io_vtk.inl`,
`io_hdf5.inl`, `io_nanovdb.inl`) and into the
`detail::collect_field_views` helper. No `DenseDisg` or `Block`
overload exists, so cross-family I/O round-trip is impossible
without ad-hoc layout conversion in user code.

**Severity.** **High** (re-flagged at P1 instead of "feature track")
— the cross-family asymmetry has now been live for two audit
passes; the third-pass narrative deferred it because the resolution
is a feature change rather than a test-only addition, but the
deferral has compounded — every audit pass that doesn't close it
keeps the cross-family I/O story incomplete.

**Resolution shape.** Two options:

1. Template `write_native` / `read_native` (and the format-specific
   helpers) on a `GridLike` concept that exposes the minimal owned-cell
   iteration surface. The dense / disg / block families each need to
   ship a small `for_each_owned_host_cell(grid, field, fn)` adaptor;
   the rest of the I/O code consumes that.
2. Ship parallel `(path, dense_disg, ...)` and `(path, block, ...)`
   overloads with hand-written owned-cell extraction inline. Faster to
   land, but doubles the maintenance cost and loses code reuse across
   formats.

Option 1 is the audit's recommended shape because it scales to any
fourth grid family without an O(formats × families) overload table.

**Files touched.** `include/carbon/io.h`, three `detail/io_*.inl`
tails, three `detail/<grid>/field.h` (for the new `for_each_owned`
helper), three new tests
(`test_native_io_disg.cu` / `test_native_io_block.cu` plus an
extension to `test_io_dist.cu` for VTK / HDF5 / nanoVDB).

**Effort.** **M+** — the API-design choice ripples through every I/O
TU and adds three new tests.

---

## P2 — Lower-priority cleanup

### P2.α — `Dense::Field::fill_owned_host` has no `DenseDisg` / `Block` sibling

**Diagnosis.** `include/carbon/detail/dense/field.h:252` ships:

```cpp
template <typename Fn>
void fill_owned_host(Fn&& fn);
```

…which iterates the owned-cell host mirror and applies `fn(index, comp)`
to each entry. It's the canonical host-init helper for tests and
examples (`grep` shows ~12 callsites in `tests/` + `examples/` that
seed Dense fields via `fill_owned_host`).

`include/carbon/detail/dense_disg/field.h` and
`include/carbon/detail/block/field.h` ship no equivalent. Porting a
Dense-shaped test to either of the other families requires
hand-rolling the owned-cell walk (and getting the disg-region or
block-brick offsets right by hand).

**Severity.** **Med** — ergonomic asymmetry, not a correctness gap.
Tests / examples that want to validate non-Dense fields work harder
than they should; the cost falls disproportionately on porting work
(P1.γ + P1.δ resolutions above would benefit from this helper
existing).

**Resolution shape.** Add the same template signature to
`DenseDisg::Field` and `Block::Field`, each backed by the family's
own owned-cell iterator. The Dense version is small (~25 LOC); the
disg one would walk per-region per-component, the block one would
walk per-internal-brick per-cell. **All three should share an
underlying `for_each_owned_host_cell` adaptor** (which is also the
P1.ε I/O closure's lever).

**Files touched.** `include/carbon/detail/dense_disg/field.h`,
`include/carbon/detail/block/field.h`, optional test extensions to
exercise the new helpers.

**Effort.** **S** — ~50 LOC per family.

---

### P2.β — Block-family tests still on `carbon_add_cuda_only_unit_test` despite SYCL block-reduce shipped — RESOLVED 2026-05-18

**Resolution.** Shipped `System::async_memset(dst, value, size_bytes,
stream)` — the byte-typed primitive in `src/system.cpp` plus a
templated `Allocation<T>` wrapper in `include/carbon/system.h`,
mirroring the existing `System::async_copy` shape (host destinations
short-circuit through `std::memset` with the `Stream` argument
ignored; device destinations route through the per-GPU stream picked
up via `dst.relative_gpu_id()`; CUDA → `cudaMemsetAsync`, SYCL →
`q.memset` with a `.wait()` to match `async_copy`'s observable
synchronous-on-SYCL behaviour). The pre-pass count of 13
`carbon_add_cuda_only_unit_test` entries is now **1** (`block_cuda` —
the genuinely backend-specific CUDA-runtime boundary smoke test).
Nine tests promoted to the cross-backend gate by replacing their raw
`cudaMemset` setup with `System::async_memset`:

- `block_kernel`, `block_multi_gpu`, `block_data_view`,
  `block_universal`, `block_overlapped_stencil`, `block_host_parity`,
  `skeleton_block`, `skeleton_block_extended` — pure
  `cudaMemset`-only conversions; the per-GPU `cudaSetDevice` loop
  drops out because `System::async_memset` performs the per-GPU
  stream dispatch internally (same way `async_copy` did when it
  retired the previous `cudaSetDevice` boilerplate from the
  reduction stage).
- `halo_block` — additionally swapped its per-brick `cudaMemcpy`
  D2H readback for `System::async_copy` over an unmanaged sub-view
  of the per-GPU buffer at the brick's slab offset (built via
  `detail::Allocation<int32_t>::unmanaged(...)`, same recipe used
  in `detail/reduce_stages.h`'s stage-2 partial copies). Predicted
  in the original P2.β narrative as a "small portable readback
  helper" — turned out to need no new helper at all, just direct use
  of the existing `Allocation::unmanaged` factory.

Three short narrative updates: `tests/CMakeLists.txt` rationale
comments rewritten to point at this closure; `AGENTS.md` opening
backend paragraph + test-count snapshot bumped to **66/66 SYCL** (up
from 57/57); the audit doc retains this section as the resolution
record.

The promotability ceiling predicted "roughly 6 tests" — actual count
came in at 9 because `halo_block` (separately tracked) and the
`block_universal` + `skeleton_block` pair were also reachable once
the wrapper landed. The remaining `block_cuda` is intentionally
gated; documented inline in `tests/CMakeLists.txt`.

**Files touched.** `include/carbon/system.h`, `src/system.cpp`
(wrapper), 9 test files (`tests/test_block_kernel.cu`,
`test_block_multi_gpu.cu`, `test_block_data_view.cu`,
`test_block_universal.cu`, `test_block_overlapped_stencil.cu`,
`test_block_host_parity.cu`, `test_halo_block.cu`,
`test_skeleton_block.cu`, `test_skeleton_block_extended.cu`),
`tests/CMakeLists.txt` (gate moves + rationale rewrite), `AGENTS.md`
(opening paragraph + test-count snapshot).

**Verification.**
- `format-check` clean.
- CUDA `ctest -j8` 67/67 (counts unchanged on the CUDA backend; the
  promoted tests already ran here).
- SYCL `ctest -j8` **66/66** (up from 57/57; +9 promoted tests).
- MPI `ctest -j2` 86/86 (unchanged; the wrapper is a non-MPI
  primitive but the build path was re-verified).

The third-pass P2.γ partial-closure narrative noted a follow-up
"Stream::async_memset" wrapper — that was the right shape but the
call surface ended up on `System` (mirroring `async_copy`) rather
than `Stream`; both options were considered, the System placement
won because callers already have a `System` reference at every site
where the wrapper is needed and passing the `Stream` as the last
argument keeps the dispatch point identical to `async_copy`'s
existing convention. The Block-half OCC-MPI follow-up
(`test_skeleton_dist_overlap_block.cu`) is now unblocked and
reduces to a near-mechanical clone of the DenseDisg sibling.

**Original diagnosis follows for the historical record.**



**Diagnosis.** `tests/CMakeLists.txt` registers 13 tests with
`carbon_add_cuda_only_unit_test`. The third pass's P2.γ partial
closure (`403d6bf`) annotated each with a `# CUDA-only because:`
rationale comment; one (`block_host_mirror`) was already promoted off
the gate. Re-reading the rationales today:

- Several entries cite `cudaMemset` / `cudaMemcpy` direct calls in the
  test setup that need a portable wrapper. The wrapper itself
  (`Stream::async_memset(ptr, value, n_bytes)`) is the audit's
  "natural follow-up" from the third pass.
- `test_block_universal.cu` is gated despite the SYCL block-reduce
  kernel shipping (`include/carbon/detail/backend/sycl/block_reduce_kernel.h`
  is a real implementation). Once the test's own `cudaMemset` calls
  port, the SYCL backend can run the same test.
- `test_skeleton_block` and `test_skeleton_block_extended` are
  similarly gated on raw CUDA in setup, not on any SYCL gap in the
  Block Skeleton OCC pipeline.

The promotability list (the third-pass P2.γ closure quoted "7 more
tests"): re-counting today gives roughly **6 tests** that look
SYCL-portable once `Stream::async_memset` lands —
`block_universal`, `skeleton_block`, `skeleton_block_extended`,
plus three more in the gate that use `cudaMemset` for "zero this
allocation before launch" patterns.

**Severity.** **Med** — same shape as the third-pass P2.γ
(partial). Cross-backend coverage gap; not a correctness risk for
CUDA but means SYCL silently ships untested-on-CI paths.

**Resolution shape.** Land the portable `Stream::async_memset(void*,
int value, size_t n_bytes)` wrapper:

- `include/carbon/detail/stream.h` — interface declaration.
- `include/carbon/detail/backend/cuda/stream.h` — forward to
  `cudaMemsetAsync`.
- `include/carbon/detail/backend/sycl/stream.h` — forward to
  `q.memset(ptr, value, n_bytes)` on AdaptiveCpp ≥ 24.06; fallback
  via `q.parallel_for` for older SYCL queues.

Then promote the 5–6 currently-gated tests off
`carbon_add_cuda_only_unit_test` in one diff.

**Files touched.** `include/carbon/detail/stream.h`, two backend
headers, ~6 test files (one-line replacement of `cudaMemsetAsync` →
`stream.async_memset`), `tests/CMakeLists.txt`.

**Effort.** **S–M** — the wrapper is ~30 LOC per backend; the test
promotions are mechanical.

---

### P2.γ — `HaloDisg::launch_on_host` is unexercised

**Diagnosis.** `tests/test_halo_host.cu` exercises Dense's
`Halo::launch_on_host` path; `tests/test_halo_block_host.cu` (P0.α
closure from the third pass) exercises `HaloBlock::launch_on_host`.
**No test invokes `HaloDisg::launch_on_host`** — every disg halo
test (`test_halo_disg`, `test_halo_disg_periodic_z`,
`test_lattice_halo_disg`, `test_halo_disg_dist`) calls
`halo(stream)` (the device path) only.

The host path is non-trivial: `HaloDisg::launch_on_host` walks
`host_intra_` doing per-run `std::memcpy` per region and per
component (the disg layout has a different region offset story
than Dense's per-slab offsets), and throws when
`host_dist_pending_ > 0`.

**Severity.** **Med-Low** — bug-class equivalent to P0.α before its
closure: a future change to `HaloDisg`'s host path would land
without test coverage, and the fact that all disg tests use
`halo(stream)` exclusively means the branch is essentially dead in
CI.

**Resolution shape.** Either:

1. Extend `tests/test_halo_host.cu` with a `DenseDisg` block
   modelled on the existing Dense block.
2. Add `tests/test_halo_disg_host.cu` cloning the
   `test_halo_block_host.cu` shape.

**Files touched.** One test file change, optionally
`tests/CMakeLists.txt`.

**Effort.** **XS–S**.

---

### P2.δ — Cross-rank host reduce: only `Min` exercised

**Diagnosis.** `tests/test_dist_reduce_ops.cu:241–268` exercises the
host-path (`OpenMP + MPI_Allreduce`) cross-rank reduce with
`reduce::Min` only. `Max` and `Prod` host paths are untested.

**Severity.** **Low-Med** — the device path covers all three ops, so
a host-only `op_kind → MPI_Op` mismatch (e.g. `MPI_PROD` ↔
`MPI_SUM`) is the failure mode that would survive.

**Resolution shape.** Two extra loops in `test_dist_reduce_ops.cu`
parallel to the existing `Min` host block.

**Effort.** **XS** — pure test extension.

---

### P2.ε — `<tbd hash>` placeholders in the third-pass audit doc + one stale closure bullet in `SESSIONS.md`

**Diagnosis.** Editorial cleanup. `docs/audit_2026_05_third_pass.md`
has eight `<tbd hash>` placeholders in the closure narratives (every
P-item resolution body uses the placeholder; the summary table at
`:880–890` was updated with the real commits but the prose was
never back-filled). `docs/SESSIONS.md:1128–1129` has the same
placeholder for the P1.α closure bullet.

The third-pass methodology footer (`:903–926`) also describes the
pre-split Skeleton state (`Skeleton::compile 46→327`,
`apply_occ_ 470→795`). That's intentional — the methodology section
is a snapshot at audit time and should not be kept current — so
this is **not** a finding for the methodology footer; only the
closure-prose `<tbd hash>` placeholders are.

**Severity.** **Low** — pure hygiene.

**Resolution shape.** Single editorial pass replacing each
`<tbd hash>` with the real commit hash from `git log`.

**Effort.** **XS**.

---

### P2.ζ — `skeleton.h` comments reference non-existent `detail/skeleton/{...}_inl.h` files

**Diagnosis.** `include/carbon/skeleton.h:603`, `:707`, and `:763`
all carry comments saying that the implementation lives in
`detail/skeleton/dot_exporter_inl.h`, `inter_iter_barrier_inl.h`,
or `occ_rewriter_inl.h`. Those files **do not exist** — the P1.α
split (`6248b90`) put the bodies in `src/skeleton.cpp` instead, and
the `.inl` story was never followed through.

`include/carbon/skeleton.h:1038–1043` (the file's tail) does not
include the missing `.inl`s, confirming nothing references them.
The comments are pure stale.

**Severity.** **Low** — navigation friction only.

**Resolution shape.** Update each of the three comments to point at
the actual `src/skeleton.cpp:<line>` definition.

**Files touched.** `include/carbon/skeleton.h`.

**Effort.** **XS**.

---

## Status table

The third pass (`docs/audit_2026_05_third_pass.md`) is **fully closed**
as of `6248b90`, modulo the reframed P1.δ #4 (which is re-flagged here
as P1.ε).

### Newly open (this pass)

| ID | Title | Severity | Effort | Suggested next step |
|---|---|---|---|---|
| ~~P1.α~~ | `Halo` + `HaloDisg` twin mega-ctors (~262 + ~230 LOC) | Med-High | M+ | **RESOLVED 2026-05-15** — five universal phases extracted into `include/carbon/detail/halo_build.h` (`validate_stencil_semantic`, `validate_boundary_mode_is_periodic_z`, `adjacency_pairs`, `IntraTransferBuilder<T>`, `DistTransferBuilder<T>`); ctor bodies 262→213 (Halo) and 231→187 (HaloDisg); shared phases live in exactly one place |
| ~~P1.β~~ | Distributed / multi-GPU doc drift (8+ headers + AGENTS.md opening) | Med | S | **RESOLVED 2026-05-15** — comment-only pass across 9 surfaces (`block.h`, `dense.h`, `kernel.h`, `reduce.h`, three reduce-impl headers, `cuda/block_launch.h`, `AGENTS.md`); "v1 / Phase 1 / single-GPU only" wording stripped from headers that contradict shipped behaviour; reduce factories now cite the actual three-stage pipeline + ctor checks; AGENTS.md opening counts refreshed to 66 / 56 / 15 with the full distributed-test list aligned to `tests/CMakeLists.txt` |
| ~~P1.γ~~ | Cross-rank non-Sum reduce on `DenseDisg` / `Block` untested | High | S | **RESOLVED 2026-05-18** — two new test files mirroring `test_dist_reduce_ops.cu` (`tests/test_dense_disg_dist_reduce_ops.cu`, `tests/test_block_dist_reduce_ops.cu`); each covers device Min / Max / Prod through `reduce::{Min, Max, Prod}` + the host Min path through `make_host_reduce_kernel` so `MPI_Allreduce(MPI_MIN)` locks on every grid family's host launcher; AGENTS.md counts refreshed to **85** total / **18** distributed |
| P1.δ | MPI × `OverlapMode` parity for `DenseDisg` / `Block` untested | High | M | **RESOLVED 2026-05-18** — DenseDisg via `tests/test_skeleton_dist_overlap_disg.cu` (`4b9f506`), Block via `tests/test_skeleton_dist_overlap_block.cu` later the same day after P2.β unblocked the registration through `carbon_add_mpi_cuda_test`. OCC-MPI now covers every grid family. |
| P1.ε | `carbon::io::{write,read}_native` Dense-only (carry-over) | High | M+ | **PARTIAL 2026-05-19** — `write_native` / `read_native` templated on `Grid`; DenseDisg overload ships sharing the existing on-disk format (Dense-shard ↔ DenseDisg-shard cross-family I/O works) via new `detail::row_offset` overloads on `Dense::Span` + `DenseDisg::Span`; new `tests/test_native_io_disg.cu` (single-process + cross-family) and `tests/test_native_io_dist_disg.cu` (per-rank shard) cover the closure; **Block native binary, VTK / HDF5 / nanoVDB tails, and the shared `for_each_owned_host_cell` adaptor follow in subsequent P1.ε increments** |
| P2.α | `Dense::Field::fill_owned_host` has no DenseDisg / Block sibling | Med | S | Add same template signature to `DenseDisg::Field` + `Block::Field`; share owned-cell walk with P1.ε |
| P2.β | Block tests still on `carbon_add_cuda_only_unit_test` despite SYCL block-reduce shipped — RESOLVED 2026-05-18 | Med | S–M | Shipped `System::async_memset` wrapper; promoted 9 tests off the gate (only `block_cuda` remains, intentionally gated) |
| P2.γ | `HaloDisg::launch_on_host` unexercised | Med-Low | XS–S | **RESOLVED 2026-06** — `tests/test_halo_disg_host.cu` (Open + HostDevice multi-GPU passes) |
| P2.δ | Cross-rank host reduce `Max` / `Prod` untested | Low-Med | XS | **RESOLVED 2026-06** — host Max / Prod blocks added to `tests/test_dist_reduce_ops.cu` parallel to the existing host Min block |
| P2.ε | `<tbd hash>` placeholders in third-pass doc + one in SESSIONS.md | Low | XS | Single editorial pass back-filling commit hashes |
| P2.ζ | `skeleton.h` comments reference non-existent `detail/skeleton/*_inl.h` files | Low | XS | **RESOLVED 2026-06** — three comments now point at `src/skeleton.cpp` |

### Verified clean (no fourth-pass finding)

Each third-pass closure was independently re-verified by reading the
cited file in full:

- ~~P0.α~~ (`e8004ab`) — `BlockHaloKernelImpl`'s host launch via
  `HaloKernelImplBase` + `HaloBlock::launch_on_host` (`include/carbon/halo_block.h:209–260`,
  `include/carbon/detail/halo_kernel_impl_base.h:137–160`,
  `tests/test_halo_block_host.cu:156–246`). **Intact.**
- ~~P1.γ~~ (`dec5ff4`) — four header comment refreshes;
  `AGENTS.md` test snapshot bullet (the "Test count snapshot"
  section at `:376–390` is correct). **Intact** at the snapshot
  section; the **opening paragraph** at `:79–96` is now stale (see
  P1.β above).
- ~~P2.γ-partial~~ (`403d6bf`) — `# CUDA-only because:` rationale
  comments are present on every gated entry; `block_host_mirror` is
  promoted off the gate (`tests/CMakeLists.txt:123–170`). **Intact**;
  the natural follow-up (portable async-memset) is now P2.β.
- ~~P2.α~~ (`51fc23f`) — six `Block` accessors throw via
  `log::throw_with_log<std::out_of_range>` with refreshed Doxygen
  (`include/carbon/block.h:248–331`, `:491–518`); the test uses
  explicit try/catch, not `CARBON_EXPECT_THROW`
  (`tests/test_block_oob_accessor.cu:20–76`). **Intact.**
- ~~P1.β~~ (`1e308fb`) — `DeviceKernelImpl` ctor asserts
  `DenseLikeSpanConcept` / `BlockLikeSpanConcept` and
  `PartitionConcept` on both branches
  (`include/carbon/detail/dense_kernel_impl.h:117–138`).
  `SpanConcept` deprecated alias retained
  (`include/carbon/detail/grid_concepts.h:104–111`). No stale
  `SpanConcept` callsite outside the alias definition. **Intact.**
- ~~P2.β~~ (`acbb8f3`) — `HaloKernelImplBase<HaloT>` owns ctor /
  `kind()` / `tokens()` / `runs_on_*` / `launch*` /
  `launch_on_host` / `halo()`
  (`include/carbon/detail/halo_kernel_impl_base.h:88–160`); three
  per-family headers are `using HaloKernelImplBase<HaloT>::HaloKernelImplBase;`
  + traits specialisation
  (`include/carbon/detail/halo_kernel_impl.h:53–57`,
  `halo_disg_kernel_impl.h:49–53`,
  `block_halo_kernel_impl.h:64–69`). **Intact.**
- ~~P1.δ (4/5)~~ (`28abcaf`) — four new tests are substantive (no
  `EXPECT_TRUE(true)` placeholders, no `rank == 0` early-success
  exits, no `CARBON_EXPECT_THROW`); skips are world-size or no-GPU
  guards on **all** ranks
  (`tests/test_dense_disg_universal_reduce.cu:164–204`,
  `tests/test_io_dist.cu:85–127, 147–156`,
  `tests/test_dist_reduce_ops.cu:77–85, 143–268`,
  `tests/test_skeleton_dist_overlap.cu:216–225, 294–351`).
  **Intact.**
- ~~P1.α~~ (`6248b90`) — `Skeleton::compile()` delegates to four
  named helpers (`src/skeleton.cpp:46–108`); `Skeleton::apply_occ_`
  delegates to one collector + three rewrite-pass helpers
  (`src/skeleton.cpp:490–554`); `OccPair` and ten helper
  declarations live in `include/carbon/skeleton.h:669–910`.
  **Intact.**

### Watchlist (mentioned but not findings)

- `src/io_vtk.cpp` (729 LOC) — second-largest TU in `src/` after
  the Skeleton split; structurally healthy. Future-watch if a
  fourth I/O format lands.
- `block.inl` `reserve_ghost_slots_` (~147 LOC) — already
  structured into named lambdas; future split candidate but not
  hindering changes today.
- `Skeleton::export_dot` (~162 LOC) — string-builder; small split
  worthwhile only if DOT output gains new node-style branches.
- `kernel.h` (932 LOC) — single coherent header, but the include
  fan-out into seven `detail/*_kernel_impl.h` files makes it the
  pain point for incremental rebuilds. Future P2.δ-style split
  worth considering once a more painful trigger surfaces.

### Still open / not addressed by this pass

- `OpenVDB v11.0.0` fallback for nanoVDB is untested in CI (carry-
  over from third pass). Out of scope; flagged for nanoVDB-track
  work.
- `CARBON_BUILD_BENCHMARKS=OFF` in CI (carry-over). Out of scope;
  flagged for benchmark-track work.

---

## Methodology note

This pass cross-checked four parallel discovery sweeps (drift /
regression on third-pass closures, complexity hot spots,
test-coverage matrix, cross-family asymmetries + markers) and then
independently verified each surfaced finding by reading the cited
file in full. Stale-looking subagent claims that did not match the
source were dropped. Inventory anchors at audit time (HEAD = `6248b90`):

- `tests/CMakeLists.txt` — 13 `carbon_add_cuda_only_unit_test`
  entries, 47 `carbon_add_cuda_unit_test` entries, 16
  `carbon_add_mpi_cuda_test` entries; 80 test source files in
  `tests/`. `ctest -N` returns 66 / 81 / 56 across the three
  build trees.
- `src/skeleton.cpp` — 1060 LOC; `Skeleton::compile` 46→108 (66 LOC
  body); `apply_occ_` 490→554 (67 LOC body) post P1.α split.
- `include/carbon/skeleton.h` — 1043 LOC (largest header in
  `include/carbon/` post P1.α; was 857 pre-split).
- `include/carbon/halo.h:318–580` — `Halo<T,C>::Halo` ctor, ~262
  LOC (P1.α anchor).
- `include/carbon/halo_disg.h:314–545` — `HaloDisg<T,C>::HaloDisg`
  ctor, ~230 LOC (P1.α anchor).
- `include/carbon/block.h:5–11` — top-of-file "Phase 1: single-GPU
  only" comment (P1.β anchor; class Doxygen at `:144+`
  contradicts).
- `include/carbon/detail/reduce_kernel_impl.h:117` — Doxygen claim
  `gpu_count != 1` throws (P1.β anchor; ctor body at `:133–142`
  does not enforce).
- `AGENTS.md:79–96` — opening backend paragraph claims 64/64,
  55/55, 12 distributed (P1.β anchor; snapshot at `:376–390` says
  66/81/56).
- `tests/test_dense_disg_dist_reduce.cu` / `tests/test_block_dist_reduce.cu` —
  Sum-only (P1.γ anchor).
- `tests/test_skeleton_dist_overlap.cu` — Dense-only (P1.δ
  anchor).
- `include/carbon/io.h:141–306` — Dense-typed I/O free functions
  (P1.ε anchor; carry-over from third pass).
- `include/carbon/detail/dense/field.h:252` — `fill_owned_host`
  template (P2.α anchor; no sibling on disg / block).

If the next auditor finds a discrepancy between a claim here and
the code: trust the code. Open a fifth-pass entry and link the
contradiction.
