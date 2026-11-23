# Carbon — project review

_Date: 2026-11-20. Branch: `carbon-bitmask` @ `f44d512`._

Opinionated review of the full Carbon tree after the SubGrid / region-mask /
Jacobi+AXPY work. This is a product and architecture view, not a line-by-line
audit. Earlier deep audits live in [`current_status.md`](current_status.md)
(2026-05 foundation) and [`carbon-review.md`](carbon-review.md) (2026-11
distributed DenseDistributed backend).

---

## Verdict

Carbon is a serious multi-GPU grid runtime with a clear centre of gravity:
**Dense + Block + Kernel + Skeleton**, validated across CUDA, SYCL, and
HIP-on-NVIDIA. The recent SubGrid + region-mask work is the right kind of
feature — it extends the scheduler without forking it. The project feels
mature where that core is concerned, and uneven where parallel “grid products”
(DenseDisg, DenseDistributed, mGrid, Block I/O) still compete for attention.

---

## What is strong

**One public API, three backends.** The `cudalike` / SYCL split behind
`detail/backend/select.h` is disciplined. Roughly 126 CUDA / 114 SYCL /
122 hip-nvidia tests with shared examples is unusual for a rewrite of this age.

**Skeleton as the product.** Auto-halo, OCC (`None` → `TwoWayExtended`), region
masks, launch-order / matrix DOT, and exact integer end-to-end tests show a
library that treats *scheduling* as a first-class concern, not an afterthought
on top of kernels.

**Dense and Block as the solid data plane.** Z-slab Dense and brick-sparse Block
are coherent, well tested, and now share SubGrid through `ParentTraits` without
a second launcher stack. The Block host-span copy fix (`c3b78cd`) was the right
kind of fix: a real use-after-free, caught by a regression that would have
bitten SubGrid hard.

**SubGrid design.** Flat Internal-first lanes, Span/Idx deriving from the
parent, no SubGrid field halo, ownership checks at kernel build — that is
Neon-sGrid shaped and OCC-compatible. The Jacobi+AXPY test (B ∥ C, Dense and
Block, 100 int64 iterations, every OCC mode) is the right pin: it proves values
*and* that masks actually parallelize.

**Engineering culture.** Append-only `docs/CHANGELOG.md`, `AGENTS.md` as living
status, compile-fail tests, Docker matrix (`scripts/docker-build-all.sh`),
mutants that must fail — that is how you keep a header-heavy C++ GPU library
honest.

---

## What is uneven

**Too many “dense-like” stories.** `Dense`, `DenseDisg`, and `DenseDistributed`
are three stacks with different I/O, halo types, and maturity. For a new user
(or agent), that naming tax is real. DenseDisg has known flakiness under load;
DenseDistributed is thinly tested relative to its surface.

**mGrid / multi-res** is present but still younger: some tests remain CUDA-only
(hand-rolled `<<<>>>`), and it is not yet first-class next to Dense/Block in the
SubGrid story.

**MPI.** One-rank-per-GPU works; SubGrid reductions and several Skeleton paths
are not exercised under MPI. Distributed solvers that put BCs on a SubGrid are
the first place that will hurt.

**HIP.** HIP-on-NVIDIA via cudalike is green; native AMD + HIP+NCCL are less of
a proven product surface than CUDA/SYCL.

**Format debt.** `carbon_format_check` still fails on pre-existing files. Small,
but it muddies “green suite” messaging.

---

## SubGrid + region masks (2026-11)

Solvers now have a clean pattern:

- **C** = bulk interior (masked parent kernel *or* a second SubGrid)
- **B** = incomplete-neighbour / BC / markers (SubGrid)
- Skeleton runs B and C without an edge; OCC gives **2 internal + 2 boundary**
  launches per Jacobi step

That matches how real codes are written. One scheduling compromise is worth
owning explicitly: **both boundary halves sit on the halo stream**. Correct, but
concurrency is left on the table when you have several region writers. Measure
that before changing the Skeleton.

The dual encoding of C (`bc_parent` vs `bc_sub`) is powerful and a footgun. A
short “when to pack vs when to mask-and-skip” note in
[`docs/skeleton/README.md`](docs/skeleton/README.md) would buy more than another
parent type.

Documented non-goals that should stay non-goals until a workload forces them:
brick-aware SubGrid lanes, nested SubGrid, DenseDisg-as-parent, multi-res Block
as SubGrid parent.

---

## Where to invest next (priority order)

1. **MPI SubGrid reduce + a small distributed Jacobi/BC path** — highest
   correctness gap relative to the claims.
2. **Clarify the product surface** — pick a primary “dense” story for
   docs/examples; demote DenseDisg / DenseDistributed to “specialized” so the
   mental model stays Dense / Block / (mGrid).
3. **OCC boundary-stream contention** — benchmark Extended with ≥2 SubGrid
   writers; only then decide whether boundary halves should fan out.
4. **Dynamic remasking** — rebuild cost and field lifetime for moving
   interfaces; untested today.
5. **Block native I/O** (already deferred in the audits) if sparse apps need
   checkpoint parity with Dense.

Do *not* chase brick-aware SubGrid lanes, nested SubGrid, or DenseDisg-as-parent
until a real workload forces it. Flat lanes match the “few special voxels” use
case SubGrid was built for.

---

## Bottom line

As a rewrite of Neon’s grid + Skeleton layer, Carbon already looks like a
**production-minded research runtime**: strong core abstractions, real
multi-backend discipline, and tests that argue about schedules and bit-exactness,
not only that something ran. SubGrid over Dense/Block with region-mask parallel
B/C is a coherent milestone and raises the bar for what “done” means here.

The risk is not that the core is weak — it is **product fragmentation** (too
many grid variants) and **distributed / HIP-native maturity** lagging the
single-node CUDA/SYCL story. Tighten the narrative around Dense + Block +
Skeleton, close the MPI SubGrid gap, and measure OCC under multiple SubGrids
before inventing more machinery.
