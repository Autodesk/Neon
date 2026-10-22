# Multi-resolution grid (mGrid) — design notes

Working design document for the **mGrid** port: a multi-resolution, block-sparse grid
built as a tower of `carbon::Block` levels (a clean-room rewrite of Neon's
`Neon::domain::details::mGrid`). This file captures the design decisions made *before*
mGrid code lands so the foundational `Block` refactors (the pre-mGrid PR series:
`StaticBlock` user-block params, `block::MicroIndex` + `Span` conversions, per-level
`Span::level()` / `spacing()`) are pointed in the right direction.

It complements:

- `AGENTS.md` — current operational state.
- `docs/CHANGELOG.md` — dated rationale for each shipped change.
- the read-only Neon reference under `Neon/libNeonDomain/include/Neon/domain/details/mGrid/`.

> **Status: runtime landed (2026-06).** The mGrid data plane ships in `include/carbon/mgrid.h`
> (tower of `Block` levels, co-resident multi-GPU/MPI partitioning, cross-level
> `mgrid::Partition`, kernel layer, per-level `HaloBlock`, Skeleton integration). This
> document remains the design contract; see `docs/CHANGELOG.md` for dated rationale.

---

## 1. Shape of the port

Neon's mGrid is a **stack of `bGrid`s**, one per refinement level, sharing the multi-res
`StaticBlock<8,8,8, 2,2,2, true>` policy (8³ memory brick = 4×4×4 = 64 user blocks of 2³
each). Level 0 is the finest; each coarser level has voxels `2×` larger per axis. A *parent*
cell at level `L+1` covers one user block (`2×2×2` cells) at level `L`; those are its
*children*.

carbon mirrors this as a tower of `carbon::Block<SBlock>` levels with
`SBlock = StaticBlock<MemX,MemY,MemZ, UserX,UserY,UserZ, true>`. Reusing `Block` per level
means the entire single-level data plane — `Span` bitmask + 27-way connectivity, `Partition`
`read`/`write`/`neighbour`, `Field` storage, `HaloBlock`, kernels, reductions — is inherited
unchanged for **intra-level** work. mGrid only adds the **cross-level** surface.

The three pre-mGrid PRs re-opened exactly the seams this needs, all defaulted so the
single-level `Block` is byte-for-byte unchanged:

| Pre-mGrid PR | Seam | Used by mGrid for |
|---|---|---|
| PR1 | `StaticBlock::UserX/Y/Z`, `BlockRatioX/Y/Z`, `IsMultiResMode` | one user block = one parent cell; branching factor per level |
| PR2 | `block::MicroIndex` + `Span::to_micro_index` / `from_micro_index` | mapping a voxel to its `(user-block, in-user-block)` tray — the unit of parent/child correspondence |
| PR3 | `Span::level()` / `Span::spacing()` | per-level scale, voxel↔world at level 0 units |

---

## 2. `mPartition` — the cross-level view

Neon's `mPartition<T,C>` derives from `bPartition<T,C,SBlock>` and adds parent/child/uncle
navigation plus per-level metadata. carbon keeps the single-level `block::Partition`
unchanged and models the cross-level partition as an **additive** layer constrained by
`MultiResPartitionConcept` (in `include/carbon/detail/grid_concepts.h`).

### 2.1 Minimum contract (`MultiResPartitionConcept`)

Re-expressed in carbon's idiom (value queries return `NeighbourValue<T>` = `{value, valid}`,
not Neon's `NghData` + bare references):

- **topology** — `has_parent(span, idx)`, `has_children(span, idx)`.
- **cross-level index** — `get_parent(span, idx) -> Idx` (coarser cell covering `idx`),
  `get_child(span, idx, child_offset) -> Idx` (finer cell at user-block `child_offset`).
  The returned `Idx` indexes the **adjacent level's** storage.
- **cross-level read** — `parent(span, idx, c) -> NeighbourValue<T>`,
  `child(span, idx, child_offset, c) -> NeighbourValue<T>`; `valid` tracks "that level's
  cell exists and is active".

Per-level scale (`level`, `spacing`, refinement factor) is **not** on the partition — it
lives on `block::Span` (PR3), since it is topology shared by every `Field` over a level, not
a property of one field's values. Neon's `getUncle` / `uncleVal` (a cell's parent's
neighbour — same-generation cross-branch read) is an **extension** left out of the minimal
contract; an mGrid that needs it adds it without widening the concept.

### 2.2 Backing data (what an `mPartition`/level must carry)

Neon threads ~16 pointers into each `mPartition` ctor. Re-grouped for carbon, the
cross-level state a level needs beyond the single-level `Block` it already owns is:

- **`parent_block_id[]`** — per child *block*, the block id of its parent on level `L+1`
  (`kInvalidBlockIdx` if none). Combined with the child's `MicroIndex` tray (PR2), this
  resolves `get_parent`: the parent's in-block coordinate is the tray index decoded into the
  parent brick.
- **`child_block_id[]`** — per parent *user-block tray*, the block id of the child brick on
  level `L-1` it refines into (`kInvalidBlockIdx` if that octant is not refined). Indexed by
  the parent cell's tray id (PR2's `to_micro_index`), this resolves `get_child` /
  `has_children`.
- **`mask_lower` / `mask_upper`** — the bitmask of the finer / coarser level, so a
  cross-level read can test activity on the *other* level without dereferencing its full
  `Span`. (Neon caches these as raw `MaskT*`; carbon can instead hand the neighbouring
  level's `Span` to the cross-level methods — see open question Q1.)
- **`mem_parent` / `mem_child`** — the field storage pointers of the adjacent levels for the
  same `(T, Components)` field, so `parent()` / `child()` can read without a separate
  partition handle. carbon's `Field` is per-level; the mGrid `Field` wrapper holds one
  `Block::Field` per level and wires the neighbouring-level pointers into each level's
  cross-level partition at launch time.

`get_parent` / `get_child` are pure index arithmetic over `MicroIndex` + these block-id
tables; they do **not** need the field pointers (only `parent()`/`child()` do). This keeps
the topology (block-id tables, masks, origins) on the per-level `Span` family and the
values on the per-level `Partition`, preserving the single-level "Span owns topology,
Partition owns values" split.

### 2.3 `getChild` / `getParent` arithmetic (reconciled to Neon)

> Reconciled with Neon's actual implementation after reading
> [mPartition_imp.h](../Neon/libNeonDomain/include/Neon/domain/details/mGrid/mPartition_imp.h)
> and [mGrid.cpp](../Neon/libNeonDomain/src/domain/details/mGrid/mGrid.cpp). The earlier draft
> here described a `MicroIndex`-tray mapping; Neon instead uses **global-index arithmetic + a
> per-voxel `childBlockID` table**. carbon mirrors Neon.

**The refinement factor `R` is the octree branching = `SBlock::UserX` (= `UserY` = `UserZ`),
not `BlockRatio`.** One parent voxel at level `L+1` has `R x R x R` child voxels at level `L`.
Neon fixes `R = 2` and enforces `descriptor.refFactor(l) == UserBlockSize` on every level
(`mGrid.cpp` lines 25-36); carbon's descriptor mirrors that (`ref_factor == SBlock::UserX`).
`BlockRatio = MemBlock / UserBlock` (4 for the default brick) is a separate quantity — the
number of user-block trays per `MemX^3` storage brick — and the `MicroIndex` tray of PR2 is
**not** on the cross-level hot path (it stays available for bGrid-internal multi-res
addressing). Coordinates below are in each level's own voxel units; adjacent levels differ by
exactly `R`.

- `get_child(parent_idx, child_offset)`, `child_offset in [0,R)^3`:
  - block: `child_block_id[ parent_idx.block_idx * VoxelCount + voxel_pitch(parent_idx.intra) ]`
    (per-voxel table; `kInvalidBlockIdx` if that voxel has no refined child);
  - in-brick: `child.intra = (R * parent_idx.intra + child_offset) mod MemX` per axis
    (Neon `mPartition_imp.h` lines 96-99; the brick-origin term drops out mod `MemX` because
    brick origins are `MemX`-aligned).
- `get_parent(child_idx)`:
  - block: `parent_block_id[ child_idx.block_idx ]` (per-block table; `kInvalidBlockIdx` if no
    parent);
  - in-brick: `parent.intra = (child_global_voxel / R) mod MemX` per axis, where
    `child_global_voxel = own_span.global_location(child_idx)` (level-local voxel coords).
    (Neon computes the same in finest units as `(g / spacing(L+1)) mod MemBlock`.)

`has_children(idx)` is `level != 0 && mem_child != null && child_block_id[...] != kInvalid`;
`has_parent(idx)` is `idx` active `&& parent_block_id[block] != kInvalid`. `parent()` /
`child()` then read the adjacent level's storage at `adjacent_span.linear_offset(other_idx, C,
c)`, with `valid` gated on the adjacent level's bitmask (the activity check carbon's
`NeighbourValue` semantics require).

These (`get_parent` / `get_child`) are the methods to implement and unit-test first
(host-side, no GPU): build a small two/three-level refined domain, then assert the
parent->child->parent round-trip and that `child_block_id` / `parent_block_id` agree with a
brute-force origin match.

### 2.4 Construction (per-level `Block` + cross-level tables)

Neon builds a dense per-level bitmask first (activation + optional overlap-cull + strong-
balance), then instantiates one `bGrid` per level and walks active blocks to fill
`parent_block_id` / `child_block_id` (`mGrid.cpp` lines 360-571). carbon's first pass keeps it
simpler and leans on the existing `Block` builder:

1. For each level `l`, build a `Block<SBlock>` over the level-`l` voxel extent
   (`ceil(finest_extent / R^l)`) with the user's per-level `is_active`. This reuses Block's
   bitmask / 27-way connectivity / origins / partitioning verbatim.
2. Build an origin->block_idx map per level (single-GPU: global ids).
3. Fill `parent_block_id[l][b]` by mapping block `b`'s origin up to level `l+1`
   (`origin / R` -> containing brick) and looking up that level's block.
4. Fill `child_block_id[l][b*VoxelCount + pitch]` for each active voxel of block `b` by mapping
   the voxel down to level `l-1` (`R * global_voxel` -> containing brick) and looking up that
   level's block.

Neon's overlap-cull and strong-balance passes are deferred follow-ups (they refine *which*
cells are active; the cross-level wiring above is independent of them). carbon's `is_active` is
supplied per level as `is_active(level, voxel)`.

---

## 4. Co-resident multi-GPU partitioning rule

> Resolves Q2. This is the load-bearing rule that keeps cross-level access cheap and, in
> particular, **Transfer-free** (the design latitude the user asked us to keep open).

### 4.1 What `Block` does today (single level)

`Block` partitions the **brick z-axis** across GPUs: brick-row `bz` is assigned to the GPU
whose `[bz_begin_per_gpu_[g], bz_end_per_gpu_[g])` range contains it (`sweep_active_bricks_`
in `block.inl`). Each GPU's slab reserves one brick-row of **ghost slots** at its lower and
upper z-edge, mirrors of the adjacent GPU's edge bricks; `HaloBlock` fills them. In
distributed mode the same cut runs across ranks (one rank per GPU) and the cross-rank edge
goes through `DistTransfer`.

### 4.2 The rule for mGrid

**Partition the coarsest level by z-brick-rows (exactly as `Block` does today); bind every
finer-level brick to the GPU that owns its ancestor at the coarsest level.**

Equivalently: a brick at level `L` with global z-origin `gz` (in level-0 voxel units) is
owned by the GPU whose coarsest-level z-slab contains `gz`. Because each coarser level's
voxels are `spacing` larger, mapping a fine brick up to the coarsest level is integer
division of its level-0 origin by the coarsest `spacing·MemZ`. Concretely the per-level
z-cuts are **the same world-space planes** on every level — chosen once at the coarsest
level and projected down — so a coarse cell and *all* of its finer descendants land on one
GPU.

### 4.3 Why

- **Cross-level access is in-partition by construction.** `get_parent` / `get_child` /
  `parent()` / `child()` only ever touch a cell and its direct ancestor/descendant, which
  by the rule are co-resident. So the cross-level methods are pure local pointer
  dereferences — **no peer copy, no `Transfer`, no NCCL on the cross-level path**. This is
  what lets mGrid avoid the `Transfer` interface for the part that would otherwise be
  hardest (irregular, level-crossing traffic).
- **Only intra-level neighbour stencils move data**, and only at z-slab edges — the exact
  shape `HaloBlock` already handles per level (§5).
- It matches Neon's mGrid, which co-locates a parent and its children on one device for the
  same reason (the descriptor's partitioning is done at the top level and inherited).

### 4.4 Interaction with `HaloBlock` z-brick partitioning

- **Alignment constraint.** The per-level z-cuts must be aligned so that a coarse brick's
  z-extent is an exact union of finer-level brick rows. Since brick sizes are fixed
  (`MemZ`) and levels differ by integer `spacing`, snapping every level's cut planes to the
  coarsest level's brick boundaries (measured in level-0 units) satisfies this with no
  remainder. The builder picks the coarsest cut first, then derives each finer level's
  `bz_begin/bz_end` by projecting those planes down.
- **Per-level ghosts stay per-level.** Each level keeps its own lower/upper ghost brick
  rows and its own `HaloBlock` for intra-level radius-1 stencils. Cross-level reads never
  use ghosts — they read the co-resident other-level brick directly.
- **Imbalance note.** Refinement is rarely uniform in z, so equal coarse-level z-rows can
  give unequal *fine*-level work. A follow-up may weight the coarse cut by descendant brick
  counts; the rule above is the correctness baseline (co-residency), load-balancing is a
  separate optimisation that must preserve it.

---

## 5. Intra-level halo strategy

> Resolves Q3.

Cross-level needs **no halo** (§4: parent/child are co-resident). The only data movement is
the per-level radius-1 neighbour stencil at z-slab edges. Two options:

- **Option A — reuse `HaloBlock` / `Transfer` per level (recommended).** Each level *is* a
  `Block`, so build one `HaloBlock<T, C, SBlock>` per level and fire each in the mGrid halo
  step. Pros: maximal reuse of tested machinery — intra-process `Transfer` peer copies and
  the cross-rank `DistTransfer` (NCCL) path come for free and are already covered by
  `halo_block` / `halo_block_dist`; zero new transfer code. Cons: one `Transfer` issue per
  level (negligible vs the copies themselves).
- **Option B — bespoke exchange on `System::async_copy` (Transfer-free).** Drive edge-brick
  copies directly through `System::async_copy` (+ a hand-rolled NCCL group for cross-rank),
  skipping the `Transfer` wrapper. Pros: lets a future optimisation fuse all levels' edge
  exchanges into one issue/one NCCL group. Cons: re-implements machinery `HaloBlock` already
  wraps and tests, including the cross-rank Send/Recv pairing — more surface, more risk.

**Decision: Option A for intra-level.** It is the smallest correct step and inherits the
distributed path. The user's "mGrid need not rely on the `Transfer` interface" is already
honoured where it matters most — the **cross-level** path is `Transfer`-free *by
construction* via §4. Option B stays a documented fallback: if profiling later shows the
per-level halo *issue* overhead (not the copies) dominates on deep towers, fuse the per-level
edge exchanges onto `System::async_copy` behind the same mGrid-halo entry point, with no API
change to user kernels.

---

## 6. Open questions

- **Q1 — cross-level masks vs neighbouring `Span`.** Pass the adjacent level's `Span`
  into the cross-level methods (carbon-idiomatic, one owner for topology) instead of
  caching raw `MaskT*` on the partition (Neon's approach). Leaning toward `Span`, pending
  the launch-layer ergonomics of getting two levels' spans into one kernel.
- **Q2 — multi-GPU partitioning across levels.** Resolved: §4 (co-resident rule).
- **Q3 — intra-level halo exchange.** Resolved: §5 (Option A, reuse `HaloBlock` per level;
  cross-level is halo-free).
