# Skeleton DAG export — reference renderings

Visual snapshots of the dependency DAG `carbon::Skeleton` infers from a Jacobi
ping-pong pipeline, in the stream-cluster format of
`Skeleton::export_streams_dot` (see `<carbon/skeleton.h>`).

`Skeleton::export_dot()` / `export_dot_file()` / `export_dot_png()` and the
`log_dot_on_compile` option render the **launch matrix** by default (same as
`export_launch_matrix_dot()`, see [Launch order listing](#launch-order-listing));
the stream-cluster graph shown here is `export_streams_dot()` /
`export_streams_dot_file()`.

| File | Source |
|------|--------|
| `jacobi_skeleton.dot` | Raw GraphViz DOT of `export_streams_dot()` for the `examples/jacobi_unified.cu` pipeline (its `--dot` log block printed this format before `export_dot` switched to the launch matrix). |
| `jacobi_skeleton.svg` | `dot -Tsvg jacobi_skeleton.dot -o jacobi_skeleton.svg` |
| `jacobi_skeleton.png` | `dot -Tpng -Gdpi=120 jacobi_skeleton.dot -o jacobi_skeleton.png` |

The pipeline is the canonical four-node Jacobi cycle:

```
halo_u -> step_ab -> halo_u_next -> step_ba
```

with two named fields (`u`, `u_next`). Every step has the previous step as a
direct predecessor, so the multi-stream scheduler collapses the schedule to a
single stream — visible in the rendering as a single `cluster_stream_0`
subgraph. Edge labels carry the conflicting field name and the access-mode
arrows (`u (RW->R)` reads as "predecessor wrote `u`, successor reads `u`");
node tooltips list each kernel's full token set.

## Regenerating

`--dot` now logs the launch matrix; to capture the stream-cluster graph again,
switch the example's `main_pipe.export_dot(...)` call to
`main_pipe.export_streams_dot(...)` first.

```bash
cmake --build build -j
build/examples/carbon_example_jacobi_unified --grid dense --dot 2>&1 \
  | awk '/--- DAG \(GraphViz DOT\) ---/{f=1; next}
         f && /^digraph/{f=2}
         f==2{print; if (/^}/) f=0}' \
  > docs/skeleton/jacobi_skeleton.dot

dot -Tsvg docs/skeleton/jacobi_skeleton.dot -o docs/skeleton/jacobi_skeleton.svg
dot -Tpng -Gdpi=120 docs/skeleton/jacobi_skeleton.dot -o docs/skeleton/jacobi_skeleton.png
```

`dot` ships with GraphViz; on a no-sudo box `pixi global install graphviz`
puts it under `~/.pixi/bin`.

## Multi-stream pattern

The Jacobi rendering is single-stream because the DAG is a chain. The
parallel-sibling subtest in `tests/test_skeleton.cu` exercises a 3-kernel
pipeline (`copy_uv`, `copy_xy`, `combine_uvxy`) where the first two kernels
share no fields and land on distinct streams; the resulting DOT has two
`cluster_stream_0` / `cluster_stream_1` subgraphs and the
`copy_xy -> combine` edge is rendered dashed-red to flag the cross-stream
event-mediated dependency. That test does not currently dump the DOT to disk
— if you want to see the multi-stream rendering, add an `export_streams_dot` call to
the subtest's `sk` instance and pipe through `dot` the same way.

## Standard overlap scheduling (Neon-style)

For `OverlapMode::None` and `OverlapMode::Standard`, `compile()` lowers a **step
graph** (data dependencies plus **hint** edges for launch order). Standard OCC
splits each `(halo, stencil-read compute|reduction)` pair into Internal and
Boundary steps; reductions add a merge step (`ReduceLaunchMode::Partial` then
`Merge`). Internal is ordered before the halo; Boundary runs on the halo stream
right after the halo. A final `compile_prune_syncs_()` pass removes redundant
waits (already covered by a prior wait, or dominated by a later stronger wait on
the same stream before the next launch — e.g. an anchor-start wait before a
stream-0 data-event wait), unused event records, and anchor joins already covered
by an earlier cross-stream wait.

Extended and TwoWayExtended are further rewrites of the same step graph
(`compile_step_graph_occ_`). Extended splits each Compute producer of a
stencil-read field into Boundary then Internal, and the halo waits on the
Boundary half only, so it overlaps the producer's Internal half. The halves
write disjoint cells, so Boundary-before-Internal is a hint edge (Neon's
`scheduling` dependency): it fixes issue order but adds no event. The Boundary
half runs on the halo stream, so the halo follows it in FIFO order without a
wait, while the Internal half runs concurrently on stream 0. TwoWayExtended
also splits Map-only Compute consumers of the stencil's output into Internal +
Boundary, and Map-pattern Reduction consumers into Partial Internal + Partial
Boundary + Merge (Neon Two-way Extended OCC), so an Internal reduce can overlap
the halo. When a consumer only reads what a split producer wrote, each half
waits on the matching half alone. A stencil reduction gets the same partial +
merge launches in every mode.

## Launch order listing

`Skeleton::export_launch_order()` prints one line per operation issued during a
single `run()` — anchor fork record/waits, per-step event waits, kernel launches
(including auto-inserted halos), event records, and anchor join record/waits.
The same sequence is what the CUDA/HIP graph capture path records when
`Options::use_cuda_graph` is enabled.

`export_streams_dot()` prefixes each schedule node with `#N` (the launch line number) and
annotates dashed sync edges with `(rec #R -> wait #W)`. When anchor fork/join is
active, extra dashed blue edges show the bracket ops.

`export_launch_matrix_dot()` — the default `export_dot()` — renders the same sequence as a grid: one column per
stream, one row per operation (anchor syncs, event waits, launches, event
records). Dashed arrows connect each event record to its waits — red for step
events, blue for the anchor fork/join — so cross-stream hand-offs read as
diagonal arrows between columns.

`export_kernel_list_dot()` is the matching table of what was added: one row per
kernel in add order (auto-inserted halos dashed), one column per accessed
resource with its access mode (R / W / RW, stencil reads flagged), the reduced
data dependencies, and the launches each kernel lowers to (`run()` sequence
number, stream, view) so rows cross-reference the launch matrix.

`benchmarks/jacobi_nvidia` accepts `--dot prefix`, which writes launch-matrix
`prefix_ab.dot` / `prefix_ba.dot`, stream-cluster sidecars `*_streams.dot`, and
matching `.txt` launch-order files; add `--dot-matrix` for the kernel lists
`prefix_ab_kernels.dot` / `prefix_ba_kernels.dot`. LBM `--export-dot` /
`--export-dot-png` likewise default to the launch matrix (`--export-dot-streams`
keeps the legacy stream-cluster graph). See `tests/test_skeleton_launch_order.cu`.

## Halo transfer matrix

`halo_messages()` lists, for this rank, every halo launch of one `run()` (user
and auto-inserted, in `run()` order, numbered like the launch-order rows) with
the cross-rank messages it issues: `DistMessage{op, peer_rank, elements, bytes, group}`
in issue order, `group` being the NCCL group within the exchange. It reads the
compiled halo kernels (`Kernel::dist_messages()`), so nothing is launched;
halos of single-process layouts report no messages.

`gather_halo_messages()`, `export_transfer_matrix_csv[_file]()` and
`export_transfer_matrix_text()` collect every rank's list (collective on a
distributed `System`) into a rank × message matrix: CSV
`rank,index,seq,kernel,group,op,peer,elements,bytes` (the file is written by
rank 0), or a log table with cells `S>3 5.24 MB (1310720)` / `R<3 ...` (size and
element count). LBM exposes it as `--export-transfers PATH` (with `--compile-only` to skip the solver loop);
`benchmarks/lbm_ldc/scripts/plotting/plot_transfers.py` draws the heatmap.

## Per-phase profiling

`Skeleton::run_profile()` issues the same schedule as `run()` but measures host
wall time around each **phase** and returns one `ProfilePhase` per phase:

```cpp
struct ProfileStep {
  std::string name;          // user kernel name, or the anchor label
  std::string view;          // "standard", "internal", "boundary"
  size_t stream = 0;
  double milliseconds = 0.0; // per-step host interval
};

struct ProfilePhase {
  std::vector<ProfileStep> steps;
  double milliseconds = 0.0; // makespan of the whole phase
};
```

A phase is a maximal region between **global joins**. A step starts a new phase
when it depends on the tail of every stream that is active in the current
phase; otherwise it joins the current overlapped region. For a schedule such as
`seed -> {left, right} -> join`, the result is three phases with step counts
`1 / 2 / 1`, and the middle phase reports the makespan of `left` and `right`,
not their sum. With `OverlapMode::None` every scheduled launch is its own phase.

`run_profile()` drains the phase's streams on the host before taking the
timestamp, so it does **not** use CUDA/HIP graph capture. Timings are host
wall-clock samples and should be treated as indicative, especially for
sub-100 µs kernels. The normal `run()` path is unchanged.

`benchmarks/cg` exposes the same data with `--profile-steps`. It reports the
mean/median/min/max of each phase across iterations, using the phase's step
labels as the phase label:

```text
iter_pre profile (3 phases, 16 samples):
  phase 0 [s:=Ap [internal]]                 4.21 ms  (mean; min 3.98, max 4.60)
  phase 1 [auto_halo_p]                      0.61 ms
  phase 2 [s:=Ap [boundary]]                 3.87 ms
  total phase mean                           8.69 ms
```

## Region masks (sub-field dependencies)

By default an access covers its whole field, so two kernels that write the same field
are always ordered. A `RegionMask` (`<carbon/region_mask.h>`, 256 bits) narrows a Map
access to a user-defined part of the domain; two accesses then conflict only when they
share a field, their modes conflict, **and** their masks intersect. Disjoint writers of
one field get no edge and may land on different streams.

```cpp
const RegionMask kInterior = RegionMask::from_label(0);
const RegionMask kBoundary = ~kInterior;            // full partition: "every other cell"

auto bc = carbon::Kernel::compute(dense, [&](carbon::KernelLoader& L) {
  L.region(kBoundary);                              // every Map access below is masked
  const auto lab = L.read(labels, RegionMask::all()); // read before the filter: everywhere
  const auto out = L.write(u);
  return [lab, out] CARBON_HD(Span s, Idx idx) {
    if (lab.read(s, idx, 0) == 0) { return; }       // the kernel filters itself
    out.write(s, idx, 0, /* Dirichlet value */ 0.0F);
  };
}, "bc");
```

Contract (Carbon trusts the declaration; a false one is a silent data race):

- **Bits are atoms.** You decide what each bit means; Carbon only assumes different bits
  are disjoint cell sets. Make them a full partition (e.g. one `uint8_t` label per cell,
  `from_label(label)`), so `~M` really is the rest of the domain.
- **Masks only declare.** They do not change the launch: the kernel still iterates its
  whole `DataView` and must skip cells outside its mask. The effective iteration set is
  the `DataView` band (Carbon-owned Internal / Boundary geometry) intersected with the
  cells the body accepts (your labels) — the two concepts are independent.
- **A mask covers every cell the access touches**, including reads made before the
  filter (declare those `RegionMask::all()`, as with `labels` above) and reads of other
  cells through the same view (Map must really be cell-local).
- **Stencil reads, halos and scalar readbacks are always whole-field.** A stencil reader
  — and the auto-halo in front of it — still waits on every masked writer of the field.
- **Same masks on every GPU and rank, fixed at kernel construction.** Under MPI the first
  `run()` of a pipeline with halos or reductions compares a hash of the pipeline shape
  (including masks) across ranks and throws `std::logic_error` on mismatch instead of
  letting NCCL deadlock.
- **An empty mask conflicts with nothing** and warns once per kernel at compile time.

API: `KernelLoader::region(M)` sets the default mask for every Map access of the kernel
(whichever order it is called in; `L.region()` reads it back, also in the launch phase),
and `read(f, M)` / `write(f, M)` / `read_write(f, M)` / `L(f, M)` pin one access.
`Skeleton::Options::ignore_region_masks` — or `CARBON_IGNORE_REGION_MASKS=1` at run time
— makes every access whole-field again, giving exactly the pre-region-mask schedule for A/B
debugging; `Skeleton::region_masks_honoured()` reports the effective setting.

Inspection: `export_kernel_list_dot()` adds a `region {…}` note under each masked access
chip, `export_streams_dot()` tooltips list each masked token's region, and both option lines show
`region_masks=ignored` when masks are off. Unmasked pipelines render exactly as before.

Costs to keep in mind (numbers from `benchmarks/skeleton_region_mask`, see
`benchmarks/README.md`):

- Every masked kernel still sweeps the whole grid, so concurrency pays off when the device
  has idle time (halo-bound multi-GPU, small grids): −12 … −16 % per iteration there,
  within noise once the device is saturated.
- Each extra stream costs roughly 1.5–3 µs of host time per iteration (events and
  fork/join). Many tiny disjoint kernels can be slower than the whole-field schedule; cap
  with `Options::max_streams` (2 was best for 16 boundary kernels).
- CUDA graph capture only covers single-stream schedules; with `use_cuda_graph` a masked
  multi-stream schedule falls back to eager launches (logged at compile).
  `max_streams = 1` keeps it capturable.

`examples/region_mask_cube.cu` is the minimal starting point: on a cube, one kernel fills
the boundary voxels and two update the interior of the same field, and the printed schedule
shows the boundary kernel running beside the interior chain (one stream with masks ignored).
`examples/region_mask_bc.cu` is the end-to-end walk-through (interior stencil + Dirichlet
boundary kernel, 3 streams vs 1, bit-identical check, `--dot` for both DOT views);
`tests/test_skeleton_region_mask.cu` pins the semantics.

## SubGrid (compact iteration over a region)

A masked Dense kernel still sweeps the whole grid and skips cells itself.
`carbon::SubGrid<Dense>` (`<carbon/sub_grid.h>`, included by `<carbon/carbon.h>`) is the
compact counterpart: `S = SubGrid(G, F, B)` stores, per GPU, the owned parent cells selected
by `F` as a lane table and launches one thread per lane. `B` is the region its kernels
declare. The parent can also be the block-sparse grid: `carbon::SubGrid<carbon::Block<>>`,
built with the same two `block.new_sub_grid(...)` factories (see "Block parent" below).

```cpp
using S = carbon::SubGrid<carbon::Dense>;
// From a uint8_t label field: the owned cells whose label bit is in the mask.
const S walls = dense.new_sub_grid(labels, RegionMask::from_label(1), "walls");
// From any predicate (evaluated once by a Dense kernel); B defaults to all.
const S shell = dense.new_sub_grid([n](carbon::KernelLoader&) {
  return [n] CARBON_HD(const Span& s, const Idx& i) { return on_face(s.global_location(i), n); };
}, kBoundary, "shell");

auto q = walls.newField<float, 1>(carbon::MemorySide::Device, "q");  // one value per lane
auto bc = carbon::Kernel::compute(walls, [&](carbon::KernelLoader& L) {
  const auto uG = L.read_stencil(u);  // parent field: own cell and neighbours (u's halo)
  const auto vG = L.write(v);         // parent field: the lane's cell
  const auto qS = L.read_write(q);    // SubGrid field
  return [=] CARBON_HD(const S::Span& s, const S::Idx& i) {
    vG.write(s, i, 0, uG.read(s, i, 0) + qS.read(s, i, 0));
  };
}, "wall_bc");
```

Rules:

- **Types.** `S::Span` and `S::Idx` derive from the parent's (the Idx adds its lane, the Span
  the lane table), so parent-field partitions accept them unchanged. SubGrid partitions take
  only `S::Span` / `S::Idx` and have no neighbour access.
- **Region.** The loader's kernel-wide region is preset to `B`, so parent-field Map accesses
  carry `B` and the Skeleton orders the kernel like a masked Dense kernel. The label
  constructor makes `F` match `B` by construction; with the predicate constructor, `B` is
  your declaration that every selected cell lies in `B` (same contract as above).
- **No halos on SubGrid fields.** `read_stencil` on one is a compile error. Stencil-reading
  a *parent* field is fine and makes the kernel an OCC candidate: lanes are stored
  parent-Internal first, then parent-Boundary, so `DataView::Internal` / `Boundary` are two
  contiguous lane ranges and Standard / Extended OCC split SubGrid kernels like Dense ones.
- **Ownership.** A SubGrid field can only be loaded by kernels of its own SubGrid; anything
  else throws `std::invalid_argument` when the kernel is built.
- **Immutable.** The lane table is built once on the host (deterministic order) and never
  changes; build a new SubGrid for a new selection. No SubGrid of a SubGrid, and no
  `compute_universal` over a SubGrid (static_assert).
- **Factories.** `Kernel::compute` / `Kernel::host_compute`, or `S.make_kernel(fn[, name][,
  shared_mem])` / `S.make_host_kernel(fn[, name])`. Tested on CUDA, SYCL and HIP (on NVIDIA).
- **Reductions.** `Kernel::reduce<T>` / `host_reduce<T>` / `reduce_universal<T>` accept a
  SubGrid, as do `S.make_reduce_kernel<T>(fn, dst[, op][, shared_mem][, options])`,
  `S.make_host_reduce_kernel<T>(fn, dst[, op])` and `S.make_universal_reduce_kernel<T>`.
  The per-cell lambda is `T(const S::Span&, const S::Idx&)` and may read parent and SubGrid
  fields. One thread per lane in fixed 256-thread blocks feeds the usual pipeline (block
  partials, multi-GPU combine, then the cross-rank stage, which no SubGrid test runs under
  MPI yet), so `ReductionResult`, the `reduce::` ops and `ReduceFinalize::BlockAtomic` work
  unchanged; `ReduceOptions::stage1_threads[_y]` throw.
  An empty SubGrid yields the op's identity. A reduction that stencil-reads a parent field
  is split by OCC into Internal / Boundary partials plus a merge, as on Dense.
- **Block parent.** `SubGrid<Block<SBlock>>` has the same API. Its lane table is flat: one
  `Block::Idx` (brick, intra-brick offset) per selected active voxel, parent-Internal bricks
  first, in brick then z / y / x order. That suits a few scattered voxels, which mostly sit
  in different bricks; a selection that fills whole bricks gains little over a Block kernel
  with a filter. Device kernels read parent fields anywhere their halo allows. Host kernels
  get spans built from the Block's host tables, which cover owned bricks only, so a host
  `neighbour` read must stay in the GPU's slab (slabs split along z). Any other parent type
  is a compile error.

```cpp
carbon::ReductionResult<double> wall_flux(sys);
auto k = walls.make_reduce_kernel<double>([&](carbon::KernelLoader& L) {
  const auto uG = L.read(u);  // parent field
  const auto qS = L.read(q);  // SubGrid field
  return [=] CARBON_HD(const S::Span& s, const S::Idx& i) -> double {
    return uG.read(s, i, 0) * qS.read(s, i, 0);
  };
}, wall_flux);                // reduce::Sum<double> by default
```

`examples/sub_grid_cube.cu` puts the cube boundary in a SubGrid, mixes SubGrid and Dense
kernels in one masked Skeleton, and times the boundary fill against the filtered Dense
kernel. On 2×RTX A4000 (µs per launch, Dense + filter / SubGrid): 64³ is launch-bound at
4.8 / 4.9, 128³ gives 9.6 / 3.6, 256³ gives 58.8 / 28.7 (SYCL 85.5 / 32.0), and 384³
gives 189 / 67. A SubGrid pays one lane-table read per thread and its parent accesses are
gathers, so it wins when the selection is a small fraction of the grid.
`examples/sub_grid_block_markers.cu` does the same over a Block ball: about one active voxel
in 1000 is a marker that gets an x/y-neighbour average, compared against a Block kernel that
filters on a label. On 2×RTX A4000 (µs per launch, Block + filter / SubGrid): 128³ gives
10.5 / 6.1, 256³ gives 79.1 / 5.8 (one GPU 155.9 / 6.0, SYCL 80.3 / 9.4, HIP on NVIDIA
79.0 / 4.9), and 256³ with one marker in 100 gives 126.2 / 23.5.
`tests/test_sub_grid*.cu` (`test_sub_grid_reduce.cu` for reductions, `test_sub_grid_block.cu`
for the Block parent) and `tests/sub_grid_compile_fail.cu` pin the semantics.
`test_sub_grid_jacobi_axpy.cu` is the end-to-end case: 100 int64 Jacobi + AXPY iterations
on Dense and Block. B (voxels missing a neighbour) and C (the rest) write disjoint regions
and run in parallel — C as a masked parent kernel or as a second SubGrid — under every OCC
mode. With OCC on, each Jacobi step has exactly two internal and two boundary launches;
values match a host reference exactly.
