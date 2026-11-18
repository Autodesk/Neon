# Skeleton DAG export — reference renderings

Visual snapshots of the dependency DAG `carbon::Skeleton` infers from a Jacobi
ping-pong pipeline, produced by `Skeleton::export_dot` (see
`<carbon/skeleton.h>`).

| File | Source |
|------|--------|
| `jacobi_skeleton.dot` | Raw GraphViz DOT, captured from `examples/jacobi_unified.cu --dot`'s `CARBON_LOG_INFO("--- DAG (GraphViz DOT) ---")` block. |
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
— if you want to see the multi-stream rendering, add an `export_dot` call to
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

`export_dot()` prefixes each schedule node with `#N` (the launch line number) and
annotates dashed sync edges with `(rec #R -> wait #W)`. When anchor fork/join is
active, extra dashed blue edges show the bracket ops.

`export_launch_matrix_dot()` renders the same sequence as a grid: one column per
stream, one row per operation (anchor syncs, event waits, launches, event
records). Dashed arrows connect each event record to its waits — red for step
events, blue for the anchor fork/join — so cross-stream hand-offs read as
diagonal arrows between columns.

`export_kernel_list_dot()` is the matching table of what was added: one row per
kernel in add order (auto-inserted halos dashed), one column per accessed
resource with its access mode (R / W / RW, stencil reads flagged), the reduced
data dependencies, and the launches each kernel lowers to (`run()` sequence
number, stream, view) so rows cross-reference the launch matrix.

`benchmarks/jacobi_nvidia` accepts `--dot prefix`, which writes `prefix_ab.dot`,
`prefix_ba.dot`, and matching `.txt` launch-order files; add `--dot-matrix` for
`prefix_ab_matrix.dot` / `prefix_ba_matrix.dot` and the kernel lists
`prefix_ab_kernels.dot` / `prefix_ba_kernels.dot`. See
`tests/test_skeleton_launch_order.cu`.

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
chip, `export_dot()` tooltips list each masked token's region, and both option lines show
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

`examples/region_mask_bc.cu` is the end-to-end walk-through (interior stencil + Dirichlet
boundary kernel, 3 streams vs 1, bit-identical check, `--dot` for both DOT views);
`tests/test_skeleton_region_mask.cu` pins the semantics.
