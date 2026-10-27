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
same-stream waits, unused event records, and anchor joins already covered by an
earlier cross-stream wait.

Extended and TwoWayExtended are further rewrites of the same step graph
(`compile_step_graph_occ_`). Extended splits each Compute producer of a
stencil-read field into Boundary then Internal, and the halo waits on the
Boundary half only, so it overlaps the producer's Internal half. TwoWayExtended
also splits the Map-only Compute consumers of the stencil's output into
Internal + Boundary; when a consumer only reads what a split producer wrote,
each half waits on the matching half alone. A stencil reduction gets the same
partial + merge launches in every mode.

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
