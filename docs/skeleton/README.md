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
