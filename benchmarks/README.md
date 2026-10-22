# carbon benchmarks

Larger / representative workloads aimed at headline performance numbers
(MLUPS, GLUPS, wall time). Functional and verification work belongs under
[`tests/`](../tests); didactic, single-concept programs belong under
[`examples/`](../examples). The benchmarks here are intended to be run and
their headline metric scraped — every program prints a single
`MLUPS=...` (or equivalent) line on stdout for easy harness consumption.

## Build

Benchmarks are built by default (toggle with `-DCARBON_BUILD_BENCHMARKS=OFF`
to skip). They share the same backend / MPI / HDF5 / nanoVDB knobs as the
rest of the carbon tree:

```bash
cmake -S . -B build
cmake --build build -j
```

Each benchmark builds as a top-level CMake target and lands under
`build/benchmarks/<name>/carbon_benchmark_<name>`.

## Programs

### `lbm_lid_driven_cavity`

3D lid-driven cavity Lattice-Boltzmann simulation on a uniform Cartesian
grid. Two-pop pull streaming, half-way bounce-back walls on five faces,
moving lid on the +y face, wrapped in a `carbon::Skeleton` so the halo
exchange between successive iterations is auto-inserted (the iteration
kernel calls `loader.read_stencil(fIn)` on its input population and the
skeleton fires a halo on that field whenever it has been written since
the last halo).

The lattice and collision model are runtime-selectable:

| `--lattice` | `--collision` | Status      | Notes                                                                                                            |
| ----------- | ------------- | ----------- | ---------------------------------------------------------------------------------------------------------------- |
| `q19`       | `bgk`         | default     | D3Q19 + standard SRT BGK. Port of `Neon/benchmarks/lbm-lid-driven-cavity-flow`.                                  |
| `q27`       | `bgk`         | supported   | D3Q27 + BGK. Same arithmetic, more directions (extra body-diagonal velocities). Modestly slower than D3Q19.       |
| `q27`       | `kbc`         | supported   | D3Q27 + Karlin-Bösch-Chikatamarla entropic stabilisation. Better numerical stability at low viscosity / high Re. |
| `q19`       | `kbc`         | rejected    | KBC's moment basis and shear-decomposition table are defined for D3Q27; not supported on D3Q19.                  |

The grid family is also runtime-selectable via `--grid dense|disg`. Both
families run the **identical** hot kernel — the pull-stream + half-way
bounce-back + collision body — through the grid-generic trusted-pull primitive
`Partition::read_pull`; only the per-slab storage layout and the auto-inserted
halo family (`Halo` vs `HaloDisg`) differ. This makes `dense` vs `disg` a
like-for-like measurement of the disaggregated layout on a real LBM workload.
`--checksum` (total domain mass) is grid-family- and GPU-count-invariant and
confirms the two families produce identical physics.

Measured (2×A100 80GB PCIe, D3Q19 + BGK, single precision, N=256):

| `--grid` | 1 GPU MLUPS | 2 GPU MLUPS (`--overlap standard`) | strong-scale |
| -------- | ----------: | ---------------------------------: | -----------: |
| `dense`  |        9469 |                              18702 |       1.98×  |
| `disg`   |        9496 |                              18759 |       1.98×  |

DenseDisg now matches Dense (within run-to-run noise) on this cavity. The earlier
24% single-GPU gap was **not** intrinsic to the five-region layout but an artefact
of `read_pull` reclassifying the disg regions on every one of the ~18 pull
directions per cell (+8 regs/thread → lower occupancy on a memory-bound kernel).
`read_pull` now uses a constant-delta fast path for same-region neighbours (all
x/y neighbours plus internal↔internal z-neighbours), reclassifying only across the
thin internal↔boundary seam — the disg fast kernel compiles to the same 48
regs/thread as Dense. This reproduces the Neon EuroPar result that dense vs
disaggregated MLUPS differ negligibly.

Build:

```bash
cmake --build build --target carbon_benchmark_lbm_lid_driven_cavity -j
```

Run (single GPU):

```bash
CUDA_VISIBLE_DEVICES=0 \
./build/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity \
  --n 256 --iters 1000 --warmup 100 --report-every 0
```

Run (all visible GPUs):

```bash
./build/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity \
  --n 256 --iters 1000 --warmup 100
```

Run (distributed, one rank per GPU). The benchmark is MPI/NCCL-aware: when
carbon is built with `-DCARBON_ENABLE_MPI=ON` and launched under `mpiexec -n R`,
each rank binds to one GPU, halos flow over NCCL between adjacent ranks (carbon's
strict one-rank-per-GPU model), and rank 0 prints the single global `MLUPS=...`
line (tagged `transport=mpi+nccl`). A single-process launch (no `mpiexec`) is
byte-for-byte unchanged and still uses every visible GPU via intra-process peer
copies. Build the MPI variant into its own tree:

```bash
cmake -S . -B build-mpi -DCARBON_ENABLE_MPI=ON
cmake --build build-mpi --target carbon_benchmark_lbm_lid_driven_cavity -j
mpiexec -n 2 \
  ./build-mpi/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity \
  --n 384 --iters 2000 --warmup 200 --overlap standard
```

**Device pinning.** If a rank sees more than one GPU it falls back to
`device = local_rank % visible_devices` (correct on a single node, with a
warning). For clean multi-node runs pin one GPU per rank at the launcher —
e.g. `srun --gpus-per-task=1`, `mpirun --map-by ppr:1:node:PE=...`, or a
`CUDA_VISIBLE_DEVICES=$LOCAL_RANK` wrapper — so each rank sees exactly its GPU.
`--overlap standard` is recommended under MPI so the NCCL halo overlaps the
interior compute instead of serialising behind it.

Measured (2×A100 80GB PCIe, D3Q19 + BGK, single precision, `--overlap standard`,
one rank per GPU over MPI+NCCL vs the single-GPU direct baseline):

| `--grid` | N   | 1 GPU MLUPS | 2 ranks (MPI+NCCL) | strong-scale |
| -------- | --- | ----------: | -----------------: | -----------: |
| `dense`  | 256 |        9452 |              18202 |       1.93×  |
| `disg`   | 256 |        9461 |              17736 |       1.87×  |
| `dense`  | 384 |        9770 |              19256 |       1.97×  |
| `disg`   | 384 |        9782 |              18921 |       1.93×  |

NCCL cross-rank halo lands within a few percent of the intra-process peer-copy
2-GPU numbers above (same node, PCIe), and `--checksum` agrees across the
single-process and MPI paths to float summation-order noise — confirming the
distributed path is numerically faithful.

> For a full runbook (build knobs, single/multi/distributed launch, device
> pinning, DGX / 8-GPU NVLink guidance, reference numbers, and the gotchas
> learned validating the distributed path) see
> [`docs/lbm_benchmark_runbook.md`](../docs/lbm_benchmark_runbook.md).

#### CLI

| Flag                  |    Default | Meaning                                                                                                     |
| --------------------- | ---------: | ----------------------------------------------------------------------------------------------------------- |
| `--n N`               |        128 | Cavity edge length in cells. Domain is `N x N x N`.                                                          |
| `--iters N`           |       1000 | Total iterations including warmup.                                                                          |
| `--warmup N`          |        100 | Warmup iterations excluded from the timing window.                                                          |
| `--re R`              |      100.0 | Reynolds number; sets the BGK / KBC relaxation through `omega = 1 / (3 nu_lb + 0.5)` with `nu_lb = ulb * (N - 1) / Re`. |
| `--ulb R`             |       0.05 | Lid velocity in lattice units; override for fixing Mach while sweeping Re.                                  |
| `--lattice {q19,q27}` |      `q19` | Lattice stencil. D3Q27 has 8 extra body-diagonal velocities and is required by KBC.                        |
| `--collision {bgk,kbc}` |    `bgk` | Collision model. KBC requires `--lattice q27`.                                                              |
| `--grid {dense,disg}` |    `dense` | Grid family. `dense` = `carbon::Dense` (aggregated Cartesian z-slabs); `disg` = `carbon::DenseDisg` (disaggregated five-region LBM layout). Both run the identical hot kernel; only the storage layout + auto-halo family differ. |
| `--checksum`          |        off | After the loop, Sum-reduce the final populations and print `CHECKSUM=<total mass>`. Grid-family- and GPU-count-invariant; used to verify `dense` vs `disg` numerical parity. |
| `--report-every N`    |        100 | Log progress every N iterations. Pass `0` to silence the loop.                                              |
| `--report PATH`       |        off | On success write a structured JSON report to `PATH` (consumed by `scripts/sweep_and_plot.py`).              |
| `--label STR`         |     `""`   | Free-form label written into the JSON report under `config.label`. Useful for tagging sweeps.               |
| `--no-warmup-skip`    |        off | Include warmup in the timing window (debug aid for short runs).                                             |

#### Output

The program prints a one-line summary on stdout that's easy to scrape:

```
MLUPS=2497.16 (cavity N=128 iters=150 gpus=1 lattice=q19 collision=bgk)
```

plus a more verbose `info`-level line through the carbon logger:

```
lbm_lid_driven_cavity: DONE  lattice=q19 collision=bgk timing_window=150 \
  iters in 0.126s  =>  2497.16 MLUPS (2497.16 MLUPS/GPU)  overall=0.170s
```

MLUPS = `(N^3 * iters_timed) / (1e6 * elapsed_seconds)`. `MLUPS/GPU` is the
total divided by the number of visible GPUs (`carbon::get_all_devices()`).

When `--report PATH` is passed, a structured JSON report is also written
(schema `carbon-lbm-report-v1`):

```json
{
  "schema": "carbon-lbm-report-v1",
  "benchmark": "lbm_lid_driven_cavity",
  "timestamp_utc": "2026-05-06T13:05:51Z",
  "config":  { "n": 256, "iters": 600, "warmup": 200, "lattice": "q19",
               "collision": "bgk", "q": 19, "re": 100.0, "ulb": 0.05,
               "nu_lb": 0.127, "omega": 1.78, "label": "single A100" },
  "system":  { "gpu_count": 1, "gpus": [{ "device_id": 0,
               "name": "NVIDIA A100 80GB PCIe", "compute_capability": "8.0",
               "total_memory_bytes": 85094825984,
               "multiprocessor_count": 108 }] },
  "results": { "timed_iters": 400, "timing_seconds": 1.31,
               "overall_seconds": 1.49, "cells_per_iter": 16777216,
               "mlups": 5133.9, "mlups_per_gpu": 5133.9,
               "effective_bandwidth_gb_s": 780.4 }
}
```

`effective_bandwidth_gb_s` assumes the pull-stream + ping-pong layout
(2 × Q × `sizeof(float)` bytes per cell update) and is what to compare
against the GPU's HBM peak when sanity-checking new sizes.

#### Sweeps + plots (Neon-style figures)

`scripts/sweep_and_plot.py` drives a parameter sweep over the binary and
plots the resulting `*.json` reports the same way the Neon paper presents
LBM throughput — MLUPS on the y-axis vs N (size sweep) or vs GPU count
(strong scaling), one curve per `(lattice, collision, label, gpus)` series.
Standard library + `matplotlib` only.

Single-GPU size sweep (10 points from 64 to 384, ≈40 s on an A100):

```bash
python3 benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py \
  --bench ./build/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity \
  --mode sweep+plot \
  --output-dir reports/single_gpu_q19_bgk \
  --sizes 64 96 128 160 192 224 256 288 320 384 \
  --gpus 1 \
  --lattice q19 --collision bgk \
  --iters 600 --warmup 200 \
  --label "single A100 80GB PCIe" \
  --plot-x size \
  --plot-out reports/single_gpu_q19_bgk/mlups_vs_n.png
```

Multi-config comparison plot (3 model combos × 8 sizes on one axes):

```bash
for combo in "q19 bgk" "q27 bgk" "q27 kbc"; do
  read lat coll <<< "$combo"
  python3 benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py \
    --mode sweep \
    --output-dir reports/model_compare \
    --sizes 64 96 128 160 192 224 256 320 \
    --lattice $lat --collision $coll \
    --label "A100 80GB PCIe"
done
python3 benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py \
  --mode plot \
  --output-dir reports/model_compare \
  --plot-x size --plot-y mlups \
  --plot-out reports/model_compare/mlups_vs_n_models.png
```

Strong-scaling sweep at fixed N=256 across 1..G GPUs:

```bash
python3 benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py \
  --mode sweep+plot \
  --output-dir reports/strong_scaling_n256 \
  --sizes 256 \
  --gpus 1 2 \
  --lattice q19 --collision bgk \
  --iters 1200 --warmup 400 \
  --plot-x gpus --plot-y mlups \
  --plot-out reports/strong_scaling_n256/mlups_vs_gpus.png
```

Useful flags:

| Flag                  | Meaning                                                                       |
| --------------------- | ----------------------------------------------------------------------------- |
| `--mode`              | `sweep`, `plot`, or `sweep+plot` (default).                                   |
| `--input-dirs`        | `plot` mode: one or more report dirs to merge onto a single figure.           |
| `--plot-x`            | `size` (MLUPS vs N, default) or `gpus` (strong scaling).                      |
| `--plot-y`            | `mlups` (default), `mlups_per_gpu`, or `bandwidth` (effective HBM GB/s).      |
| `--dry-run`           | `sweep` mode: print every command without executing.                          |

GPU sweeps select consecutive devices via `CUDA_VISIBLE_DEVICES=0,...,g-1`,
so `--gpus 1 2 4` requires at least 4 visible devices. For carbon built
with `-DCARBON_ENABLE_MPI=ON`, replace the binary invocation in the script
(see `run_one`) with an `mpiexec -n $gpus ...` wrapper if you want
distributed sweeps; the JSON report's `gpu_count` will then reflect ranks
× one-GPU-per-rank.

#### Reference data points

On a single NVIDIA A100 80 GB PCIe with carbon built `Release` against CUDA
12.x. The two columns below are taken from the saturated-throughput plateau
(N ≥ 192) and the launch-bound regime (N = 128) so it's clear how much of
the headline number is real steady-state throughput:

| `--lattice` | `--collision` | MLUPS @ N=128 | MLUPS @ N≥192 (saturated) | Effective HBM GB/s |
| ----------- | ------------- | ------------: | ------------------------: | -----------------: |
| `q19`       | `bgk`         |          2520 |                  **5130** |               ~780 |
| `q27`       | `bgk`         |          2330 |                  **3500** |               ~755 |
| `q27`       | `kbc`         |          1990 |                  **1950** |               ~420 |

Effective bandwidth = `mlups × 2 × Q × sizeof(float)` (the pull-stream +
ping-pong layout reads `Q` populations and writes `Q` populations per cell).
On A100 80 GB PCIe the HBM peak is ~1935 GB/s, so D3Q19 + BGK runs at
~40 % of peak — typical for a real LBM kernel — and D3Q27 + BGK is just
slightly behind because the larger Q amortises the per-cell rho/u work
better. KBC's bottleneck is compute, not bandwidth: the per-cell entropic
ratio + shear-decomposition adds ~3× the FLOP count of BGK, which is why
its effective bandwidth drops well below the BGK runs at the same lattice.

Numbers scale with `N^3` once the kernel is compute / memory-bound
(typically `N ≥ 192`); at very small `N` the per-iteration launch + halo
overhead dominates and the headline figure understates steady-state
throughput. Pick a size that gives at least a few hundred milliseconds of
timed work — the JSON report includes `timing_seconds` so it's easy to
spot.

For comparison with the published Neon LBM numbers
([Mahmoud et al., IPDPS 2024 grid-refinement LBM
paper](https://arxiv.org/pdf/2308.08085), single A100 40 GB):

| Implementation                        | Hardware            | Lattice / collision | MLUPS    |
| ------------------------------------- | ------------------- | ------------------- | -------: |
| **Carbon** (this benchmark, N=256)    | A100 80 GB PCIe ×1  | D3Q19 / BGK         | **5130** |
| Neon dGrid uniform (IPDPS 2022)       | A100 40 GB SXM ×1   | D3Q19 / BGK         |    ~1900 |
| Neon AMR (IPDPS 2024)                 | A100 40 GB ×1       | D3Q19 / BGK         |    >2250 |

Caveat: the Neon paper's A100 is the 40 GB SXM4 (1.555 TB/s HBM2) and our
A100 is the 80 GB PCIe (1.935 TB/s HBM2e), so ~1.24× of the gap is "free"
from extra memory bandwidth. The remaining ~2× comes from the pull-stream
+ wall-bitmask kernel structure (one branchless read of a precomputed
27-bit neighbour mask vs. classifying each neighbour cell per iteration).

#### Implementation notes

- Two `carbon::Dense::Field<float, Q>` populations (ping-pong, with `Q ∈
  {19, 27}`); one `carbon::Dense::Field<int32_t, 1>` cell-type field
  carrying both the classification (Bulk / BounceBack / MovingWall) and
  the precomputed `wall_neighbour_bits` mask packed into a single int
  (4-bit class + up to 28-bit mask, plenty for D3Q27's 27-direction mask).
- The wall-mask precomputation is one shot before the timing window —
  amortises the per-iteration cost of branching on each neighbour's
  classification down to a single packed-int read inside the hot kernel.
- One iteration kernel does the full collide-stream as a fused pull (read
  Q neighbours, compute rho/u, apply collision, write Q outputs) and is
  the only kernel inside each `carbon::Skeleton`. Two skeletons (one per
  ping-pong direction) are precompiled before the timing loop, so the
  hot loop is a tight `sk.run()` per iteration with no allocation /
  recompilation.
- The hot kernel is templated on a `Lattice` traits struct (`D3Q19` /
  `D3Q27` in `src/d3q19.h` / `src/d3q27.h`) and a `Collision` policy
  (`BgkCollision` / `KbcCollision` in `src/lbm_kernels.h`). Every
  `(Lattice, Collision)` pair lives in the binary; main dispatches on
  the CLI flags. nvcc's `#pragma unroll` on the per-Q loops produces
  fully-unrolled code that's competitive with a hand-unrolled body
  (within ~1% for D3Q19 + BGK).
- KBC implementation mirrors `Neon/apps/lbmMultiRes/collide.h::collideKBC`
  bit-for-bit: per-cell second-moment stress `Pi[6]` projection through
  the lattice's `moment(q)` matrix, per-direction shear-decomposition
  via `shear_decompose(q, Pi)`, entropic ratio `e0 / e1`, per-cell
  `gamma = 1/beta - (2 - 1/beta) * e0 / (tiny + e1)`, then the standard
  `f_q^post = f_q - beta * (2 * deltaS_q + gamma * deltaH_q)` write-back.
  `tiny = 1e-7` guards the `1/feq` divisions inside the inner products
  when `feq` rounds to zero.
- Multi-GPU correctness comes for free from `Skeleton::compile`'s auto-halo
  insertion driven by the `loader.read_stencil(fIn)` call inside the
  iteration kernel — no manual halo plumbing in the driver, same body
  for D3Q19 and D3Q27.
- Compared to the Neon reference: same physics, same magic numbers
  (lattice weights, BGK formula, bounce-back rules, KBC moment matrix
  and shear-decomposition table), same two-pop scheme. The scheduler /
  dependency tracking layer differs — Neon's `Skeleton::sequence` is
  built from `Neon::set::Container`s carrying `Pattern::STENCIL` /
  `Pattern::MAP` annotations; carbon's `carbon::Skeleton` infers the
  same DAG from `KernelLoader::read_stencil` / `read` / `write` tokens.

### `reduce_throughput`

Times repeated three-stage `Sum` reductions over Dense, DenseDisg, and Block
(sphere predicate) grids. Prints one scrapeable line per family (median over
`--repeats` timed windows):

```
REDUCE_GBs=388.6 ELEMENTS_PER_S=9.7e10 elapsed_seconds=0.086 grid=dense gpus=1 n=256 iters=500 batch_iters=32 repeats=5
```

| Flag            | Default | Meaning                                              |
| --------------- | ------: | ---------------------------------------------------- |
| `--n N`         |     256 | Cube edge length                                     |
| `--iters N`     |     500 | Timed outer iterations per repeat                    |
| `--batch-iters N` |   1   | Reductions batched inside each timed iteration       |
| `--warmup N`    |      50 | Warmup iterations (excluded)                         |
| `--repeats N`   |       5 | Independent timed samples; stdout reports median     |
| `--gpus N`      |     all | Limit to first N visible GPUs                        |
| `--report PATH` |     off | JSON report base path (`.dense` / `.disg` / `.block` suffix) |

**Interpretation:** at `--batch-iters 1`, multi-GPU GB/s is dominated by stage-2/3
sync jitter and is **not** a strong-scaling metric. Use `--batch-iters 32` (or higher)
for scaling sweeps. Dense and DenseDisg should track each other; Block reduce iterates
active voxels only.

Build target: `carbon_benchmark_reduce_throughput`.

### `skeleton_overlap`

Dense Jacobi ping-pong under each `OverlapMode`; one scrapeable line per mode:

```
MLUPS=19740 elapsed_seconds=0.170 overlap=none gpus=1 n=256 iters=200 repeats=3
```

| Flag               | Default | Meaning                                    |
| -------------------- | ------: | ------------------------------------------ |
| `--n N`              |     256 | Cube edge length                           |
| `--iters N`          |     200 | Total Jacobi sweeps (even)                 |
| `--warmup N`         |      40 | Warmup sweeps (excluded)                   |
| `--repeats N`        |       3 | Median over independent timed samples      |
| `--min-elapsed-ms N` |     200 | Warn on stderr if median window is shorter  |
| `--gpus N`           |     all | Limit to first N visible GPUs              |
| `--report PATH`      |     off | JSON base (`.none` / `.standard` / … suffix) |

**Interpretation:** on cheap Jacobi + PCIe (no NVLink), `OverlapMode::None` often
beats overlap modes because OCC splits each step into Internal+Boundary launches plus
cross-stream events while the hidden halo payload is small. OCC pays off when
compute-per-step dominates halo latency (heavier kernels, NVLink). Check
`elapsed_seconds=` — values below ~200 ms are launch-bound.

Build target: `carbon_benchmark_skeleton_overlap`.

### `block_sparse`

Block Jacobi on a sphere-inscribed cube; reports MLUPS over **active** owned
cells plus the geometric active fraction:

```
MLUPS=20166 elapsed_seconds=0.052 halo_pair_count=2 active_fraction=0.31 grid=block gpus=2 n=256 iters=200 repeats=3
```

CLI matches `skeleton_overlap` (without `--min-elapsed-ms`). Also prints
`halo_pair_count=` (coalesced device halo transfers) so regressions in
`HaloBlock` coalescing are visible without re-profiling.

Build target: `carbon_benchmark_block_sparse`.

### `halo_scalability`

Isolated halo-exchange bandwidth probe (Dense vs DenseDisg, standard vs lattice
semantic). Appends CSV rows for plotting — does **not** run a stencil kernel.

```bash
./build/benchmarks/halo_scalability/carbon_benchmark_halo_scalability \
  --N 256 --gpus 2 --components 1,19,27 --semantic standard \
  --iters 200 --warmup 50 --csv reports/halo.csv
```

| Flag           | Default              | Meaning                         |
| -------------- | -------------------- | ------------------------------- |
| `--N N`        |                  256 | Cube edge length                |
| `--gpus N`     |                  all | Limit to first N GPUs           |
| `--components` | 1,3,9,19,27          | Component counts (standard sem.)|
| `--semantic`   | standard             | `standard` or `lattice`         |
| `--lattice`    | required for lattice | `d3q19`, `d3q27`, `*_grouped`  |
| `--iters N`    |                  200 | Timed halo iterations           |
| `--warmup N`   |                   50 | Warmup iterations               |
| `--csv PATH`   |                  off | Append CSV rows                 |

Use this to measure raw halo transfer GB/s; use `skeleton_overlap` for end-to-end
Skeleton + OCC behaviour on a real stencil pipeline.

Build target: `carbon_benchmark_halo_scalability`. See also
`benchmarks/halo_scalability/scripts/run_local.sh` and `plot.py`.

## Unified harness

Run the three micro-benchmarks with median aggregation and JSON reports:

```bash
cmake --build build -j --target \
  carbon_benchmark_reduce_throughput \
  carbon_benchmark_skeleton_overlap \
  carbon_benchmark_block_sparse

# Quick matrix: n=256, gpus=1,2, repeats=3 (~1 min on 2×A100)
python3 benchmarks/scripts/run_benchmarks.py \
  --build-dir build --output-dir reports/bench-quick --mode quick

# Full matrix: n=128,192,256 × gpus=1,2, repeats=5
python3 benchmarks/scripts/run_benchmarks.py \
  --build-dir build --output-dir reports/bench-full --mode full

# Or one shell wrapper (builds targets, runs quick, plots summary.png)
bash benchmarks/scripts/run_local.sh build reports/bench-quick quick
```

Outputs under `--output-dir`:

- `summary.csv` / `summary.md` — all runs with metric, elapsed, topo
- Per-run JSON (`carbon-bench-report-v1`) when binaries receive `--report`
- `summary.png` from `benchmarks/scripts/plot_summary.py` (optional)

Set `CARBON_BENCH_TOPO` to override interconnect label in JSON (otherwise parsed
from `nvidia-smi topo -m` GPU0→GPU1 entry).

## How to read results

| Symptom | Likely cause | What to do |
| ------- | ------------ | ---------- |
| `elapsed_seconds` ≪ 0.2 s | Launch-bound / short window | Increase `--iters` or `--batch-iters`; heed `WARNING launch_bound` |
| `none` MLUPS > overlap modes on Jacobi | Cheap kernel + OCC overhead | Expected on PCIe; try heavier kernel or NVLink box |
| `reduce` 2-GPU GB/s swings 2× at batch=1 | Sync-dominated single reduce | Use `--batch-iters 32+` and `--repeats 5` |
| `block_sparse` 2-GPU ≪ 1-GPU | Per-brick halo copies (pre-coalescing) | Should show `halo_pair_count=2`; if hundreds, coalescing regressed |
| Multi-GPU MLUPS ≪ 1-GPU / gpus | PCIe halo latency | Check topo (NODE/PHB vs NV#); not a carbon bug by itself |

## Reference data (2× NVIDIA A100 80GB PCIe, NODE interconnect, Release, n=256)

Captured with `run_benchmarks.py --mode quick --repeats 3` after Block halo
coalescing (2026-07). Your box may differ — use the harness to refresh.

| Benchmark | Config | 1 GPU | 2 GPU |
| --------- | ------ | ----: | ----: |
| block_sparse | active-cell MLUPS | 20027 | 20131 |
| skeleton_overlap | none | 19898 | 5624 |
| skeleton_overlap | standard | 10983 | 2919 |
| reduce_throughput | dense, batch=1 | 388 GB/s | 473 GB/s |
| reduce_throughput | dense, batch=32 | 391 GB/s | 475 GB/s |
| reduce_throughput | disg, batch=32 | 358 GB/s | 469 GB/s |

Notes:

- `block_sparse` 2-GPU matches 1-GPU after halo coalescing (`halo_pair_count=2`).
- `skeleton_overlap` 2-GPU drop is PCIe halo bound; overlap modes ~2× slower than `none`.
- `reduce` disg tracks dense at batch=32; batch=1 multi-GPU numbers are noisy.
