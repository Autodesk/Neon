# `lbm_lid_driven_cavity` scripts

Drivers and plotters for the LBM lid-driven cavity benchmark
(`carbon_benchmark_lbm_lid_driven_cavity`). Every script talks to the binary
through its `--report PATH` JSON output (schema `carbon-lbm-report-v1`), so the
run scripts and the plotters can be used independently: re-plotting never
requires re-running. For a short tour with examples, start with the
[quick start](../README.md).

All paths below are relative to the carbon repository root; every shell script
`cd`s there itself, so they can be launched from anywhere. Run scripts live in
`scripts/`, the plotters in `scripts/plotting/`.

| Script | Kind | What it produces |
| --- | --- | --- |
| [`sweep_and_plot.py`](#sweep_and_plotpy) | generic sweep + plot | MLUPS per GPU (or total MLUPS / bandwidth) vs N / vs GPUs / per config, for any single-process parameter grid |
| [`run_local_single_gpu_n256.sh`](#run_local_single_gpu_n256sh) | local quick run | bar chart of the 3 lattice/collision configs, 1 GPU, N=256 |
| [`run_local_full_matrix.sh`](#run_local_full_matrixsh) | local quick run | bar chart of configs × backend (CUDA/SYCL) × grid × {1, all} GPUs |
| [`lbm-dgx-sweep-run.sh`](#dgx-sweep) | DGX sweep | any lattice/collision × precision × sizes × GPU counts × grids × decompositions × transports × halo priority, from one YAML |
| [`plotting/plot_scaling.py`](#plotting-a-sweep) | plotter | MLUPS per GPU vs GPUs / vs N, hp-halo speedup |
| [`plotting/plot_disg_vs_dense.py`](#plotting-a-sweep) | plotter | DenseDisg / Dense MLUPS ratio of a sweep that ran both layouts |
| [`plotting/plot_best_family.py`](#plotting-a-sweep) | plotter | N × GPU-count colormap of the fastest of Dense / DenseDisg × Peeraccess / NCCL |
| `../data/<set>/reproduce.sh` | result set | re-runs and / or re-plots one checked-in result set from its frozen YAML |

`plotting/lbm_reports.py` (report loading) and `lbm_sweep_config.py` (sweep
YAML, run matrix and report labels) are the libraries the scripts share.

## Terminology

The DGX sweeps compare *families*, encoded in the report label and filename
(`<prefix>-<lattice>-<collision>-<precision>-<label>-n<N>-g<G>.json`):

| Label | Grid (`--grid`) | Transport |
| --- | --- | --- |
| `dense-intra` / `disg-intra` | `dense` / `disg` | **Peeraccess** — one process, all GPUs, intra-process NVLink P2P copies |
| `dense-mpi` / `disg-mpi` | `dense` / `disg` | **NCCL** — `mpiexec -n G`, one rank per GPU, halos over NCCL (needs the `-DCARBON_ENABLE_MPI=ON` build) |
| `dense-mpip2p` / `disg-mpip2p` | `dense` / `disg` | **MPI (P2P)** — same MPI build, CUDA-aware MPI instead of NCCL (`CARBON_DIST_TRANSPORT=mpi`) |
| `<grid>-<decomposition>-<transport>-hp<off\|on>` | `dense_distributed` / `disg_distributed` | NCCL or MPI (P2P), one rank per GPU, the domain split by `--decomposition slab\|pencil\|brick` |

A `-hpon` suffix on a `dense` / `disg` label marks a run with `--hp-halo`
(high-priority halo stream).

Every scaling plot shows **MLUPS per GPU**:

```
MLUPS per GPU = MLUPS(config, N, G) / G
```

Perfect strong scaling is a horizontal line, drawn dashed at the 1-GPU
`dense-intra` run of the same N (the *reference*). G comes from the report's
file name: MPI reports written before the driver counted every rank carry
`system.gpu_count = 1`, and their `results.mlups_per_gpu` is the total.

## Generic tool

### `sweep_and_plot.py`

The general-purpose driver. Two modes, combinable:

- `--mode sweep` runs the binary over the cartesian product of `--sizes`,
  `--gpus`, `--grids` (`dense`, `disg`) and `--lattice/--collision` (or
  `--configs q19:bgk q27:kbc ...`) and writes one `run-NNN-...json` per run
  into `--output-dir`. GPU counts are selected with
  `CUDA_VISIBLE_DEVICES=0..g-1`.
- `--mode plot` loads every report from `--output-dir` (or several
  `--input-dirs`, to overlay sweeps) and renders one curve per
  `(grid, lattice, collision, label, gpus)` series.
- `--mode sweep+plot` (default) does both.

Plot axes: `--plot-x size|gpus|config` and
`--plot-y mlups_per_gpu|mlups|bandwidth` (default `mlups_per_gpu`). `--dry-run`
prints the commands without running them. Needs `matplotlib`.

```bash
python3 benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py \
  --bench ./build/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity \
  --output-dir reports/strong_scaling_n256 \
  --sizes 256 --gpus 1 2 4 8 --grids dense disg \
  --lattice q19 --collision bgk \
  --plot-x gpus --plot-out reports/strong_scaling_n256/mlups_per_gpu_vs_gpus.png
```

It only drives single-process runs; for MPI sweeps use the [DGX sweep](#dgx-sweep).
Plot mode reads MPI reports too.

## Local quick runs

Both scripts are thin wrappers around `sweep_and_plot.py` meant for a
workstation with an existing native build.

### `run_local_single_gpu_n256.sh`

One GPU, N=256, the three valid configs (`q19/bgk`, `q27/bgk`, `q27/kbc`).
Builds the binary in `BUILD_DIR` if it is missing (the tree must already be
configured). Output: `reports/single_gpu_n256_local/` with three JSON reports
and `mlups_by_config.png`.

Knobs: `ITERS=600`, `WARMUP=200`, `BUILD_DIR=build`, `OUT_DIR=...`, `JOBS=8`.

### `run_local_full_matrix.sh`

Same three configs at N=256, crossed with every backend whose binary it finds
(CUDA: `build/`, `build-mpi/`, `build-cuda/`; SYCL: `build-sycl/`,
`build-sycl-mpi/`), with `GRIDS` and with GPU counts `1` and "all GPUs".
Missing backends are skipped. Output: `reports/full_matrix_local/` with one
JSON per cell and `mlups_per_gpu_full_matrix.png` (an all-GPU bar as tall as
the 1-GPU bar means perfect scaling).

Knobs: `ITERS`, `WARMUP`, `N=256`, `GRIDS=dense` (`dense disg`), `GPUS_LIST`,
`OUT_DIR`, and `CUDA_BENCH=` / `SYCL_BENCH=` to point at a binary directly.

## DGX sweeps (8× A100-SXM4-40GB, NVLink)

`lbm-dgx-sweep-run.sh` builds Release trees **inside Docker** (`build/` for
intra-process, `build-mpi/` with `-DCARBON_ENABLE_MPI=ON`), runs the matrix a
YAML file describes, and drops JSON into `../data/<prefix>/json/`; the
plotters take the same YAML, read from there and write the PNGs next to it.
The checked-in PNGs under `../data/` are the reference results, each set with
its frozen YAML and a `reproduce.sh` (see the
[result sets](#result-sets)).

Two Docker images are needed: `carbon-cuda:local` (`image_intra`, from
`docker/Dockerfile.cuda`) compiles both trees and does the Peeraccess runs;
`carbon-cuda-nccl-pinned:local` (`image_mpi`) does the MPI runs with NCCL
2.27.7 for CUDA 12.2, because the NCCL of plain `carbon-cuda` needs a newer
driver and fails `ncclCommInitRank` on this DGX.

### DGX sweep

```bash
./benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-sweep-run.sh \
  benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-sweep.yml
```

- Parameters come from an optional YAML file (see
  [`lbm-dgx-sweep.yml`](lbm-dgx-sweep.yml), which documents every key), whose
  keys are the lowercase knob names (`configs`, `sizes`, `gpus`, `grids`,
  `decompositions`, `hp_halo`, `reps`, ...). List-valued keys accept a list or
  a space-separated string; `sizes` / `gpus` also accept an inclusive
  `{start, stop, step}` range. Unknown keys, invalid values and contradictory
  combinations (a distributed grid without an MPI transport, `hp_halo: on`
  with `overlap: none`) are rejected. Env vars (`CONFIGS`, `SIZES`,
  `GPUS_LIST`, `GRIDS`, ...) override the file; `DRY_RUN=1` prints the
  resolved configuration and every run, then exits.
- The run matrix is `CONFIGS` × `PRECISIONS` × `SIZES` × the
  (grid, decomposition, transport, hp-halo, GPU count) cells that
  `lbm_sweep_config.runs` expands:
  - `dense` / `disg` run every `TRANSPORTS` entry (`intra mpi mpip2p`) on
    every `GPUS_LIST` count (default 1..8);
  - `dense_distributed` / `disg_distributed` run under `mpiexec`, one rank per
    GPU, for every `DECOMPOSITIONS` entry: slab on `GPUS_LIST`, pencil on
    `GPUS_PENCIL` (default 4 8), brick on `GPUS_BRICK` (default 8); they skip
    `intra`;
  - NCCL and MPI (P2P) are skipped at 1 GPU, unless `MPI_SINGLE_GPU`
    (`mpi_single_gpu: true`) runs them for `dense` / `disg` as one rank
    pinned to `CUDA_VISIBLE_DEVICES=0` (a plain `mpiexec -n 1` would pick up
    every visible GPU);
  - every cell runs once per `HP_HALO` entry (`off` default, `on` adds
    `--hp-halo`), with `--overlap OVERLAP` (default `standard`).
- `CONFIGS` holds lattice:collision pairs, any of `q19:bgk` (D3Q19 + BGK,
  the default), `q27:bgk` (D3Q27 + BGK) and `q27:kbc` (D3Q27 + KBC; KBC
  exists only for D3Q27). `PRECISIONS` is `fp32` (default) and/or `fp64`
  (binary `--precision`, every grid). fp64 roughly halves the MLUPS and
  doubles the memory: on a 40 GB A100 the 1-GPU run at N=512 does not fit in
  fp64 (N=448 does), so that run fails out of memory and the fp64 series has
  no reference at N=512. `q27:kbc` with `fp64` does not launch yet ("too many
  resources requested for launch": the kernel needs more registers than the
  library's fixed 512-thread blocks allow), so those runs fail too.
- Timing is set by `WARMUP` (default 400 untimed solver steps), `REPS`
  (default 4 timed repetitions) and `ITERS_PER_REP` (default 400 solver steps
  averaged per repetition). They map to the binary's
  `--iters WARMUP+REPS*ITERS_PER_REP --warmup WARMUP --reps REPS`: one process
  per run, GPUs synced between repetitions. `results.mlups` is the MLUPS over
  all timed steps; `results.repetitions` holds the per-repetition seconds and
  MLUPS plus their mean, min, max and stddev.
- Reports go to `data/<PREFIX>/json/` (default `PREFIX=lbm-dgx-sweep`) as
  `<PREFIX>-<lattice>-<collision>-<precision>-<label>-n<N>-g<G>.json` with the
  labels of the [terminology table](#terminology). Each
  (lattice, collision, precision) is therefore its own series.
- A failing or hung run (`RUN_TIMEOUT`, default 1800 s) is recorded in
  `failed-{intra,mpi}.txt` and the sweep continues.

### Plotting a sweep

The plotters take the same YAML via `--config` and plot every
`configs` × `precisions` series of the sweep from `data/<prefix>/json/` into
`data/<prefix>/`, one panel per grid (per grid × decomposition for the
distributed grids) and one curve per transport × hp-halo:

```bash
cfg=benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-sweep.yml
python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/plot_scaling.py --config $cfg
#   -> <series>-mlups-per-gpu-vs-gpus-n<N>.png   for every size N            (plots: vs_gpus)
#   -> <series>-mlups-per-gpu-vs-n-g<G>.png      at the largest GPU count G  (plots: vs_n)
#   -> <series>-hp-speedup-<transport>.png       when hp_halo has off and on (plots: hp_speedup)
#   -> <series>-grids-n<N>.png                   when 2+ grids, every size N (plots: grids)
python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/plot_disg_vs_dense.py --config $cfg
#   -> <series>-disg-vs-dense-<transport>-hp<off|on>.png and <series>-disg-vs-dense.csv
python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/plot_best_family.py --config $cfg
#   -> <series>-best-family.png
```

with `<series> = <prefix>-<lattice>-<collision>-<precision>`. The YAML's
`plots` key picks which of the four `plot_scaling.py` figures are drawn
(default all). The `grids` figure puts every grid on shared axes: the left
panel is the same transport for all (NCCL, slab, hp-halo off), the right one
each grid's fastest run per GPU count, with a table of the winning
configuration (transport, decomposition, hp-halo) under it. The reference
(dashed ideal line) of size N is the 1-GPU run
`<reference_dir>/<reference_prefix>-<lattice>-<collision>-<precision>-<reference_label>-n<N>-g1.json`;
by default the sweep's own `dense-intra` run. A sweep without 1-GPU runs
(only distributed grids) points `reference_dir` at another sweep's `json/`.
`--input-dir`, `--output-dir` and `--sizes` override the file; series without
reports are skipped with a warning. Env vars such as `SIZES` only affect the
run script, not the plotters. Distributed reports written by the retired
`lbm-dist-sweep-run.sh` (`<series>-<decomposition>-<transport>-hp<x>-...`)
are still found.

`plot_best_family.py` needs grids `dense` and `disg` with transports `intra`
and `mpi`. It colours each (N, G) cell by the fastest of the four families
Dense / DenseDisg × Peeraccess / NCCL (red / blue / green / purple) and
blends in white for every other family within `--threshold` (default 10 %)
of it, so a fully white cell means all four tie. Each cell is labelled with
the winner, its MLUPS per GPU (in thousands) and the near-tie count; cells
with a missing report are grey, so set `mpi_single_gpu` to fill the 1-GPU row.

### Result sets

Each folder under `../data/` holds the PNGs, the frozen YAML(s) of the sweep
and a `reproduce.sh [--plot-only]` that re-runs it (or only re-plots from
`json/`). All are D3Q19 + BGK, fp32, `--overlap standard` on 8× A100-SXM4-40GB.
The reports of these sets are committed, so `--plot-only` works on a fresh
clone. `*.json` is git-ignored, so reports of a new sweep stay local until
added with `git add -f`.

| Folder | Matrix | Plots |
| --- | --- | --- |
| `data/lbm-dgx-sweep/` | all four grids, N=256..512, hp-halo off / on: `dense` / `disg` × Peeraccess / NCCL / MPI (P2P) on 1..8 GPUs; `dense_distributed` / `disg_distributed` × NCCL / MPI (P2P) × slab (2..8) / pencil (4, 8) / brick (8) | `plot_scaling` (all four figures), `plot_disg_vs_dense` |
| `data/lbm-dgx-a100-40gb/` | `dense` / `disg` × all three transports at 8 GPUs plus the 1-GPU reference, N=128..608 step 32 (`lbm-dgx-a100-40gb.yml`); `lbm-dgx-a100-40gb-colormap.yml` adds Peeraccess / NCCL on 1, 2 and 4 GPUs in the same `json/` | `plot_scaling` (vs N), `plot_best_family` |
| `data/lbm-n512-gpu-scaling/` | `dense` / `disg` × all three transports on 1, 2, 4, 8 GPUs, N=512 | `plot_scaling` (vs GPUs) |

Domain sizes stop at N=608: N=640 does not fit the two Q=19 population fields
on one 40 GB A100, and a size without a single-GPU reference cannot be
compared.
