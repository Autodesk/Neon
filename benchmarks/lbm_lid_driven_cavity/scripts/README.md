# `lbm_lid_driven_cavity` scripts

Drivers and plotters for the LBM lid-driven cavity benchmark
(`carbon_benchmark_lbm_lid_driven_cavity`). Every script talks to the binary
through its `--report PATH` JSON output (schema `carbon-lbm-report-v1`), so the
run scripts and the plotters can be used independently: re-plotting never
requires re-running.

All paths below are relative to the carbon repository root; every shell script
`cd`s there itself, so they can be launched from anywhere. Run scripts live in
`scripts/`, the plotters in `scripts/plotting/`.

| Script | Kind | What it produces |
| --- | --- | --- |
| [`sweep_and_plot.py`](#sweep_and_plotpy) | generic sweep + plot | MLUPS vs N / vs GPUs / per config, for any parameter grid |
| [`run_local_single_gpu_n256.sh`](#run_local_single_gpu_n256sh) | local quick run | bar chart of the 3 lattice/collision configs, 1 GPU, N=256 |
| [`run_local_full_matrix.sh`](#run_local_full_matrixsh) | local quick run | bar chart of configs × backend (CUDA/SYCL) × {1, all} GPUs |
| [`lbm-dgx-a100-40gb-run.sh`](#neon-fig-7-domain-size-sweep-8-gpus) | DGX sweep | 8-GPU parallel efficiency vs N, per grid × transport |
| [`plotting/lbm-dgx-a100-40gb-plot.py`](#neon-fig-7-domain-size-sweep-8-gpus) | plotter | `data/lbm-dgx-a100-40gb/lbm-dgx-a100-40gb.png` |
| [`plotting/plot_neon_fig7.py`](#neon-fig-7-domain-size-sweep-8-gpus) | plotter (library) | the Fig. 7 renderer used by the wrapper above |
| [`lbm-dgx-a100-40gb-run-colormap.sh`](#best-family-colormap-n--gpus) | DGX sweep | fills the N × GPU-count matrix for the 4 families |
| [`plotting/lbm-dgx-a100-40gb-run-colormap.py`](#best-family-colormap-n--gpus) | plotter | `data/lbm-dgx-a100-40gb/lbm-dgx-a100-40gb-run-colormap.png` |
| [`lbm-n512-gpu-scaling-run.sh`](#n512-strong-scaling) | DGX sweep | strong scaling at N=512 over 1/2/4/8 GPUs |
| [`plotting/lbm-n512-gpu-scaling-plot.py`](#n512-strong-scaling) | plotter | `data/lbm-n512-gpu-scaling/*.png` (efficiency + MLUPS) |
| [`lbm-dgx-sweep-run.sh`](#generic-dgx-sweep-sizes--gpus--transports) | DGX sweep | any lattice/collision × precision × sizes × GPU counts × grids × transports |

## Terminology

The DGX scripts compare a fixed set of *families*, encoded in the report label
and filename (`<prefix>-<label>-n<N>-g<G>.json`):

| Label | Grid (`--grid`) | Transport |
| --- | --- | --- |
| `dense-intra` / `disg-intra` | `dense` / `disg` | **Peeraccess** — one process, all GPUs, intra-process NVLink P2P copies |
| `dense-mpi` / `disg-mpi` | `dense` / `disg` | **NCCL** — `mpiexec -n G`, one rank per GPU, halos over NCCL (needs the `-DCARBON_ENABLE_MPI=ON` build) |
| `dense-mpip2p` / `disg-mpip2p` | `dense` / `disg` | **MPI (P2P)** — same MPI build, CUDA-aware MPI instead of NCCL (`CARBON_DIST_TRANSPORT=mpi`) |

Parallel efficiency in the Fig. 7 style plots is always normalised to a fixed
single-GPU dense baseline:

```
eff = MLUPS(config, N, G) / (MLUPS_baseline(dense-intra, 1 GPU) * G)
```

so GLUPS on the secondary axis is an exact linear rescale of efficiency.

## Generic tool

### `sweep_and_plot.py`

The general-purpose driver. Two modes, combinable:

- `--mode sweep` runs the binary over the cartesian product of `--sizes`,
  `--gpus` and `--lattice/--collision` (or `--configs q19:bgk q27:kbc ...`)
  and writes one `run-NNN.json` per run into `--output-dir`. GPU counts are
  selected with `CUDA_VISIBLE_DEVICES=0..g-1`.
- `--mode plot` loads every report from `--output-dir` (or several
  `--input-dirs`, to overlay sweeps) and renders one curve per
  `(lattice, collision, label, gpus)` series.
- `--mode sweep+plot` (default) does both.

Plot axes: `--plot-x size|gpus|config` and `--plot-y mlups|mlups_per_gpu|bandwidth`.
`--dry-run` prints the commands without running them. Needs `matplotlib`.

```bash
python3 benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py \
  --bench ./build/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity \
  --output-dir reports/strong_scaling_n256 \
  --sizes 256 --gpus 1 2 4 8 \
  --lattice q19 --collision bgk \
  --plot-x gpus --plot-out reports/strong_scaling_n256/mlups_vs_gpus.png
```

It only drives single-process runs; for MPI sweeps use the DGX scripts below.

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
`build-sycl-mpi/`) and with GPU counts `1` and "all GPUs". Missing backends are
skipped. Output: `reports/full_matrix_local/` with one JSON per cell and
`mlups_full_matrix.png`.

Knobs: `ITERS`, `WARMUP`, `N=256`, `GPUS_LIST`, `OUT_DIR`, and
`CUDA_BENCH=` / `SYCL_BENCH=` to point at a binary directly.

## DGX reproduction pipelines (8× A100-SXM4-40GB, NVLink)

These scripts reproduce the Neon EuroPar Fig. 7 style results. Each is a
`*-run.sh` / `*-plot.py` pair: the run script builds Release trees **inside
Docker** (`build/` for intra-process, `build-mpi/` with `-DCARBON_ENABLE_MPI=ON`),
runs the matrix, and drops JSON into `../data/<name>/json/`; the plotter reads
from there and writes the PNG next to it. The checked-in PNGs under `../data/`
are the reference results. All DGX runs use `--overlap standard`, and all
except the generic sweep use D3Q19 + BGK in single precision.

Shared knobs (env vars):

| Variable | Default | Meaning |
| --- | --- | --- |
| `ITERS` / `WARMUP` | `2000` / `400` | iterations per run / excluded warmup |
| `SM_ARCH` | `80` | CUDA arch (A100 = 80, H100 = 90) |
| `IMAGE_INTRA` | `carbon-cuda:local` | image for builds and Peeraccess runs |
| `IMAGE_MPI` | `carbon-cuda-nccl-pinned:local` | image for MPI runs (driver-pinned NCCL; plain `carbon-cuda` fails `ncclCommInitRank` on the DGX) |
| `SKIP_BUILD` | `0` (`1` for n512) | `1` reuses the existing `build/` and `build-mpi/` |
| `FORCE` | `0` (`1` for n512) | `1` re-runs cells whose JSON already exists |

Domain sizes stop at N=608: N=640 does not fit the two Q=19 population fields
on one 40 GB A100, and a size without a single-GPU baseline cannot be
normalised.

### Neon Fig. 7: domain-size sweep, 8 GPUs

```bash
./benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-a100-40gb-run.sh
python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/lbm-dgx-a100-40gb-plot.py
```

- `lbm-dgx-a100-40gb-run.sh` sweeps `SIZES` (128..608, step 32) and runs, at
  `GPUS=8`, all six families (dense/disg × Peeraccess/NCCL/MPI (P2P)), plus
  the single-GPU `dense-intra` baseline at every size.
- `lbm-dgx-a100-40gb-plot.py` is a wrapper that pins input/output to
  `data/lbm-dgx-a100-40gb/` and calls `plot_neon_fig7.py`; extra flags
  (`--sizes`, `--output`, `--show`) are passed through.
- `plot_neon_fig7.py` holds the actual renderer: two panels (Dense |
  DenseDisg) of 8-GPU parallel efficiency vs N, one line per transport
  (Peeraccess, NCCL, MPI (P2P)), with a GLUPS secondary axis. Use it directly to plot reports stored elsewhere
  (`--input-dir`, `--prefix`, `--output`).

### Best-family colormap (N × GPUs)

```bash
./benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-a100-40gb-run-colormap.sh
python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/lbm-dgx-a100-40gb-run-colormap.py
```

- The run script fills the full `SIZES` × `GPUS_LIST` (1 2 4 8) matrix for the
  four families `dense-intra`, `dense-mpi`, `disg-intra`, `disg-mpi`. It
  writes into the same `data/lbm-dgx-a100-40gb/json/` folder with the same
  naming as the Fig. 7 sweep, so the 8-GPU cells already produced there are
  reused and only missing cells are run. The 1-GPU MPI cell is pinned to
  `CUDA_VISIBLE_DEVICES=0`, because `mpiexec -n 1` would otherwise pick up
  every visible GPU.
- The plotter colours each (N, G) cell by the fastest family (red / blue /
  green / purple) and blends in white for every other family within 10 % of
  it; a fully white cell means all four tie. Cells with a missing report are
  grey.

### N=512 strong scaling

```bash
./benchmarks/lbm_lid_driven_cavity/scripts/lbm-n512-gpu-scaling-run.sh
python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/lbm-n512-gpu-scaling-plot.py
```

- The run script fixes `N=512` and sweeps `GPUS_LIST` (1 2 4 8) for all six
  families. It reuses existing builds and overwrites existing JSON by default
  (`SKIP_BUILD=1`, `FORCE=1`). Reports go to `data/lbm-n512-gpu-scaling/json/`
  with prefix `lbm-n512-scaling`.
- The plotter writes two figures into `data/lbm-n512-gpu-scaling/`:
  `lbm-n512-gpu-scaling.png` (parallel efficiency vs GPUs, Dense | DenseDisg
  panels) and `lbm-n512-gpu-scaling-mlups.png` (absolute MLUPS, same layout).
  `--n`, `--gpus`, `--prefix` and `--input-dir` point it at any other sweep.

### Generic DGX sweep (sizes × GPUs × transports)

```bash
./benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-sweep-run.sh \
  benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-sweep.yml
```

- Parameters come from an optional YAML file (see
  [`lbm-dgx-sweep.yml`](lbm-dgx-sweep.yml)), whose keys are the lowercase
  knob names (`configs`, `sizes`, `gpus`, `reps`, ...). List-valued keys
  accept a list or a space-separated string; `sizes` / `gpus` also accept an
  inclusive `{start, stop, step}` range. Unknown keys and invalid values are
  rejected. Env vars (`CONFIGS`, `SIZES`, `GPUS_LIST`, ...) override the
  file; `DRY_RUN=1` prints the resolved configuration and exits.
- Does one benchmark run per combination of `CONFIGS` × `PRECISIONS` ×
  `SIZES` (default 256..512, step 64) × `GPUS_LIST` (default 1..8) × `GRIDS`
  (`dense disg`) × `TRANSPORTS` (`intra mpi mpip2p`). NCCL and MPI (P2P) are
  skipped at 1 GPU.
- `CONFIGS` holds lattice:collision pairs, any of `q19:bgk` (D3Q19 + BGK,
  the default), `q27:bgk` (D3Q27 + BGK) and `q27:kbc` (D3Q27 + KBC; KBC
  exists only for D3Q27). `PRECISIONS` is `fp32` (default) and/or `fp64`
  (binary `--precision`). fp64 roughly halves the MLUPS and doubles the
  memory: on a 40 GB A100 the 1-GPU run at N=512 does not fit in fp64 (N=448
  does), so that run fails out of memory and the fp64 series has no
  efficiency baseline at N=512. `q27:kbc` with `fp64` does not launch yet
  ("too many resources requested for launch": the kernel needs more registers
  than the library's fixed 512-thread blocks allow), so those runs fail too.
- Timing is set by `WARMUP` (default 400 untimed solver steps), `REPS`
  (default 4 timed repetitions) and `ITERS_PER_REP` (default 400 solver steps
  averaged per repetition). They map to the binary's
  `--iters WARMUP+REPS*ITERS_PER_REP --warmup WARMUP --reps REPS`: one process
  per run, GPUs synced between repetitions. `results.mlups` is the MLUPS over
  all timed steps; `results.repetitions` holds the per-repetition seconds and
  MLUPS plus their mean, min, max and stddev.
- Reports go to `data/<PREFIX>/json/` (default `PREFIX=lbm-dgx-sweep`) as
  `<PREFIX>-<lattice>-<collision>-<precision>-<label>-n<N>-g<G>.json`. Each
  (lattice, collision, precision) is therefore its own series in the shared
  `<prefix>-<label>-n<N>-g<G>.json` naming: the plotters above read it with
  `--prefix <PREFIX>-<lattice>-<collision>-<precision>` and normalise it to
  its own 1-GPU dense-intra run.
- A failing or hung run (`RUN_TIMEOUT`, default 1800 s) is recorded in
  `failed-{intra,mpi}.txt` and the sweep continues.

The plotters take the same YAML via `--config` and plot every
`configs` × `precisions` series of the sweep from `data/<prefix>/json/` into
`data/<prefix>/`: efficiency and MLUPS vs GPU count for every size, and
efficiency vs size at the largest GPU count:

```bash
cfg=benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-sweep.yml
python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/lbm-n512-gpu-scaling-plot.py --config $cfg
#   -> <prefix>-<lattice>-<collision>-<precision>-{eff,mlups}-vs-gpus-n<N>.png
python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/plot_neon_fig7.py --config $cfg
#   -> <prefix>-<lattice>-<collision>-<precision>-eff-vs-n-g<G>.png
```

`--input-dir`, `--output-dir`, `--n` / `--sizes` and `--gpus` override the
file; series without reports are skipped with a warning. Env vars such as
`SIZES` only affect the run script, not the plotters. The YAML is parsed by
[`lbm_sweep_config.py`](lbm_sweep_config.py), which the run script uses too,
so both see the same keys, defaults and validation.
