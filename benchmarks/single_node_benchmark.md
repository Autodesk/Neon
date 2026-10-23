# Single-node strong-scaling harness (carbon-nvidia-jacobi)

The orchestrator script lives in the **parent** [carbon-nvidia-jacobi](https://github.com/massimim/ai-repos) repo (alongside the NVIDIA Jacobi reference tree). It drives Carbon benchmarks built from this `Neon/` tree plus optional NVIDIA OpenMP/MPI Jacobi binaries.

| Artifact | Location (monorepo layout) |
| -------- | --------------------------- |
| Script | `single_node_benchmark.py` at repo root |
| Wrapper | `single_node_benchmark.sh` |
| Longer copy of this doc | `docs/single_node_benchmark.md` (repo root) |

If you only clone **Neon**, check out **carbon-nvidia-jacobi** for the script, or copy `single_node_benchmark.py` and point `--carbon-src` / `--carbon-build-dir` at your Neon checkout.

## What it does

- **Strong scaling:** fixed problem size, GPU count `g = 1 … max_gpus`.
- **Suites:** `jacobi`, `lbm`, `carbon-micro`, `halo` (select with `--suites`).
- **Outputs:** `bench_results/<timestamp>/results.csv`, `run.log`, `plots/*.png` (needs **matplotlib**).

Carbon binaries are expected under `build/benchmarks/<name>/carbon_benchmark_<name>` after a normal Neon CMake build.

## Build Carbon benchmarks

From the **monorepo root** (CMake source = `Neon/`):

```bash
cmake -S Neon -B build -DCARBON_BUILD_BENCHMARKS=ON
cmake --build build -j --target \
  carbon_benchmark_jacobi_nvidia \
  carbon_benchmark_lbm_lid_driven_cavity \
  carbon_benchmark_reduce_throughput \
  carbon_benchmark_skeleton_overlap \
  carbon_benchmark_block_sparse \
  carbon_benchmark_halo_scalability
```

From **inside Neon** only:

```bash
cmake -S . -B build -DCARBON_BUILD_BENCHMARKS=ON
cmake --build build -j --target carbon_benchmark_lbm_lid_driven_cavity
# … other targets as needed
```

## Quick start

From **carbon-nvidia-jacobi** root (defaults: `--carbon-src Neon`, `--carbon-build-dir build`):

```bash
python3 single_node_benchmark.py --quick --no-mpi
# or
./single_node_benchmark.sh --quick --no-mpi
```

Custom Neon path / build tree:

```bash
python3 single_node_benchmark.py \
  --carbon-src /path/to/Neon \
  --carbon-build-dir /path/to/Neon/build \
  --quick --no-mpi
```

Full run (8192² Jacobi, LBM N=128, micro/halo N=128, all visible GPUs):

```bash
python3 single_node_benchmark.py --no-mpi
```

## Suites and Carbon programs

| Suite | Carbon targets | Notes |
| ----- | -------------- | ----- |
| `lbm` | `carbon_benchmark_lbm_lid_driven_cavity` | Passes `--gpus g` and sets `CUDA_VISIBLE_DEVICES`; JSON under `lbm/*.json`. See [lbm_lid_driven_cavity](README.md#lbm_lid_driven_cavity). |
| `carbon-micro` | `reduce_throughput`, `skeleton_overlap`, `block_sparse` | Uses [scripts/run_benchmarks.py](scripts/run_benchmarks.py); `reduce` at `--batch-iters 32`. |
| `halo` | `carbon_benchmark_halo_scalability` | Halo GB/s probe; appends `halo/halo.csv`. |
| `jacobi` | `carbon_benchmark_jacobi_nvidia` | Also runs NVIDIA `multi-gpu-programming-models/*` if present. |

In-repo micro harness only (no outer script):

```bash
python3 benchmarks/scripts/run_benchmarks.py --build-dir build --output-dir reports/bench-quick --mode quick
```

LBM-only sweep/plot (size or GPU ladders, Neon-style figures):

```bash
python3 benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py \
  --bench ./build/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity \
  --mode sweep+plot --output-dir reports/lbm-scaling \
  --sizes 256 --gpus 1 2 --plot-x gpus --plot-out reports/lbm-scaling/mlups_vs_gpus.png
```

## Useful options

| Flag | Purpose |
| ---- | ------- |
| `--suites jacobi,lbm` | Run a subset |
| `--max-gpus N` | Cap scaling sweep |
| `--device-list 0,1,…` | Physical GPU IDs for `CUDA_VISIBLE_DEVICES` |
| `--no-mpi` | Skip NVIDIA MPI/NCCL Jacobi variants |
| `--lbm-n`, `--lbm-iters`, `--lbm-lattice`, `--lbm-collision`, `--lbm-grid` | LBM cavity config |
| `--carbon-n` | Cube edge for micro-benchmarks + halo |
| `--jacobi-nx`, `--jacobi-ny`, `--jacobi-niter` | Jacobi strong-scaling grid |
| `--output-dir DIR` | Result root (default `bench_results/<UTC>`) |
| `--plot-only` | Re-render plots from existing `results.csv` |

## Multi-GPU LBM

The cavity binary supports **`--gpus N`** (first N visible CUDA devices). The orchestrator sets `CUDA_VISIBLE_DEVICES=0,…,g-1` and passes **`--gpus g`**, then checks logs for `GpuGroup: g CUDA device(s)`.

If you see `WARN … GpuGroup size 1 != requested 2`, the process only sees one CUDA device (`cudaGetDeviceCount()`), even when `nvidia-smi` lists two cards. Fix visibility (driver, cgroup, MIG, container device passthrough) before expecting 2-GPU MLUPS scaling.

For multi-GPU LBM on one node, prefer **`--overlap standard`** when `g > 1` (the script sets this automatically).

## Reading results

- **`results.csv`:** `speedup` and `efficiency_pct` vs `g=1` per series.
- **Jacobi on PCIe:** cheap stencil + sync often yields ~1× wall time at 2 GPUs (~50% efficiency); see [README — How to read results](README.md#how-to-read-results).
- **Per-benchmark detail:** [benchmarks/README.md](README.md).

## Related docs

- [benchmarks/README.md](README.md) — individual programs, CLI, MLUPS interpretation
- [docs/lbm_benchmark_runbook.md](../docs/lbm_benchmark_runbook.md) — LBM build, MPI, DGX notes (if present in your tree)
