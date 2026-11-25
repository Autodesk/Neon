# LBM lid-driven cavity — scripts quick start

How to run the `carbon_benchmark_lbm_ldc` benchmark and plot
its results with the scripts in [`scripts/`](scripts/). This page is a short
tour with a few examples; [`scripts/README.md`](scripts/README.md) is the full
reference (every key, report naming, the reference results), and the
benchmark itself (CLI flags, JSON report) is described in
[`../README.md`](../README.md#lbm_ldc).

Run every command from the carbon repository root.

## The scripts at a glance

| Script | Use it to |
| --- | --- |
| `scripts/lbm-sweep-run.sh` | build the benchmark and run the experiments of a YAML file, in Docker (MPI included) or on the host |
| `scripts/plotting/plot_scaling.py` | plot MLUPS per GPU vs GPUs and vs N, the hp-halo speedup, and all grids on shared axes |
| `scripts/plotting/plot_disg_vs_dense.py` | plot the DenseDisg / Dense ratio |
| `scripts/plotting/plot_best_family.py` | colour an N × GPU-count map by the fastest of Dense / DenseDisg × Peeraccess / NCCL |

A YAML file lists *experiments*, each with a type (`strong_scaling`,
`domain_size`, `best_family`) that decides its figures, and its own run
matrix. The sweep script runs every distinct run once, so experiments
sharing a run share its JSON report, and every plotter takes the same YAML
via `--config`. Reports go to `data/json/`, figures to
`data/plots/<experiment>/`, progress to `data/trace.log`; `data/` is
git-ignored and [`reference/`](reference/) holds the committed results.

All plots show **MLUPS per GPU** (`MLUPS / G`). Perfect strong scaling is a
flat line, drawn dashed at the 1-GPU run of the same N.

## Example 1 — one run by hand

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target carbon_benchmark_lbm_ldc

mkdir -p reports   # --report does not create directories
CUDA_VISIBLE_DEVICES=0 \
./build/benchmarks/lbm_ldc/carbon_benchmark_lbm_ldc \
  --n 256 --iters 1000 --warmup 200 --report-every 0 --checksum \
  --report reports/one-run.json
```

The last stdout lines are the scrapeable result:

```
MLUPS=... (cavity N=256 iters=800 gpus=1 ... grid=dense decomposition=slab lattice=q19 collision=bgk precision=fp32)
CHECKSUM=... MOMENTUM_X=... (grid=dense decomposition=slab precision=fp32 N=256 iters=1000)
```

## Example 2 — quick check on a workstation, no Docker

```bash
cfg=benchmarks/lbm_ldc/scripts/examples/local-quick.yml
./benchmarks/lbm_ldc/scripts/lbm-sweep-run.sh $cfg
python3 benchmarks/lbm_ldc/scripts/plotting/plot_scaling.py --config $cfg
# -> data/plots/local_quick/q19-bgk-fp32-mlups-per-gpu-vs-n-g1.png, q27-bgk-..., q27-kbc-...
```

`runner: local` builds `build/` with the host's `cmake` and `nvcc`, then runs
`q19/bgk`, `q27/bgk` and `q27/kbc` on one GPU at N=64..256 (12 runs). Needs
PyYAML for the sweep and matplotlib for the plots.

## Example 3 — full DGX sweep

[`scripts/lbm-sweep.yml`](scripts/lbm-sweep.yml) documents every key and
holds the four experiments of the reference results (all grids, domain size,
best family, N=512). Check what it will run, then run and plot it:

```bash
cfg=benchmarks/lbm_ldc/scripts/lbm-sweep.yml

DRY_RUN=1 ./benchmarks/lbm_ldc/scripts/lbm-sweep-run.sh $cfg
./benchmarks/lbm_ldc/scripts/lbm-sweep-run.sh $cfg   # 1050 runs, a few hours on 8 GPUs
tail -f benchmarks/lbm_ldc/data/trace.log            # completion bar + ETA

python3 benchmarks/lbm_ldc/scripts/plotting/plot_scaling.py --config $cfg
python3 benchmarks/lbm_ldc/scripts/plotting/plot_disg_vs_dense.py --config $cfg
python3 benchmarks/lbm_ldc/scripts/plotting/plot_best_family.py --config $cfg
```

The runner builds `build/` and `build-mpi/` inside Docker (images
`carbon-cuda:local` and `carbon-cuda-nccl-pinned:local`) and writes one JSON
per run to `data/json/`. `--only grids,n512` runs (or plots) a subset of the
experiments. An interrupted sweep resumes where it stopped: reports whose
timing and overlap match the YAML are kept, others are redone (`FORCE=1`
redoes everything). Failed or hung runs are listed in
`data/failed-{intra,mpi}.txt` and the sweep continues.

## Example 4 — your own sweep: distributed grids in fp64

Write a YAML with the experiments you need; every key not given takes the
default documented in `scripts/lbm-sweep.yml`. The distributed grids have no
1-GPU run of their own, so a second experiment provides the 1-GPU
`dense-intra` run that draws their ideal line (`plots: []`: it only adds
runs):

```yaml
# my-dist-fp64.yml
configs: [q19:bgk]
precisions: [fp64]
sizes: [256, 384]

experiments:
  dist_fp64:
    type: strong_scaling
    grids: [dense_distributed, disg_distributed]
    decompositions: [slab, pencil, brick]
    gpus: [2, 4, 8]          # slab
    transports: [mpi]        # NCCL; add mpip2p for CUDA-aware MPI
    hp_halo: ["off", "on"]
  ideal:
    type: domain_size
    gpus: [1]
    grids: [dense]
    transports: [intra]
    plots: []
```

```bash
DRY_RUN=1 ./benchmarks/lbm_ldc/scripts/lbm-sweep-run.sh my-dist-fp64.yml
./benchmarks/lbm_ldc/scripts/lbm-sweep-run.sh my-dist-fp64.yml
python3 benchmarks/lbm_ldc/scripts/plotting/plot_scaling.py --config my-dist-fp64.yml
# -> data/plots/dist_fp64/q19-bgk-fp64-mlups-per-gpu-vs-gpus-n256.png, ...
```

Unknown keys and contradictory settings (a distributed grid with only
`intra`, `hp_halo: on` with `overlap: none`) are rejected before anything
runs. `q27:kbc` does not launch in fp64 yet.

## Example 5 — re-plot the reference results

```bash
cfg=benchmarks/lbm_ldc/reference/lbm-sweep.yml
python3 benchmarks/lbm_ldc/scripts/plotting/plot_scaling.py --config $cfg
# -> reference/plots/grids/q19-bgk-fp32-grids-n512.png (all four grids), ...
```

[`reference/`](reference/) is a committed copy of a full DGX sweep: its YAML,
1050 JSON reports and the figures. To plot your own `data/` against the
same experiments, pass `--data-dir benchmarks/lbm_ldc/data`.
