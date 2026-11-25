# LBM lid-driven cavity — scripts quick start

How to run the `carbon_benchmark_lbm_lid_driven_cavity` benchmark and plot
its results with the scripts in [`scripts/`](scripts/). This page is a short
tour with a few examples; [`scripts/README.md`](scripts/README.md) is the full
reference (every knob, report naming, result sets), and the benchmark itself
(CLI flags, JSON report) is described in
[`../README.md`](../README.md#lbm_lid_driven_cavity).

Run every command from the carbon repository root.

## The scripts at a glance

| Script | Use it to |
| --- | --- |
| `scripts/sweep_and_plot.py` | sweep sizes, GPU counts, grids and lattice/collision pairs on a workstation (single process), then plot |
| `scripts/run_local_single_gpu_n256.sh` | get a quick 1-GPU bar chart of the three lattice/collision configs at N=256 |
| `scripts/run_local_full_matrix.sh` | compare configs × backends (CUDA / SYCL) × grids at 1 GPU and all GPUs |
| `scripts/lbm-dgx-sweep-run.sh` | run any matrix on a multi-GPU node in Docker, MPI included, from one YAML file |
| `scripts/plotting/plot_scaling.py` | plot a sweep as MLUPS per GPU vs GPUs and vs N, the hp-halo speedup, and all grids on shared axes |
| `scripts/plotting/plot_disg_vs_dense.py` | plot the DenseDisg / Dense ratio of a sweep that ran both layouts |
| `scripts/plotting/plot_best_family.py` | colour an N × GPU-count map by the fastest of Dense / DenseDisg × Peeraccess / NCCL |
| `data/<result-set>/reproduce.sh` | re-run or re-plot one of the checked-in result sets |

All DGX runs go through `lbm-dgx-sweep-run.sh` and a YAML file, and every
plotter takes the same YAML via `--config`.

All plots show **MLUPS per GPU** (`MLUPS / G`). Perfect strong scaling is a
flat line, drawn dashed at the 1-GPU run of the same N.

## Example 1 — one run by hand

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target carbon_benchmark_lbm_lid_driven_cavity

mkdir -p reports   # --report does not create directories
CUDA_VISIBLE_DEVICES=0 \
./build/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity \
  --n 256 --iters 1000 --warmup 200 --report-every 0 --checksum \
  --report reports/one-run.json
```

The last stdout lines are the scrapeable result:

```
MLUPS=... (cavity N=256 iters=800 gpus=1 ... grid=dense decomposition=slab lattice=q19 collision=bgk precision=fp32)
CHECKSUM=... MOMENTUM_X=... (grid=dense decomposition=slab precision=fp32 N=256 iters=1000)
```

## Example 2 — quick local check

```bash
./benchmarks/lbm_lid_driven_cavity/scripts/run_local_single_gpu_n256.sh
# -> reports/single_gpu_n256_local/mlups_by_config.png
```

Runs `q19/bgk`, `q27/bgk` and `q27/kbc` on one GPU at N=256 (building the
binary in `build/` if it is missing). `ITERS`, `WARMUP` and `BUILD_DIR` are
env knobs.

## Example 3 — size sweep on a workstation, Dense vs DenseDisg

```bash
python3 benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py \
  --bench ./build/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity \
  --output-dir reports/size_sweep \
  --sizes 128 192 256 320 384 --gpus 1 --grids dense disg \
  --lattice q19 --collision bgk \
  --plot-x size --plot-out reports/size_sweep/mlups_per_gpu_vs_n.png
```

Add `--dry-run` to print the commands first, `--gpus 1 2 4` with
`--plot-x gpus` for strong scaling, or `--mode plot` to re-plot existing
reports.

## Example 4 — full DGX sweep from a YAML file

[`scripts/lbm-dgx-sweep.yml`](scripts/lbm-dgx-sweep.yml) is the documented
template (Dense and DenseDisg over Peeraccess / NCCL / MPI (P2P), 1..8 GPUs,
N=256..512). Check what it will run, then run and plot it:

```bash
cfg=benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-sweep.yml

DRY_RUN=1 ./benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-sweep-run.sh $cfg
./benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-sweep-run.sh $cfg   # ~1 h on 8 GPUs

python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/plot_scaling.py --config $cfg
python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/plot_disg_vs_dense.py --config $cfg
python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/plot_best_family.py --config $cfg
```

The runner builds `build/` and `build-mpi/` inside Docker (images
`carbon-cuda:local` and `carbon-cuda-nccl-pinned:local`), writes one JSON per
run to `data/lbm-dgx-sweep/json/`, and the plotters write the PNGs to
`data/lbm-dgx-sweep/`. Failed or hung runs are listed in
`json/failed-{intra,mpi}.txt` and the sweep continues. Env vars override the
YAML for the runner only, e.g. `SIZES="256 512" GPUS_LIST="1 8"`. The
best-family map's 1-GPU row stays grey unless the YAML sets
`mpi_single_gpu: true`, because NCCL is otherwise not run on 1 GPU.

## Example 5 — your own sweep: distributed grids in fp64

Copy the template and change the keys you need. This one runs both
distributed grids on every decomposition in double precision. The
distributed grids have no 1-GPU run of their own, so the ideal line comes
from the 1-GPU `dense-intra` runs of Example 4; run that sweep with
`PRECISIONS="fp32 fp64"` first, otherwise the plots have no ideal line.

```yaml
# my-dist-fp64.yml
configs: [q19:bgk]
precisions: [fp64]
sizes: [256, 384]
grids: [dense_distributed, disg_distributed]
decompositions: [slab, pencil, brick]
gpus: [2, 4, 8]          # slab
gpus_pencil: [4, 8]
gpus_brick: [8]
transports: [mpi]        # NCCL; add mpip2p for CUDA-aware MPI
hp_halo: ["off", "on"]
reference_dir: benchmarks/lbm_lid_driven_cavity/data/lbm-dgx-sweep/json
reference_prefix: lbm-dgx-sweep
prefix: my-dist-fp64
```

```bash
DRY_RUN=1 ./benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-sweep-run.sh my-dist-fp64.yml
./benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-sweep-run.sh my-dist-fp64.yml
python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/plot_scaling.py --config my-dist-fp64.yml
# -> data/my-dist-fp64/my-dist-fp64-q19-bgk-fp64-mlups-per-gpu-vs-gpus-n256.png, ...
```

Unknown keys and contradictory settings (a distributed grid with only
`intra`, `hp_halo: on` with `overlap: none`) are rejected before anything
runs. `q27:kbc` does not launch in fp64 yet.

## Example 6 — re-plot a checked-in result set

```bash
./benchmarks/lbm_lid_driven_cavity/data/lbm-dgx-sweep/reproduce.sh --plot-only
# -> data/lbm-dgx-sweep/lbm-dgx-sweep-q19-bgk-fp32-grids-n512.png (all four grids), ...
```

Each folder under [`data/`](data/) (`lbm-dgx-sweep`: all four grids;
`lbm-dgx-a100-40gb`: N up to 608; `lbm-n512-gpu-scaling`) holds its frozen
YAML, its committed JSON reports and a `reproduce.sh`; without
`--plot-only` it re-runs the whole sweep first. `*.json` is git-ignored, so
the reports of your own sweeps stay local unless you `git add -f` them.
