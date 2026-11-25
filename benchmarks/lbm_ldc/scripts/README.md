# `lbm_ldc` scripts

One sweep script and three plotters for the LBM lid-driven cavity benchmark
(`carbon_benchmark_lbm_ldc`). Both sides read the same YAML file
and talk through the binary's `--report PATH` JSON output (schema
`carbon-lbm-report-v1`), so re-plotting never requires re-running. For a short
tour with examples, start with the [quick start](../README.md).

All paths below are relative to the carbon repository root; the sweep script
`cd`s there itself, so it can be launched from anywhere.

| Script | What it does |
| --- | --- |
| [`lbm-sweep-run.sh`](#the-sweep) | builds the benchmark and runs every experiment of a YAML, in Docker or on the host |
| [`plotting/plot_scaling.py`](#plotting) | MLUPS per GPU vs GPU count / vs N, hp-halo speedup, all grids compared |
| [`plotting/plot_disg_vs_dense.py`](#plotting) | DenseDisg / Dense MLUPS ratio |
| [`plotting/plot_best_family.py`](#plotting) | N × GPU-count colormap of the fastest of Dense / DenseDisg × Peeraccess / NCCL |

| YAML | Contents |
| --- | --- |
| [`lbm-sweep.yml`](lbm-sweep.yml) | documents every key; the full DGX sweep behind the [reference results](../reference/) |
| [`examples/local-quick.yml`](examples/local-quick.yml) | 12 runs on one GPU, every lattice / collision pair, no Docker |
| [`../reference/lbm-sweep.yml`](../reference/lbm-sweep.yml) | the same experiments as `lbm-sweep.yml`, reading and writing `reference/` |

`lbm_sweep_config.py` (YAML loading and validation, run matrix, report labels)
and `plotting/lbm_reports.py` (report loading, shared plot styles) are the
libraries the scripts share.

## Experiments

A sweep YAML has two parts:

- **Global keys**, shared by every experiment: `runner` (`docker` or `local`),
  the timing (`warmup`, `iters_per_rep`, `reps`), `overlap`, `data_dir`,
  `reference_label`, `sm_arch`, the Docker images, the build trees, `skip_build`,
  `force` and `run_timeout`. An env var with the key's name in upper case
  (`RUNNER=local`, `FORCE=1`, `WARMUP=200`, ...) overrides the file.
- **`experiments`**, a mapping from a name to a `type` plus its run matrix
  (`configs`, `precisions`, `sizes`, `gpus`, `grids`, `decompositions`,
  `gpus_pencil`, `gpus_brick`, `transports`, `mpi_single_gpu`, `hp_halo`,
  `plots`). Matrix keys at the top level are defaults for every experiment.

The type decides which figures the plotters draw; `plots` picks a subset:

| Type | Figures (`plots`) | Plotter |
| --- | --- | --- |
| `strong_scaling` | `vs_gpus`, `hp_speedup`, `grids`, `disg_vs_dense` | `plot_scaling.py`, `plot_disg_vs_dense.py` |
| `domain_size` | `vs_n` | `plot_scaling.py` |
| `best_family` | `best_family` | `plot_best_family.py` |

Because the timing is global, a run is identified by its matrix cell alone:
two experiments that both need, say, the 1-GPU `dense-intra` run at N=512 get
it from the same report, and the sweep script runs it once. Unknown keys,
invalid values and contradictory combinations (a distributed grid without an
MPI transport, `hp_halo: on` with `overlap: none`, a figure that does not
belong to the type, a `best_family` experiment without Dense, DenseDisg,
Peeraccess and NCCL) are rejected before anything runs.

## Terminology

Reports are named `<lattice>-<collision>-<precision>-<label>-n<N>-g<G>.json`
and the first three parts are the *series*; the label encodes the family:

| Label | Grid (`--grid`) | Transport |
| --- | --- | --- |
| `dense-intra` / `disg-intra` | `dense` / `disg` | **Peeraccess**: one process, all GPUs, intra-process NVLink P2P copies |
| `dense-mpi` / `disg-mpi` | `dense` / `disg` | **NCCL**: `mpiexec -n G`, one rank per GPU, halos over NCCL (needs the `-DCARBON_ENABLE_MPI=ON` build) |
| `dense-mpip2p` / `disg-mpip2p` | `dense` / `disg` | **MPI (P2P)**: same MPI build, CUDA-aware MPI instead of NCCL (`CARBON_DIST_TRANSPORT=mpi`) |
| `<grid>-<decomposition>-<transport>-hp<off\|on>` | `dense_distributed` / `disg_distributed` | NCCL or MPI (P2P), one rank per GPU, the domain split by `--decomposition slab\|pencil\|brick` |

A `-hpon` suffix on a `dense` / `disg` label marks a run with `--hp-halo`
(high-priority halo stream).

Every scaling plot shows **MLUPS per GPU**:

```
MLUPS per GPU = MLUPS(config, N, G) / G
```

Perfect strong scaling is a horizontal line, drawn dashed at the 1-GPU run of
the same series and N with label `reference_label` (default `dense-intra`). G
comes from the report's file name: MPI reports written before the driver
counted every rank carry `system.gpu_count = 1`, and their
`results.mlups_per_gpu` is the total.

## The sweep

```bash
cfg=benchmarks/lbm_ldc/scripts/lbm-sweep.yml
DRY_RUN=1 ./benchmarks/lbm_ldc/scripts/lbm-sweep-run.sh $cfg   # list the runs
./benchmarks/lbm_ldc/scripts/lbm-sweep-run.sh $cfg
./benchmarks/lbm_ldc/scripts/lbm-sweep-run.sh $cfg --only grids,n512
tail -f benchmarks/lbm_ldc/data/trace.log
```

- **Build.** With `runner: docker` (default) the script configures and builds
  Release trees inside `image_intra`: `build_dir` (default `build/`) and, when
  an experiment has NCCL or MPI (P2P) runs, `build_mpi_dir` (`build-mpi/`, with
  `-DCARBON_ENABLE_MPI=ON`). With `runner: local` it does the same with the
  host's `cmake` / `nvcc` (Ninja if installed) and runs `mpiexec` from the
  `PATH`. `sm_arch: auto` reads the GPU's compute capability from
  `nvidia-smi`; `skip_build: true` uses the trees as they are.
- **Images.** `carbon-cuda:local` (`image_intra`, from `docker/Dockerfile.cuda`)
  compiles both trees and does the Peeraccess runs;
  `carbon-cuda-nccl-pinned:local` (`image_mpi`) does the MPI runs with NCCL
  2.27.7 for CUDA 12.2, because the NCCL of plain `carbon-cuda` needs a newer
  driver and fails `ncclCommInitRank` on this DGX.
- **Run matrix.** Per experiment, `configs` × `precisions` × `sizes` × the
  (grid, decomposition, transport, hp-halo, GPU count) cells that
  `lbm_sweep_config.runs` expands:
  - `dense` / `disg` run every `transports` entry (`intra mpi mpip2p`) on
    every `gpus` count (default 1..8);
  - `dense_distributed` / `disg_distributed` run under `mpiexec`, one rank per
    GPU, for every `decompositions` entry: slab on `gpus`, pencil on
    `gpus_pencil` (default 4 8), brick on `gpus_brick` (default 8); they skip
    `intra`;
  - NCCL and MPI (P2P) are skipped at 1 GPU, unless `mpi_single_gpu: true`
    runs them for `dense` / `disg` as one rank pinned to
    `CUDA_VISIBLE_DEVICES=0` (a plain `mpiexec -n 1` would pick up every
    visible GPU);
  - every cell runs once per `hp_halo` entry (`off` default, `on` adds
    `--hp-halo`), with `--overlap OVERLAP` (default `standard`).

  The union over the selected experiments is run once per distinct cell.
- **Configs and precisions.** `configs` holds lattice:collision pairs, any of
  `q19:bgk` (D3Q19 + BGK, the default), `q27:bgk` and `q27:kbc` (KBC exists
  only for D3Q27). `precisions` is `fp32` (default) and/or `fp64`. fp64
  roughly halves the MLUPS and doubles the memory: on a 40 GB A100 the 1-GPU
  run at N=512 does not fit (N=448 does), so the fp64 series has no reference
  at N=512. `q27:kbc` with `fp64` does not launch yet (the kernel needs more
  registers than the library's fixed 512-thread blocks allow).
- **Timing.** `warmup` (default 400 untimed solver steps), `reps` (default 5)
  and `iters_per_rep` (default 1000) map to the binary's
  `--iters WARMUP+REPS*ITERS_PER_REP --warmup WARMUP --reps REPS`: one process
  per run, GPUs synced between repetitions. `results.repetitions` holds the
  per-repetition seconds and MLUPS plus their mean, min, max and stddev.
- **Reuse.** With `force: false` a report that already exists is kept if its
  iters, warmup, reps and overlap match the YAML, so an interrupted sweep
  resumes where it stopped and a bigger matrix only does the new runs. A report
  with other settings is redone; `force: true` redoes everything.
- **Output.** Everything goes to `data_dir` (default
  `benchmarks/lbm_ldc/data/`, git-ignored): the reports in
  `json/`, the progress in `trace.log` and failed or hung runs
  (`run_timeout`, default 1800 s) in `failed-{intra,mpi}.txt`; the sweep moves
  on after a failure.
- **Progress.** `trace.log` gets a header (config, experiments, run counts),
  one line per run and a footer with the totals. Each run line carries a
  completion bar over the whole sweep, the elapsed time, an ETA from the
  average run time so far, the status (`ok` or `redo` with the MLUPS and run
  time, `skip`, `FAIL`, `TIMEOUT`) and the report name:

  ```
  2026-11-24 10:12:03 [#########.....................]  312/1050  29%  elapsed 1h02m  ETA 2h28m   | ok      7712 MLUPS   11.2s | q19-bgk-fp32-dense-mpi-n512-g8
  ```

## Plotting

The plotters take the same YAML and read `<data_dir>/json/`; every figure of
experiment `<name>` goes to `<data_dir>/plots/<name>/`:

```bash
cfg=benchmarks/lbm_ldc/scripts/lbm-sweep.yml
python3 benchmarks/lbm_ldc/scripts/plotting/plot_scaling.py --config $cfg
#   strong_scaling: <series>-mlups-per-gpu-vs-gpus-n<N>.png   every size N        (vs_gpus)
#                   <series>-hp-speedup-<transport>.png       hp_halo off and on  (hp_speedup)
#                   <series>-grids-n<N>.png                   2+ grids, every N   (grids)
#   domain_size:    <series>-mlups-per-gpu-vs-n-g<G>.png      largest GPU count G (vs_n)
python3 benchmarks/lbm_ldc/scripts/plotting/plot_disg_vs_dense.py --config $cfg
#   strong_scaling: <series>-disg-vs-dense-<transport>-hp<off|on>.png, <series>-disg-vs-dense.csv
python3 benchmarks/lbm_ldc/scripts/plotting/plot_best_family.py --config $cfg
#   best_family:    <series>-best-family.png
```

Common options: `--data-dir DIR` reads and writes another folder (for example
`benchmarks/lbm_ldc/reference`), `--only a,b` plots a subset of
the experiments and `--show` pops each figure up. Series without reports are
skipped with a warning. The plotters need `matplotlib`.

Scaling panels are one per grid (per grid × decomposition for the distributed
grids), with one curve per transport × hp-halo; distributed panels label each
GPU count with the process grid ProcessGrid chose. The `grids` figure puts
every grid on shared axes: the left panel is the same transport for all
(NCCL, slab, hp-halo off), the right one each grid's fastest run per GPU
count, with a table of the winning configuration under it.

`plot_best_family.py` colours each (N, G) cell by the fastest of the four
families Dense / DenseDisg × Peeraccess / NCCL (red / blue / green / purple)
and blends in white for every other family within `--threshold` (default
10 %) of it, so a fully white cell means all four tie. Each cell is labelled
with the winner, its MLUPS per GPU (in thousands) and the near-tie count;
cells with a missing report are grey, so set `mpi_single_gpu` to fill the
1-GPU row.

## Reference results

[`../reference/`](../reference/) is the committed copy of a full DGX sweep
(8× A100-SXM4-40GB, D3Q19 + BGK, fp32, `--overlap standard`): 1050 reports in
`json/`, the figures in `plots/<experiment>/` and the YAML that produced them.
Its four experiments:

| Experiment | Type | Matrix |
| --- | --- | --- |
| `grids` | `strong_scaling` | all four grids, N=256..512 step 64, hp-halo off / on: `dense` / `disg` × Peeraccess / NCCL / MPI (P2P) on 1..8 GPUs; `dense_distributed` / `disg_distributed` × NCCL / MPI (P2P) × slab (2..8) / pencil (4, 8) / brick (8) |
| `domain_size` | `domain_size` | `dense` / `disg` × all three transports at 8 GPUs plus the 1-GPU reference, N=128..608 step 32 |
| `best_family` | `best_family` | `dense` / `disg` × Peeraccess / NCCL on 1, 2, 4, 8 GPUs, N=128..608 step 32 |
| `n512` | `strong_scaling` | `dense` / `disg` × all three transports on 1, 2, 4, 8 GPUs, N=512 |

Re-plot it with `--config benchmarks/lbm_ldc/reference/lbm-sweep.yml`
(its `data_dir` is `reference/`). The 840 `grids` runs use the current timing
(400 + 5 × 1000 steps); the other 210 come from earlier sweeps with
400 + 1 × 1600 steps, as the YAML's header notes. To refresh it, run the sweep
into `data/` and copy `data/json/` and `data/plots/` over.

Domain sizes stop at N=608: N=640 does not fit the two Q=19 population fields
on one 40 GB A100, and a size without a single-GPU reference cannot be
compared.
