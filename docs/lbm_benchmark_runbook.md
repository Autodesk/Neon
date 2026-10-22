# LBM benchmark runbook (single-GPU → DGX multi-GPU)

Operational memory for running `benchmarks/lbm_lid_driven_cavity` at scale —
written after the 2026-07 grid-generic pull-stream + `read_pull` fast-path +
distributed-validation work (see `docs/CHANGELOG.md` 2026-07 entries and
`benchmarks/README.md`). Everything below was validated on a 2×A100 80GB PCIe
host; the DGX section extrapolates the knobs for an 8-GPU NVLink/NVSwitch node.

## What the benchmark is

3D lid-driven-cavity Lattice-Boltzmann on a uniform `N×N×N` cavity. Two-pop pull
streaming + half-way bounce-back + collision, wrapped in a `carbon::Skeleton`
(auto halo). Headline metric is **MLUPS** over post-warmup iters:
`MLUPS = N^3 * timed_iters / (1e6 * seconds)`. Prints one scrapeable
`MLUPS=...` line (rank 0 only under MPI).

Runtime knobs that matter for scaling:

- `--grid dense|disg` — `carbon::Dense` (aggregated SoA z-slabs) vs
  `carbon::DenseDisg` (disaggregated five-region LBM layout). **Both run the
  identical hot kernel** via the `Partition::read_pull` trusted-pull primitive;
  after the read_pull fast path they are at MLUPS parity (see below).
- `--lattice q19|q27`, `--collision bgk|kbc` (KBC needs q27). Default q19+bgk is
  the SOTA symmetry-unrolled fast kernel (48 regs/thread on both grids).
- `--overlap none|standard|...` — **use `standard` for multi-GPU / distributed**
  so the halo overlaps interior compute on a second stream instead of
  serialising. `none` is the single-GPU baseline.
- `--cuda-graph` — collapses per-iter host launch overhead; single-GPU /
  single-stream only (no-op fallback otherwise). Worth trying on DGX for small N
  where launch overhead bites.
- `--checksum` — Sum-reduces total domain mass (`sum_q pop_q == rho`). Grid- and
  GPU-count-invariant to float summation-order noise; the correctness oracle.
- `--report PATH` — structured JSON (schema `carbon-lbm-report-v1`), consumed by
  `benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py`.

## Build

`clang-format` 20+ must be on PATH for `format-check`
(`pip install --user clang-format`; `export PATH="$HOME/.local/bin:$PATH"`).

Single-node (intra-process multi-GPU, no MPI):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target carbon_benchmark_lbm_lid_driven_cavity
```

Distributed (MPI + NCCL, one rank per GPU) — **separate build tree**:

```bash
cmake -S . -B build-mpi -DCARBON_ENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-mpi -j --target carbon_benchmark_lbm_lid_driven_cavity
```

On DGX set the GPU arch if CMake doesn't autodetect: A100 `sm_80`, H100 `sm_90`
(e.g. `-DCMAKE_CUDA_ARCHITECTURES=90`). NCCL must be discoverable
(`NCCL_INCLUDE_DIR` / `NCCL_LIBRARY`, or a system install); on DGX the CUDA/NCCL
stack is normally already present.

## Run

Binary path: `build[-mpi]/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity`.

Single GPU (true baseline — pin ONE visible GPU):

```bash
CUDA_VISIBLE_DEVICES=0 ./build/.../carbon_benchmark_lbm_lid_driven_cavity \
  --n 256 --iters 1000 --warmup 200 --report-every 0
```

All visible GPUs, intra-process (peer copy / NVLink):

```bash
./build/.../carbon_benchmark_lbm_lid_driven_cavity \
  --n 512 --iters 2000 --warmup 200 --overlap standard
```

Distributed, one rank per GPU (NCCL halo):

```bash
mpiexec -n 8 ./build-mpi/.../carbon_benchmark_lbm_lid_driven_cavity \
  --n 768 --iters 2000 --warmup 200 --overlap standard
```

## Gotchas learned this session (read before benchmarking)

- **`mpiexec -n 1` grabs ALL visible GPUs.** The driver treats `world_size()==1`
  as "not distributed" and falls through to `get_all_devices()`. For a real
  1-GPU baseline use a direct single-process run with `CUDA_VISIBLE_DEVICES=0`,
  not `mpiexec -n 1`.
- **Device pinning.** If a rank sees >1 GPU it falls back to
  `device = local_rank % visible_devices` (correct on one node, emits a
  warning). For clean/multi-node runs pin at the launcher — one GPU per rank:
  `srun --gpus-per-task=1`, `mpirun --map-by ppr:1:node:PE=...`, or wrap the
  binary in `CUDA_VISIBLE_DEVICES=$LOCAL_RANK` (Intel MPI: `$MPI_LOCALRANKID`;
  Open MPI: `$OMPI_COMM_WORLD_LOCAL_RANK`).
- **Strict one-rank-per-GPU.** Under MPI each rank owns exactly one GPU
  (`gpu_group().size()==1`); the global domain is z-partitioned across
  `world_size`. `--n` is always the GLOBAL edge length; MLUPS is global.
- **`--overlap standard` for multi-GPU**, always. On PCIe the default `none`
  regresses below 1-GPU because the halo serialises every iteration; on NVLink
  the gap is smaller but standard still wins.
- **VTK dump (`--vtk-out`) is Dense-only** (`carbon::io::write_vtk` has no
  DenseDisg overload) — `--grid disg` warns and skips. Use `--checksum` for
  disg correctness instead.
- **Small N is launch/halo bound.** N=96 gave 1.8× scaling; N≥256 gives
  1.9–2.0×. On DGX push N up (512/768/1024) and iters up (≥2000) for meaningful
  strong-scaling numbers; the per-GPU domain must be big enough to hide the
  halo.

## Reference numbers (2×A100 80GB PCIe, D3Q19+BGK, fp32)

Use as a sanity floor; a DGX (NVLink/NVSwitch, more GPUs) should scale better.

| grid  | N   | 1 GPU | 2 GPU intra-proc | 2 rank MPI+NCCL | scale |
| ----- | --- | ----: | ---------------: | --------------: | ----: |
| dense | 256 |  9469 |            18702 |           18202 | ~1.9× |
| disg  | 256 |  9496 |            18759 |           17736 | ~1.9× |
| dense | 384 |  9770 |            19602 |           19256 | ~2.0× |
| disg  | 384 |  9782 |            19426 |           18921 | ~1.9× |

- Single-GPU ~9.5k MLUPS (climbs with N as fixed overhead amortises). The
  D3Q19+BGK fast kernel is 48 regs/thread, ~74% of A100 HBM peak.
- **Dense ≡ DenseDisg** at MLUPS parity after the `read_pull` fast path (the
  earlier 24% disg gap was addressing-code register pressure, now gone —
  matches the Neon EuroPar dense-vs-disaggregated parity result).
- Checksum at N=96: single-GPU `830592.16792`; 2-GPU/2-rank agree to ~1e-7
  relative (float summation order across partition seams).

## Verify after any code change (the triad)

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure      # 85 default (CUDA)
cmake --build build --target format-check
```

Distributed tests need the MPI build: `ctest --test-dir build-mpi` adds the
`distributed`-labelled suite (dense/disg/block/mgrid halo, reduce, native I/O,
skeleton overlap, checkpoint).

## Where the code lives

- Driver: `benchmarks/lbm_lid_driven_cavity/lbm_lid_driven_cavity.cu`
  (templated `run_benchmark<Grid, Lattice, Collision>`; MPI branches under
  `#ifdef CARBON_HAVE_MPI_NCCL`).
- Kernels: `benchmarks/lbm_lid_driven_cavity/src/lbm_kernels.h`
  (grid-generic via `Partition::read_pull`).
- Fast pull: `include/carbon/detail/dense/partition.h` (Dense) and
  `include/carbon/detail/dense_disg/partition.h` (DenseDisg constant-delta fast
  path + seam fallback) + `dense_disg/span.h::pull_idx`.
