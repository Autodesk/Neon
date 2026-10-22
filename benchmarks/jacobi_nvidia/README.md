# Jacobi NVIDIA parity benchmark

Ports the [NVIDIA multi-gpu-programming-models](https://github.com/NVIDIA/multi-gpu-programming-models) 2D Laplace Jacobi solver onto `carbon::Dense` with extent `{nx, 1, ny - 2}` and `BoundaryMode::PeriodicZ`.

## Build

```bash
cmake -S . -B build -DCARBON_BUILD_BENCHMARKS=ON
cmake --build build --target carbon_benchmark_jacobi_nvidia
```

## CLI (matches reference where noted)

- `-nx`, `-ny`, `-niter`, `-nccheck`, `-csv` — same as NVIDIA `single_gpu/jacobi`
- `-gpus N` — limit visible devices (intra-node multi-GPU)
- `--overlap none|standard|extended|two-way-extended` — Skeleton overlap (cf. `mpi_overlap` / `nccl_overlap`)
- `--block 64x1x8` — dense **compute/halo** launch block (default `64x1x8`; does not tune the fused Jacobi reduce step)
- `--reduce-block N` or `--reduce-block BxY` — fused Jacobi **reduce** launch: 1D (`256`, `512`, …) or **2D** (`32x32` matches NVIDIA `jacobi_kernel`; also `16x16`, `64x64`, …). Default 1D @ 256 threads.
- `--dump path` — write row-major `float32` field (`ny * nx`) after the solve
- `--skip-validation` — skip single-GPU reference check
- `--report path` — JSON timing report (via `bench_report.h`)

## Examples

```bash
# Single GPU CSV line (compare with multi-gpu-programming-models/single_gpu/jacobi)
./build/benchmarks/jacobi_nvidia/carbon_benchmark_jacobi_nvidia -csv -nx 4096 -ny 4096 -niter 100

# Two GPUs, overlap
CUDA_VISIBLE_DEVICES=0,1 ./build/benchmarks/jacobi_nvidia/carbon_benchmark_jacobi_nvidia \
  -csv -nx 8192 -ny 8192 --overlap standard

# MPI + NCCL (one rank per GPU)
mpiexec -n 2 ./build/benchmarks/jacobi_nvidia/carbon_benchmark_jacobi_nvidia -csv -nx 8192 -ny 8192
```

## Cross-codebase checks

```bash
scripts/compare_jacobi_perf.sh   # env: NX, NY, NITER, GPUS

# NVIDIA reference binary (multi-gpu-programming-models/single_gpu)
cd ../../../multi-gpu-programming-models/single_gpu
make BUILD_SM_ARCH=86    # or plain `make` on CUDA 12.x; see Makefile
./jacobi --dump ref.bin -nx 512 -ny 512 -niter 50 --fixed-iters

# Carbon candidate (same -nx/-ny/-niter; use --warmup 0 for A/B timing)
./carbon_benchmark_jacobi_nvidia --dump cand.bin -nx 512 -ny 512 -niter 50 --warmup 0 --fixed-iters
python3 scripts/compare_dumps.py ref.bin cand.bin -nx 512 -ny 512
# Default interior tolerance is 1e-6 (float32 ULP vs NVIDIA reference).
```

## Build note (GPU arch)

Set `CMAKE_CUDA_ARCHITECTURES` to match your GPU (e.g. `86` for RTX A4000). Wrong arch causes CUDA error 222 at runtime.

The NVIDIA reference `Makefile` now includes `sm_86` in the default fat binary. On CUDA 13+ (which drops `sm_70`), use `make BUILD_SM_ARCH=86` on Ampere if the default target fails.
