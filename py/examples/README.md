# Carbon Python examples

Runnable demos for the Warp + `libcarbonpy` bindings. They mirror the C++
programs under `examples/` but use Python containers and Warp `@wp.func` kernels.

## Prerequisites

```bash
git submodule update --init --recursive warp
./scripts/setup_warp.sh

cmake -S . -B build-py -DCARBON_ENABLE_PYTHON=ON
cmake --build build-py -j
```

## Running

From the repo root:

```bash
export PYTHONPATH=warp:py
python3 py/examples/jacobi_step.py
python3 py/examples/salt_pepper_filter.py
python3 py/examples/salt_pepper_filter.py --output /tmp/filtered.pgm --noisy-output /tmp/noisy.pgm
```

On multi-GPU hosts, restrict to one device for PGM output and simpler demos:

```bash
CUDA_VISIBLE_DEVICES=0 python3 py/examples/salt_pepper_filter.py --output /tmp/filtered.pgm
```

The salt-and-pepper demo defaults to `examples/data/astronaut.pgm` (same bundled
image as `examples/salt_pepper_filter.cu`).

## Scripts

| Script | C++ counterpart | Notes |
|--------|-----------------|-------|
| `jacobi_step.py` | Jacobi step in `examples/jacobi_unified.cu` | Minimal 8³ Dense grid, two-field container |
| `salt_pepper_filter.py` | `examples/salt_pepper_filter.cu` | PGM load, noise, 3×3 median filter, PSNR |

Shared helpers: `pgm_io.py` (P5 read/write), `_carbon_env.py` (path + Warp fork check).

## Limitations vs C++

The Python port currently covers the **Dense** grid on a **single GPU**:

- No `--grid disg|block`, MPI, or Skeleton overlap modes yet
- Field upload and median filter use `wp.launch` with Carbon partition builtins
- Open-boundary neighbour reads are implemented manually (carbonpy Dense uses an empty stencil today)
- A `DenseContainer` Driver-path median (matching `jacobi_step.py`) is straightforward to add once stencil/halo tokens land
