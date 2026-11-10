# 2-D stage-1 reduce over flattened `(y, z)` — implementation report

Date: 2026-11-07

## Problem

The opt-in 2-D stage-1 reduce path (`ReduceOptions::stage1_threads_y > 0`) mapped a
CUDA/HIP block over `(x, z)` and always called

```cpp
span.set_and_validate(view, ix, 0, iz);
```

so it reduced only the `y == 0` plane. The launch grid and
`ReduceLaunchTraits::stage1_unit_count` also used `ext.z` only, ignoring `ext.y`.

This was correct for the intended NVIDIA Jacobi-style `Dense` slab shape where
`iter_extent(view).y == 1`, but it silently under-counted any grid with `y > 1`:

- `Dense` / `DenseDistributed` `Standard` and `Internal` views reduced only `y == 0`.
- The `DenseDistributed` compact-shell `Boundary` view reduced only its first 128
  shell cells, because its launch box was `(128, ceil(S / 128), 1)` and
  `t = iy * ext.x + ix` with `iy == 0` covered only the first row.

## Fix

The second block dimension is now a flattened `(y, z)` index:

```cpp
const int64_t yz_extent = ext.y * ext.z;
const int64_t flat_yz = blockIdx.y * BlockDimY + threadIdx.y;
if (ix < ext.x && flat_yz < yz_extent) {
  const int64_t iy = flat_yz % ext.y;
  const int64_t iz = flat_yz / ext.y;
  const ValidatedIdx vi = span.set_and_validate(view, ix, iy, iz);
  ...
}
```

The launch grid and stage-1 unit count use the same flattened extent:

```cpp
grid_y = ceil_div(ext.y * ext.z, block_dim_y);
stage1_unit_count = ceil_div(ext.x, block_dim_x) * grid_y;
```

For `y == 1` this reduces to the old `(x, z)` NVIDIA Jacobi shape. For `y > 1` it
covers every `(y, z)` cell.

## Changes by file

### Kernel mapping

- `include/carbon/detail/backend/cudalike/dense_reduce_kernel.h`
  - Normal and `BlockAtomic` 2-D kernels now decode `flat_yz` into `(iy, iz)` and pass
    all three indices to `set_and_validate`.
  - Updated the kernel comment to describe the flattened `(y, z)` second dimension.

- `include/carbon/detail/backend/cudalike/dense_distributed_reduce_kernel.h`
  - Same mapping change for the `DenseDistributed` normal and `BlockAtomic` 2-D
    kernels. This is the part that fixes compact-shell `Boundary` under-counting.

### Launch geometry

- `include/carbon/detail/backend/cudalike/dense_reduce_launch.h`
  - `grid_y` is computed from `ext.y * ext.z` for both normal and `BlockAtomic`
    launches.
  - The grid/block-count consistency error now says the grid is over `(x, y*z)`.
  - Removed dead dispatch branches: only `32x32` is reachable under
    `normalize_dense_stage1_block_2d`. The removed `16x16`, `8x8`, `32x16`,
    `16x32`, and `64x64` branches were either unreachable or invalid (`64x64` is
    4096 threads, above the 1024-thread limit).

- `include/carbon/detail/backend/cudalike/dense_distributed_reduce_launch.h`
  - Same launch-grid and dead-branch changes for `DenseDistributed`.

### Unit count and scratch sizing

- `include/carbon/detail/launch_traits.h`
  - 2-D `stage1_unit_count` now uses
    `ceil_div(ext.x, bx) * ceil_div(ext.y * ext.z, by)`.
  - Updated the trait documentation to distinguish 1-D volume tiling from 2-D
    flattened `(y, z)` tiling.

- `include/carbon/detail/reduce_kernel_impl.h`
  - Added `stage1_scratch_blocks_per_gpu_`, sized from the maximum
    `stage1_unit_count` over `Standard`, `Internal`, and `Boundary`.
  - Partials and CUB temp storage now use this maximum instead of the
    `Standard`-only `block_count_per_gpu_`.
  - This is required because the compact-shell `Boundary` launch box can contain
    more blocks than `Standard` due to flattened `(y, z)` row padding. Sizing from
    `Standard` would otherwise reject or overflow the `Boundary` launch.
  - Public `block_count()` remains tied to the `Standard` view.
  - Updated the sanity-check and error text to refer to the per-view maximum.

### Public documentation

- `include/carbon/reduce.h`
  - `ReduceOptions::stage1_threads` and `stage1_threads_y` now document CUDA/HIP,
    `Dense` / `DenseDistributed`, and the flattened `(y, z)` second dimension.
  - Documented that `y == 1` keeps the NVIDIA Jacobi `32x32` `x/z` shape.
- `docs/CHANGELOG.md`
  - Added the 2026-11 entry describing the bug, the flattened mapping, the scratch
    sizing change, and the removed dead dispatch branches.

## Regression tests

- `tests/test_reduce_sum.cu`
  - New Case 6: `Dense` grid `24 x 32 x 16` (so `kHeight == 32` and the `z` halo
    is non-empty) with `stage1_threads = 32`, `stage1_threads_y = 32`.
  - Checks `Standard` against the host reference and checks
    `Internal + Boundary == Standard` plus `host_value_combined == Standard`.
  - A regression to the old `y == 0` mapping fails all three checks.

- `tests/test_reduce_atomic.cu`
  - New 2-D `BlockAtomic` case with the same `32x32` options, checked against the
    host reference. This exercises the atomic kernel's flattened `(y, z)` mapping.

- `tests/test_dense_distributed_dist.cu`
  - Added `run_reduce_2d` and calls it for every existing `DenseDistributed` case
    (slab, pencil, brick, open/compact boundary modes).
  - Compares `Standard` to a global host reference and checks
    `Internal + Boundary == Standard` and `host_value_combined`.
  - This covers the compact-shell `Boundary` path and the new scratch-max sizing.

## Verification performed

Local CUDA build (RTX 3050 Ti, compute 8.6, CUDA 13.4, MPI + NCCL enabled):

```text
test_reduce_sum    PASS  (including 2D Standard = Internal + Boundary)
test_reduce_atomic PASS  (expected negative BlockAtomic non-Sum log only)
test_reduce_ops    PASS
```

`test_dense_distributed_dist` was built and launched with
`mpirun --oversubscribe -n 4`. It correctly skipped because this machine exposes
only one CUDA device (`needs >= 4 visible devices, have 1; skipping`), so the
multi-rank 2-D reduce regression still needs a multi-GPU run.

An ad-hoc single-GPU `DenseDistributed` smoke test (not committed; 1x1x1
`ProcessGrid`, `9 x 8 x 11`, six-point stencil, `32x32` stage-1) also passed:

```text
dd2d: PASS Standard=39880368 Internal=32629392 Boundary=7250976
```

This directly exercises the compact-shell `Boundary` mapping and the new
scratch-max sizing without MPI.

## Supported grid families and 3D stencil fields

The 2-D stage-1 option (`stage1_threads_y > 0`) is available on plain `Dense`
and `DenseDistributed`:

- **`DenseDistributed` with a true 3-D stencil** (x/y/z halo, pencil/brick, or
  periodic X/Y): supported. `Standard` iterates every owned cell, and the
  flattened `(y, z)` mapping covers `Internal` plus the compact-shell
  `Boundary`. Use `stage1_threads = 32`, `stage1_threads_y = 32`.
- **`Dense`**: the reduce itself covers all owned x/y/z cells and the 2-D path
  works, but `Dense` allocates only a z-halo (`halo_z = stencil.radius()`). It
  is therefore fine for an open x/y boundary where the x/y neighbours lie in
  the full owned plane, but it cannot express x/y periodic halos. Use
  `DenseDistributed` for multi-axis halo/periodicity.
- **`DenseDisg` and `Block`**: the default 1-D reduce works, but
  `stage1_threads_y` is rejected at construction, so the 2-D stage-1 path is
  not available.
- Only the `32x32` 2-D block shape is reachable under the current
  `normalize_dense_stage1_block_2d`.

## Remaining validation

- On Leonardo (AMD MI250X, HIP + MPI), rebuild and run the HIP/MPI checks:

  ```bash
  cmake --build build-mpi-nccl --target test_reduce_sum test_reduce_atomic \
      test_dense_distributed_dist -j
  ./build-mpi-nccl/tests/test_reduce_sum
  ./build-mpi-nccl/tests/test_reduce_atomic
  mpirun -n 4 ./build-mpi-nccl/tests/test_dense_distributed_dist
  # If 8 GPUs are available, also exercise the full brick (2,2,2):
  mpirun -n 8 ./build-mpi-nccl/tests/test_dense_distributed_dist
  ```

  `test_reduce_atomic` should exit 0 with only the expected negative
  `ReduceFinalize::BlockAtomic requires reduce::Sum<T>` log.
  `test_dense_distributed_dist` should run its 4- or 8-rank cases instead of
  self-skipping.
- The `gridDim.y` limit is 65535; the existing 1-D flat path remains the
  fallback for grids whose flattened `(y, z)` extent exceeds that limit.
