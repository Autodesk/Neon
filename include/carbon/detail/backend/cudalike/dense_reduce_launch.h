#pragma once

/**
 * Host-side launch dispatch for @c Dense grid reduce kernels at variable CUB block
 * sizes (compile-time @c BlockThreads template on the device kernels).
 */

#include <carbon/data_view.h>
#include <carbon/detail/backend/cuda_like.h>
#include <carbon/detail/backend/cudalike/dense_reduce_kernel.h>
#include <carbon/detail/gpu.h>
#include <carbon/logging.h>

#include <cstdint>
#include <stdexcept>

namespace carbon::detail::dense {

#if CARBON_CUDALIKE_DEVICE_TU

template <typename F,
          typename T,
          typename Op>
auto
launch_dense_grid_reduce_kernel(int block_threads,
                                const ::carbon::detail::Gpu& gpu,
                                int64_t blocks,
                                size_t shared_mem,
                                gpu_stream_t stream,
                                Span span,
                                ::carbon::DataView view,
                                F& fn,
                                Op& op,
                                T identity,
                                T* partials,
                                int64_t total_cells) -> gpu_error_t {
#define CARBON_LAUNCH_DENSE_REDUCE(BT)                                                       \
  return gpu.template typed_launch<&dense_grid_reduce_kernel_t<BT, F, T, Op>>(               \
      dim3(static_cast<unsigned>(blocks), 1U, 1U), dim3(static_cast<unsigned>(BT), 1U, 1U),  \
      shared_mem, stream, span, view, fn, op, identity, partials, total_cells)

  switch (block_threads) {
    case 32:
      CARBON_LAUNCH_DENSE_REDUCE(32);
    case 64:
      CARBON_LAUNCH_DENSE_REDUCE(64);
    case 128:
      CARBON_LAUNCH_DENSE_REDUCE(128);
    case 256:
      CARBON_LAUNCH_DENSE_REDUCE(256);
    case 512:
      CARBON_LAUNCH_DENSE_REDUCE(512);
    case 1024:
      CARBON_LAUNCH_DENSE_REDUCE(1024);
    default:
      log::throw_with_log<std::invalid_argument>(
          "Kernel::reduce: unsupported Dense stage1_threads {} (use a power-of-two 32–1024)",
          block_threads);
  }
#undef CARBON_LAUNCH_DENSE_REDUCE
}

template <typename F,
          typename T,
          typename Op>
auto
launch_dense_grid_reduce_block_atomic_kernel(int block_threads,
                                             const ::carbon::detail::Gpu& gpu,
                                             int64_t blocks,
                                             size_t shared_mem,
                                             gpu_stream_t stream,
                                             Span span,
                                             ::carbon::DataView view,
                                             F& fn,
                                             Op& op,
                                             T identity,
                                             T* global_out,
                                             int64_t total_cells) -> gpu_error_t {
#define CARBON_LAUNCH_DENSE_REDUCE_ATOMIC(BT)                                                \
  return gpu.template typed_launch<&dense_grid_reduce_block_atomic_kernel_t<BT, F, T, Op>>(  \
      dim3(static_cast<unsigned>(blocks), 1U, 1U), dim3(static_cast<unsigned>(BT), 1U, 1U), \
      shared_mem, stream, span, view, fn, op, identity, global_out, total_cells)

  switch (block_threads) {
    case 32:
      CARBON_LAUNCH_DENSE_REDUCE_ATOMIC(32);
    case 64:
      CARBON_LAUNCH_DENSE_REDUCE_ATOMIC(64);
    case 128:
      CARBON_LAUNCH_DENSE_REDUCE_ATOMIC(128);
    case 256:
      CARBON_LAUNCH_DENSE_REDUCE_ATOMIC(256);
    case 512:
      CARBON_LAUNCH_DENSE_REDUCE_ATOMIC(512);
    case 1024:
      CARBON_LAUNCH_DENSE_REDUCE_ATOMIC(1024);
    default:
      log::throw_with_log<std::invalid_argument>(
          "Kernel::reduce: unsupported Dense stage1_threads {} (use a power-of-two 32–1024)",
          block_threads);
  }
#undef CARBON_LAUNCH_DENSE_REDUCE_ATOMIC
}

template <typename F,
          typename T,
          typename Op>
auto
launch_dense_grid_reduce_kernel_2d(int block_dim_x,
                                   int block_dim_y,
                                   const ::carbon::detail::Gpu& gpu,
                                   int64_t blocks,
                                   size_t shared_mem,
                                   gpu_stream_t stream,
                                   Span span,
                                   ::carbon::DataView view,
                                   F& fn,
                                   Op& op,
                                   T identity,
                                   T* partials,
                                   int64_t total_cells) -> gpu_error_t {
  const ::carbon::int64_3d ext = span.iter_extent(view);
  const unsigned grid_x =
      static_cast<unsigned>((ext.x + block_dim_x - 1) / block_dim_x);
  const unsigned grid_y =
      static_cast<unsigned>((ext.z + block_dim_y - 1) / block_dim_y);
  if (static_cast<int64_t>(grid_x) * static_cast<int64_t>(grid_y) != blocks) {
    log::throw_with_log<std::invalid_argument>(
        "Kernel::reduce: 2D launch grid {}x{} != expected block count {}", grid_x, grid_y,
        blocks);
  }

#define CARBON_LAUNCH_DENSE_REDUCE_2D(BX, BY)                                                \
  return gpu.template typed_launch<&dense_grid_reduce_kernel_2d_t<BX, BY, F, T, Op>>(        \
      dim3(grid_x, grid_y, 1U), dim3(static_cast<unsigned>(BX), static_cast<unsigned>(BY),  \
           1U),                                                                              \
      shared_mem, stream, span, view, fn, op, identity, partials, total_cells)

  if (block_dim_x == 32 && block_dim_y == 32) {
    CARBON_LAUNCH_DENSE_REDUCE_2D(32, 32);
  }
  if (block_dim_x == 16 && block_dim_y == 16) {
    CARBON_LAUNCH_DENSE_REDUCE_2D(16, 16);
  }
  if (block_dim_x == 8 && block_dim_y == 8) {
    CARBON_LAUNCH_DENSE_REDUCE_2D(8, 8);
  }
  if (block_dim_x == 32 && block_dim_y == 16) {
    CARBON_LAUNCH_DENSE_REDUCE_2D(32, 16);
  }
  if (block_dim_x == 16 && block_dim_y == 32) {
    CARBON_LAUNCH_DENSE_REDUCE_2D(16, 32);
  }
  if (block_dim_x == 64 && block_dim_y == 64) {
    CARBON_LAUNCH_DENSE_REDUCE_2D(64, 64);
  }
  log::throw_with_log<std::invalid_argument>(
      "Kernel::reduce: unsupported 2D reduce block {}x{} (try 32x32, 16x16, …)", block_dim_x,
      block_dim_y);
#undef CARBON_LAUNCH_DENSE_REDUCE_2D
}

template <typename F,
          typename T,
          typename Op>
auto
launch_dense_grid_reduce_block_atomic_kernel_2d(int block_dim_x,
                                              int block_dim_y,
                                              const ::carbon::detail::Gpu& gpu,
                                              int64_t blocks,
                                              size_t shared_mem,
                                              gpu_stream_t stream,
                                              Span span,
                                              ::carbon::DataView view,
                                              F& fn,
                                              Op& op,
                                              T identity,
                                              T* global_out,
                                              int64_t total_cells) -> gpu_error_t {
  const ::carbon::int64_3d ext = span.iter_extent(view);
  const unsigned grid_x =
      static_cast<unsigned>((ext.x + block_dim_x - 1) / block_dim_x);
  const unsigned grid_y =
      static_cast<unsigned>((ext.z + block_dim_y - 1) / block_dim_y);
  if (static_cast<int64_t>(grid_x) * static_cast<int64_t>(grid_y) != blocks) {
    log::throw_with_log<std::invalid_argument>(
        "Kernel::reduce: 2D launch grid {}x{} != expected block count {}", grid_x, grid_y,
        blocks);
  }

#define CARBON_LAUNCH_DENSE_REDUCE_ATOMIC_2D(BX, BY)                                         \
  return gpu.template typed_launch<                                                          \
      &dense_grid_reduce_block_atomic_kernel_2d_t<BX, BY, F, T, Op>>(                        \
      dim3(grid_x, grid_y, 1U), dim3(static_cast<unsigned>(BX), static_cast<unsigned>(BY),  \
           1U),                                                                              \
      shared_mem, stream, span, view, fn, op, identity, global_out, total_cells)

  if (block_dim_x == 32 && block_dim_y == 32) {
    CARBON_LAUNCH_DENSE_REDUCE_ATOMIC_2D(32, 32);
  }
  if (block_dim_x == 16 && block_dim_y == 16) {
    CARBON_LAUNCH_DENSE_REDUCE_ATOMIC_2D(16, 16);
  }
  if (block_dim_x == 8 && block_dim_y == 8) {
    CARBON_LAUNCH_DENSE_REDUCE_ATOMIC_2D(8, 8);
  }
  if (block_dim_x == 32 && block_dim_y == 16) {
    CARBON_LAUNCH_DENSE_REDUCE_ATOMIC_2D(32, 16);
  }
  if (block_dim_x == 16 && block_dim_y == 32) {
    CARBON_LAUNCH_DENSE_REDUCE_ATOMIC_2D(16, 32);
  }
  if (block_dim_x == 64 && block_dim_y == 64) {
    CARBON_LAUNCH_DENSE_REDUCE_ATOMIC_2D(64, 64);
  }
  log::throw_with_log<std::invalid_argument>(
      "Kernel::reduce: unsupported 2D reduce block {}x{} (try 32x32, 16x16, …)", block_dim_x,
      block_dim_y);
#undef CARBON_LAUNCH_DENSE_REDUCE_ATOMIC_2D
}

#endif  // CARBON_CUDALIKE_DEVICE_TU

}  // namespace carbon::detail::dense
