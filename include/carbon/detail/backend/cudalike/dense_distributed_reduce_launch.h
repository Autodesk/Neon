#pragma once

/**
 * Host-side launch dispatch for @c DenseDistributed grid reduce kernels at variable CUB block
 * sizes (compile-time @c BlockThreads template on the device kernels).
 */

#include <carbon/data_view.h>
#include <carbon/detail/backend/cuda_like.h>
#include <carbon/detail/backend/cudalike/dense_distributed_reduce_kernel.h>
#include <carbon/detail/gpu.h>
#include <carbon/logging.h>

#include <cstdint>
#include <stdexcept>

namespace carbon::detail::dense_distributed {

#if CARBON_CUDALIKE_DEVICE_TU

template <::carbon::DataView View,
          typename F,
          typename T,
          typename Op>
auto
launch_dense_grid_reduce_kernel_view(int block_threads,
                                     const ::carbon::detail::Gpu& gpu,
                                     int64_t blocks,
                                     size_t shared_mem,
                                     gpu_stream_t stream,
                                     Span span,
                                     F& fn,
                                     Op& op,
                                     T identity,
                                     T* partials,
                                     int64_t total_cells) -> gpu_error_t {
#define CARBON_LAUNCH_DENSE_REDUCE(BT)                                                      \
  return gpu.template typed_launch<&dense_grid_reduce_kernel_t<BT, View, F, T, Op>>(        \
      dim3(static_cast<unsigned>(blocks), 1U, 1U), dim3(static_cast<unsigned>(BT), 1U, 1U), \
      shared_mem, stream, span, fn, op, identity, partials, total_cells)

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
          "Kernel::reduce: unsupported DenseDistributed stage1_threads {} (use a power-of-two "
          "32–1024)",
          block_threads);
  }
#undef CARBON_LAUNCH_DENSE_REDUCE
}

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
  switch (view) {
    case ::carbon::DataView::Standard:
      return launch_dense_grid_reduce_kernel_view<::carbon::DataView::Standard>(
          block_threads, gpu, blocks, shared_mem, stream, span, fn, op, identity, partials,
          total_cells);
    case ::carbon::DataView::Internal:
      return launch_dense_grid_reduce_kernel_view<::carbon::DataView::Internal>(
          block_threads, gpu, blocks, shared_mem, stream, span, fn, op, identity, partials,
          total_cells);
    case ::carbon::DataView::Boundary:
      return launch_dense_grid_reduce_kernel_view<::carbon::DataView::Boundary>(
          block_threads, gpu, blocks, shared_mem, stream, span, fn, op, identity, partials,
          total_cells);
  }
  return gpu_error_not_supported;
}

template <::carbon::DataView View,
          typename F,
          typename T,
          typename Op>
auto
launch_dense_grid_reduce_block_atomic_kernel_view(int block_threads,
                                                  const ::carbon::detail::Gpu& gpu,
                                                  int64_t blocks,
                                                  size_t shared_mem,
                                                  gpu_stream_t stream,
                                                  Span span,
                                                  F& fn,
                                                  Op& op,
                                                  T identity,
                                                  T* global_out,
                                                  int64_t total_cells) -> gpu_error_t {
#define CARBON_LAUNCH_DENSE_REDUCE_ATOMIC(BT)                                               \
  return gpu.template typed_launch<&dense_grid_reduce_block_atomic_kernel_t<BT, View, F, T, Op>>( \
      dim3(static_cast<unsigned>(blocks), 1U, 1U), dim3(static_cast<unsigned>(BT), 1U, 1U), \
      shared_mem, stream, span, fn, op, identity, global_out, total_cells)

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
          "Kernel::reduce: unsupported DenseDistributed stage1_threads {} (use a power-of-two "
          "32–1024)",
          block_threads);
  }
#undef CARBON_LAUNCH_DENSE_REDUCE_ATOMIC
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
  switch (view) {
    case ::carbon::DataView::Standard:
      return launch_dense_grid_reduce_block_atomic_kernel_view<::carbon::DataView::Standard>(
          block_threads, gpu, blocks, shared_mem, stream, span, fn, op, identity, global_out,
          total_cells);
    case ::carbon::DataView::Internal:
      return launch_dense_grid_reduce_block_atomic_kernel_view<::carbon::DataView::Internal>(
          block_threads, gpu, blocks, shared_mem, stream, span, fn, op, identity, global_out,
          total_cells);
    case ::carbon::DataView::Boundary:
      return launch_dense_grid_reduce_block_atomic_kernel_view<::carbon::DataView::Boundary>(
          block_threads, gpu, blocks, shared_mem, stream, span, fn, op, identity, global_out,
          total_cells);
  }
  return gpu_error_not_supported;
}

template <::carbon::DataView View,
          int BlockDimX,
          int BlockDimY,
          typename F,
          typename T,
          typename Op>
auto
launch_dense_grid_reduce_kernel_2d_view(int block_dim_x,
                                        int block_dim_y,
                                        unsigned grid_x,
                                        unsigned grid_y,
                                        const ::carbon::detail::Gpu& gpu,
                                        size_t shared_mem,
                                        gpu_stream_t stream,
                                        Span span,
                                        F& fn,
                                        Op& op,
                                        T identity,
                                        T* partials,
                                        int64_t total_cells) -> gpu_error_t {
  if (block_dim_x == BlockDimX && block_dim_y == BlockDimY) {
    return gpu.template typed_launch<
        &dense_grid_reduce_kernel_2d_t<BlockDimX, BlockDimY, View, F, T, Op>>(
        dim3(grid_x, grid_y, 1U),
        dim3(static_cast<unsigned>(BlockDimX), static_cast<unsigned>(BlockDimY), 1U), shared_mem,
        stream, span, fn, op, identity, partials, total_cells);
  }
  log::throw_with_log<std::invalid_argument>(
      "Kernel::reduce: unsupported 2D reduce block {}x{} (currently only 32x32)", block_dim_x,
      block_dim_y);
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
  const unsigned grid_x = static_cast<unsigned>((ext.x + block_dim_x - 1) / block_dim_x);
  const int64_t yz_extent = ext.y * ext.z;
  const unsigned grid_y = static_cast<unsigned>((yz_extent + block_dim_y - 1) / block_dim_y);
  if (static_cast<int64_t>(grid_x) * static_cast<int64_t>(grid_y) != blocks) {
    log::throw_with_log<std::invalid_argument>(
        "Kernel::reduce: 2D launch grid {}x{} over (x, y*z) != expected block count {}", grid_x,
        grid_y, blocks);
  }

  switch (view) {
    case ::carbon::DataView::Standard:
      return launch_dense_grid_reduce_kernel_2d_view<::carbon::DataView::Standard, 32, 32>(
          block_dim_x, block_dim_y, grid_x, grid_y, gpu, shared_mem, stream, span, fn, op,
          identity, partials, total_cells);
    case ::carbon::DataView::Internal:
      return launch_dense_grid_reduce_kernel_2d_view<::carbon::DataView::Internal, 32, 32>(
          block_dim_x, block_dim_y, grid_x, grid_y, gpu, shared_mem, stream, span, fn, op,
          identity, partials, total_cells);
    case ::carbon::DataView::Boundary:
      return launch_dense_grid_reduce_kernel_2d_view<::carbon::DataView::Boundary, 32, 32>(
          block_dim_x, block_dim_y, grid_x, grid_y, gpu, shared_mem, stream, span, fn, op,
          identity, partials, total_cells);
  }
  return gpu_error_not_supported;
}

template <::carbon::DataView View,
          int BlockDimX,
          int BlockDimY,
          typename F,
          typename T,
          typename Op>
auto
launch_dense_grid_reduce_block_atomic_kernel_2d_view(int block_dim_x,
                                                     int block_dim_y,
                                                     unsigned grid_x,
                                                     unsigned grid_y,
                                                     const ::carbon::detail::Gpu& gpu,
                                                     size_t shared_mem,
                                                     gpu_stream_t stream,
                                                     Span span,
                                                     F& fn,
                                                     Op& op,
                                                     T identity,
                                                     T* global_out,
                                                     int64_t total_cells) -> gpu_error_t {
  if (block_dim_x == BlockDimX && block_dim_y == BlockDimY) {
    return gpu.template typed_launch<
        &dense_grid_reduce_block_atomic_kernel_2d_t<BlockDimX, BlockDimY, View, F, T, Op>>(
        dim3(grid_x, grid_y, 1U),
        dim3(static_cast<unsigned>(BlockDimX), static_cast<unsigned>(BlockDimY), 1U), shared_mem,
        stream, span, fn, op, identity, global_out, total_cells);
  }
  log::throw_with_log<std::invalid_argument>(
      "Kernel::reduce: unsupported 2D reduce block {}x{} (currently only 32x32)", block_dim_x,
      block_dim_y);
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
  const unsigned grid_x = static_cast<unsigned>((ext.x + block_dim_x - 1) / block_dim_x);
  const int64_t yz_extent = ext.y * ext.z;
  const unsigned grid_y = static_cast<unsigned>((yz_extent + block_dim_y - 1) / block_dim_y);
  if (static_cast<int64_t>(grid_x) * static_cast<int64_t>(grid_y) != blocks) {
    log::throw_with_log<std::invalid_argument>(
        "Kernel::reduce: 2D launch grid {}x{} over (x, y*z) != expected block count {}", grid_x,
        grid_y, blocks);
  }

  switch (view) {
    case ::carbon::DataView::Standard:
      return launch_dense_grid_reduce_block_atomic_kernel_2d_view<
          ::carbon::DataView::Standard, 32, 32>(block_dim_x, block_dim_y, grid_x, grid_y, gpu,
                                                shared_mem, stream, span, fn, op, identity,
                                                global_out, total_cells);
    case ::carbon::DataView::Internal:
      return launch_dense_grid_reduce_block_atomic_kernel_2d_view<
          ::carbon::DataView::Internal, 32, 32>(block_dim_x, block_dim_y, grid_x, grid_y, gpu,
                                                shared_mem, stream, span, fn, op, identity,
                                                global_out, total_cells);
    case ::carbon::DataView::Boundary:
      return launch_dense_grid_reduce_block_atomic_kernel_2d_view<
          ::carbon::DataView::Boundary, 32, 32>(block_dim_x, block_dim_y, grid_x, grid_y, gpu,
                                                shared_mem, stream, span, fn, op, identity,
                                                global_out, total_cells);
  }
  return gpu_error_not_supported;
}

#endif  // CARBON_CUDALIKE_DEVICE_TU

}  // namespace carbon::detail::dense_distributed
