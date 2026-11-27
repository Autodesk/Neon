#pragma once

/**
 * Host-side launch dispatch for @c DenseDistributed grid reduce kernels at variable CUB block
 * sizes (compile-time @c BlockThreads template on the device kernels).
 *
 * A @ref DataView is realised as one or more @ref ViewWindow boxes on the host; each box is
 * stamped into a span copy with @c Span::with_window and launched independently. The per-block
 * partials of every window are concatenated, so the caller still performs a single
 * @c cub::DeviceReduce over the summed block count.
 */

#include <carbon/data_view.h>
#include <carbon/detail/backend/cuda_like.h>
#include <carbon/detail/backend/cudalike/dense_distributed_reduce_kernel.h>
#include <carbon/detail/dense_distributed/window_launch.h>
#include <carbon/detail/gpu.h>
#include <carbon/logging.h>

#include <cstdint>
#include <stdexcept>

namespace carbon::detail::dense_distributed {

#if CARBON_CUDALIKE_DEVICE_TU

template <typename F,
          typename T,
          typename Op>
auto
launch_dense_grid_reduce_window(int block_threads,
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
#define CARBON_LAUNCH_DENSE_REDUCE(BT)                                                     \
  return gpu.template typed_launch<&dense_grid_reduce_kernel_t<BT, F, T, Op>>(             \
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
  const ViewWindowSet windows = span.view_windows(view);
  int64_t block_offset = 0;
  int64_t cell_count = 0;
  for (int i = 0; i < windows.count; ++i) {
    const ViewWindow& w = windows.windows[i];
    const int64_t window_blocks = window_flat_units(w, block_threads);
    if (window_blocks == 0) {
      continue;
    }
    const gpu_error_t err = launch_dense_grid_reduce_window<F, T, Op>(
        block_threads, gpu, window_blocks, shared_mem, stream, span.with_window(w), fn, op,
        identity, partials + block_offset, w.volume());
    if (err != gpu_success) {
      return err;
    }
    block_offset += window_blocks;
    cell_count += w.volume();
  }
  if (block_offset != blocks || cell_count != total_cells) {
    log::throw_with_log<std::invalid_argument>(
        "Kernel::reduce: DenseDistributed window plan produced {} block(s) / {} cell(s), "
        "expected {} / {} for view {}",
        block_offset, cell_count, blocks, total_cells, ::carbon::to_string(view));
  }
  return gpu_success;
}

template <typename F,
          typename T,
          typename Op>
auto
launch_dense_grid_reduce_block_atomic_window(int block_threads,
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
  return gpu.template typed_launch<&dense_grid_reduce_block_atomic_kernel_t<BT, F, T, Op>>( \
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
  const ViewWindowSet windows = span.view_windows(view);
  int64_t block_count = 0;
  int64_t cell_count = 0;
  for (int i = 0; i < windows.count; ++i) {
    const ViewWindow& w = windows.windows[i];
    const int64_t window_blocks = window_flat_units(w, block_threads);
    if (window_blocks == 0) {
      continue;
    }
    const gpu_error_t err = launch_dense_grid_reduce_block_atomic_window<F, T, Op>(
        block_threads, gpu, window_blocks, shared_mem, stream, span.with_window(w), fn, op,
        identity, global_out, w.volume());
    if (err != gpu_success) {
      return err;
    }
    block_count += window_blocks;
    cell_count += w.volume();
  }
  if (block_count != blocks || cell_count != total_cells) {
    log::throw_with_log<std::invalid_argument>(
        "Kernel::reduce: DenseDistributed block-atomic window plan produced {} block(s) / {} "
        "cell(s), expected {} / {} for view {}",
        block_count, cell_count, blocks, total_cells, ::carbon::to_string(view));
  }
  return gpu_success;
}

template <int BlockDimX,
          int BlockDimY,
          typename F,
          typename T,
          typename Op>
auto
launch_dense_grid_reduce_window_2d(int block_dim_x,
                                   int block_dim_y,
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
  if (block_dim_x != BlockDimX || block_dim_y != BlockDimY) {
    log::throw_with_log<std::invalid_argument>(
        "Kernel::reduce: unsupported 2D reduce block {}x{} (currently only {}x{})", block_dim_x,
        block_dim_y, BlockDimX, BlockDimY);
  }
  const ::carbon::int64_3d ext = span.window_extent();
  const unsigned grid_x =
      static_cast<unsigned>((ext.x + static_cast<int64_t>(BlockDimX) - 1) / BlockDimX);
  const int64_t yz_extent = ext.y * ext.z;
  const unsigned grid_y =
      static_cast<unsigned>((yz_extent + static_cast<int64_t>(BlockDimY) - 1) / BlockDimY);
  if (static_cast<int64_t>(grid_x) * static_cast<int64_t>(grid_y) != blocks) {
    log::throw_with_log<std::invalid_argument>(
        "Kernel::reduce: 2D launch grid {}x{} over (x, y*z) != expected block count {}", grid_x,
        grid_y, blocks);
  }
  return gpu.template typed_launch<
      &dense_grid_reduce_kernel_2d_t<BlockDimX, BlockDimY, F, T, Op>>(
      dim3(grid_x, grid_y, 1U),
      dim3(static_cast<unsigned>(BlockDimX), static_cast<unsigned>(BlockDimY), 1U), shared_mem,
      stream, span, fn, op, identity, partials, total_cells);
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
  const ViewWindowSet windows = span.view_windows(view);
  int64_t block_offset = 0;
  int64_t cell_count = 0;
  for (int i = 0; i < windows.count; ++i) {
    const ViewWindow& w = windows.windows[i];
    const int64_t window_blocks = window_2d_units(w, block_dim_x, block_dim_y);
    if (window_blocks == 0) {
      continue;
    }
    const gpu_error_t err = launch_dense_grid_reduce_window_2d<32, 32, F, T, Op>(
        block_dim_x, block_dim_y, gpu, window_blocks, shared_mem, stream, span.with_window(w), fn,
        op, identity, partials + block_offset, w.volume());
    if (err != gpu_success) {
      return err;
    }
    block_offset += window_blocks;
    cell_count += w.volume();
  }
  if (block_offset != blocks || cell_count != total_cells) {
    log::throw_with_log<std::invalid_argument>(
        "Kernel::reduce: DenseDistributed 2-D window plan produced {} block(s) / {} cell(s), "
        "expected {} / {} for view {}",
        block_offset, cell_count, blocks, total_cells, ::carbon::to_string(view));
  }
  return gpu_success;
}

template <int BlockDimX,
          int BlockDimY,
          typename F,
          typename T,
          typename Op>
auto
launch_dense_grid_reduce_block_atomic_window_2d(int block_dim_x,
                                                int block_dim_y,
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
  if (block_dim_x != BlockDimX || block_dim_y != BlockDimY) {
    log::throw_with_log<std::invalid_argument>(
        "Kernel::reduce: unsupported 2D reduce block {}x{} (currently only {}x{})", block_dim_x,
        block_dim_y, BlockDimX, BlockDimY);
  }
  const ::carbon::int64_3d ext = span.window_extent();
  const unsigned grid_x =
      static_cast<unsigned>((ext.x + static_cast<int64_t>(BlockDimX) - 1) / BlockDimX);
  const int64_t yz_extent = ext.y * ext.z;
  const unsigned grid_y =
      static_cast<unsigned>((yz_extent + static_cast<int64_t>(BlockDimY) - 1) / BlockDimY);
  if (static_cast<int64_t>(grid_x) * static_cast<int64_t>(grid_y) != blocks) {
    log::throw_with_log<std::invalid_argument>(
        "Kernel::reduce: 2D launch grid {}x{} over (x, y*z) != expected block count {}", grid_x,
        grid_y, blocks);
  }
  return gpu.template typed_launch<
      &dense_grid_reduce_block_atomic_kernel_2d_t<BlockDimX, BlockDimY, F, T, Op>>(
      dim3(grid_x, grid_y, 1U),
      dim3(static_cast<unsigned>(BlockDimX), static_cast<unsigned>(BlockDimY), 1U), shared_mem,
      stream, span, fn, op, identity, global_out, total_cells);
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
  const ViewWindowSet windows = span.view_windows(view);
  int64_t block_count = 0;
  int64_t cell_count = 0;
  for (int i = 0; i < windows.count; ++i) {
    const ViewWindow& w = windows.windows[i];
    const int64_t window_blocks = window_2d_units(w, block_dim_x, block_dim_y);
    if (window_blocks == 0) {
      continue;
    }
    const gpu_error_t err = launch_dense_grid_reduce_block_atomic_window_2d<32, 32, F, T, Op>(
        block_dim_x, block_dim_y, gpu, window_blocks, shared_mem, stream, span.with_window(w), fn,
        op, identity, global_out, w.volume());
    if (err != gpu_success) {
      return err;
    }
    block_count += window_blocks;
    cell_count += w.volume();
  }
  if (block_count != blocks || cell_count != total_cells) {
    log::throw_with_log<std::invalid_argument>(
        "Kernel::reduce: DenseDistributed 2-D block-atomic window plan produced {} block(s) / "
        "{} cell(s), expected {} / {} for view {}",
        block_count, cell_count, blocks, total_cells, ::carbon::to_string(view));
  }
  return gpu_success;
}

#endif  // CARBON_CUDALIKE_DEVICE_TU

}  // namespace carbon::detail::dense_distributed
