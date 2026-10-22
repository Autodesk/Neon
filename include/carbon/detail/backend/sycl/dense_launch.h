#pragma once

/**
 * SYCL twin of @c detail/backend/cudalike/dense_launch.h. Provides the same
 * @c launch_dense_grid free function name (different signature: takes a
 * @c sycl::queue& instead of a @c cudaStream_t, returns @c void instead of
 * @c cudaError_t — SYCL surfaces failures as exceptions). Internal API; only
 * @c carbon::detail::DeviceKernelImpl reaches into here.
 *
 * @par 3D launch mapping
 * The CUDA @c dense_grid_kernel maps each thread to @c (ix, iy, iz) =
 * @c blockIdx * @c blockDim + @c threadIdx, with x as @c {.x} (fastest dim) and z as
 * @c {.z} (slowest). SYCL @c nd_range / @c nd_item index by dimension @b 0 (slowest
 * dim) through @c d-1 (fastest), so the carbon X axis maps to SYCL dim 2, Y to dim 1,
 * Z to dim 0. The @c LaunchShape coming in (built by @c compute_dense_grid) is in
 * (x, y, z) order regardless of backend, so we mirror the CUDA call's per-axis values
 * into the SYCL @c range<3> with that swap.
 *
 * @par Shared memory
 * The CUDA backend supports a non-zero @p shared_mem dynamic-shared bytes argument
 * on launch. SYCL exposes the same concept through @c sycl::local_accessor, but the
 * existing carbon kernel signature @c fn(span, idx) does not include a local-memory
 * pointer parameter — adding one is a public-API change that belongs in a later phase
 * once a compute kernel actually needs it. For now, @p shared_mem != 0 throws so the
 * mismatch surfaces loudly at the call site instead of silently dropping the request.
 *
 * @par Errors
 * The implementation throws @c std::out_of_range when @p relative_gpu_id is invalid
 * and @c std::runtime_error on any @c sycl::exception escaping the @c parallel_for
 * submission. Callers in @c dense_kernel_impl.h forward the throw upward; carbon's
 * existing test suite asserts on exception messages, not error codes, so this is
 * symmetric with the CUDA path's @c throw_with_log behaviour.
 */

#include <carbon/data_view.h>
#include <carbon/dense.h>
#include <carbon/detail/dense/kernel.h>
#include <carbon/detail/gpu.h>
#include <carbon/launch_shape.h>
#include <carbon/logging.h>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace carbon::detail::dense {

template <typename F>
requires dense_kernel_fn<F> inline auto
launch_dense_grid(const ::carbon::Dense& grid,
                  LaunchShape grid_shape,
                  LaunchShape block_shape,
                  size_t relative_gpu_id,
                  size_t shared_mem,
                  sycl::queue& queue,
                  ::carbon::DataView view,
                  F fn) -> void {
  if (relative_gpu_id >= grid.gpu_count()) {
    ::carbon::log::throw_with_log<std::out_of_range>(
        "launch_dense_grid: relative_gpu_id out of range (relative_gpu_id={}, gpu_count={})",
        relative_gpu_id, grid.gpu_count());
  }
  if (shared_mem != 0U) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "launch_dense_grid: dynamic shared memory ({} bytes) is not yet supported under "
        "the SYCL backend. Phase 5 will plumb a sycl::local_accessor into the per-cell "
        "callable signature.",
        shared_mem);
  }
  if (!grid_shape.is_nonzero() || !block_shape.is_nonzero()) {
    // Empty iteration. Match the CUDA backend's "non-fatal no-op" behaviour: a zero-volume
    // launch is legitimate (e.g. a Boundary view on a slab whose Z extent is fully owned)
    // and the CUDA driver also silently skips it.
    return;
  }

  const Span span = grid.spans()[relative_gpu_id];

  // SYCL nd_range orders dimensions slowest..fastest; carbon's LaunchShape is (x, y, z) in
  // the CUDA convention (x fastest). Swap when constructing the range so dim 2 maps to x,
  // dim 1 to y, dim 0 to z.
  const sycl::range<3> local{static_cast<size_t>(block_shape.z), static_cast<size_t>(block_shape.y),
                             static_cast<size_t>(block_shape.x)};
  const sycl::range<3> global{
      static_cast<size_t>(grid_shape.z) * static_cast<size_t>(block_shape.z),
      static_cast<size_t>(grid_shape.y) * static_cast<size_t>(block_shape.y),
      static_cast<size_t>(grid_shape.x) * static_cast<size_t>(block_shape.x)};

  try {
    queue.parallel_for(sycl::nd_range<3>{global, local}, [=](sycl::nd_item<3> it) {
      const int64_t ix = static_cast<int64_t>(it.get_global_id(2));
      const int64_t iy = static_cast<int64_t>(it.get_global_id(1));
      const int64_t iz = static_cast<int64_t>(it.get_global_id(0));
      const ValidatedIdx r = span.set_and_validate(view, ix, iy, iz);
      if (!r.valid) {
        return;
      }
      fn(span, r.idx);
    });
  } catch (const sycl::exception& ex) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "launch_dense_grid: SYCL parallel_for submission failed (relative_gpu_id={}): {}",
        relative_gpu_id, ex.what());
  }
}

}  // namespace carbon::detail::dense
