#pragma once

/**
 * SYCL twin of the CUDA backend's @c ReduceFinalize::BlockAtomic stage 1 for row-major
 * dense spans (@c dense::Span and @c dense_distributed::Span ): each work-group sums its
 * cells with @c sycl::reduce_over_group and adds the group sum into the per-GPU scalar with
 * one device-scope atomic. No partials buffer and no second pass. Only @c reduce::Sum is
 * allowed for BlockAtomic, so the group algorithm uses @c sycl::plus (custom operators are
 * not portable group-algorithm operators). Floating-point sums are not bitwise-deterministic
 * across launches (atomic order), as on CUDA.
 */

#include <carbon/data_view.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace carbon::detail::sycl_reduce {

/** Work-group size of the block-atomic launch (same default as the CUDA path). */
constexpr size_t kBlockAtomicGroupSize = 256;

/**
 * @brief Zero @p out on @p queue , then add @c fn(span, idx) over every owned cell of
 *        @p view into it.
 *
 * @param out          Device pointer to one @c T on @p queue 's device.
 * @param total_cells  @c span.iter_extent(view).volume() .
 */
template <typename SpanT,
          typename F,
          typename T>
inline auto
launch_dense_block_atomic_sum(sycl::queue& queue,
                              SpanT span,
                              ::carbon::DataView view,
                              F fn,
                              T* out,
                              int64_t total_cells) -> void {
  try {
    queue.memset(out, 0, sizeof(T));
    if (total_cells <= 0) {
      return;
    }
    const ::carbon::int64_3d ext = span.iter_extent(view);
    const size_t total = static_cast<size_t>(total_cells);
    const size_t global =
        (total + kBlockAtomicGroupSize - 1U) / kBlockAtomicGroupSize * kBlockAtomicGroupSize;
    queue.parallel_for(sycl::nd_range<1>(global, kBlockAtomicGroupSize), [=](sycl::nd_item<1> it) {
      const size_t flat_u = it.get_global_id(0);
      T v{};
      if (flat_u < total) {
        const int64_t flat = static_cast<int64_t>(flat_u);
        const int64_t ix = flat % ext.x;
        const int64_t rest = flat / ext.x;
        const int64_t iy = rest % ext.y;
        const int64_t iz = rest / ext.y;
        const auto vi = span.set_and_validate(view, ix, iy, iz);
        if (vi.valid) {
          v = fn(span, vi.idx);
        }
      }
      const T group_sum = sycl::reduce_over_group(it.get_group(), v, sycl::plus<T>());
      if (it.get_local_id(0) == 0U) {
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            acc(*out);
        acc.fetch_add(group_sum);
      }
    });
  } catch (const sycl::exception& ex) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "launch_dense_block_atomic_sum: SYCL submission failed: {}", ex.what());
  }
}

}  // namespace carbon::detail::sycl_reduce
