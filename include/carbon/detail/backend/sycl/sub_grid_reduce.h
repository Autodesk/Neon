#pragma once

/**
 * SYCL reduction over the lanes of a @c carbon::SubGrid @ref DataView : a single-pass
 * @c sycl::reduction (twin of @c launch_dense_reduce ) and the @c ReduceFinalize::BlockAtomic
 * variant (twin of @c launch_dense_block_atomic_sum ). @c ReduceLaunchTraits<SubGrid> is the
 * only caller.
 */

#include <carbon/data_view.h>
#include <carbon/logging.h>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace carbon::detail::sub_grid {

template <typename IdxT,
          typename SpanT,
          typename F,
          typename T,
          typename Op>
inline auto
launch_lane_reduce(sycl::queue& queue,
                   SpanT span,
                   ::carbon::DataView view,
                   F fn,
                   Op op,
                   T init,
                   T* out) -> void {
  const int64_t begin = span.sub().lane_offset(view);
  const int64_t count = span.sub().lane_count(view);
  try {
    if (count <= 0) {
      queue.fill(out, init, 1U);
      return;
    }
    queue.submit([&](sycl::handler& cgh) {
      auto reducer =
          sycl::reduction(out, init, op, sycl::property::reduction::initialize_to_identity{});
      cgh.parallel_for(sycl::range<1>(static_cast<size_t>(count)), reducer,
                       [=](sycl::id<1> k, auto& red) {
                         const int64_t lane = begin + static_cast<int64_t>(k[0]);
                         red.combine(fn(span, IdxT(span.sub().parent_idx(lane), lane)));
                       });
    });
  } catch (const sycl::exception& ex) {
    ::carbon::log::throw_with_log<std::runtime_error>("SubGrid reduce: SYCL submission failed: {}",
                                                      ex.what());
  }
}

/** Work-group size of the block-atomic launch (same as the CUDA stage 1). */
inline constexpr size_t kBlockAtomicGroupSize = 256;

template <typename IdxT,
          typename SpanT,
          typename F,
          typename T>
inline auto
launch_lane_block_atomic_sum(sycl::queue& queue,
                             SpanT span,
                             ::carbon::DataView view,
                             F fn,
                             T* out) -> void {
  const int64_t begin = span.sub().lane_offset(view);
  const int64_t count = span.sub().lane_count(view);
  try {
    queue.memset(out, 0, sizeof(T));
    if (count <= 0) {
      return;
    }
    const size_t total = static_cast<size_t>(count);
    const size_t global =
        (total + kBlockAtomicGroupSize - 1U) / kBlockAtomicGroupSize * kBlockAtomicGroupSize;
    queue.parallel_for(sycl::nd_range<1>(global, kBlockAtomicGroupSize), [=](sycl::nd_item<1> it) {
      const size_t k = it.get_global_id(0);
      T v{};
      if (k < total) {
        const int64_t lane = begin + static_cast<int64_t>(k);
        v = fn(span, IdxT(span.sub().parent_idx(lane), lane));
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
        "SubGrid block-atomic reduce: SYCL submission failed: {}", ex.what());
  }
}

}  // namespace carbon::detail::sub_grid
