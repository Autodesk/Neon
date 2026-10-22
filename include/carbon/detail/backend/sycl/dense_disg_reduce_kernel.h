#pragma once

/**
 * SYCL twin of @c detail/backend/cudalike/dense_disg_reduce_kernel.h . Drives the
 * grid-wide reduction over a @c DenseDisg slab through @c sycl::reduction over
 * @c sycl::queue::parallel_for .
 */

#include <carbon/data_view.h>
#include <carbon/detail/dense_disg/span.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>

#include <sycl/sycl.hpp>

#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace carbon::detail::dense_disg {

template <typename F, typename T>
concept disg_reduce_fn = std::is_invocable_r_v<T, F, Span, Idx>;

constexpr int kDisgReduceBlockThreads = 256;

template <typename F,
          typename T,
          typename Op>
inline auto
launch_disg_reduce(sycl::queue& queue,
                   Span span,
                   ::carbon::DataView view,
                   F fn,
                   Op op,
                   T init,
                   T* out,
                   int64_t total_cells) -> void {
  if (total_cells <= 0) {
    try {
      queue.fill(out, init, 1U);
    } catch (const sycl::exception& ex) {
      ::carbon::log::throw_with_log<std::runtime_error>(
          "launch_disg_reduce: empty-domain queue.fill failed: {}", ex.what());
    }
    return;
  }

  const ::carbon::int64_3d ext = span.iter_extent(view);

  try {
    queue.submit([&](sycl::handler& cgh) {
      auto reducer =
          sycl::reduction(out, init, op, sycl::property::reduction::initialize_to_identity{});
      cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_cells)), reducer,
                       [=](sycl::id<1> i, auto& red) {
                         const int64_t flat = static_cast<int64_t>(i[0]);
                         const int64_t ix = flat % ext.x;
                         const int64_t rest = flat / ext.x;
                         const int64_t iy = rest % ext.y;
                         const int64_t iz = rest / ext.y;
                         const ValidatedIdx vi = span.set_and_validate(view, ix, iy, iz);
                         if (vi.valid) {
                           red.combine(fn(span, vi.idx));
                         }
                       });
    });
  } catch (const sycl::exception& ex) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "launch_disg_reduce: SYCL parallel_for submission failed: {}", ex.what());
  }
}

}  // namespace carbon::detail::dense_disg
