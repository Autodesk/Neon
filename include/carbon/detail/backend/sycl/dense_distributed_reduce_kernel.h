#pragma once

/**
 * SYCL twin of @c detail/backend/cudalike/dense_distributed_reduce_kernel.h. Drives the grid-wide
 * reduction through @c sycl::reduction over @c sycl::queue::parallel_for, replacing the
 * CUDA backend's two-stage @c cub::BlockReduce + @c cub::DeviceReduce::Reduce pipeline
 * with the single SYCL primitive — the runtime picks the appropriate per-target reduce
 * tree internally (CUB on cuda, OMP on the host fallback, etc.).
 *
 * @par Concept
 * Same @c dense_reduce_fn<F, T> concept as the CUDA backend so user lambdas authored
 * against the public @c DenseDistributed::make_reduce_kernel API satisfy both. The concept itself
 * is host-portable (no CUDA / SYCL types) and lives at file scope.
 *
 * @par Why no per-block scratch
 * CUDA needs a @c partials_ array (one @c T per launched block) plus a CUB temp-storage
 * scratch buffer because the host orchestrates the second pass. SYCL's @c sycl::reduction
 * subsumes both stages: the reduction object names the destination scalar directly and
 * the runtime allocates whatever per-work-group scratch it needs internally. The
 * @c reduce_kernel_impl.h SYCL branch therefore skips both allocations entirely.
 *
 * @par Windows, not DataView templates
 * The view is translated to one or more @ref ViewWindow boxes on the host. Standard and
 * Internal are a single box; Boundary is up to six disjoint boxes. The kernel maps its flat
 * logical index onto those boxes, so no @ref DataView template parameter is instantiated.
 */

#include <carbon/data_view.h>
#include <carbon/detail/backend/sycl/block_atomic_reduce.h>
#include <carbon/detail/dense_distributed/span.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>

#include <sycl/sycl.hpp>

#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace carbon::detail::dense_distributed {

/**
 * Same shape as the CUDA-backend concept: callable as @c (Span, Idx) -> T returned by
 * value. No @c std::is_trivially_copyable_v check, matching the CUDA side's permissive
 * stance toward extended host-device lambda wrappers (clang's SYCL closures fail the
 * standard trait too).
 */
template <typename F, typename T>
concept dense_reduce_fn = std::is_invocable_r_v<T, F, Span, Idx>;

/**
 * Same constant the CUDA backend uses for its block size. Kept here so @c
 * reduce_kernel_impl.h's ctor can publish a @c block_count() value with the same
 * meaning across backends — the SYCL launch itself does NOT honour this in a literal
 * sense (the runtime picks a work-group size of its own choosing inside @c parallel_for
 * with @c sycl::reduction), but exposing the same constant keeps the per-test
 * @c block_count assertions backend-symmetric.
 */
constexpr int kReduceBlockThreads = 256;

/**
 * @brief One @c sycl::reduction over the concatenated logical cells of @p windows.
 *
 * Standard / Internal have one window, so the loop resolves on its first iteration.
 * Boundary has up to six; the flat index is shifted by each preceding window volume.
 */
template <typename F,
          typename T,
          typename Op>
inline auto
launch_dense_reduce_windows(sycl::queue& queue,
                            Span span,
                            ViewWindowSet windows,
                            F fn,
                            Op op,
                            T init,
                            T* out,
                            int64_t total_cells) -> void {
  try {
    queue.submit([&](sycl::handler& cgh) {
      auto reducer =
          sycl::reduction(out, init, op, sycl::property::reduction::initialize_to_identity{});
      cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_cells)), reducer,
                       [=](sycl::id<1> i, auto& red) {
                         const int64_t flat = static_cast<int64_t>(i[0]);
                         int window_index = 0;
                         int64_t local = flat;
                         for (; window_index < windows.count; ++window_index) {
                           const int64_t volume = windows.windows[window_index].volume();
                           if (local < volume) {
                             break;
                           }
                           local -= volume;
                         }
                         if (window_index >= windows.count) {
                           return;
                         }
                         const ViewWindow w = windows.windows[window_index];
                         const int64_t ix = local % w.extent.x;
                         const int64_t rest = local / w.extent.x;
                         const int64_t iy = rest % w.extent.y;
                         const int64_t iz = rest / w.extent.y;
                         red.combine(fn(span, Idx(w.origin.x + ix, w.origin.y + iy,
                                                   w.origin.z + iz)));
                       });
    });
  } catch (const sycl::exception& ex) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "launch_dense_reduce: SYCL parallel_for submission failed: {}", ex.what());
  }
}

/**
 * @brief Launch a fused map+reduce over @c span 's window(s) for @p view and write the final
 *        scalar into @p out .
 *
 * @tparam F   Per-cell device callable returning the per-cell @c T contribution.
 * @tparam T   Reduction value type.
 * @tparam Op  Binary reduction functor; must be commutative-associative and accept
 *             @c op(a, b) -> T .
 *
 * @param queue        SYCL queue to enqueue on (already pinned to the right device).
 * @param span         Slab geometry.
 * @param view         Iteration view, translated on the host via @c span.view_windows(view).
 * @param fn           Per-cell device callable.
 * @param op           Reduction operator.
 * @param init         Identity element for @p op .
 * @param out          Device pointer to a single @c T receiving the final result. The
 *                     value at @p out before the launch is ignored —
 *                     @c sycl::property::reduction::initialize_to_identity ensures the
 *                     reduction object writes @p init in-place before combining.
 * @param total_cells  Owned-cell count for @p view (sum of the window volumes).
 *
 * @throws std::runtime_error on any @c sycl::exception escaping the @c parallel_for
 *         submission (signature symmetric with @c launch_dense_grid).
 */
template <typename F,
          typename T,
          typename Op>
inline auto
launch_dense_reduce(sycl::queue& queue,
                    Span span,
                    ::carbon::DataView view,
                    F fn,
                    Op op,
                    T init,
                    T* out,
                    int64_t total_cells) -> void {
  const ViewWindowSet windows = span.view_windows(view);
  if (windows.total_volume() != total_cells) {
    ::carbon::log::throw_with_log<std::invalid_argument>(
        "launch_dense_reduce: window plan covers {} cell(s), expected {}", windows.total_volume(),
        total_cells);
  }
  if (total_cells <= 0) {
    // Empty domain: write the identity directly to the output and return. Matches the
    // CUDA backend's "degenerate cub::DeviceReduce::Reduce with num_items=0" branch.
    try {
      queue.fill(out, init, 1U);
    } catch (const sycl::exception& ex) {
      ::carbon::log::throw_with_log<std::runtime_error>(
          "launch_dense_reduce: empty-domain queue.fill failed: {}", ex.what());
    }
    return;
  }

  launch_dense_reduce_windows(queue, span, windows, fn, op, init, out, total_cells);
}

/**
 * @brief Block-atomic Sum stage 1 over @p view 's window(s).
 *
 * The output is zeroed once; each work-group reduces its cells and adds one atomic. The flat
 * index is mapped across the concatenated window volumes exactly as in @ref launch_dense_reduce.
 */
template <typename F,
          typename T>
inline auto
launch_dense_reduce_block_atomic(sycl::queue& queue,
                                 Span span,
                                 ::carbon::DataView view,
                                 F fn,
                                 T* out,
                                 int64_t total_cells) -> void {
  ViewWindowSet windows = span.view_windows(view);
  if (windows.total_volume() != total_cells) {
    ::carbon::log::throw_with_log<std::invalid_argument>(
        "launch_dense_reduce_block_atomic: window plan covers {} cell(s), expected {}",
        windows.total_volume(), total_cells);
  }
  try {
    queue.memset(out, 0, sizeof(T));
    if (total_cells <= 0) {
      return;
    }
    const size_t total = static_cast<size_t>(total_cells);
    constexpr size_t group_size = ::carbon::detail::sycl_reduce::kBlockAtomicGroupSize;
    const size_t global = (total + group_size - 1U) / group_size * group_size;
    queue.parallel_for(sycl::nd_range<1>(global, group_size), [=](sycl::nd_item<1> it) {
      const size_t flat_u = it.get_global_id(0);
      T value{};
      if (flat_u < total) {
        int window_index = 0;
        int64_t local = static_cast<int64_t>(flat_u);
        for (; window_index < windows.count; ++window_index) {
          const int64_t volume = windows.windows[window_index].volume();
          if (local < volume) {
            break;
          }
          local -= volume;
        }
        if (window_index < windows.count) {
          const ViewWindow w = windows.windows[window_index];
          const int64_t ix = local % w.extent.x;
          const int64_t rest = local / w.extent.x;
          const int64_t iy = rest % w.extent.y;
          const int64_t iz = rest / w.extent.y;
          value = fn(span, Idx(w.origin.x + ix, w.origin.y + iy, w.origin.z + iz));
        }
      }
      const T group_sum = sycl::reduce_over_group(it.get_group(), value, sycl::plus<T>());
      if (it.get_local_id(0) == 0U) {
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            acc(*out);
        acc.fetch_add(group_sum);
      }
    });
  } catch (const sycl::exception& ex) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "launch_dense_reduce_block_atomic: SYCL submission failed: {}", ex.what());
  }
}

}  // namespace carbon::detail::dense_distributed
