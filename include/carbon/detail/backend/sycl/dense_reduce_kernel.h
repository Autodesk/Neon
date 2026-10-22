#pragma once

/**
 * SYCL twin of @c detail/backend/cudalike/dense_reduce_kernel.h. Drives the grid-wide
 * reduction through @c sycl::reduction over @c sycl::queue::parallel_for, replacing the
 * CUDA backend's two-stage @c cub::BlockReduce + @c cub::DeviceReduce::Reduce pipeline
 * with the single SYCL primitive — the runtime picks the appropriate per-target reduce
 * tree internally (CUB on cuda, OMP on the host fallback, etc.).
 *
 * @par Concept
 * Same @c dense_reduce_fn<F, T> concept as the CUDA backend so user lambdas authored
 * against the public @c Dense::make_reduce_kernel API satisfy both. The concept itself
 * is host-portable (no CUDA / SYCL types) and lives at file scope.
 *
 * @par Why no per-block scratch
 * CUDA needs a @c partials_ array (one @c T per launched block) plus a CUB temp-storage
 * scratch buffer because the host orchestrates the second pass. SYCL's @c sycl::reduction
 * subsumes both stages: the reduction object names the destination scalar directly and
 * the runtime allocates whatever per-work-group scratch it needs internally. The
 * @c reduce_kernel_impl.h SYCL branch therefore skips both allocations entirely.
 */

#include <carbon/data_view.h>
#include <carbon/detail/dense/span.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>

#include <sycl/sycl.hpp>

#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace carbon::detail::dense {

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
 * @brief Launch a fused map+reduce over @c span 's iteration extent for @p view and
 *        write the final scalar into @p out .
 *
 * @tparam F   Per-cell device callable returning the per-cell @c T contribution.
 * @tparam T   Reduction value type.
 * @tparam Op  Binary reduction functor; must be commutative-associative and accept
 *             @c op(a, b) -> T .
 *
 * @param queue        SYCL queue to enqueue on (already pinned to the right device).
 * @param span         Slab geometry.
 * @param view         Iteration view; @c iter_extent and @c set_and_validate are both
 *                     parameterised on this so Internal / Boundary launches index only
 *                     their owned z-band.
 * @param fn           Per-cell device callable.
 * @param op           Reduction operator.
 * @param init         Identity element for @p op .
 * @param out          Device pointer to a single @c T receiving the final result. The
 *                     value at @p out before the launch is ignored —
 *                     @c sycl::property::reduction::initialize_to_identity ensures the
 *                     reduction object writes @p init in-place before combining.
 * @param total_cells  Owned-cell count for @p view
 *                     (== @c span.iter_extent(view).volume() ).
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

  const ::carbon::int64_3d ext = span.iter_extent(view);

  try {
    queue.submit([&](sycl::handler& cgh) {
      auto reducer =
          sycl::reduction(out, init, op, sycl::property::reduction::initialize_to_identity{});
      cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_cells)), reducer,
                       [=](sycl::id<1> i, auto& red) {
                         const int64_t flat = static_cast<int64_t>(i[0]);
                         // Row-major decoding (z slowest, x fastest), same as the CUDA
                         // dense_grid_reduce_kernel — keeps the per-cell traversal order
                         // identical so any non-associative-op corner case lands on the
                         // same accumulation tree shape on both backends (modulo the
                         // runtime's own internal reduction tree, which is associative
                         // by contract).
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
        "launch_dense_reduce: SYCL parallel_for submission failed: {}", ex.what());
  }
}

}  // namespace carbon::detail::dense
