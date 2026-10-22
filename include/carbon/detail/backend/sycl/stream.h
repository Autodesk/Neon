#pragma once

/**
 * SYCL-backend wrapper around a single @c sycl::queue. Selected by the dispatcher in
 * include/carbon/detail/stream.h when CARBON_BACKEND_IS_SYCL.
 *
 * @par Class name
 * The class is still called @c CudaStream for cross-backend symmetry with the CUDA wrapper
 * at @c detail/backend/cudalike/stream.h. The "Cuda" prefix is residual — under the SYCL
 * backend, this owns a @c sycl::queue (which is a SYCL handle that may be backed by a
 * CUDA, HIP, OpenCL, or OpenMP runtime depending on the AdaptiveCpp target). Renaming to
 * a backend-neutral name (e.g. @c NativeStream) is a follow-up cleanup; doing it now would
 * mean touching ~15 detail-layer call sites, which is out of scope for the minimal
 * compile-only milestone of phase 2.3.
 *
 * @par Public surface vs the CUDA wrapper
 * The CUDA wrapper exposes @c get() returning @c cudaStream_t. The SYCL wrapper
 * deliberately does NOT have a @c get() method — it exposes @c queue() returning
 * @c sycl::queue&. The two backends require different consumer code at each call site
 * (e.g. @c cudaMemcpyAsync vs @c sycl::queue::memcpy), so making the accessor name
 * differ keeps fork-points explicit at the call site rather than silently producing
 * different types from the same expression.
 */

#include <carbon/export.h>

#include <sycl/sycl.hpp>

#include <memory>
#include <utility>

namespace carbon {
namespace detail {

class Gpu;

/**
 * Owning view of a single @c sycl::queue; shallow copies share the same underlying
 * queue via @c std::shared_ptr. Created only through @ref Gpu::stream().
 *
 * @c sycl::queue itself is a SYCL handle (refcounted internally), so wrapping it in
 * @c std::shared_ptr is a minor redundancy preserved for symmetry with the CUDA wrapper
 * at @c detail/backend/cudalike/stream.h. Both wrappers thus offer the same shallow-copy
 * semantics and the same "moved-from is empty (queue() UB but @c bool conversion is
 * @c false)" contract.
 */
class CARBON_API CudaStream {
  friend class Gpu;

  CudaStream() = default;
  explicit CudaStream(std::shared_ptr<sycl::queue> q) noexcept
      : queue_(std::move(q)) {}

public:
  CudaStream(const CudaStream&) = default;
  CudaStream&
  operator=(const CudaStream&) = default;
  CudaStream(CudaStream&&) noexcept = default;
  CudaStream&
  operator=(CudaStream&&) noexcept = default;

  /** Underlying @c sycl::queue. UB to call on a default-constructed / moved-from instance. */
  [[nodiscard]] auto
  queue() const noexcept -> sycl::queue& {
    return *queue_;
  }

  [[nodiscard]] explicit
  operator bool() const noexcept {
    return static_cast<bool>(queue_);
  }

private:
  std::shared_ptr<sycl::queue> queue_{};
};

}  // namespace detail
}  // namespace carbon
