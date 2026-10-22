#pragma once

// CUDA-backend wrapper around a single gpu_stream_t. Selected by the dispatcher in
// include/carbon/detail/stream.h when CARBON_BACKEND_IS_CUDALIKE. The SYCL backend will
// provide its own equivalent at include/carbon/detail/backend/sycl/stream.h (phase 2.3).

#include <carbon/export.h>

#include <carbon/detail/backend/cuda_like.h>

#include <memory>

namespace carbon {
namespace detail {

class Gpu;

/**
 * Owning view of a single CUDA stream; shallow copies share the same underlying
 * @c gpu_stream_t via @c std::shared_ptr. Created only through @ref Gpu::stream().
 *
 * @par Naming
 * Distinct from the public @c carbon::Stream (which is an alias for @ref StreamGroup,
 * carbon's singular handle over a per-GPU collection of these). Inside @c carbon::detail
 * unqualified @c Stream would shadow @c carbon::Stream, so this type uses the
 * @c CudaStream name to keep both meanings unambiguous; user-facing code should never
 * touch this type directly — go through @c Stream / @ref StreamGroup instead.
 */
class CARBON_API CudaStream {
  friend class Gpu;

  CudaStream() = default;
  explicit CudaStream(std::shared_ptr<void> inner) noexcept;

public:
  CudaStream(const CudaStream&) = default;
  CudaStream&
  operator=(const CudaStream&) = default;
  CudaStream(CudaStream&&) noexcept = default;
  CudaStream&
  operator=(CudaStream&&) noexcept = default;

  /** Underlying stream handle, or @c nullptr if empty. */
  [[nodiscard]] auto
  get() const noexcept -> gpu_stream_t {
    return static_cast<gpu_stream_t>(inner_.get());
  }

  [[nodiscard]] explicit
  operator bool() const noexcept {
    return static_cast<bool>(inner_);
  }

private:
  std::shared_ptr<void> inner_{};
};

inline CudaStream::CudaStream(std::shared_ptr<void> inner) noexcept
    : inner_(std::move(inner)) {}

}  // namespace detail
}  // namespace carbon
