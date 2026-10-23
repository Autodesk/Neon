#pragma once

#include <carbon/detail/backend/select.h>
#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cuda_like.h>
#endif
#include <carbon/logging.h>

#include <cstddef>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace carbon {
namespace detail {

template <typename F>
auto
GpuGroup::for_each_device(F&& f) const -> void {
  for (const auto& g : gpus_) {
    g.set_device();
    std::invoke(std::forward<F>(f), g);
  }
}

template <typename F>
auto
GpuGroup::parallel_for_each_device(F&& f) const -> void {
  if (gpus_.empty()) {
    CARBON_LOG_WARN("parallel_for_each_device: no GPUs in this GpuGroup instance; nothing to run");
    return;
  }
  std::exception_ptr first_exception{};
  const std::vector<Gpu>& vec = gpus_;
  const int n = static_cast<int>(vec.size());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    try {
      vec[static_cast<size_t>(i)].set_device();
      std::invoke(f, vec[static_cast<size_t>(i)]);
    } catch (...) {
#pragma omp critical(carbon_parallel_for_each)
      {
        if (!first_exception) {
          first_exception = std::current_exception();
        }
      }
    }
  }
  if (first_exception) {
    CARBON_LOG_ERROR("parallel_for_each_device: rethrowing exception from parallel body");
    std::rethrow_exception(first_exception);
  }
}

template <typename T>
auto
GpuGroup::buffer(const std::vector<size_t>& counts_per_gpu) const -> BufferGroup<T> {
  if (counts_per_gpu.size() != size()) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "GpuGroup::buffer: counts_per_gpu.size() ({}) does not match GpuGroup::size() ({})",
        counts_per_gpu.size(), size());
  }
  auto vec = std::make_shared<std::vector<Allocation<T>>>();
  vec->reserve(counts_per_gpu.size());
  for (size_t i = 0; i < counts_per_gpu.size(); ++i) {
    vec->push_back(gpus_[i].buffer<T>(counts_per_gpu[i], static_cast<int>(i)));
  }
  return BufferGroup<T>(vec);
}

inline auto
GpuGroup::stream() const -> StreamGroup {
  auto vec = std::make_shared<std::vector<CudaStream>>();
  vec->reserve(gpus_.size());
  for (const auto& g : gpus_) {
    vec->push_back(g.stream());
  }
  return StreamGroup(vec);
}

inline auto
GpuGroup::stream(StreamPriority priority) const -> StreamGroup {
  auto vec = std::make_shared<std::vector<CudaStream>>();
  vec->reserve(gpus_.size());
  if (priority == StreamPriority::High && !gpus_.empty()) {
    int least = 0;
    int greatest = 0;
    gpus_[0].set_device();
#if CARBON_BACKEND_IS_CUDALIKE
    (void)detail::gpu_stream_priority_range(&least, &greatest);
#endif
    for (const auto& g : gpus_) {
      vec->push_back(g.stream_with_priority(greatest));
    }
  } else {
    for (const auto& g : gpus_) {
      vec->push_back(g.stream());
    }
  }
  return StreamGroup(vec);
}

inline auto
GpuGroup::event() const -> EventGroup {
  auto vec = std::make_shared<std::vector<Event>>();
  vec->reserve(gpus_.size());
  for (const auto& g : gpus_) {
    vec->push_back(g.event());
  }
  return EventGroup(vec);
}

}  // namespace detail
}  // namespace carbon
