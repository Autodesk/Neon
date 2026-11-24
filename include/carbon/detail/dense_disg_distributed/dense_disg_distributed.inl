#pragma once

// Out-of-class template definitions for carbon::DenseDisgDistributed. Pulled in at the bottom
// of <carbon/dense_disg_distributed.h>, mirroring <carbon/dense_disg.h> + dense_disg.inl.

#include <carbon/data_view.h>
#include <carbon/detail/dense_disg_distributed/field.h>

#include <cstddef>
#include <utility>

namespace carbon {

template <typename T,
          int Components>
auto
DenseDisgDistributed::newField(MemorySide side,
                               std::string name) const -> Field<T,
                                                                Components> {
  if (gpu_count() == 0U) {
    log::throw_with_log<std::runtime_error>(
        "DenseDisgDistributed::newField: backing System has no GPUs (gpu_count() == 0)");
  }

  std::vector<size_t> totals_per_gpu(gpu_count());
  for (size_t i = 0; i < gpu_count(); ++i) {
    totals_per_gpu[i] = static_cast<size_t>(spans_[i].linear_storage_elements(Components));
  }

  ::carbon::Buffer<T> gpu_buffers{};
  ::carbon::Buffer<T> cpu_buffers{};
  std::vector<detail::dense_disg_distributed::Partition<T, Components>> partitions;
  std::vector<detail::dense_disg_distributed::Partition<T, Components>> host_partitions;
  partitions.reserve(gpu_count());

  if (side == MemorySide::Device || side == MemorySide::HostDevice) {
    gpu_buffers = system_.buffer<T>(totals_per_gpu);
    for (size_t i = 0; i < gpu_count(); ++i) {
      const auto& sp = spans_[i];
      partitions.emplace_back(gpu_buffers[i].data(), carbon::DataView::Standard, sp.origin(),
                              sp.spacing());
    }
  }

  if (side == MemorySide::Host) {
    auto host_allocs = system_.make_vector([&](size_t i) {
      return ::carbon::Cpu::buffer<T>(totals_per_gpu[i]);
    });
    cpu_buffers = ::carbon::detail::BufferGroup<T>::from_vector(std::move(host_allocs));
    host_partitions.reserve(gpu_count());
    for (size_t i = 0; i < gpu_count(); ++i) {
      const auto& sp = spans_[i];
      host_partitions.emplace_back(cpu_buffers[i].data(), carbon::DataView::Standard, sp.origin(),
                                   sp.spacing());
    }
    partitions.clear();
    partitions.reserve(gpu_count());
    for (size_t i = 0; i < gpu_count(); ++i) {
      const auto& sp = spans_[i];
      partitions.emplace_back(cpu_buffers[i].data(), carbon::DataView::Standard, sp.origin(),
                              sp.spacing());
    }
  }

  return Field<T, Components>(*this, std::move(gpu_buffers), std::move(cpu_buffers),
                              std::move(partitions), std::move(host_partitions), side,
                              std::move(name));
}

}  // namespace carbon
