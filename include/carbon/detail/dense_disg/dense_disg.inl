#pragma once

// Out-of-class template definitions for carbon::DenseDisg. Pulled in at the bottom of
// <carbon/dense_disg.h> so user code that just needs the grid type sees the inline
// definitions without an extra include, mirroring <carbon/dense.h> + dense.inl.

#include <carbon/data_view.h>
#include <carbon/detail/dense_disg/field.h>

#include <cstddef>
#include <utility>

namespace carbon {

template <typename T,
          int Components>
auto
DenseDisg::newField(MemorySide side,
                    std::string name) const -> Field<T,
                                                     Components> {
  if (gpu_count() == 0U) {
    log::throw_with_log<std::runtime_error>(
        "DenseDisg::newField: backing System has no GPUs (gpu_count() == 0)");
  }

  // Per-slab total elements: (Lz + 2) * C * xy_pitch.
  std::vector<size_t> totals_per_gpu(gpu_count());
  for (size_t i = 0; i < gpu_count(); ++i) {
    totals_per_gpu[i] = static_cast<size_t>(spans_[i].linear_storage_elements(Components));
  }

  ::carbon::Buffer<T> gpu_buffers{};
  ::carbon::Buffer<T> cpu_buffers{};
  std::vector<detail::dense_disg::Partition<T, Components>> partitions;
  std::vector<detail::dense_disg::Partition<T, Components>> host_partitions;
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
    // For Host-only fields, the device partitions are unused; bind partitions to host
    // pointers so user code can still pass the field through host-only APIs without a
    // separate accessor.
    partitions.clear();
    partitions.reserve(gpu_count());
    for (size_t i = 0; i < gpu_count(); ++i) {
      const auto& sp = spans_[i];
      partitions.emplace_back(cpu_buffers[i].data(), carbon::DataView::Standard, sp.origin(),
                              sp.spacing());
    }
  }

  // HostDevice: the host mirror is left lazy (allocated on first cpu_buffers / host_partitions
  // access via Field::ensure_host_storage_). Pass empty cpu_buffers / host_partitions through.
  return Field<T, Components>(*this, std::move(gpu_buffers), std::move(cpu_buffers),
                              std::move(partitions), std::move(host_partitions), side,
                              std::move(name));
}

}  // namespace carbon
