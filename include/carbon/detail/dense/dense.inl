#pragma once

#include <carbon/cpu.h>
#include <carbon/detail/dense/field.h>  // brings in carbon::Buffer via system.h

#include <carbon/logging.h>
#include <carbon/memory_side.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace carbon {

template <typename T,
          int Components>
inline auto
Dense::newField(MemorySide side,
                std::string name) const -> Field<T,
                                                 Components> {
  static_assert(Components >= 1,
                "Dense::newField requires Components >= 1 (1 = scalar, 3 = vector, ...). "
                "The Components value sizes the per-cell SoA storage; Components == 0 would "
                "allocate zero bytes per slab.");
  const size_t n = gpu_count();
  if (n == 0U) {
    log::throw_with_log<std::runtime_error>(
        "Dense::newField: gpu_count is zero; cannot allocate field buffers");
  }
  // spans_[i] already carries the haloed storage layout (built in Dense's ctor from the grid's
  // stencil radius), so per-component sizes and partition z-offsets come straight from it. The
  // total buffer size is Components * linear_storage_elements() because Partition::linear_offset
  // uses `component_id * linear_storage_elements() + ...` (SoA): a vector field with Components=3
  // needs three component planes back-to-back per slab.
  const auto counts_per_gpu = system_.make_vector([&](size_t i) {
    const size_t per_component = static_cast<size_t>(spans_[i].linear_storage_elements());
    return per_component * static_cast<size_t>(Components);
  });

  Buffer<T> gpu_buffers{};
  if (has_device(side)) {
    gpu_buffers = system_.gpu_group().buffer<T>(counts_per_gpu);
  }

  // Host mirror is allocated eagerly only for @c MemorySide::Host . For
  // @c MemorySide::HostDevice the mirror is allocated lazily on first access (see
  // @c Field::ensure_host_storage_ ) — the default-empty @c Buffer<T> is passed
  // through to the field's @c HostStorage where the once_flag faults it in on
  // demand. For @c MemorySide::Device the mirror stays default-empty for the field's
  // lifetime.
  Buffer<T> cpu_buffers{};
  if (side == MemorySide::Host) {
    auto host_allocs = system_.make_vector([&](size_t i) {
      return Cpu::buffer<T>(counts_per_gpu[i]);
    });
    cpu_buffers = detail::BufferGroup<T>::from_vector(std::move(host_allocs));
  }

  // Device-bound partitions (drive Field::partitions and the device kernel launch path).
  // Bind to the device buffer when present; for Host-only fields fall back to the host
  // buffer so existing single-side device callers still see a non-null pointer (the
  // launch will fail at the first device read of a host pointer, but that mismatch is
  // covered by Field's MemorySide invariants and we do not change them here).
  auto parts = system_.make_vector([&](size_t i) {
    T* const ptr = has_device(side) ? gpu_buffers[i].data() : cpu_buffers[i].data();
    const auto& sp = spans_[i];
    return detail::dense::Partition<T, Components>(ptr, carbon::DataView::Standard, sp.origin(),
                                                   sp.spacing());
  });
  // Host-bound partitions (drive Field::host_partitions and the OpenMP host-kernel
  // launch path). Populated only for @c MemorySide::Host (eager); for
  // @c MemorySide::HostDevice they are built alongside the lazy @c cpu_buffers in
  // @c Field::ensure_host_storage_ . Host kernels run against an unallocated mirror
  // throw a clean @c std::invalid_argument at launch time with the offending side.
  std::vector<detail::dense::Partition<T, Components>> host_parts;
  if (side == MemorySide::Host) {
    host_parts = system_.make_vector([&](size_t i) {
      const auto& sp = spans_[i];
      return detail::dense::Partition<T, Components>(
          cpu_buffers[i].data(), carbon::DataView::Standard, sp.origin(), sp.spacing());
    });
  }
  return detail::dense::Field<T, Components>(*this, std::move(gpu_buffers), std::move(cpu_buffers),
                                             std::move(parts), std::move(host_parts), side,
                                             std::move(name));
}

}  // namespace carbon
