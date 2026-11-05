#pragma once

/**
 * @file halo_distributed_kernel_impl.h
 * @brief @c Kernel adapter for @ref HaloDistributed.
 *
 * Identical in shape to @c HaloKernelImpl (the Z-slab @ref Halo adapter): all launch/ordering
 * logic lives in @c HaloKernelImplBase, and only the two field-identity accessors differ per halo
 * type.
 */

#include <carbon/detail/halo_kernel_impl_base.h>
#include <carbon/halo_distributed.h>

#include <cstdint>
#include <string>

namespace carbon::detail {

template <typename T, int Components>
struct halo_kernel_impl_traits<::carbon::HaloDistributed<T, Components>> {
  static auto
  field_uid(const ::carbon::HaloDistributed<T,
                                            Components>& h) noexcept -> std::uint64_t {
    return h.field().gpu_buffers().get_uid();
  }
  static auto
  field_name(const ::carbon::HaloDistributed<T,
                                             Components>& h) noexcept -> const std::string& {
    return h.field().name();
  }
};

template <typename T, int Components>
class HaloDistributedKernelImpl final
    : public HaloKernelImplBase<::carbon::HaloDistributed<T, Components>> {
public:
  using HaloKernelImplBase<::carbon::HaloDistributed<T, Components>>::HaloKernelImplBase;
};

}  // namespace carbon::detail
