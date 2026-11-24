#pragma once

/**
 * Halo-exchange concrete impl behind ::carbon::Kernel for @c DenseDisgDistributed : wraps a
 * @c HaloDisgDistributed<T, Components> by value and delegates @c launch to the halo's
 * stream-issued @c operator() . Twin of @ref HaloDisgKernelImpl ; the shared boilerplate lives
 * in @ref HaloKernelImplBase .
 */

#include <carbon/detail/halo_kernel_impl_base.h>
#include <carbon/halo_disg_distributed.h>

#include <cstdint>
#include <string>

namespace carbon::detail {

template <typename T, int Components>
struct halo_kernel_impl_traits<::carbon::HaloDisgDistributed<T, Components>> {
  static auto
  field_uid(const ::carbon::HaloDisgDistributed<T,
                                                Components>& h) noexcept -> std::uint64_t {
    return h.field().gpu_buffers().get_uid();
  }
  static auto
  field_name(const ::carbon::HaloDisgDistributed<T,
                                                 Components>& h) noexcept -> const std::string& {
    return h.field().name();
  }
};

/**
 * Concrete @ref KernelImpl that issues a @c HaloDisgDistributed exchange on the user's stream.
 *
 * @tparam T          Element type of the field being exchanged.
 * @tparam Components Per-cell component count.
 */
template <typename T, int Components>
class HaloDisgDistributedKernelImpl final
    : public HaloKernelImplBase<::carbon::HaloDisgDistributed<T, Components>> {
public:
  using HaloKernelImplBase<::carbon::HaloDisgDistributed<T, Components>>::HaloKernelImplBase;
};

}  // namespace carbon::detail
