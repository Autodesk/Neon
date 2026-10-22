#pragma once

/**
 * Halo-exchange concrete impl behind ::carbon::Kernel for the disaggregated dense
 * layout: wraps a @c HaloDisg<T, Components> by value and delegates @c launch to
 * the halo's stream-issued @c operator() . The halo exchange ignores the @ref
 * DataView argument — a halo has no notion of an iteration sub-domain.
 *
 * Mirrors @ref HaloKernelImpl (Dense) and @ref BlockHaloKernelImpl (Block) one-
 * to-one. Header-only; no nvcc gating needed because @c HaloDisg::operator()
 * forwards directly to @c Transfer<T> / @c DistTransfer<T> .
 *
 * Implementation note: the kind / tokens / runs_on_* / launch / launch_on_device /
 * launch_on_host / halo() boilerplate lives in @ref HaloKernelImplBase (P2.β
 * closure of @c docs/audit_2026_05_third_pass.md ); this header is a thin
 * subclass plus the @ref halo_kernel_impl_traits specialisation that pulls
 * @c (field_uid, field_name) out of @c HaloDisg for the dependency token.
 */

#include <carbon/detail/halo_kernel_impl_base.h>
#include <carbon/halo_disg.h>

#include <cstdint>
#include <string>

namespace carbon::detail {

template <typename T, int Components>
struct halo_kernel_impl_traits<::carbon::HaloDisg<T, Components>> {
  static auto
  field_uid(const ::carbon::HaloDisg<T,
                                     Components>& h) noexcept -> std::uint64_t {
    return h.field().gpu_buffers().get_uid();
  }
  static auto
  field_name(const ::carbon::HaloDisg<T,
                                      Components>& h) noexcept -> const std::string& {
    return h.field().name();
  }
};

/**
 * Concrete @ref KernelImpl that issues a DenseDisg @c HaloDisg exchange on the
 * user's stream.
 *
 * @tparam T          Element type of the field being exchanged.
 * @tparam Components Per-cell component count.
 */
template <typename T, int Components>
class HaloDisgKernelImpl final : public HaloKernelImplBase<::carbon::HaloDisg<T, Components>> {
public:
  using HaloKernelImplBase<::carbon::HaloDisg<T, Components>>::HaloKernelImplBase;
};

}  // namespace carbon::detail
