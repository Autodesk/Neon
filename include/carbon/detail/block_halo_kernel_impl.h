#pragma once

/**
 * Halo-exchange concrete impl behind @c ::carbon::Kernel for the block-sparse
 * @c carbon::Block grid family: wraps a @c HaloBlock<T, Components, SBlock> by
 * value and delegates @c launch to its stream-issued @c operator() . The halo
 * ignores the @ref DataView argument (a halo has no notion of an iteration
 * sub-domain — the same brick pairs are exchanged regardless of which compute
 * view the surrounding pipeline is running, exactly like the dense
 * @c HaloKernelImpl ).
 *
 * Parallel to @c carbon::detail::HaloKernelImpl<T, Components> (which wraps
 * @c Halo<T, Components> for @c Dense fields) and @c HaloDisgKernelImpl (which
 * wraps @c HaloDisg<T, Components> for @c DenseDisg ). Both impls expose the
 * same @c KernelKind::Halo / token / launch surface, so @c carbon::Skeleton +
 * @c carbon::OverlappedStencil consume them uniformly.
 *
 * Header-only; no nvcc gating needed because @c HaloBlock::operator() forwards
 * directly to @c Transfer<T> / @c DistTransfer<T> which are themselves host
 * code issuing @c cudaMemcpyAsync / NCCL ops on the user's stream.
 *
 * Implementation note: the kind / tokens / runs_on_* / launch / launch_on_device /
 * launch_on_host / halo() boilerplate lives in @ref HaloKernelImplBase (P2.β
 * closure of @c docs/audit_2026_05_third_pass.md ); this header is a thin
 * subclass plus the @ref halo_kernel_impl_traits specialisation that pulls
 * @c (field_uid, field_name) out of @c HaloBlock (which exposes those
 * accessors directly because the block halo doesn't keep the owning @c Field
 * around).
 */

#include <carbon/detail/halo_kernel_impl_base.h>
#include <carbon/halo_block.h>

#include <cstdint>
#include <string>

namespace carbon::detail {

template <typename T, int Components, typename SBlock>
struct halo_kernel_impl_traits<::carbon::HaloBlock<T, Components, SBlock>> {
  static auto
  field_uid(const ::carbon::HaloBlock<T,
                                      Components,
                                      SBlock>& h) noexcept -> std::uint64_t {
    return h.field_uid();
  }
  static auto
  field_name(const ::carbon::HaloBlock<T,
                                       Components,
                                       SBlock>& h) noexcept -> const std::string& {
    return h.field_name();
  }
};

/**
 * Concrete @ref KernelImpl that issues a @c HaloBlock exchange on the user's
 * stream. Pairs with @c Block::Field 's storage layout (SoA per-block, ghost
 * slots at @c [n_owned, n_owned + n_ghost) ).
 *
 * @tparam T          Element type of the field being exchanged.
 * @tparam Components Per-cell component count.
 * @tparam SBlock     @c StaticBlock policy of the underlying @c Block .
 */
template <typename T, int Components, typename SBlock>
class BlockHaloKernelImpl final
    : public HaloKernelImplBase<::carbon::HaloBlock<T, Components, SBlock>> {
public:
  using HaloKernelImplBase<::carbon::HaloBlock<T, Components, SBlock>>::HaloKernelImplBase;
};

}  // namespace carbon::detail
