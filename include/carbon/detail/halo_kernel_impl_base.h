#pragma once

/**
 * Cross-family @c KernelImpl base for the three halo kernel impls
 * (@c HaloKernelImpl for @c Dense , @c HaloDisgKernelImpl for @c DenseDisg ,
 * @c BlockHaloKernelImpl for @c Block ). Closes P2.β of
 * @c docs/audit_2026_05_third_pass.md : the three concrete impls used to be
 * ~85% identical (kind / tokens / runs_on_* / launch / launch_on_device /
 * launch_on_host / halo() boilerplate), differing only in the wrapped halo
 * type and in how to extract the @c (field_uid, field_name) pair from that
 * halo for @c AccessToken construction. This header captures that shape once;
 * each concrete impl is now a thin two-line subclass that pulls in the base's
 * inherited ctors and supplies a @ref halo_kernel_impl_traits specialisation.
 *
 * The trait shape:
 *
 * @code
 * template <>
 * struct halo_kernel_impl_traits<MyHaloT> {
 *   static auto field_uid(const MyHaloT&) noexcept -> uint64_t;
 *   static auto field_name(const MyHaloT&) noexcept -> const std::string&;
 * };
 * @endcode
 *
 * is the only family-specific surface. @c Dense / @c DenseDisg pull the pair
 * via @c halo.field().gpu_buffers().get_uid() / @c halo.field().name() ;
 * @c Block exposes @c HaloBlock::field_uid() / @c HaloBlock::field_name()
 * directly because the block halo doesn't keep the owning @c Field around.
 *
 * Header-only. No nvcc gating needed because every halo type's
 * @c operator() forwards directly to @c Transfer<T> / @c DistTransfer<T> ,
 * which are themselves host code (issuing @c cudaMemcpyAsync / NCCL ops on
 * the user's stream).
 */

#include <carbon/access_token.h>
#include <carbon/data_view.h>
#include <carbon/detail/kernel_impl.h>
#include <carbon/system.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace carbon::detail {

/**
 * Trait that each halo-impl specialises to extract the
 * @c (field_uid, field_name) pair from the wrapped halo type. Required by
 * @ref HaloKernelImplBase 's ctor for the @c AccessToken it pushes into
 * @c tokens_ . Default is intentionally undefined — instantiating
 * @ref HaloKernelImplBase over a halo type without a specialisation fails
 * to compile with a clean diagnostic at the trait reference, not deep inside
 * @c AccessToken .
 *
 * Required surface:
 *
 * - @c static auto field_uid(const HaloT&) noexcept -> uint64_t ;
 * - @c static auto field_name(const HaloT&) noexcept -> const std::string& .
 */
template <typename HaloT>
struct halo_kernel_impl_traits;

/**
 * Concrete @ref KernelImpl that issues a halo exchange on the user's stream.
 * Parameterised on the wrapped halo type @p HaloT (e.g. @c carbon::Halo<T, C>
 * for Dense, @c carbon::HaloDisg<T, C> for DenseDisg, @c carbon::HaloBlock<T,
 * C, SBlock> for Block).
 *
 * Required surface on @p HaloT :
 *
 * - @c HaloT() default ctor (the impl's halo_ default-init path);
 * - @c HaloT(HaloT&&) move ctor (kernel.h's factory hands the halo by value);
 * - @c operator()(const Stream&) — device-side launch entry point, issues
 *   the halo on the user's stream;
 * - @c launch_on_host(const Stream&) — host-side launch entry point, replays
 *   the halo via @c std::memcpy between adjacent host mirrors;
 * - a matching @ref halo_kernel_impl_traits<HaloT> specialisation (see above).
 *
 * The base owns the conservative single-token @c AccessToken push (one
 * @c ReadWrite + Map token on the field's @c BufferGroup is enough for the
 * @c Skeleton 's pairwise dependency analysis — the @c Stencil pattern lives
 * on the kernel that @b reads the halo, not on the halo itself), the
 * polymorphic @c launch -> @c launch_on_device default routing, and every
 * other piece of boilerplate that used to be duplicated three ways.
 */
template <typename HaloT>
class HaloKernelImplBase : public KernelImpl {
public:
  HaloKernelImplBase(HaloT halo,
                     std::string name)
      : KernelImpl(std::move(name)),
        halo_(std::move(halo)) {
    using Traits = halo_kernel_impl_traits<HaloT>;
    // Halo writes the field's halo / ghost cells from the adjacent slab's
    // owned cells of the same field — strictly speaking a Read on the source
    // and a Write on the destination, but they're the same BufferGroup (a
    // halo only ever touches one field). One conservative ReadWrite + Map
    // token suffices for dependency analysis: any kernel that touches this
    // field will be sequenced behind the halo, which is the contract halos
    // serve in a stencil pipeline.
    tokens_.push_back(AccessToken{Traits::field_uid(halo_), ResourceKind::Field,
                                  AccessMode::ReadWrite, AccessPattern::Map,
                                  Traits::field_name(halo_),
                                  /*halo_factory=*/nullptr});
  }

  [[nodiscard]] auto
  kind() const noexcept -> KernelKind override {
    return KernelKind::Halo;
  }

  [[nodiscard]] auto
  tokens() const noexcept -> const std::vector<AccessToken>& override {
    return tokens_;
  }

  [[nodiscard]] auto
  runs_on_device() const noexcept -> bool override {
    return true;
  }

  [[nodiscard]] auto
  runs_on_host() const noexcept -> bool override {
    return true;
  }

  /**
   * @brief Issue the halo exchange. @p view is ignored — a halo update has no
   *        iteration sub-domain; the same set of slab / brick pairs is touched
   *        regardless of the surrounding compute view.
   *
   * Default polymorphic launch routes through the device path (matching every
   * other device-natural kernel kind); switch to @ref launch_on_host
   * explicitly when the surrounding pipeline runs on the CPU mirror.
   */
  auto
  launch(const ::carbon::Stream& stream,
         DataView view) -> void override {
    this->launch_on_device(stream, view);
  }

  auto
  launch_on_device(const ::carbon::Stream& stream,
                   DataView /*view*/) -> void override {
    halo_(stream);
  }

  /**
   * @brief CPU-side halo exchange — @c std::memcpy between adjacent slabs' /
   *        bricks' host mirrors. See the wrapped halo's @c launch_on_host
   *        override for the family-specific contract; every family throws
   *        with a pointed diagnostic when the field has no host mirror or
   *        when the layout has cross-rank entries.
   */
  auto
  launch_on_host(const ::carbon::Stream& stream,
                 DataView /*view*/) -> void override {
    halo_.launch_on_host(stream);
  }

  /** The wrapped halo descriptor. */
  [[nodiscard]] auto
  halo() const noexcept -> const HaloT& {
    return halo_;
  }

private:
  HaloT halo_{};
  std::vector<AccessToken> tokens_{};
};

}  // namespace carbon::detail
