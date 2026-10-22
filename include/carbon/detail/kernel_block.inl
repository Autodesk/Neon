// kernel_block.inl — Block + Block::Field convenience members. Split out of
// <carbon/kernel.h>. See the matching header at the top of kernel_dense.inl
// for the include-context contract — this file is a tail include of
// <carbon/kernel.h>, not a standalone header.

// ----- Block::Field convenience member -----
//
// Builds a real HaloBlock-backed @c Kernel when the field has a back-reference
// to its owning @c Block<SBlock> (set automatically by @c Block::newField ;
// see @c detail::block::Field::set_block for the lifetime contract). Falls
// back to a no-op anchor when the back-reference is null — that path keeps
// default-constructed / detached fields well-formed (they have nothing to
// halo-exchange anyway) and is also what the @c KernelLoader::read_stencil
// auto-halo path sees on a single-GPU non-distributed @c Block (where the
// halo plan is empty and the anchor + a real @c HaloBlock would be
// behaviourally indistinguishable). The non-anchor path routes through
// @c Kernel::halo_block — same shape as @c Dense::Field::make_halo_kernel
// above.
template <typename T,
          int Components,
          typename SBlock>
inline auto
detail::block::Field<T,
                     Components,
                     SBlock>::make_halo_kernel(std::string halo_name) const -> ::carbon::Kernel {
  std::string resolved =
      halo_name.empty() ? std::string{"halo_block_"} + name_ : std::move(halo_name);
  if (block_ == nullptr) {
    return ::carbon::Kernel::anchor(std::move(resolved));
  }
  // The HaloBlock ctor walks block_'s halo plan + this field's per-GPU
  // device buffers; it does not retain a reference to the Block beyond the
  // ctor. Building one here is cheap (a few Transfer / DistTransfer
  // descriptor entries) and side-effect-free.
  using HaloBlockT = ::carbon::HaloBlock<T, Components, SBlock>;
  HaloBlockT halo(*block_, *const_cast<detail::block::Field<T, Components, SBlock>*>(this));
  return ::carbon::Kernel::halo_block(std::move(halo), std::move(resolved));
}

// ----- Block::make_kernel convenience member -----
template <typename SBlock>
template <typename L>
[[nodiscard]] inline auto
Block<SBlock>::make_kernel(L&& loading_fn,
                           size_t shared_mem) const -> ::carbon::Kernel {
  return ::carbon::Kernel::compute(*this, std::forward<L>(loading_fn), /*name=*/{}, shared_mem);
}

template <typename SBlock>
template <typename T,
          typename L,
          typename Op>
[[nodiscard]] inline auto
Block<SBlock>::make_reduce_kernel(L&& loading_fn,
                                  ::carbon::ReductionResult<T>& dst,
                                  Op op,
                                  size_t shared_mem) const -> ::carbon::Kernel {
  // Kernel::reduce dispatches on Grid type at compile time — when the grid
  // exposes a @c BlockPolicy typedef (which @c Block<SBlock> does), it
  // builds a @c BlockReduceKernelImpl ; otherwise the dense impl. Same
  // shape as @c Dense::make_reduce_kernel above.
  return ::carbon::Kernel::reduce<T>(*this, std::forward<L>(loading_fn), dst, std::move(op),
                                     /*name=*/{}, shared_mem);
}

// ----- Block::make_host_kernel / make_host_reduce_kernel convenience members -----
template <typename SBlock>
template <typename L>
[[nodiscard]] inline auto
Block<SBlock>::make_host_kernel(L&& loading_fn) const -> ::carbon::Kernel {
  return ::carbon::Kernel::host_compute(*this, std::forward<L>(loading_fn), /*name=*/{});
}

template <typename SBlock>
template <typename T,
          typename L,
          typename Op>
[[nodiscard]] inline auto
Block<SBlock>::make_host_reduce_kernel(L&& loading_fn,
                                       ::carbon::ReductionResult<T>& dst,
                                       Op op) const -> ::carbon::Kernel {
  return ::carbon::Kernel::host_reduce<T>(*this, std::forward<L>(loading_fn), dst, std::move(op),
                                          /*name=*/{});
}

// ----- Block::make_universal_kernel / make_universal_reduce_kernel convenience members -----
template <typename SBlock>
template <typename L>
[[nodiscard]] inline auto
Block<SBlock>::make_universal_kernel(L&& loading_fn,
                                     size_t shared_mem) const -> ::carbon::Kernel {
  return ::carbon::Kernel::compute_universal(*this, std::forward<L>(loading_fn), /*name=*/{},
                                             shared_mem);
}

template <typename SBlock>
template <typename T,
          typename L,
          typename Op>
[[nodiscard]] inline auto
Block<SBlock>::make_universal_reduce_kernel(L&& loading_fn,
                                            ::carbon::ReductionResult<T>& dst,
                                            Op op,
                                            size_t shared_mem) const -> ::carbon::Kernel {
  return ::carbon::Kernel::reduce_universal<T>(*this, std::forward<L>(loading_fn), dst,
                                               std::move(op), /*name=*/{}, shared_mem);
}
