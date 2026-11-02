// kernel_dense_distributed.inl — DenseDistributed + DenseDistributed::Field convenience members.
// Included from <carbon/kernel.h> exactly once, after Kernel's static factories are defined and
// after the dense impl detail headers have brought DeviceKernelImpl / ReduceKernelImpl etc. into
// scope. Not a standalone header (same contract as kernel_dense.inl).

// ----- DenseDistributed convenience members -----
//
// Forward to the static factories on @c Kernel; same naming convention as @c Dense:
//   make_kernel             -> device-only compute
//   make_universal_kernel   -> dual-target compute (device + host)
//   make_host_kernel        -> host-only compute
// Same applies to the reduce twins.

template <typename L>
[[nodiscard]] inline auto
DenseDistributed::make_kernel(L&& loading_fn,
                              size_t shared_mem) const -> Kernel {
  return Kernel::compute(*this, std::forward<L>(loading_fn), /*name=*/{}, shared_mem);
}

template <typename L>
[[nodiscard]] inline auto
DenseDistributed::make_universal_kernel(L&& loading_fn,
                                        size_t shared_mem) const -> Kernel {
  return Kernel::compute_universal(*this, std::forward<L>(loading_fn), /*name=*/{}, shared_mem);
}

template <typename T,
          typename L,
          typename Op>
[[nodiscard]] inline auto
DenseDistributed::make_reduce_kernel(L&& loading_fn,
                                     ReductionResult<T>& dst,
                                     Op op,
                                     size_t shared_mem,
                                     reduce::ReduceOptions options) const -> Kernel {
  return Kernel::reduce<T>(*this, std::forward<L>(loading_fn), dst, std::move(op), /*name=*/{},
                           shared_mem, options);
}

template <typename T,
          typename L,
          typename Op>
[[nodiscard]] inline auto
DenseDistributed::make_universal_reduce_kernel(L&& loading_fn,
                                               ReductionResult<T>& dst,
                                               Op op,
                                               size_t shared_mem) const -> Kernel {
  return Kernel::reduce_universal<T>(*this, std::forward<L>(loading_fn), dst, std::move(op),
                                     /*name=*/{}, shared_mem);
}

template <typename L>
[[nodiscard]] inline auto
DenseDistributed::make_host_kernel(L&& loading_fn) const -> Kernel {
  return Kernel::host_compute(*this, std::forward<L>(loading_fn), /*name=*/{});
}

template <typename T,
          typename L,
          typename Op>
[[nodiscard]] inline auto
DenseDistributed::make_host_reduce_kernel(L&& loading_fn,
                                          ReductionResult<T>& dst,
                                          Op op) const -> Kernel {
  return Kernel::host_reduce<T>(*this, std::forward<L>(loading_fn), dst, std::move(op),
                                /*name=*/{});
}

// ----- DenseDistributed::Field convenience member -----
//
// Declared in <carbon/detail/dense_distributed/field.h> against a forward-declared Kernel;
// defined here once HaloDistributed and Kernel are fully visible.
template <typename T,
          int Components>
inline auto
detail::dense_distributed::Field<T,
                                 Components>::make_halo_kernel(std::string name) const
    -> ::carbon::Kernel {
  return ::carbon::Kernel::halo_distributed(::carbon::HaloDistributed<T, Components>(*this),
                                            std::move(name));
}

template <typename T,
          int Components>
inline auto
detail::dense_distributed::Field<T,
                                 Components>::make_halo_kernel(std::string name,
                                                               Stencil stencil,
                                                               StencilSemantic semantic) const
    -> ::carbon::Kernel {
  return ::carbon::Kernel::halo_distributed(
      ::carbon::HaloDistributed<T, Components>(*this, std::move(stencil), semantic),
      std::move(name));
}
