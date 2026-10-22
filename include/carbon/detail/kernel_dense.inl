// kernel_dense.inl — Dense + Dense::Field convenience members. Split out of
// <carbon/kernel.h> to keep the umbrella header focused on the cross-grid
// Kernel / KernelLoader machinery. Included from <carbon/kernel.h> exactly
// once, after Kernel's static factories are defined and after the dense
// impl detail headers have brought DeviceKernelImpl / ReduceKernelImpl etc.
// into scope. Not a standalone header — has no include guard for the same
// reason `kernel.h` includes the detail impl headers directly: this file
// only makes sense in the kernel.h tail-include context.

// ----- Dense convenience members -----
//
// Forward to the static factories on @c Kernel. The naming convention rewards the
// common case with the shortest spelling:
//   make_kernel             -> device-only compute (was make_device_kernel).
//   make_universal_kernel   -> dual-target compute (was the old make_kernel).
//   make_host_kernel        -> host-only compute (unchanged).
// Same applies to the reduce twins.

template <typename L>
[[nodiscard]] inline auto
Dense::make_kernel(L&& loading_fn,
                   size_t shared_mem) const -> Kernel {
  return Kernel::compute(*this, std::forward<L>(loading_fn), /*name=*/{}, shared_mem);
}

template <typename L>
[[nodiscard]] inline auto
Dense::make_universal_kernel(L&& loading_fn,
                             size_t shared_mem) const -> Kernel {
  return Kernel::compute_universal(*this, std::forward<L>(loading_fn), /*name=*/{}, shared_mem);
}

template <typename T,
          typename L,
          typename Op>
[[nodiscard]] inline auto
Dense::make_reduce_kernel(L&& loading_fn,
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
Dense::make_universal_reduce_kernel(L&& loading_fn,
                                    ReductionResult<T>& dst,
                                    Op op,
                                    size_t shared_mem) const -> Kernel {
  return Kernel::reduce_universal<T>(*this, std::forward<L>(loading_fn), dst, std::move(op),
                                     /*name=*/{}, shared_mem);
}

template <typename L>
[[nodiscard]] inline auto
Dense::make_host_kernel(L&& loading_fn) const -> Kernel {
  return Kernel::host_compute(*this, std::forward<L>(loading_fn), /*name=*/{});
}

template <typename T,
          typename L,
          typename Op>
[[nodiscard]] inline auto
Dense::make_host_reduce_kernel(L&& loading_fn,
                               ReductionResult<T>& dst,
                               Op op) const -> Kernel {
  return Kernel::host_reduce<T>(*this, std::forward<L>(loading_fn), dst, std::move(op),
                                /*name=*/{});
}

// ----- Dense::Field convenience member -----
//
// Declared in <carbon/detail/dense/field.h> against a forward-declared Kernel; defined
// here once Halo and Kernel are fully visible. Mirrors the Dense::make_kernel pattern.
template <typename T,
          int Components>
inline auto
detail::dense::Field<T,
                     Components>::make_halo_kernel(std::string name) const -> ::carbon::Kernel {
  return ::carbon::Kernel::halo(::carbon::Halo<T, Components>(*this), std::move(name));
}

template <typename T,
          int Components>
inline auto
detail::dense::Field<T,
                     Components>::make_halo_kernel(std::string name,
                                                   Stencil stencil,
                                                   StencilSemantic semantic) const
    -> ::carbon::Kernel {
  return ::carbon::Kernel::halo(::carbon::Halo<T, Components>(*this, std::move(stencil), semantic),
                                std::move(name));
}
