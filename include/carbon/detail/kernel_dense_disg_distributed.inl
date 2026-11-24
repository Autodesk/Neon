// kernel_dense_disg_distributed.inl — DenseDisgDistributed + DenseDisgDistributed::Field
// convenience members. Tail include of <carbon/kernel.h>, not a standalone header (see
// kernel_dense.inl for the include-context contract).

// ----- DenseDisgDistributed::Field convenience members -----

template <typename T,
          int Components>
inline auto
detail::dense_disg_distributed::Field<T,
                                      Components>::make_halo_kernel(std::string halo_name) const
    -> ::carbon::Kernel {
  std::string resolved =
      halo_name.empty() ? std::string{"halo_disg_distributed_"} + name_ : std::move(halo_name);
  using HaloT = ::carbon::HaloDisgDistributed<T, Components>;
  HaloT halo(*this);
  return ::carbon::Kernel::halo_disg_distributed(std::move(halo), std::move(resolved));
}

template <typename T,
          int Components>
inline auto
detail::dense_disg_distributed::Field<T,
                                      Components>::make_halo_kernel(std::string halo_name,
                                                                    Stencil stencil,
                                                                    StencilSemantic semantic) const
    -> ::carbon::Kernel {
  std::string resolved =
      halo_name.empty() ? std::string{"halo_disg_distributed_"} + name_ : std::move(halo_name);
  using HaloT = ::carbon::HaloDisgDistributed<T, Components>;
  HaloT halo(*this, std::move(stencil), semantic);
  return ::carbon::Kernel::halo_disg_distributed(std::move(halo), std::move(resolved));
}

// ----- DenseDisgDistributed convenience members -----
//
// The DenseDisgDistributedPolicy marker routes the per-cell launch through
// dense_disg_distributed::launch_disg_grid (and reduce through disg_grid_reduce_kernel /
// launch_disg_reduce) inside detail::DeviceLaunchTraits / detail::ReduceLaunchTraits.

template <typename L>
[[nodiscard]] inline auto
DenseDisgDistributed::make_kernel(L&& loading_fn,
                                  size_t shared_mem) const -> Kernel {
  return Kernel::compute(*this, std::forward<L>(loading_fn), /*name=*/{}, shared_mem);
}

template <typename L>
[[nodiscard]] inline auto
DenseDisgDistributed::make_universal_kernel(L&& loading_fn,
                                            size_t shared_mem) const -> Kernel {
  return Kernel::compute_universal(*this, std::forward<L>(loading_fn), /*name=*/{}, shared_mem);
}

template <typename L>
[[nodiscard]] inline auto
DenseDisgDistributed::make_host_kernel(L&& loading_fn) const -> Kernel {
  return Kernel::host_compute(*this, std::forward<L>(loading_fn), /*name=*/{});
}

template <typename T,
          typename L,
          typename Op>
[[nodiscard]] inline auto
DenseDisgDistributed::make_reduce_kernel(L&& loading_fn,
                                         ReductionResult<T>& dst,
                                         Op op,
                                         size_t shared_mem) const -> Kernel {
  return Kernel::reduce<T>(*this, std::forward<L>(loading_fn), dst, std::move(op), /*name=*/{},
                           shared_mem);
}

template <typename T,
          typename L,
          typename Op>
[[nodiscard]] inline auto
DenseDisgDistributed::make_universal_reduce_kernel(L&& loading_fn,
                                                   ReductionResult<T>& dst,
                                                   Op op,
                                                   size_t shared_mem) const -> Kernel {
  return Kernel::reduce_universal<T>(*this, std::forward<L>(loading_fn), dst, std::move(op),
                                     /*name=*/{}, shared_mem);
}

template <typename T,
          typename L,
          typename Op>
[[nodiscard]] inline auto
DenseDisgDistributed::make_host_reduce_kernel(L&& loading_fn,
                                              ReductionResult<T>& dst,
                                              Op op) const -> Kernel {
  return Kernel::host_reduce<T>(*this, std::forward<L>(loading_fn), dst, std::move(op),
                                /*name=*/{});
}
