// kernel_dense_disg.inl — DenseDisg + DenseDisg::Field convenience members.
// Split out of <carbon/kernel.h>. See the matching header at the top of
// kernel_dense.inl for the include-context contract — this file is a tail
// include of <carbon/kernel.h>, not a standalone header.

// ----- DenseDisg::Field convenience member -----
//
// Mirror of Dense::Field::make_halo_kernel for the disaggregated layout. Always
// produces a real HaloDisg-backed @c Kernel; single-GPU non-distributed layouts
// where the halo plan is empty still get a well-formed kernel whose operator()
// is a no-op. Used by the Skeleton auto-halo path on stencil-tagged DenseDisg
// reads and by user-driven OverlappedStencil pipelines.
template <typename T,
          int Components>
inline auto
detail::dense_disg::Field<T,
                          Components>::make_halo_kernel(std::string halo_name) const
    -> ::carbon::Kernel {
  std::string resolved =
      halo_name.empty() ? std::string{"halo_disg_"} + name_ : std::move(halo_name);
  using HaloDisgT = ::carbon::HaloDisg<T, Components>;
  HaloDisgT halo(*this);
  return ::carbon::Kernel::halo_disg(std::move(halo), std::move(resolved));
}

// Lattice-aware twin: forwards @p stencil / @p semantic to HaloDisg so the
// Skeleton's read_stencil(field, stencil, Lattice) auto-halo narrows the
// per-adjacency Send/Recv to the stencil's c_q.z run-length profile.
template <typename T,
          int Components>
inline auto
detail::dense_disg::Field<T,
                          Components>::make_halo_kernel(std::string halo_name,
                                                        Stencil stencil,
                                                        StencilSemantic semantic) const
    -> ::carbon::Kernel {
  std::string resolved =
      halo_name.empty() ? std::string{"halo_disg_"} + name_ : std::move(halo_name);
  using HaloDisgT = ::carbon::HaloDisg<T, Components>;
  HaloDisgT halo(*this, std::move(stencil), semantic);
  return ::carbon::Kernel::halo_disg(std::move(halo), std::move(resolved));
}

// ----- DenseDisg convenience members -----
//
// DenseDisg carries a @c DisgPolicy marker that the kernel factories use to route
// the per-cell launch through @c launch_disg_grid (and reduce through
// @c disg_grid_reduce_kernel / @c launch_disg_reduce ) instead of the aggregated
// dense backend helpers. The factories themselves are uniform across grid families
// — same code path as @c Dense::make_kernel — only the compile-time dispatch
// inside the @c DeviceKernelImpl / @c ReduceKernelImpl / ... bodies differs.

template <typename L>
[[nodiscard]] inline auto
DenseDisg::make_kernel(L&& loading_fn,
                       size_t shared_mem) const -> Kernel {
  return Kernel::compute(*this, std::forward<L>(loading_fn), /*name=*/{}, shared_mem);
}

template <typename L>
[[nodiscard]] inline auto
DenseDisg::make_universal_kernel(L&& loading_fn,
                                 size_t shared_mem) const -> Kernel {
  return Kernel::compute_universal(*this, std::forward<L>(loading_fn), /*name=*/{}, shared_mem);
}

template <typename L>
[[nodiscard]] inline auto
DenseDisg::make_host_kernel(L&& loading_fn) const -> Kernel {
  return Kernel::host_compute(*this, std::forward<L>(loading_fn), /*name=*/{});
}

template <typename T,
          typename L,
          typename Op>
[[nodiscard]] inline auto
DenseDisg::make_reduce_kernel(L&& loading_fn,
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
DenseDisg::make_universal_reduce_kernel(L&& loading_fn,
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
DenseDisg::make_host_reduce_kernel(L&& loading_fn,
                                   ReductionResult<T>& dst,
                                   Op op) const -> Kernel {
  return Kernel::host_reduce<T>(*this, std::forward<L>(loading_fn), dst, std::move(op),
                                /*name=*/{});
}
