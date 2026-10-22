// kernel_mgrid.inl — MultiResBlock + MultiResField cross-level kernel convenience. Split out
// of <carbon/kernel.h>. See the header note at the top of kernel_dense.inl for the
// include-context contract — this file is a tail include of <carbon/kernel.h> (it needs the
// fully-defined ::carbon::Kernel and Block::make_kernel from kernel_block.inl), not a
// standalone header.

// ----- MultiResField::LevelView intra-level halo factory -----
//
// Forwards to the level's block::Field::make_halo_kernel (defined in kernel_block.inl above),
// so KernelLoader::read_stencil on a level view auto-inserts that level's HaloBlock. The
// cross-level (parent/child) path is halo-free by the co-residency rule; only same-level
// stencils need a halo, which is exactly what the underlying Block field provides.
template <typename T,
          int Components,
          typename SBlock>
inline auto
detail::mgrid::Field<T,
                     Components,
                     SBlock>::LevelView::make_halo_kernel(std::string halo_name) const
    -> ::carbon::Kernel {
  return level_field_->make_halo_kernel(std::move(halo_name));
}

// ----- MultiResField::make_halo_kernel (intra-level halo) -----
//
// The user-facing entry for a standalone same-level halo exchange; forwards to that level's
// block::Field::make_halo_kernel (HaloBlock). Distinct from LevelView::make_halo_kernel above,
// which exists for KernelLoader::read_stencil's auto-halo discovery.
template <typename T,
          int Components,
          typename SBlock>
inline auto
detail::mgrid::Field<T,
                     Components,
                     SBlock>::make_halo_kernel(int level,
                                               std::string halo_name) const -> ::carbon::Kernel {
  return level_field(level).make_halo_kernel(std::move(halo_name));
}

// ----- MultiResField::bind_cross_level cross-level dependency declaration -----
//
// The own-level call returns the full cross-level mgrid::Partition (with parent/child pointers)
// the per-cell body needs; the optional adjacent calls bind the cheap block::Field purely to
// register an AccessMode::Read token whose field_uid matches level_view(level±1) — that is what
// makes a cross-level kernel orderable by the grid-agnostic Skeleton. The adjacent block::Partition
// returned by loader.read is intentionally discarded (the body reads parent/child through `p`).
template <typename T,
          int Components,
          typename SBlock>
template <typename Loader>
[[nodiscard]] inline auto
detail::mgrid::Field<T,
                     Components,
                     SBlock>::bind_cross_level(const Loader& loader,
                                               int level,
                                               ::carbon::AccessMode own_mode,
                                               bool read_child,
                                               bool read_parent) const -> PartitionT {
  PartitionT p = (own_mode == ::carbon::AccessMode::Read)    ? loader.read(level_view(level))
                 : (own_mode == ::carbon::AccessMode::Write) ? loader.write(level_view(level))
                                                             : loader.read_write(level_view(level));
  if (read_child && level > 0) {
    (void)loader.read(level_field(level - 1));
  }
  if (read_parent && level + 1 < depth()) {
    (void)loader.read(level_field(level + 1));
  }
  return p;
}

// ----- MultiResBlock::make_kernel convenience member -----
//
// The launch geometry is the level's Block (DeviceLaunchTraits<Block> drives the per-brick /
// per-voxel launch over every GPU); the cross-level wiring rides in through the
// mgrid::Partition values the loading lambda pulls from field.level_view(level).
template <typename SBlock>
template <typename L>
[[nodiscard]] inline auto
MultiResBlock<SBlock>::make_kernel(int level,
                                   L&& loading_fn,
                                   size_t shared_mem) const -> ::carbon::Kernel {
  if (level < 0 || level >= depth()) {
    log::throw_with_log<std::out_of_range>(
        "MultiResBlock::make_kernel: level {} out of range [0, {}).", level, depth());
  }
  return grid(level).make_kernel(std::forward<L>(loading_fn), shared_mem);
}

template <typename SBlock>
template <typename L>
[[nodiscard]] inline auto
MultiResBlock<SBlock>::make_host_kernel(int level,
                                        L&& loading_fn) const -> ::carbon::Kernel {
  if (level < 0 || level >= depth()) {
    log::throw_with_log<std::out_of_range>(
        "MultiResBlock::make_host_kernel: level {} out of range [0, {}).", level, depth());
  }
  return grid(level).make_host_kernel(std::forward<L>(loading_fn));
}

// ----- MultiResBlock::make_reduce_kernel convenience member -----
//
// Reduction twin of make_kernel: the launch + 3-stage reduce (per-brick stage 1, intra-process
// combine, cross-rank ncclAllReduce) ride the Block reducer; the cross-level wiring rides in
// through the mgrid::Partition values the loading lambda pulls from field.level_view(level).
template <typename SBlock>
template <typename T,
          typename L,
          typename Op>
[[nodiscard]] inline auto
MultiResBlock<SBlock>::make_reduce_kernel(int level,
                                          L&& loading_fn,
                                          ::carbon::ReductionResult<T>& dst,
                                          Op op,
                                          size_t shared_mem) const -> ::carbon::Kernel {
  if (level < 0 || level >= depth()) {
    log::throw_with_log<std::out_of_range>(
        "MultiResBlock::make_reduce_kernel: level {} out of range [0, {}).", level, depth());
  }
  return grid(level).template make_reduce_kernel<T>(std::forward<L>(loading_fn), dst, std::move(op),
                                                    shared_mem);
}

template <typename SBlock>
template <typename T,
          typename L,
          typename Op>
[[nodiscard]] inline auto
MultiResBlock<SBlock>::make_host_reduce_kernel(int level,
                                               L&& loading_fn,
                                               ::carbon::ReductionResult<T>& dst,
                                               Op op) const -> ::carbon::Kernel {
  if (level < 0 || level >= depth()) {
    log::throw_with_log<std::out_of_range>(
        "MultiResBlock::make_host_reduce_kernel: level {} out of range [0, {}).", level, depth());
  }
  return grid(level).template make_host_reduce_kernel<T>(std::forward<L>(loading_fn), dst,
                                                         std::move(op));
}
