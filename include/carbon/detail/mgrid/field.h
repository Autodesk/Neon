#pragma once

#include <carbon/access_token.h>
#include <carbon/detail/block/field.h>
#include <carbon/detail/mgrid/partition.h>

#include <stdint.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace carbon {
template <typename>
class MultiResBlock;  // forward (defined in <carbon/mgrid.h>)
class Kernel;         // forward (defined in <carbon/kernel.h>); only LevelView::make_halo_kernel
                      // returns one, and that member is defined in the kernel.h tail include.
}  // namespace carbon

namespace carbon::detail::mgrid {

/**
 * Multi-resolution field: one @c block::Field<T, Components, SBlock> per level of the mGrid
 * tower, plus the wiring that lets a kernel build a cross-level @ref Partition for any level.
 * Mirrors Neon's @c mField .
 *
 * Storage is genuinely per-level — each level's @c block::Field owns its own per-GPU SoA
 * buffer over that level's retained bricks. The cross-level connection is made at
 * @ref make_partition time: the returned @ref Partition carries this level's storage pointer
 * plus the @b adjacent levels' storage pointers, spans, and the parent / child block-id
 * tables owned by the @c MultiResBlock . That keeps @c Field a pure storage aggregate and
 * pushes the (level-dependent) cross-level pointer gathering into one place.
 *
 * @tparam T          Scalar / element type per voxel and component.
 * @tparam Components Per-cell component count.
 * @tparam SBlock     @c StaticBlock policy shared by every level's @c Block .
 */
template <typename T, int Components, typename SBlock>
class Field {
public:
  using value_type = T;
  using PartitionT = Partition<T, Components, SBlock>;
  using LevelField = block::Field<T, Components, SBlock>;
  static constexpr int components = Components;

  Field() = default;

  Field(std::vector<LevelField> levels,
        const ::carbon::MultiResBlock<SBlock>* owner,
        std::string name)
      : levels_(std::move(levels)),
        owner_(owner),
        name_(std::move(name)) {}

  /** Number of levels (== the owning @c MultiResBlock 's depth). */
  [[nodiscard]] auto
  depth() const noexcept -> int {
    return static_cast<int>(levels_.size());
  }

  /** Per-level @c block::Field for @p level (0 = finest). */
  [[nodiscard]] auto
  level_field(int level) -> LevelField& {
    return levels_.at(static_cast<size_t>(level));
  }

  [[nodiscard]] auto
  level_field(int level) const -> const LevelField& {
    return levels_.at(static_cast<size_t>(level));
  }

  [[nodiscard]] auto
  name() const noexcept -> const std::string& {
    return name_;
  }

  auto
  set_name(std::string name) noexcept -> void {
    name_ = std::move(name);
  }

  /**
   * Build the cross-level @ref Partition for @p level on @p gpu_id . Gathers this level's
   * storage pointer plus the adjacent levels' storage / spans / block-id tables from the
   * owning @c MultiResBlock . Defined out-of-line in @c mgrid.inl (needs the full
   * @c MultiResBlock definition).
   */
  [[nodiscard]] auto
  make_partition(int level,
                 size_t gpu_id = 0) const -> PartitionT;

  /**
   * Host-mirror variant of @ref make_partition : binds @c host_partitions() / @c cpu_buffers()
   * pointers and the host parent / child block-id tables. Empty when the level field has no
   * host mirror (@c MemorySide::Device ).
   */
  [[nodiscard]] auto
  make_host_partition(int level,
                      size_t gpu_id = 0) const -> PartitionT;

  /**
   * @brief Loader-bindable per-level view of this multi-resolution field.
   *
   * A @c MultiResField spans the whole tower, but a compute kernel runs over one level at a
   * time (the launch geometry is that level's @c Block ). @ref LevelView is the thin adapter
   * that lets the existing @c KernelLoader bind a single level: it exposes exactly the
   * duck-typed surface @c KernelLoader::read / @c write / @c operator() consume
   * (@c partitions() , @c host_partitions() , @c gpu_buffers() , @c name() ), where
   * @c partitions()[gpu] is the cross-level @ref Partition for @c (level, gpu) . The per-cell
   * body therefore reads/writes its own level and walks to parent/child levels through one
   * partition value — no peer copy, transfer-free by the co-residency rule.
   *
   * Host mirrors: when the underlying @c block::Field was built with @c MemorySide::Host or
   * @c HostDevice , @c host_partitions()[gpu] is a cross-level @ref Partition bound to the CPU
   * buffer (same layout as @c partitions() ). Empty for @c MemorySide::Device fields.
   */
  class LevelView {
  public:
    LevelView(const LevelField* level_field,
              std::vector<PartitionT> partitions,
              std::vector<PartitionT> host_partitions,
              std::string name)
        : level_field_(level_field),
          partitions_(std::move(partitions)),
          host_partitions_(std::move(host_partitions)),
          name_(std::move(name)) {}

    /** Per-GPU cross-level @ref Partition vector ( @c [gpu] indexes the launch's GPU). */
    [[nodiscard]] auto
    partitions() const noexcept -> const std::vector<PartitionT>& {
      return partitions_;
    }

    /** Per-GPU host-mirror cross-level @ref Partition vector (empty for device-only fields). */
    [[nodiscard]] auto
    host_partitions() const noexcept -> const std::vector<PartitionT>& {
      return host_partitions_;
    }

    /** Underlying level @c Buffer<T> whose @c get_uid() is the field identity for tokens. */
    [[nodiscard]] auto
    gpu_buffers() const noexcept -> const ::carbon::Buffer<T>& {
      return level_field_->gpu_buffers();
    }

    /** Diagnostic label ( @c "<field>:L<level>" ) snapshotted into recorded access tokens. */
    [[nodiscard]] auto
    name() const noexcept -> const std::string& {
      return name_;
    }

    /**
     * Intra-level halo factory for @c KernelLoader::read_stencil — forwards to this level's
     * @c block::Field::make_halo_kernel (the cross-level path is halo-free; only same-level
     * stencils need a halo). Defined in the @c <carbon/kernel.h> tail include.
     */
    [[nodiscard]] auto
    make_halo_kernel(std::string halo_name = {}) const -> ::carbon::Kernel;

  private:
    const LevelField* level_field_{nullptr};
    std::vector<PartitionT> partitions_{};
    std::vector<PartitionT> host_partitions_{};
    std::string name_{};
  };

  /**
   * Build the loader-bindable @ref LevelView for @p level (gathers one cross-level
   * @ref Partition per GPU). Defined out-of-line in @c mgrid.inl (needs the full
   * @c MultiResBlock definition for the GPU count).
   */
  [[nodiscard]] auto
  level_view(int level) const -> LevelView;

  /**
   * @brief Bind @p level for a cross-level compute kernel @b and declare its parent / child read
   *        dependencies in one call, so a @ref ::carbon::Skeleton sequences the kernel correctly.
   *
   * A cross-level body reads the adjacent levels through the returned @ref Partition (
   * @c p.parent(...) / @c p.child(...) ), but @c loader.read / @c write on @ref level_view only
   * records a token for the @b own level — the adjacent-level reads are invisible to the
   * @c Skeleton 's token-driven dependency graph. This helper closes that gap atomically: it
   * binds the own level with @p own_mode (returning the full cross-level @ref Partition for the
   * per-cell body) and, when requested, records an @c AccessMode::Read token on the child (
   * @c level-1 ) and/or parent ( @c level+1 ) level. Those adjacent reads bind the cheap
   * @c block::Field (the discarded @c block::Partition is never used) but carry the @b same
   * @c field_uid as @c level_view(level±1) , so e.g. a restriction kernel ( @c own_mode=Write ,
   * @c read_child=true ) is correctly ordered after the fine-level write that produced its input.
   *
   * Same-level @e stencil reads are a separate concern: call
   * @c loader.read_stencil(field.level_view(level)) for those (that path also snapshots the
   * level's halo factory so the @c Skeleton can auto-insert the intra-level halo). Out-of-range
   * @c level±1 declarations are silently skipped (the finest level has no child, the coarsest no
   * parent), so a kernel that walks the tower edges composes without special-casing.
   *
   * Defined in the @c <carbon/kernel.h> tail include (needs the full @c KernelLoader );
   * templated on @c Loader so this header stays decoupled from @c <carbon/kernel.h> .
   *
   * @param loader      The @c KernelLoader handed to the loading lambda.
   * @param level       Level to iterate (0 = finest). @c [0, depth()) .
   * @param own_mode    Access mode for this level ( @c Read / @c Write / @c ReadWrite ).
   * @param read_child  Declare a read of the child ( @c level-1 ) level (default false).
   * @param read_parent Declare a read of the parent ( @c level+1 ) level (default false).
   * @return The cross-level @ref Partition for @c (level, loader's GPU) — same as
   *         @c loader.<own_mode>(level_view(level)) .
   */
  template <typename Loader>
  [[nodiscard]] auto
  bind_cross_level(const Loader& loader,
                   int level,
                   ::carbon::AccessMode own_mode,
                   bool read_child = false,
                   bool read_parent = false) const -> PartitionT;

  /**
   * @brief Build the intra-level halo @c Kernel for @p level — a @c HaloBlock over that level's
   *        @c block::Field .
   *
   * mGrid's cross-level (parent/child) path is transfer-free by co-residency, so the @b only
   * data movement a multi-resolution stencil needs is the @e same-level neighbour exchange:
   * this is that exchange, one level at a time. Forwards to the level's
   * @c block::Field::make_halo_kernel (the back-reference to the level's @c Block is set by
   * @c MultiResBlock::newField , so a real @c HaloBlock is built — intra-process @c Transfer plus,
   * under @c CARBON_HAVE_MPI_NCCL , the cross-rank @c DistTransfer companion). Launching the
   * returned @c Kernel runs the exchange for @p level ; iterate @c level in @c [0, depth()) for a
   * whole-tower halo. Defined in the @c <carbon/kernel.h> tail include.
   *
   * @throws std::out_of_range via @c level_field when @p level is outside @c [0, depth()) .
   */
  [[nodiscard]] auto
  make_halo_kernel(int level,
                   std::string halo_name = {}) const -> ::carbon::Kernel;

private:
  std::vector<LevelField> levels_{};
  const ::carbon::MultiResBlock<SBlock>* owner_{nullptr};
  std::string name_{};
};

}  // namespace carbon::detail::mgrid
