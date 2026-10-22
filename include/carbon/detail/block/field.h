#pragma once

#include <carbon/detail/block/partition.h>
#include <carbon/detail/block/span.h>
#include <carbon/detail/block/static_block.h>
#include <carbon/system.h>

#include <stdint.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace carbon {
class Kernel;  // forward (defined in <carbon/kernel.h>)

template <typename>
class Block;  // forward (defined in <carbon/block.h>; needed for the field's back-ref)
}  // namespace carbon

namespace carbon::detail::block {

/**
 * Per-grid field storage over a @c carbon::Block layout: owns the per-device SoA buffer
 * (and an optional per-device host mirror) plus the matching @ref Partition views.
 * Mirrors @c carbon::detail::dense::Field at the surface level: per-GPU device buffer,
 * per-GPU host mirror under @c MemorySide::Host / @c MemorySide::HostDevice ,
 * @c make_halo_kernel for cross-slab transfers via @c HaloBlock , and back-reference
 * to the owning @c Block for kernel construction.
 *
 * Storage layout per device (matches Neon's @c bField , see analysis report §4):
 *
 * @code
 * elements_per_card = block_count · MemX · MemY · MemZ
 * total_elements    = Components · elements_per_card
 * linear_index      = (block_idx · Components + component_id) · MemX · MemY · MemZ +
 *                     (x + MemX · y + MemX · MemY · z)
 * @endcode
 *
 * That is: structure-of-arrays per-block (one slab of @c VoxelCount elements per
 * cardinality, all cardinalities of one block contiguous, then the next block).
 *
 * @par KernelLoader contract
 *
 * The loader treats fields as duck-typed via
 *   - @c partitions() : per-GPU device @ref Partition vector. Each partition is now a
 *     thin @c { T* } wrapper — bitmask + 27-way connectivity live on @c Span (one
 *     copy per @c (Block, GPU) ), so @c Field carries no @c Block -borrowed pointers
 *     and the kernel reads topology metadata through its captured @c Span on every
 *     @c neighbour() call.
 *   - @c host_partitions() : per-GPU host mirror @ref Partition vector. Empty when the
 *     field was built with @c MemorySide::Device (no CPU mirror was allocated). For
 *     fields built with @c MemorySide::Host or @c MemorySide::HostDevice the mirror
 *     is eagerly populated by @c Block::newField with the same per-GPU sizes as
 *     the device buffer (owned + ghost slots).
 *   - @c cpu_buffers() : per-GPU host @ref Buffer<T> for raw host-side read / write.
 *     Sized identically to @c gpu_buffers (one host @ref Allocation<T> per local
 *     GPU, each of length @c (n_owned + n_ghost_lo + n_ghost_up) * VoxelCount *
 *     Components scalars). Empty on a @c MemorySide::Device field.
 *   - @c gpu_buffers() : the underlying @ref Buffer<T> ; its @c get_uid() identifies
 *     the field across kernels for the @c Skeleton's dependency analysis.
 *   - @c make_halo_kernel() : called by @c KernelLoader::read_stencil to insert the
 *     pre-stencil halo. Returns a real @c HaloBlock<T, Components, SBlock> when the
 *     field has been bound to a @c Block via @ref set_block ; falls back to a no-op
 *     anchor for an unbound field.
 *   - @c name() : diagnostic label propagated to logs.
 *
 * @tparam T          Scalar / element type stored per voxel and component.
 * @tparam Components Per-cell component count (1 = scalar, 3 = vector, ...).
 * @tparam SBlock     @ref StaticBlock policy; same brick size used to build the owning
 *                    @c carbon::Block .
 */
template <typename T, int Components, typename SBlock>
class Field {
  static_assert(Components >= 1,
                "carbon::Block::Field<T, Components, SBlock> requires Components >= 1");

public:
  using value_type = T;
  using PartitionT = Partition<T, Components, SBlock>;
  using SpanT = Span<SBlock>;
  using BitMask = typename SBlock::BitMask;
  static constexpr int components = Components;

  Field() = default;

  /**
   * Build a field bound to @p device_buffer (typically allocated through
   * @c carbon::Block::newField on the system's @c GpuGroup ). @p name is a debug
   * label propagated to logging.
   *
   * Bitmask + 27-way connectivity are @b not stored on the field; they live on
   * the per-GPU @c Span owned by @c Block and are read through it on every
   * @c Partition::neighbour walk. That keeps @c Field free of @c Block -borrowed
   * pointers and collapses the lifetime contract to "the @c Block (and therefore
   * its @c Span s) must outlive any kernel that references this field" — which is
   * already required for @c make_kernel / @c make_halo_kernel anyway.
   */
  Field(Buffer<T> device_buffer,
        std::string name)
      : device_buffer_(std::move(device_buffer)),
        name_(std::move(name)) {
    rebuild_partitions_();
  }

  /**
   * As above, but also adopts a per-GPU host mirror @p host_buffer parallel to
   * @p device_buffer (same per-GPU element counts, host-resident @ref
   * Allocation<T> entries). Built by @c Block::newField when the user passes
   * @c MemorySide::Host or @c MemorySide::HostDevice . The mirror feeds
   * @ref host_partitions and @ref cpu_buffers ; the device side is unchanged
   * and continues to feed @ref partitions / @ref gpu_buffers . Topology
   * metadata (bitmask + connectivity) is shared between sides via the owning
   * @c Block 's per-GPU @c Span — see the single-buffer constructor above.
   */
  Field(Buffer<T> device_buffer,
        Buffer<T> host_buffer,
        std::string name)
      : device_buffer_(std::move(device_buffer)),
        host_buffer_(std::move(host_buffer)),
        name_(std::move(name)) {
    rebuild_partitions_();
  }

  /**
   * Back-reference to the owning @c Block<SBlock> . Set by @c Block::newField so
   * that @ref make_halo_kernel can build a real @c HaloBlock<T, Components, SBlock>
   * rather than the no-op anchor that default-constructed fields fall back to,
   * and so that kernels can fetch the per-GPU @c Span (bitmask + connectivity)
   * for @c Partition::neighbour walks.
   *
   * The pointer is non-owning; the caller (typically the same user code that
   * created the @c Block ) is responsible for keeping the underlying @c Block
   * alive for as long as any kernel built over the field may run. Since carbon's
   * usage pattern is "construct @c Block , then build fields + kernels, then
   * launch", this matches how users already drive @c Block .
   *
   * This mirrors @c Dense::Field 's @c dense_ back-reference, with the
   * simplification that @c Block does not yet have a shallow-shared
   * @c shared_ptr<State> indirection — so a back-pointer plus the lifetime
   * constraint above is the cheapest correct option.
   */
  auto
  set_block(const ::carbon::Block<SBlock>* block) noexcept -> void {
    block_ = block;
  }

  [[nodiscard]] auto
  block_back_ref() const noexcept -> const ::carbon::Block<SBlock>* {
    return block_;
  }

  /** Bound buffer (size 1 in phase 1; one entry per local GPU once multi-GPU lands). */
  [[nodiscard]] auto
  gpu_buffers() noexcept -> Buffer<T>& {
    return device_buffer_;
  }

  [[nodiscard]] auto
  gpu_buffers() const noexcept -> const Buffer<T>& {
    return device_buffer_;
  }

  /** @c KernelLoader-facing alias of @ref gpu_buffers . */
  [[nodiscard]] auto
  device_buffer() noexcept -> Buffer<T>& {
    return device_buffer_;
  }

  [[nodiscard]] auto
  device_buffer() const noexcept -> const Buffer<T>& {
    return device_buffer_;
  }

  [[nodiscard]] auto
  partitions() noexcept -> std::vector<PartitionT>& {
    return partitions_;
  }

  [[nodiscard]] auto
  partitions() const noexcept -> const std::vector<PartitionT>& {
    return partitions_;
  }

  /**
   * Per-GPU host mirror partitions. Empty when the field was built with
   * @c MemorySide::Device (no CPU mirror allocated); for @c MemorySide::Host
   * and @c MemorySide::HostDevice each entry points into @ref cpu_buffers
   * with the same indexing the device @ref partitions use.
   */
  [[nodiscard]] auto
  host_partitions() noexcept -> std::vector<PartitionT>& {
    return host_partitions_;
  }

  [[nodiscard]] auto
  host_partitions() const noexcept -> const std::vector<PartitionT>& {
    return host_partitions_;
  }

  /**
   * Per-GPU host mirror @ref Buffer<T> . Sized identically to
   * @ref gpu_buffers — one host allocation per local GPU, length
   * @c (n_owned + n_ghost_lo + n_ghost_up) * VoxelCount * Components
   * scalars — when the field was built with @c MemorySide::Host or
   * @c MemorySide::HostDevice ; empty otherwise.
   */
  [[nodiscard]] auto
  cpu_buffers() noexcept -> Buffer<T>& {
    return host_buffer_;
  }

  [[nodiscard]] auto
  cpu_buffers() const noexcept -> const Buffer<T>& {
    return host_buffer_;
  }

  [[nodiscard]] auto
  name() const noexcept -> const std::string& {
    return name_;
  }

  /**
   * Diagnostic label setter — mirror of @c Dense::Field::set_name and
   * @c DenseDisg::Field::set_name so cross-grid generic code (examples, benchmarks,
   * skeleton-orchestrated pipelines) can rename a field after construction without
   * branching on grid family.
   */
  auto
  set_name(std::string name) noexcept -> void {
    name_ = std::move(name);
  }

  /**
   * @c KernelLoader::read_stencil hook. Returns a real @c HaloBlock<T, Components,
   * SBlock> Kernel when the field is bound to a @c Block via @ref set_block ; falls
   * back to a no-op anchor for an unbound field (e.g. a default-constructed @c Field
   * not produced through @c Block::newField ).
   *
   * Defined out-of-line in @c block.inl so the @c Kernel forward declaration above is
   * sufficient here (the @ref Kernel type is fully defined where @c block.inl is
   * included).
   */
  [[nodiscard]] auto
  make_halo_kernel(std::string halo_name = {}) const -> ::carbon::Kernel;

  /**
   * Build a device-visible @ref Partition view of this field. Equivalent to
   * @c partitions()[0] ; kept for ergonomic use in single-GPU user code that wants
   * a shortcut without touching the per-GPU vector.
   */
  [[nodiscard]] auto
  device_partition() const -> PartitionT {
    if (partitions_.empty()) {
      return PartitionT{};
    }
    return partitions_.front();
  }

private:
  auto
  rebuild_partitions_() -> void {
    partitions_.clear();
    host_partitions_.clear();
    if (device_buffer_.empty()) {
      return;
    }
    partitions_.reserve(device_buffer_.size());
    for (size_t i = 0; i < device_buffer_.size(); ++i) {
      partitions_.emplace_back(const_cast<T*>(device_buffer_[i].data()));
    }
    if (host_buffer_.empty()) {
      return;
    }
    // Topology metadata (bitmask + connectivity) lives on the per-GPU Span owned
    // by Block, not on the partition; both device and host partitions are pure
    // value-views and share the same Span at the kernel-body site. This matches
    // Dense::Field where host_partitions reuse the same Span metadata as device.
    host_partitions_.reserve(host_buffer_.size());
    for (size_t i = 0; i < host_buffer_.size(); ++i) {
      host_partitions_.emplace_back(const_cast<T*>(host_buffer_[i].data()));
    }
  }

  Buffer<T> device_buffer_{};
  Buffer<T> host_buffer_{};
  std::vector<PartitionT> partitions_{};
  std::vector<PartitionT> host_partitions_{};
  std::string name_{};
  const ::carbon::Block<SBlock>* block_{nullptr};
};

}  // namespace carbon::detail::block
