#pragma once

#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/system.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace carbon::detail::sub_grid {

/**
 * Per-GPU storage of a @ref carbon::SubGrid field: one packed SoA buffer per GPU holding
 * @c Components planes of @c count(gpu) elements, plus one @c Partition per GPU.
 *
 * No halo and no halo kernel: SubGrid fields only support same-cell access (see
 * @ref Partition ). Built only through @c SubGrid::newField .
 *
 * The SubGrid is stored by value (a shallow handle), so a field keeps its lane tables alive.
 * Copies share the underlying buffers, like @c Dense::Field .
 *
 * @tparam Grid       The owning @c carbon::SubGrid<Parent> .
 * @tparam T          Element type per component.
 * @tparam Components Components per cell.
 */
template <typename Grid, typename T, int Components>
class Field {
  static_assert(Components >= 1,
                "carbon::SubGrid::Field<T, Components> requires Components >= 1.");

public:
  using value_type = T;
  static constexpr int components = Components;
  using Partition = typename Grid::template Partition<T, Components>;

  /** Empty placeholder (no grid, no buffers, no partitions). */
  Field() = default;

  [[nodiscard]] auto
  sub_grid() const noexcept -> const Grid& {
    return grid_;
  }

  /** Identity of the SubGrid this field is laid out for; @c KernelLoader checks it. */
  [[nodiscard]] auto
  sub_grid_uid() const noexcept -> std::uint64_t {
    return grid_.uid();
  }

  /** Diagnostic name (Skeleton DOT labels, logs); identity is the buffer uid. */
  [[nodiscard]] auto
  name() const noexcept -> const std::string& {
    return name_;
  }

  auto
  set_name(std::string name) -> Field& {
    name_ = std::move(name);
    return *this;
  }

  [[nodiscard]] auto
  side() const noexcept -> ::carbon::MemorySide {
    return side_;
  }

  [[nodiscard]] auto
  gpu_buffers() noexcept -> ::carbon::Buffer<T>& {
    return gpu_buffers_;
  }

  [[nodiscard]] auto
  gpu_buffers() const noexcept -> const ::carbon::Buffer<T>& {
    return gpu_buffers_;
  }

  /** Host mirror buffers; empty for @c MemorySide::Device fields. */
  [[nodiscard]] auto
  cpu_buffers() noexcept -> ::carbon::Buffer<T>& {
    return cpu_buffers_;
  }

  [[nodiscard]] auto
  cpu_buffers() const noexcept -> const ::carbon::Buffer<T>& {
    return cpu_buffers_;
  }

  /** Per-GPU partitions bound to the device buffers (host buffers for Host-only fields). */
  [[nodiscard]] auto
  partitions() const noexcept -> const std::vector<Partition>& {
    return partitions_;
  }

  /** Per-GPU partitions bound to the host mirror; empty for @c MemorySide::Device fields. */
  [[nodiscard]] auto
  host_partitions() const noexcept -> const std::vector<Partition>& {
    return host_partitions_;
  }

  /**
   * Fill every lane on the host mirror with @p fn applied to the lane's parent cell in global
   * coordinates and the component — the SubGrid twin of @c Dense::Field::fill_owned_host .
   * Push to the device afterwards with a @c Transfer , as for Dense fields.
   *
   * @tparam Fn Callable as @c fn(carbon::int64_3d global, int component) returning @c T .
   * @throws std::invalid_argument when the field has no host mirror.
   */
  template <typename Fn>
  void
  fill_owned_host(Fn&& fn) {
    static_assert(std::is_invocable_r_v<T, Fn&, ::carbon::int64_3d, int>,
                  "SubGrid::Field::fill_owned_host expects a callable invocable as "
                  "fn(carbon::int64_3d global, int component) returning T.");
    const size_t n_gpus = grid_.gpu_count();
    if (cpu_buffers_.size() != n_gpus) {
      log::throw_with_log<std::invalid_argument>(
          "SubGrid::Field::fill_owned_host: field '{}' has no host mirror "
          "(cpu_buffers().size()={}, "
          "gpu_count={}). Construct it with MemorySide::Host or MemorySide::HostDevice.",
          name_, cpu_buffers_.size(), n_gpus);
    }
    for (size_t g = 0; g < n_gpus; ++g) {
      const int64_t n = grid_.count(g);
      T* h = cpu_buffers_[g].data();
      for (int c = 0; c < Components; ++c) {
        for (int64_t lane = 0; lane < n; ++lane) {
          h[static_cast<int64_t>(c) * n + lane] =
              static_cast<T>(fn(grid_.global_location(g, lane), c));
        }
      }
    }
  }

private:
  friend Grid;

  Field(Grid grid,
        ::carbon::Buffer<T> gpu_buffers,
        ::carbon::Buffer<T> cpu_buffers,
        std::vector<Partition> partitions,
        std::vector<Partition> host_partitions,
        ::carbon::MemorySide side,
        std::string name)
      : grid_(std::move(grid)),
        gpu_buffers_(std::move(gpu_buffers)),
        cpu_buffers_(std::move(cpu_buffers)),
        partitions_(std::move(partitions)),
        host_partitions_(std::move(host_partitions)),
        side_(side),
        name_(std::move(name)) {}

  Grid grid_{};
  ::carbon::Buffer<T> gpu_buffers_{};
  ::carbon::Buffer<T> cpu_buffers_{};
  std::vector<Partition> partitions_{};
  std::vector<Partition> host_partitions_{};
  ::carbon::MemorySide side_{::carbon::MemorySide::Device};
  std::string name_{};
};

}  // namespace carbon::detail::sub_grid
