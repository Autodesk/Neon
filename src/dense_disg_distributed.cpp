#include <carbon/dense_disg_distributed.h>
#include <carbon/detail/dense/launch_shape.h>
#include <carbon/detail/dense_disg_distributed/layout.h>
#include <carbon/detail/dense_distributed/layout.h>
#include <carbon/distributed.h>
#include <carbon/launch_shape.h>
#include <carbon/logging.h>
#include <carbon/process_grid.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

// DenseDisgDistributed is DenseDisg's disaggregated storage behind DenseDistributed's ProcessGrid
// construction surface. Slab, Pencil and Brick process grids are accepted; the per-chunk split goes
// through dense_distributed::rank_chunk, whose rule (earlier coordinates get the remainder) is the
// same as DenseDisg's split_extent_z, so every slab span matches DenseDisg for a radius-1 stencil.

namespace {

[[nodiscard]] auto
resolve_world_size(const carbon::System& system) -> int {
#ifdef CARBON_HAVE_MPI_NCCL
  if (const carbon::Distributed* dist = system.distributed(); dist != nullptr) {
    return dist->world_size();
  }
#endif
  return static_cast<int>(system.gpu_group().size());
}

[[nodiscard]] auto
resolve_world_rank([[maybe_unused]] const carbon::System& system) -> int {
#ifdef CARBON_HAVE_MPI_NCCL
  if (const carbon::Distributed* dist = system.distributed(); dist != nullptr) {
    return dist->world_rank();
  }
#endif
  return 0;
}

[[nodiscard]] auto
linear_count_for_slab(int64_t nx,
                      int64_t ny,
                      int64_t nz) -> size_t {
  if (nx < 0 || ny < 0 || nz < 0) {
    carbon::log::throw_with_log<std::invalid_argument>(
        "DenseDisgDistributed: extents must be non-negative (nx={}, ny={}, nz={})", nx, ny, nz);
  }
  const int64_t prod = nx * ny;
  if (nx != 0 && prod / nx != ny) {
    carbon::log::throw_with_log<std::overflow_error>(
        "DenseDisgDistributed: x*y overflow (nx={}, ny={})", nx, ny);
  }
  const int64_t line = prod * nz;
  if (prod != 0 && line / prod != nz) {
    carbon::log::throw_with_log<std::overflow_error>(
        "DenseDisgDistributed: linear element count overflow (nx={}, ny={}, nz={})", nx, ny, nz);
  }
  if (static_cast<uint64_t>(line) > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    carbon::log::throw_with_log<std::overflow_error>(
        "DenseDisgDistributed: linear element count too large for size_t (line={})", line);
  }
  return static_cast<size_t>(line);
}

}  // namespace

namespace carbon {

void
DenseDisgDistributed::init_common_(System system,
                                   int64_3d global_extent,
                                   Stencil stencil,
                                   BoundaryModeDistributed boundary_mode,
                                   float32_3d origin,
                                   float spacing,
                                   ProcessGrid process_grid,
                                   std::optional<LaunchShape> block_override) {
  if (!std::isfinite(spacing) || spacing <= 0.0F) {
    log::throw_with_log<std::invalid_argument>(
        "DenseDisgDistributed: spacing must be finite and strictly positive (spacing={})", spacing);
  }
  if (!std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z)) {
    log::throw_with_log<std::invalid_argument>(
        "DenseDisgDistributed: origin components must be finite (origin=({}, {}, {}))", origin.x,
        origin.y, origin.z);
  }
  if (system.gpu_group().size() == 0U) {
    log::throw_with_log<std::invalid_argument>(
        "DenseDisgDistributed: System has zero GPUs; cannot build a grid");
  }
  if (global_extent.x < 0 || global_extent.y < 0 || global_extent.z < 0) {
    log::throw_with_log<std::invalid_argument>(
        "DenseDisgDistributed: global x, y, z extents must be non-negative (x={}, y={}, z={})",
        global_extent.x, global_extent.y, global_extent.z);
  }
  const int64_t halo_z = detail::dense_disg_distributed::halo_z_from_stencil(stencil);
  (void)detail::dense_disg_distributed::to_boundary_mode(boundary_mode);
  const int64_t ghost_x = detail::dense_disg_distributed::ghost_x_for(process_grid);
  const int64_t ghost_y = detail::dense_disg_distributed::ghost_y_for(process_grid, boundary_mode);

  global_ = global_extent;
  system_ = std::move(system);
  stencil_ = std::move(stencil);
  halo_z_ = halo_z;
  halo_y_ = ghost_y * halo_z;
  halo_x_ = ghost_x * halo_z;
  boundary_mode_ = boundary_mode;
  origin_ = origin;
  spacing_ = spacing;
  process_grid_ = std::move(process_grid);

  const size_t local_n = system_.gpu_group().size();
  const int world_rank = resolve_world_rank(system_);
  bool distributed = false;

#ifdef CARBON_HAVE_MPI_NCCL
  const Distributed* dist = system_.distributed();
  distributed = dist != nullptr;
  if (dist != nullptr) {
    const int world_size = dist->world_size();
    if (local_n != 1U) {
      log::throw_with_log<std::invalid_argument>(
          "DenseDisgDistributed: distributed System must own exactly one local GPU per rank "
          "(got {})",
          local_n);
    }
    if (world_size <= 0 || world_rank < 0 || world_rank >= world_size) {
      log::throw_with_log<std::invalid_argument>(
          "DenseDisgDistributed: invalid Distributed state (world_size={}, world_rank={})",
          world_size, world_rank);
    }
    if (process_grid_.world_size() != world_size) {
      log::throw_with_log<std::invalid_argument>(
          "DenseDisgDistributed: ProcessGrid world_size ({}) does not match Distributed "
          "world_size ({})",
          process_grid_.world_size(), world_size);
    }
  } else
#endif
  {
    if (process_grid_.world_size() != static_cast<int>(local_n)) {
      log::throw_with_log<std::invalid_argument>(
          "DenseDisgDistributed: ProcessGrid world_size ({}) does not match local GPU count ({})",
          process_grid_.world_size(), local_n);
    }
  }

  const int32_3d dims = process_grid_.dims();
  const int32_3d coords = process_grid_.coords_of(world_rank);
  global_slab_count_ = static_cast<size_t>(dims.z);
  local_slab_offset_ = distributed ? static_cast<size_t>(coords.z) : 0U;
  local_rank_offset_ = distributed ? world_rank : 0;

  const size_t max_nx =
      detail::dense_distributed::split_extent(global_.x, static_cast<size_t>(dims.x)).front();
  const size_t max_ny =
      detail::dense_distributed::split_extent(global_.y, static_cast<size_t>(dims.y)).front();
  for (const size_t nz :
       detail::dense_distributed::split_extent(global_.z, static_cast<size_t>(dims.z))) {
    (void)linear_count_for_slab(
        static_cast<int64_t>(max_nx) +
            ghost_x * (2 + detail::dense_disg_distributed::kInternalPitchAlign),
        static_cast<int64_t>(max_ny) + 2 * ghost_y,
        static_cast<int64_t>(nz) + 2 * detail::dense_disg_distributed::kGhostRadius + 8 * ghost_y);
  }

  spans_ = system_.make_vector([&](size_t i) {
    const int32_3d gpu_coords =
        distributed ? coords : process_grid_.coords_of(static_cast<int32_t>(i));
    const detail::dense_distributed::RankChunk chunk = detail::dense_disg_distributed::disg_chunk(
        global_, process_grid_, gpu_coords, halo_z_, boundary_mode_);
    return Span(chunk.haloed, global_, chunk.xy_pitch, chunk.offset, chunk.halo, boundary_mode_,
                origin_, spacing_, ghost_y, ghost_x);
  });

  for (size_t i = 0; i < local_n; ++i) {
    const int64_3d owned = spans_[i].logic_extent();
    if (ghost_x != 0 && owned.x < 2 * Span::boundary_radius) {
      log::throw_with_log<std::invalid_argument>(
          "DenseDisgDistributed: local x-extent {} is smaller than 2 * boundary_radius {}; the "
          "brick layout needs distinct x boundary columns. Grow the global x extent or use fewer "
          "x ranks",
          owned.x, 2 * Span::boundary_radius);
    }
    const int64_t units = spans_[i].per_card_elements() / spans_[i].storage_unit();
    if (units > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      log::throw_with_log<std::overflow_error>(
          "DenseDisgDistributed: chunk {} needs {} storage units per component, more than the "
          "32-bit region fields of Idx can address; use more ranks",
          i, units);
    }
    if (owned.z < 2 * Span::boundary_radius) {
      log::throw_with_log<std::invalid_argument>(
          "DenseDisgDistributed: local z-extent {} is smaller than 2 * boundary_radius {}; "
          "the disaggregated layout needs distinct down / up boundary slabs. Grow the global z "
          "extent or use fewer z ranks",
          owned.z, 2 * Span::boundary_radius);
    }
    if (ghost_y != 0 && owned.y < 2 * Span::boundary_radius) {
      log::throw_with_log<std::invalid_argument>(
          "DenseDisgDistributed: local y-extent {} is smaller than 2 * boundary_radius {}; the "
          "pencil layout needs distinct y boundary rows. Grow the global y extent or use fewer "
          "y ranks",
          owned.y, 2 * Span::boundary_radius);
    }
  }

  for (auto view : data_view_options()) {
    const auto vi = static_cast<size_t>(view);
    gpu_block_per_view_[vi] = system_.make_vector([&](size_t i) {
      if (block_override.has_value()) {
        return *block_override;
      }
      return dense_make_kernel_block(system_.gpu_group().devices()[i]);
    });
    gpu_grid_per_view_[vi] = system_.make_vector([&](size_t i) {
      const detail::Gpu& gpu = system_.gpu_group().devices()[i];
      LaunchShape grid;
      detail::compute_dense_grid(spans_[i].iter_extent(view), gpu_block_per_view_[vi][i],
                                 gpu.properties().max_threads_per_block, grid);
      return grid;
    });
  }
}

DenseDisgDistributed::DenseDisgDistributed(System system,
                                           int64_3d global_extent,
                                           Stencil stencil,
                                           BoundaryModeDistributed boundary_mode,
                                           float32_3d origin,
                                           float spacing,
                                           std::optional<LaunchShape> block_override)
    : DenseDisgDistributed() {
  const int world_size = resolve_world_size(system);
  const ProcessGrid pg =
      world_size > 0 ? ProcessGrid::make(world_size, DecompositionMode::Slab) : ProcessGrid{};
  init_common_(std::move(system), global_extent, std::move(stencil), boundary_mode, origin, spacing,
               pg, block_override);
}

DenseDisgDistributed::DenseDisgDistributed(System system,
                                           int64_3d global_extent,
                                           DecompositionMode decomposition,
                                           Stencil stencil,
                                           BoundaryModeDistributed boundary_mode,
                                           float32_3d origin,
                                           float spacing,
                                           std::optional<LaunchShape> block_override)
    : DenseDisgDistributed() {
  const int world_size = resolve_world_size(system);
  const ProcessGrid pg =
      world_size > 0 ? ProcessGrid::make(world_size, decomposition) : ProcessGrid{};
  init_common_(std::move(system), global_extent, std::move(stencil), boundary_mode, origin, spacing,
               pg, block_override);
}

DenseDisgDistributed::DenseDisgDistributed(System system,
                                           int64_3d global_extent,
                                           ProcessGrid process_grid,
                                           Stencil stencil,
                                           BoundaryModeDistributed boundary_mode,
                                           float32_3d origin,
                                           float spacing,
                                           std::optional<LaunchShape> block_override)
    : DenseDisgDistributed() {
  init_common_(std::move(system), global_extent, std::move(stencil), boundary_mode, origin, spacing,
               std::move(process_grid), block_override);
}

auto
DenseDisgDistributed::process_grid() const noexcept -> const ProcessGrid& {
  return process_grid_;
}

auto
DenseDisgDistributed::global_extent() const noexcept -> const int64_3d& {
  return global_;
}

auto
DenseDisgDistributed::stencil() const noexcept -> const Stencil& {
  return stencil_;
}

auto
DenseDisgDistributed::halo_z() const noexcept -> int64_t {
  return halo_z_;
}

auto
DenseDisgDistributed::halo_y() const noexcept -> int64_t {
  return halo_y_;
}

auto
DenseDisgDistributed::halo_x() const noexcept -> int64_t {
  return halo_x_;
}

auto
DenseDisgDistributed::boundary_mode() const noexcept -> BoundaryModeDistributed {
  return boundary_mode_;
}

auto
DenseDisgDistributed::origin() const noexcept -> float32_3d {
  return origin_;
}

auto
DenseDisgDistributed::spacing() const noexcept -> float {
  return spacing_;
}

auto
DenseDisgDistributed::system() noexcept -> System& {
  return system_;
}

auto
DenseDisgDistributed::system() const noexcept -> const System& {
  return system_;
}

auto
DenseDisgDistributed::gpu_count() const noexcept -> size_t {
  return system_.gpu_group().size();
}

auto
DenseDisgDistributed::global_slab_count() const noexcept -> size_t {
  return global_slab_count_;
}

auto
DenseDisgDistributed::local_slab_offset() const noexcept -> size_t {
  return local_slab_offset_;
}

auto
DenseDisgDistributed::local_rank_offset() const noexcept -> int32_t {
  return local_rank_offset_;
}

auto
DenseDisgDistributed::local_z(size_t gpu) const -> int64_t {
  if (gpu >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "DenseDisgDistributed::local_z: gpu index out of range (gpu={}, gpu_count={})", gpu,
        gpu_count());
  }
  return spans_[gpu].logic_extent().z;
}

auto
DenseDisgDistributed::z_offset_global(size_t gpu) const -> int64_t {
  if (gpu >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "DenseDisgDistributed::z_offset_global: gpu index out of range (gpu={}, gpu_count={})", gpu,
        gpu_count());
  }
  return spans_[gpu].z_offset() + Span::ghost_radius;
}

auto
DenseDisgDistributed::local_extent(size_t gpu) const -> int64_3d {
  if (gpu >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "DenseDisgDistributed::local_extent: gpu index out of range (gpu={}, gpu_count={})", gpu,
        gpu_count());
  }
  return spans_[gpu].logic_extent();
}

auto
DenseDisgDistributed::linear_elements(size_t gpu) const -> size_t {
  if (gpu >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "DenseDisgDistributed::linear_elements: gpu index out of range (gpu={}, gpu_count={})", gpu,
        gpu_count());
  }
  const int64_3d owned = spans_[gpu].logic_extent();
  return linear_count_for_slab(owned.x, owned.y, owned.z);
}

auto
DenseDisgDistributed::linear_counts_per_gpu() const -> std::vector<size_t> {
  return system_.make_vector([&](size_t i) {
    return linear_elements(i);
  });
}

}  // namespace carbon
