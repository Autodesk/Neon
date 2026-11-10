#include <carbon/dense_distributed.h>
#include <carbon/detail/dense/launch_shape.h>
#include <carbon/detail/dense_distributed/layout.h>
#include <carbon/distributed.h>
#include <carbon/launch_shape.h>
#include <carbon/logging.h>

#include <carbon/process_grid.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using carbon::detail::dense_distributed::split_extent;

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
resolve_world_rank(const carbon::System& system) -> int {
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
                      size_t nz_slab) -> size_t {
  if (nx < 0 || ny < 0) {
    carbon::log::throw_with_log<std::invalid_argument>(
        "DenseDistributed: global x and y extents must be non-negative (nx={}, ny={})", nx, ny);
  }
  const int64_t nx64 = nx;
  const int64_t ny64 = ny;
  const int64_t nz64 = static_cast<int64_t>(nz_slab);
  if (nz_slab > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    carbon::log::throw_with_log<std::overflow_error>(
        "DenseDistributed: per-GPU z slab too large (nz_slab={})", nz_slab);
  }
  const int64_t prod = nx64 * ny64;
  if (nx64 != 0 && prod / nx64 != ny64) {
    carbon::log::throw_with_log<std::overflow_error>(
        "DenseDistributed: x*y overflow (nx={}, ny={})", nx64, ny64);
  }
  const int64_t line = prod * nz64;
  if (prod != 0 && line / prod != nz64) {
    carbon::log::throw_with_log<std::overflow_error>(
        "DenseDistributed: linear element count overflow (nx={}, ny={}, nz_slab={})", nx64, ny64,
        nz_slab);
  }
  if (line < 0) {
    carbon::log::throw_with_log<std::overflow_error>(
        "DenseDistributed: negative linear element count (nx={}, ny={}, nz_slab={})", nx64, ny64,
        nz_slab);
  }
  if (static_cast<uint64_t>(line) > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    carbon::log::throw_with_log<std::overflow_error>(
        "DenseDistributed: linear element count too large for size_t (line={})", line);
  }
  return static_cast<size_t>(line);
}

}  // namespace

namespace carbon {
namespace {
// Validates a BoundaryConditions spec against the grid it is attached to. A face without a
// condition (BcKind::None) is never checked, so the default spec is always accepted.
void
validate_boundary_conditions(const BoundaryConditions& bc,
                             BoundaryModeDistributed boundary_mode,
                             const int64_3d& global_extent,
                             const Stencil& stencil) {
  constexpr const char* kFaceName[kFaceCount] = {"XLow", "XHigh", "YLow", "YHigh", "ZLow", "ZHigh"};
  const Axis axis_of_face[kFaceCount] = {Axis::X, Axis::X, Axis::Y, Axis::Y, Axis::Z, Axis::Z};
  const int64_t extent_of_axis[3] = {global_extent.x, global_extent.y, global_extent.z};
  const int32_3d radius = stencil.radius3();
  const int64_t radius_of_axis[3] = {radius.x, radius.y, radius.z};
  for (int f = 0; f < kFaceCount; ++f) {
    const FaceBc& fbc = bc.faces[f];
    if (fbc.kind == BcKind::None) {
      continue;
    }
    const Axis axis = axis_of_face[f];
    const int a = static_cast<int>(axis);
    if (is_periodic(boundary_mode, axis)) {
      log::throw_with_log<std::invalid_argument>(
          "DenseDistributed: boundary condition on face {} conflicts with a periodic boundary on "
          "that axis; use either a periodic axis or a per-face condition, not both",
          kFaceName[f]);
    }
    if (!std::isfinite(fbc.value)) {
      log::throw_with_log<std::invalid_argument>(
          "DenseDistributed: boundary condition on face {} must have a finite value (got {})",
          kFaceName[f], fbc.value);
    }
    if (extent_of_axis[a] <= radius_of_axis[a]) {
      log::throw_with_log<std::invalid_argument>(
          "DenseDistributed: boundary condition on face {} needs a global extent larger than the "
          "stencil radius on that axis so every mirror cell is inside the domain (extent={}, "
          "radius={})",
          kFaceName[f], extent_of_axis[a], radius_of_axis[a]);
    }
  }
}

}  // namespace

void
DenseDistributed::init_common_(System system,
                               int64_3d global_extent,
                               Stencil stencil,
                               BoundaryModeDistributed boundary_mode,
                               float32_3d origin,
                               float spacing,
                               ProcessGrid process_grid,
                               std::optional<LaunchShape> block_override,
                               BoundaryConditions boundary_conditions) {
  if (!std::isfinite(spacing) || spacing <= 0.0F) {
    log::throw_with_log<std::invalid_argument>(
        "DenseDistributed: spacing must be finite and strictly positive (spacing={})", spacing);
  }
  if (!std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z)) {
    log::throw_with_log<std::invalid_argument>(
        "DenseDistributed: origin components must be finite (origin=({}, {}, {}))", origin.x,
        origin.y, origin.z);
  }
  if (system.gpu_group().size() == 0U) {
    log::throw_with_log<std::invalid_argument>(
        "DenseDistributed: System has zero GPUs; cannot build a grid (matches the Block / "
        "DenseDisg "
        "policy -- empty systems fail at the grid ctor rather than at the first newField "
        "call, so the diagnostic points at the actual misconfiguration)");
  }
  if (global_extent.x < 0 || global_extent.y < 0 || global_extent.z < 0) {
    log::throw_with_log<std::invalid_argument>(
        "DenseDistributed: global x, y, z extents must be non-negative (x={}, y={}, z={})",
        global_extent.x, global_extent.y, global_extent.z);
  }

  validate_boundary_conditions(boundary_conditions, boundary_mode, global_extent, stencil);

  global_ = global_extent;
  system_ = std::move(system);
  stencil_ = std::move(stencil);
  boundary_mode_ = boundary_mode;
  boundary_conditions_ = boundary_conditions;
  origin_ = origin;
  spacing_ = spacing;
  process_grid_ = std::move(process_grid);

  const size_t local_n = system_.gpu_group().size();
  const int world_size = resolve_world_size(system_);
  const int world_rank = resolve_world_rank(system_);
  bool distributed = false;

#ifdef CARBON_HAVE_MPI_NCCL
  const Distributed* dist = system_.distributed();
  distributed = dist != nullptr;
  if (dist != nullptr) {
    if (local_n != 1U) {
      log::throw_with_log<std::invalid_argument>(
          "DenseDistributed: distributed System must own exactly one local GPU per rank (got {}); "
          "carbon's "
          "distributed model is one rank per GPU",
          local_n);
    }
    if (world_size <= 0 || world_rank < 0 || world_rank >= world_size) {
      log::throw_with_log<std::invalid_argument>(
          "DenseDistributed: invalid Distributed state (world_size={}, world_rank={})", world_size,
          world_rank);
    }
    if (process_grid_.world_size() != world_size) {
      log::throw_with_log<std::invalid_argument>(
          "DenseDistributed: ProcessGrid world_size ({}) does not match Distributed world_size "
          "({})",
          process_grid_.world_size(), world_size);
    }
  } else
#endif
  {
    if (process_grid_.world_size() != static_cast<int>(local_n)) {
      log::throw_with_log<std::invalid_argument>(
          "DenseDistributed: ProcessGrid world_size ({}) does not match local GPU count ({})",
          process_grid_.world_size(), local_n);
    }
  }

  const int32_3d dims = process_grid_.dims();
  if (dims.x <= 0 || dims.y <= 0 || dims.z <= 0) {
    log::throw_with_log<std::invalid_argument>(
        "DenseDistributed: ProcessGrid dimensions must be positive (dims=({}, {}, {}))", dims.x,
        dims.y, dims.z);
  }
  if (local_n > 1U && (dims.x > 1 || dims.y > 1)) {
    log::throw_with_log<std::invalid_argument>(
        "DenseDistributed: single-process Pencil/Brick with multiple local GPUs is not supported; "
        "use MPI with one rank per GPU (local GPUs={}, dims=({}, {}, {}))",
        local_n, dims.x, dims.y, dims.z);
  }
  const int32_3d coords = process_grid_.coords_of(world_rank);
  global_slab_count_ = static_cast<size_t>(dims.z);
  local_slab_offset_ = distributed ? static_cast<size_t>(coords.z) : 0U;

  // Validate every global chunk's owned-element count before allocating fields.
  for (const size_t nx : split_extent(global_.x, static_cast<size_t>(dims.x))) {
    for (const size_t ny : split_extent(global_.y, static_cast<size_t>(dims.y))) {
      for (const size_t nz : split_extent(global_.z, static_cast<size_t>(dims.z))) {
        (void)linear_count_for_slab(static_cast<int64_t>(nx), static_cast<int64_t>(ny), nz);
      }
    }
  }

  const bool legacy_slab = detail::dense_distributed::is_legacy_slab(process_grid_, boundary_mode_);
  const int64_3d halo =
      detail::dense_distributed::halo_extent(process_grid_, boundary_mode_, stencil_);

  spans_ = system_.make_vector([&](size_t i) {
    const int32_3d gpu_coords =
        distributed ? coords : process_grid_.coords_of(static_cast<int32_t>(i));
    const detail::dense_distributed::RankChunk chunk =
        detail::dense_distributed::rank_chunk(global_, process_grid_, gpu_coords, halo);

    return detail::dense_distributed::Span(chunk.haloed, global_, chunk.xy_pitch, chunk.offset,
                                           chunk.halo, boundary_mode_, origin_, spacing_,
                                           boundary_conditions_);
  });

  // The legacy Slab Boundary is still two z-bands; keep its old overlap guard.
  // Multi-axis Stage-A Boundary is full-owned and rejects internal cells instead.
  if (legacy_slab && halo.z > 0) {
    for (size_t i = 0; i < local_n; ++i) {
      const int64_t lz = spans_[i].logic_extent().z;
      if (lz < 2 * halo.z) {
        log::throw_with_log<std::invalid_argument>(
            "DenseDistributed: local slab z-extent {} is smaller than 2 * halo_z {} (stencil "
            "radius {}); "
            "the Internal / Boundary DataView split would overlap. Either grow the slab or "
            "shrink the stencil",
            lz, 2 * halo.z, stencil_.radius());
      }
    }
  }

  if (!legacy_slab) {
    for (size_t i = 0; i < local_n; ++i) {
      const int64_3d owned = spans_[i].logic_extent();
      const int64_3d h = spans_[i].halo();

      if ((h.x > 0 && owned.x < h.x) || (h.y > 0 && owned.y < h.y) || (h.z > 0 && owned.z < h.z)) {
        log::throw_with_log<std::invalid_argument>(
            "DenseDistributed: local owned extent ({}, {}, {}) is smaller than halo ({}, {}, {}) "
            "on a "
            "multi-axis decomposition; the halo would overlap the owned region",
            owned.x, owned.y, owned.z, h.x, h.y, h.z);
      }
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

DenseDistributed::DenseDistributed(System system,
                                   int64_3d global_extent,
                                   Stencil stencil,
                                   BoundaryModeDistributed boundary_mode,
                                   float32_3d origin,
                                   float spacing,
                                   std::optional<LaunchShape> block_override,
                                   BoundaryConditions boundary_conditions)
    : DenseDistributed() {
  const int world_size = resolve_world_size(system);
  const ProcessGrid pg =
      world_size > 0 ? ProcessGrid::make(world_size, DecompositionMode::Slab) : ProcessGrid{};
  init_common_(std::move(system), global_extent, std::move(stencil), boundary_mode, origin, spacing,
               pg, block_override, boundary_conditions);
}

DenseDistributed::DenseDistributed(System system,
                                   int64_3d global_extent,
                                   DecompositionMode decomposition,
                                   Stencil stencil,
                                   BoundaryModeDistributed boundary_mode,
                                   float32_3d origin,
                                   float spacing,
                                   std::optional<LaunchShape> block_override,
                                   BoundaryConditions boundary_conditions)
    : DenseDistributed() {
  const int world_size = resolve_world_size(system);
  const ProcessGrid pg =
      world_size > 0 ? ProcessGrid::make(world_size, decomposition) : ProcessGrid{};
  init_common_(std::move(system), global_extent, std::move(stencil), boundary_mode, origin, spacing,
               pg, block_override, boundary_conditions);
}

DenseDistributed::DenseDistributed(System system,
                                   int64_3d global_extent,
                                   ProcessGrid process_grid,
                                   Stencil stencil,
                                   BoundaryModeDistributed boundary_mode,
                                   float32_3d origin,
                                   float spacing,
                                   std::optional<LaunchShape> block_override,
                                   BoundaryConditions boundary_conditions)
    : DenseDistributed() {
  init_common_(std::move(system), global_extent, std::move(stencil), boundary_mode, origin, spacing,
               std::move(process_grid), block_override, boundary_conditions);
}

auto
DenseDistributed::process_grid() const noexcept -> const ProcessGrid& {
  return process_grid_;
}

auto
DenseDistributed::local_extent(size_t gpu) const -> int64_3d {
  if (gpu >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "DenseDistributed::local_extent: gpu index out of range (gpu={}, gpu_count={})", gpu,
        gpu_count());
  }
  return spans_[gpu].logic_extent();
}

auto
DenseDistributed::linear_elements(size_t gpu) const -> size_t {
  if (gpu >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "DenseDistributed::linear_elements: gpu index out of range (gpu={}, gpu_count={})", gpu,
        gpu_count());
  }
  const int64_3d owned = spans_[gpu].logic_extent();
  return linear_count_for_slab(owned.x, owned.y, static_cast<size_t>(owned.z));
}

auto
DenseDistributed::global_extent() const noexcept -> const int64_3d& {
  return global_;
}

auto
DenseDistributed::stencil() const noexcept -> const Stencil& {
  return stencil_;
}

auto
DenseDistributed::boundary_mode() const noexcept -> BoundaryModeDistributed {
  return boundary_mode_;
}

auto
DenseDistributed::boundary_conditions() const noexcept -> const BoundaryConditions& {
  return boundary_conditions_;
}

auto
DenseDistributed::origin() const noexcept -> float32_3d {
  return origin_;
}

auto
DenseDistributed::spacing() const noexcept -> float {
  return spacing_;
}

auto
DenseDistributed::system() noexcept -> System& {
  return system_;
}

auto
DenseDistributed::system() const noexcept -> const System& {
  return system_;
}

auto
DenseDistributed::gpu_count() const noexcept -> size_t {
  return system_.gpu_group().size();
}

auto
DenseDistributed::global_slab_count() const noexcept -> size_t {
  return global_slab_count_;
}

auto
DenseDistributed::local_slab_offset() const noexcept -> size_t {
  return local_slab_offset_;
}

auto
DenseDistributed::local_z(size_t gpu) const -> int64_t {
  if (gpu >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "DenseDistributed::local_z: gpu index out of range (gpu={}, gpu_count={})", gpu,
        gpu_count());
  }
  return spans_[gpu].logic_extent().z;
}

auto
DenseDistributed::z_offset_global(size_t gpu) const -> int64_t {
  if (gpu >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "DenseDistributed::z_offset_global: gpu index out of range (gpu={}, gpu_count={})", gpu,
        gpu_count());
  }
  // Span::z_offset() is the start of the lower halo; the first owned layer is halo_z above it.
  return spans_[gpu].z_offset() + spans_[gpu].halo_z();
}

auto
DenseDistributed::linear_counts_per_gpu() const -> std::vector<size_t> {
  return system_.make_vector([&](size_t i) {
    return linear_elements(i);
  });
}

}  // namespace carbon
