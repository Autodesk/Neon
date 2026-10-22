#include <carbon/dense_disg.h>
#include <carbon/detail/dense/launch_shape.h>
#include <carbon/distributed.h>
#include <carbon/launch_shape.h>
#include <carbon/logging.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// DenseDisg is the disaggregated counterpart of carbon::Dense (port of Neon's
// Neon::domain::details::disaggregated::dGrid). The z-slab partitioning math is identical
// to Dense -- both grids split the global z extent across global_slab_count() slabs as
// evenly as possible -- so this TU mirrors src/dense.cpp closely. What differs is:
//
//   - The hard pin to halo_z ∈ {0, 1}: the disaggregated layout's five-region z map
//     (lower ghost / down boundary / internal / up boundary / upper ghost) does not
//     generalise to thicker boundaries without changing the region count, see
//     carbon::detail::dense_disg::Span class docs.
//   - Per-slab Span construction uses dense_disg::Span (not dense::Span); spans carry
//     the disg pitch formula via Span::linear_offset (multi-region), not the aggregated
//     SoA-on-cardinality stride.
//   - No Stencil object is stored: the disg layout's halo radius is fully determined by
//     halo_z ∈ {0, 1}, so a stencil descriptor is redundant here. Future stencil-aware
//     features (boundary mode, neighbour clipping) read off boundary_mode() instead.

namespace {

[[nodiscard]] auto
split_extent_z(int64_t nz,
               size_t num_gpus) -> std::vector<size_t> {
  if (nz < 0) {
    carbon::log::throw_with_log<std::invalid_argument>(
        "DenseDisg: global z extent must be non-negative (nz={})", nz);
  }
  if (static_cast<std::uint64_t>(nz) >
      static_cast<std::uint64_t>(std::numeric_limits<size_t>::max())) {
    carbon::log::throw_with_log<std::overflow_error>(
        "DenseDisg: global z extent too large for this platform (nz={})", nz);
  }
  const size_t nz_u = static_cast<size_t>(nz);
  if (num_gpus == 0U) {
    return {};
  }
  std::vector<size_t> counts(num_gpus);
  const size_t base = nz_u / num_gpus;
  const size_t rem = nz_u % num_gpus;
  for (size_t i = 0; i < num_gpus; ++i) {
    counts[i] = base + (i < rem ? 1U : 0U);
  }
  return counts;
}

}  // namespace

namespace carbon {

DenseDisg::DenseDisg(System system,
                     int64_3d global_extent,
                     int64_t halo_z,
                     BoundaryMode boundary_mode,
                     float32_3d origin,
                     float spacing)
    : global_(global_extent),
      system_(std::move(system)),
      halo_z_(halo_z),
      boundary_mode_(boundary_mode),
      origin_(origin),
      spacing_(spacing) {
  if (!std::isfinite(spacing) || spacing <= 0.0F) {
    log::throw_with_log<std::invalid_argument>(
        "DenseDisg: spacing must be finite and strictly positive (spacing={})", spacing);
  }
  if (!std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z)) {
    log::throw_with_log<std::invalid_argument>(
        "DenseDisg: origin components must be finite (origin=({}, {}, {}))", origin.x, origin.y,
        origin.z);
  }
  if (system_.gpu_group().size() == 0U) {
    log::throw_with_log<std::invalid_argument>(
        "DenseDisg: System has zero GPUs; cannot build a grid");
  }
  if (halo_z != 0 && halo_z != 1) {
    log::throw_with_log<std::invalid_argument>(
        "DenseDisg: halo_z must be 0 or 1 (the disaggregated layout pins boundary radius to 1); "
        "got halo_z={}",
        halo_z);
  }
  if (global_.x < 0 || global_.y < 0 || global_.z < 0) {
    log::throw_with_log<std::invalid_argument>(
        "DenseDisg: global x, y, z extents must be non-negative (x={}, y={}, z={})", global_.x,
        global_.y, global_.z);
  }
  const size_t local_n = system_.gpu_group().size();

  // Resolve cross-rank partitioning. Mirrors Dense::Dense; in MPI builds a Distributed-bound
  // System forces one rank per GPU, otherwise the global partition collapses to the local one.
#ifdef CARBON_HAVE_MPI_NCCL
  const Distributed* dist = system_.distributed();
  if (dist != nullptr) {
    if (local_n != 1U) {
      log::throw_with_log<std::invalid_argument>(
          "DenseDisg: distributed System must own exactly one local GPU per rank (got {})",
          local_n);
    }
    const int world_size = dist->world_size();
    const int world_rank = dist->world_rank();
    if (world_size <= 0 || world_rank < 0 || world_rank >= world_size) {
      log::throw_with_log<std::invalid_argument>(
          "DenseDisg: invalid Distributed state (world_size={}, world_rank={})", world_size,
          world_rank);
    }
    global_slab_count_ = static_cast<size_t>(world_size);
    local_slab_offset_ = static_cast<size_t>(world_rank);
  } else
#endif
  {
    global_slab_count_ = local_n;
    local_slab_offset_ = 0U;
  }

  const std::vector<size_t> z_counts_global = split_extent_z(global_.z, global_slab_count_);

  // Cumulative z up to (but not including) this rank's first local slab.
  int64_t running_z = 0;
  for (size_t g = 0; g < local_slab_offset_; ++g) {
    running_z += static_cast<int64_t>(z_counts_global[g]);
  }

  spans_ = system_.make_vector([&](size_t i) {
    const int64_t owned_z = static_cast<int64_t>(z_counts_global[local_slab_offset_ + i]);
    const int64_t z_off = running_z;
    running_z += owned_z;
    const int64_3d owned = global_.with_z(owned_z);
    const int64_3d haloed{owned.x, owned.y, owned.z + 2 * halo_z_};
    const int64_t xy_pitch = haloed.x * haloed.y;
    return detail::dense_disg::Span(haloed, global_, xy_pitch, z_off - halo_z_, halo_z_,
                                    boundary_mode_, origin_, spacing_);
  });

  // Reject geometries that are too thin for the Internal / Boundary split. With halo_z = 1
  // the internal band has Lz - 2 layers and the boundary spans 2; a slab with Lz < 2 makes
  // the boundary view's iter extent collide with the internal one. Same shape of check as
  // src/dense.cpp.
  if (halo_z_ > 0) {
    for (size_t i = 0; i < local_n; ++i) {
      const int64_t lz = spans_[i].logic_extent().z;
      if (lz < 2 * halo_z_) {
        log::throw_with_log<std::invalid_argument>(
            "DenseDisg: local slab z-extent {} is smaller than 2 * halo_z {}; the Internal / "
            "Boundary DataView split would overlap. Either grow the slab or reduce halo_z to 0",
            lz, 2 * halo_z_);
      }
    }
  }

  for (auto view : data_view_options()) {
    const auto vi = static_cast<size_t>(view);
    gpu_block_per_view_[vi] = system_.make_vector([&](size_t i) {
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

auto
DenseDisg::global_extent() const noexcept -> const int64_3d& {
  return global_;
}

auto
DenseDisg::halo_z() const noexcept -> int64_t {
  return halo_z_;
}

auto
DenseDisg::boundary_mode() const noexcept -> BoundaryMode {
  return boundary_mode_;
}

auto
DenseDisg::origin() const noexcept -> float32_3d {
  return origin_;
}

auto
DenseDisg::spacing() const noexcept -> float {
  return spacing_;
}

auto
DenseDisg::system() noexcept -> System& {
  return system_;
}

auto
DenseDisg::system() const noexcept -> const System& {
  return system_;
}

auto
DenseDisg::gpu_count() const noexcept -> size_t {
  return system_.gpu_group().size();
}

auto
DenseDisg::global_slab_count() const noexcept -> size_t {
  return global_slab_count_;
}

auto
DenseDisg::local_slab_offset() const noexcept -> size_t {
  return local_slab_offset_;
}

auto
DenseDisg::local_z(size_t gpu) const -> int64_t {
  if (gpu >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "DenseDisg::local_z: gpu index out of range (gpu={}, gpu_count={})", gpu, gpu_count());
  }
  return spans_[gpu].logic_extent().z;
}

auto
DenseDisg::z_offset_global(size_t gpu) const -> int64_t {
  if (gpu >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "DenseDisg::z_offset_global: gpu index out of range (gpu={}, gpu_count={})", gpu,
        gpu_count());
  }
  return spans_[gpu].z_offset() + spans_[gpu].halo_z();
}

}  // namespace carbon
