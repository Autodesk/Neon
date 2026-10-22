#include <carbon/dense.h>
#include <carbon/detail/dense/launch_shape.h>
#include <carbon/distributed.h>
#include <carbon/launch_shape.h>
#include <carbon/logging.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

[[nodiscard]] auto
split_extent_z(int64_t nz,
               size_t num_gpus) -> std::vector<size_t> {
  if (nz < 0) {
    carbon::log::throw_with_log<std::invalid_argument>(
        "Dense: global z extent must be non-negative (nz={})", nz);
  }
  if (static_cast<uint64_t>(nz) > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    carbon::log::throw_with_log<std::overflow_error>(
        "Dense: global z extent too large for this platform (nz={})", nz);
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

[[nodiscard]] auto
linear_count_for_slab(int64_t nx,
                      int64_t ny,
                      size_t nz_slab) -> size_t {
  if (nx < 0 || ny < 0) {
    carbon::log::throw_with_log<std::invalid_argument>(
        "Dense: global x and y extents must be non-negative (nx={}, ny={})", nx, ny);
  }
  const int64_t nx64 = nx;
  const int64_t ny64 = ny;
  const int64_t nz64 = static_cast<int64_t>(nz_slab);
  if (nz_slab > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    carbon::log::throw_with_log<std::overflow_error>("Dense: per-GPU z slab too large (nz_slab={})",
                                                     nz_slab);
  }
  const int64_t prod = nx64 * ny64;
  if (nx64 != 0 && prod / nx64 != ny64) {
    carbon::log::throw_with_log<std::overflow_error>("Dense: x*y overflow (nx={}, ny={})", nx64,
                                                     ny64);
  }
  const int64_t line = prod * nz64;
  if (prod != 0 && line / prod != nz64) {
    carbon::log::throw_with_log<std::overflow_error>(
        "Dense: linear element count overflow (nx={}, ny={}, nz_slab={})", nx64, ny64, nz_slab);
  }
  if (line < 0) {
    carbon::log::throw_with_log<std::overflow_error>(
        "Dense: negative linear element count (nx={}, ny={}, nz_slab={})", nx64, ny64, nz_slab);
  }
  if (static_cast<uint64_t>(line) > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    carbon::log::throw_with_log<std::overflow_error>(
        "Dense: linear element count too large for size_t (line={})", line);
  }
  return static_cast<size_t>(line);
}

}  // namespace

namespace carbon {

Dense::Dense(System system,
             int64_3d global_extent,
             Stencil stencil,
             BoundaryMode boundary_mode,
             float32_3d origin,
             float spacing,
             std::optional<LaunchShape> block_override)
    : global_(global_extent),
      system_(std::move(system)),
      stencil_(std::move(stencil)),
      boundary_mode_(boundary_mode),
      origin_(origin),
      spacing_(spacing) {
  if (!std::isfinite(spacing) || spacing <= 0.0F) {
    log::throw_with_log<std::invalid_argument>(
        "Dense: spacing must be finite and strictly positive (spacing={})", spacing);
  }
  if (!std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z)) {
    log::throw_with_log<std::invalid_argument>(
        "Dense: origin components must be finite (origin=({}, {}, {}))", origin.x, origin.y,
        origin.z);
  }
  if (system_.gpu_group().size() == 0U) {
    log::throw_with_log<std::invalid_argument>(
        "Dense: System has zero GPUs; cannot build a grid (matches the Block / DenseDisg "
        "policy -- empty systems fail at the grid ctor rather than at the first newField "
        "call, so the diagnostic points at the actual misconfiguration)");
  }
  if (global_.x < 0 || global_.y < 0 || global_.z < 0) {
    log::throw_with_log<std::invalid_argument>(
        "Dense: global x, y, z extents must be non-negative (x={}, y={}, z={})", global_.x,
        global_.y, global_.z);
  }
  const size_t local_n = system_.gpu_group().size();

  // Resolve the cross-rank partitioning. In a distributed build the global slab count equals
  // world_size and this rank's local slab offset equals world_rank (carbon's one-rank-per-GPU
  // model); in single-process mode the global partition collapses to the local one. The
  // Distributed methods are only linked when CARBON_ENABLE_MPI=ON; in non-MPI builds nobody can
  // construct a Distributed (its ctor lives in src/distributed.cpp), so the pointer is
  // unconditionally null and the gate below short-circuits. Guard the call site on
  // CARBON_HAVE_MPI_NCCL so the linker has no dangling symbols in OFF builds.
#ifdef CARBON_HAVE_MPI_NCCL
  const Distributed* dist = system_.distributed();
  if (dist != nullptr) {
    if (local_n != 1U) {
      log::throw_with_log<std::invalid_argument>(
          "Dense: distributed System must own exactly one local GPU per rank (got {}); carbon's "
          "distributed model is one rank per GPU",
          local_n);
    }
    const int world_size = dist->world_size();
    const int world_rank = dist->world_rank();
    if (world_size <= 0 || world_rank < 0 || world_rank >= world_size) {
      log::throw_with_log<std::invalid_argument>(
          "Dense: invalid Distributed state (world_size={}, world_rank={})", world_size,
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

  // Validate per-slab owned-element counts up front so any overflow surfaces during Dense
  // construction rather than later in a buffer allocation. The values themselves are not
  // cached: linear_elements() recomputes them on demand from spans_.
  for (size_t g = 0; g < global_slab_count_; ++g) {
    (void)linear_count_for_slab(global_.x, global_.y, z_counts_global[g]);
  }

  // Cumulative z up to (but not including) this rank's first local slab — i.e. the global z
  // index of @c local_slab_offset_'s first owned layer. Span::z_offset is then the running
  // accumulator minus halo_z (lower-halo origin).
  int64_t running_z = 0;
  for (size_t g = 0; g < local_slab_offset_; ++g) {
    running_z += static_cast<int64_t>(z_counts_global[g]);
  }

  // Prefix scan over the @b local subset of z_counts_global. System::make_vector invokes the
  // callback in ascending order 0..local_n-1; the captured accumulator continues across local
  // slabs without resetting.
  const int64_t halo_z = static_cast<int64_t>(stencil_.radius());
  spans_ = system_.make_vector([&](size_t i) {
    const int64_t owned_z = static_cast<int64_t>(z_counts_global[local_slab_offset_ + i]);
    const int64_t z_off = running_z;
    running_z += owned_z;
    const int64_3d owned = global_.with_z(owned_z);
    const int64_3d haloed{owned.x, owned.y, owned.z + 2 * halo_z};
    // No XY pitch padding today; use the minimum compact pitch from the haloed extent.
    const int64_t xy_pitch = haloed.x * haloed.y;
    return detail::dense::Span(haloed, global_, xy_pitch, z_off - halo_z, halo_z, boundary_mode_,
                               origin_, spacing_);
  });

  // The Internal / Boundary split partitions the owned z-extent into a lower halo-thick band,
  // an interior of (Lz - 2*halo_z) layers, and an upper halo-thick band. If any local slab is
  // thinner than 2*halo_z those bands would overlap, so reject the configuration up front
  // rather than letting set_and_validate quietly double-write cells.
  if (halo_z > 0) {
    for (size_t i = 0; i < local_n; ++i) {
      const int64_t lz = spans_[i].logic_extent().z;
      if (lz < 2 * halo_z) {
        log::throw_with_log<std::invalid_argument>(
            "Dense: local slab z-extent {} is smaller than 2 * halo_z {} (stencil radius {}); "
            "the Internal / Boundary DataView split would overlap. Either grow the slab or "
            "shrink the stencil",
            lz, 2 * halo_z, stencil_.radius());
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
    // Each view has its own iteration extent; compute_dense_grid maps that extent to a launch
    // grid using the chosen block. With halo_z == 0 the Internal extent equals Standard's
    // and Boundary's z is 0 (a 0-thread launch), so the no-stencil path stays unchanged.
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
Dense::global_extent() const noexcept -> const int64_3d& {
  return global_;
}

auto
Dense::stencil() const noexcept -> const Stencil& {
  return stencil_;
}

auto
Dense::boundary_mode() const noexcept -> BoundaryMode {
  return boundary_mode_;
}

auto
Dense::origin() const noexcept -> float32_3d {
  return origin_;
}

auto
Dense::spacing() const noexcept -> float {
  return spacing_;
}

auto
Dense::system() noexcept -> System& {
  return system_;
}

auto
Dense::system() const noexcept -> const System& {
  return system_;
}

auto
Dense::gpu_count() const noexcept -> size_t {
  return system_.gpu_group().size();
}

auto
Dense::global_slab_count() const noexcept -> size_t {
  return global_slab_count_;
}

auto
Dense::local_slab_offset() const noexcept -> size_t {
  return local_slab_offset_;
}

auto
Dense::local_z(size_t gpu) const -> int64_t {
  if (gpu >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "Dense::local_z: gpu index out of range (gpu={}, gpu_count={})", gpu, gpu_count());
  }
  return spans_[gpu].logic_extent().z;
}

auto
Dense::z_offset_global(size_t gpu) const -> int64_t {
  if (gpu >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "Dense::z_offset_global: gpu index out of range (gpu={}, gpu_count={})", gpu, gpu_count());
  }
  // Span::z_offset() is the start of the lower halo; the first owned layer is halo_z above it.
  return spans_[gpu].z_offset() + spans_[gpu].halo_z();
}

auto
Dense::local_extent(size_t gpu) const -> int64_3d {
  return global_.with_z(local_z(gpu));
}

auto
Dense::linear_elements(size_t gpu) const -> size_t {
  if (gpu >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "Dense::linear_elements: gpu index out of range (gpu={}, gpu_count={})", gpu, gpu_count());
  }
  return linear_count_for_slab(global_.x, global_.y, static_cast<size_t>(local_z(gpu)));
}

auto
Dense::linear_counts_per_gpu() const -> std::vector<size_t> {
  return system_.make_vector([&](size_t i) {
    return linear_elements(i);
  });
}

}  // namespace carbon
