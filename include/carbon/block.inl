#pragma once

// Inline definitions for carbon::Block. Implemented in the header (template + non-type
// SBlock) because every brick size is its own instantiation; pre-declaring an extern
// template for StaticBlock<8, 8, 8> is a possible future optimisation if instantiation
// cost shows up in a profile.

#include <carbon/block.h>
#include <carbon/cpu.h>
#include <carbon/logging.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/distributed.h>
#endif

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace carbon {

namespace detail::block::impl {

/** Ceiling integer division for positive arguments. */
[[nodiscard]] inline auto
ceil_div_pos(int64_t a,
             int64_t b) noexcept -> int64_t {
  return (a + b - 1) / b;
}

/**
 * Distribute @p extent brick rows evenly across @p n GPUs and return per-GPU
 * @c [begin, end) intervals. Mirrors the @c divide_extent_z helper @c Dense uses for
 * its z-slab partition: every GPU gets @c floor(extent / n) bricks, and the first
 * @c (extent mod n) GPUs get one extra. Sums to @p extent .
 */
[[nodiscard]] inline auto
partition_extent_z(int64_t extent,
                   size_t n) -> std::pair<std::vector<int64_t>,
                                          std::vector<int64_t>> {
  std::vector<int64_t> begin(n, 0);
  std::vector<int64_t> end(n, 0);
  if (n == 0U || extent <= 0) {
    return {std::move(begin), std::move(end)};
  }
  const int64_t base = extent / static_cast<int64_t>(n);
  const int64_t remainder = extent % static_cast<int64_t>(n);
  int64_t cursor = 0;
  for (size_t g = 0; g < n; ++g) {
    const int64_t take = base + ((static_cast<int64_t>(g) < remainder) ? 1 : 0);
    begin[g] = cursor;
    cursor += take;
    end[g] = cursor;
  }
  return {std::move(begin), std::move(end)};
}

}  // namespace detail::block::impl

template <typename SBlock>
Block<SBlock>::Block(System system,
                     int64_3d domain_extent,
                     const ActiveCellPredicate& is_active,
                     std::string name)
    : Block(std::move(system),
            domain_extent,
            is_active,
            ZBrickPartition{},
            std::move(name)) {}

template <typename SBlock>
Block<SBlock>::Block(System system,
                     int64_3d domain_extent,
                     const ActiveCellPredicate& is_active,
                     ZBrickPartition z_partition,
                     std::string name)
    : system_(std::move(system)),
      domain_extent_(domain_extent),
      name_(std::move(name)) {
  if (system_.gpu_group().size() == 0U) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::Block: system has no GPUs (gpu_group().size() == 0)");
  }
  if (domain_extent.x <= 0 || domain_extent.y <= 0 || domain_extent.z <= 0) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::Block: domain_extent must have positive components (got ({}, {}, {})).",
        domain_extent.x, domain_extent.y, domain_extent.z);
  }

  BuildContext ctx{};
  ctx.explicit_z_partition = z_partition.begin.empty() ? nullptr : &z_partition;
  initialise_partition_(domain_extent, ctx);
  sweep_active_bricks_(domain_extent, is_active, ctx);
  split_internal_boundary_(ctx);
  reserve_ghost_slots_(domain_extent, is_active, ctx);
  build_connectivity_(ctx);
  build_intra_halo_plan_(ctx);
  build_dist_halo_plan_(ctx);
  upload_topology_and_build_spans_(ctx);
}

template <typename SBlock>
auto
Block<SBlock>::initialise_partition_(const int64_3d& domain_extent,
                                     BuildContext& ctx) -> void {
  using detail::block::impl::ceil_div_pos;
  using detail::block::impl::partition_extent_z;

  ctx.n_gpus = system_.gpu_group().size();
  candidate_extent_ =
      int64_3d{ceil_div_pos(domain_extent.x, MemX), ceil_div_pos(domain_extent.y, MemY),
               ceil_div_pos(domain_extent.z, MemZ)};
  ctx.total_candidates = candidate_extent_.x * candidate_extent_.y * candidate_extent_.z;

  // Resolve the global slab count + this rank's local slab offset. In single-
  // process mode we partition across the local GpuGroup directly. In distributed
  // mode (one-rank-per-GPU under CARBON_HAVE_MPI_NCCL) the global partition has
  // world_size slabs and this rank only owns one of them — gpu_count() must be 1.
  global_slab_count_ = ctx.n_gpus;
  local_slab_offset_ = 0U;
#ifdef CARBON_HAVE_MPI_NCCL
  if (system_.distributed() != nullptr) {
    if (ctx.n_gpus != 1U) {
      log::throw_with_log<std::invalid_argument>(
          "carbon::Block: distributed System must have exactly one GPU per rank "
          "(gpu_count={}, world_size={}); the multi-GPU-per-rank hybrid is not yet "
          "supported.",
          ctx.n_gpus, system_.distributed()->world_size());
    }
    global_slab_count_ = static_cast<size_t>(system_.distributed()->world_size());
    local_slab_offset_ = static_cast<size_t>(system_.distributed()->world_rank());
  }
#endif
  if (static_cast<size_t>(candidate_extent_.z) < global_slab_count_) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::Block: candidate brick z-extent ({}) is smaller than global_slab_count ({}); "
        "every slab must own at least one brick row. Either widen the domain z-extent "
        "(>= {}) or rebuild the System with fewer slabs.",
        candidate_extent_.z, global_slab_count_, global_slab_count_ * static_cast<int64_t>(MemZ));
  }

  // Global brick z-row partition (Z-axis split, even with first remainder slabs
  // taking one extra brick row each). bz_*_per_gpu_ stores only this rank's
  // local slabs (a single contiguous range of size n_gpus starting at
  // local_slab_offset_); the global ranges live on @c ctx so cross-rank
  // ownership lookups during ghost-slot reservation can resolve them and then
  // get discarded at ctor exit.
  std::vector<int64_t> bz_begin_global;
  std::vector<int64_t> bz_end_global;
  if (ctx.explicit_z_partition != nullptr) {
    // Caller-supplied global cut (MultiResBlock projects the coarsest level's cut down to
    // every level). Validate the ZBrickPartition contract: per-slab counts, contiguity,
    // non-empty slabs, full coverage of the candidate z-extent.
    const auto& zp = *ctx.explicit_z_partition;
    if (zp.begin.size() != global_slab_count_ || zp.end.size() != global_slab_count_) {
      log::throw_with_log<std::invalid_argument>(
          "carbon::Block: explicit z-partition must hold global_slab_count ({}) entries per "
          "side (got begin={}, end={}).",
          global_slab_count_, zp.begin.size(), zp.end.size());
    }
    for (size_t s = 0; s < global_slab_count_; ++s) {
      const int64_t expected_begin = (s == 0U) ? 0 : zp.end[s - 1U];
      if (zp.begin[s] != expected_begin || zp.end[s] <= zp.begin[s]) {
        log::throw_with_log<std::invalid_argument>(
            "carbon::Block: explicit z-partition is not a contiguous non-empty cover at slab "
            "{} (begin={}, end={}, expected begin={}).",
            s, zp.begin[s], zp.end[s], expected_begin);
      }
    }
    if (zp.end.back() != candidate_extent_.z) {
      log::throw_with_log<std::invalid_argument>(
          "carbon::Block: explicit z-partition end ({}) does not cover the candidate brick "
          "z-extent ({}).",
          zp.end.back(), candidate_extent_.z);
    }
    bz_begin_global = zp.begin;
    bz_end_global = zp.end;
  } else {
    std::tie(bz_begin_global, bz_end_global) =
        partition_extent_z(candidate_extent_.z, global_slab_count_);
  }
  ctx.bz_begin_global = std::move(bz_begin_global);
  ctx.bz_end_global = std::move(bz_end_global);
  bz_begin_per_gpu_.assign(ctx.n_gpus, 0);
  bz_end_per_gpu_.assign(ctx.n_gpus, 0);
  for (size_t i = 0; i < ctx.n_gpus; ++i) {
    bz_begin_per_gpu_[i] = ctx.bz_begin_global[local_slab_offset_ + i];
    bz_end_per_gpu_[i] = ctx.bz_end_global[local_slab_offset_ + i];
  }

  // Initialise per-GPU storage vectors. Reserved 64 entries / GPU as a starter — the
  // realloc cost is negligible vs. the brick discovery sweep below.
  host_origins_per_gpu_.assign(ctx.n_gpus, {});
  host_bitmasks_per_gpu_.assign(ctx.n_gpus, {});
  host_connectivity_per_gpu_.assign(ctx.n_gpus, {});
  for (size_t g = 0; g < ctx.n_gpus; ++g) {
    host_origins_per_gpu_[g].reserve(64U);
    host_bitmasks_per_gpu_[g].reserve(64U);
  }

  // Map each candidate brick to its (gpu_id, local_block_id) once we discover it.
  // Used by the second-pass connectivity build to look up neighbour bricks.
  ctx.candidate_to_local.assign(static_cast<size_t>(ctx.total_candidates),
                                typename BuildContext::CandidateMap{});
}

template <typename SBlock>
auto
Block<SBlock>::sweep_active_bricks_(const int64_3d& domain_extent,
                                    const ActiveCellPredicate& is_active,
                                    BuildContext& ctx) -> void {
  active_voxel_count_ = 0;

  // Local sweep over only the z-rows this rank owns. In distributed mode that's
  // the [bz_begin_global[local_offset], bz_end_global[local_offset+n_gpus-1])
  // range; in single-process mode it's the entire candidate z-extent. The
  // candidate_to_local map is still sized for the full lattice so the second-
  // pass connectivity build can look up neighbours that fall in the adjacent
  // (potentially remote) rank's slab and resolve them as ghosts.
  const int64_t local_bz_lo = bz_begin_per_gpu_.empty() ? 0 : bz_begin_per_gpu_.front();
  const int64_t local_bz_hi = bz_end_per_gpu_.empty() ? 0 : bz_end_per_gpu_.back();
  for (int64_t bz = local_bz_lo; bz < local_bz_hi; ++bz) {
    // Determine owning local GPU for this brick z-row. Linear scan (n_gpus is
    // small — typically <= 8) avoids precomputing a bz->gpu lookup table.
    size_t owner = 0;
    for (size_t g = 0; g < ctx.n_gpus; ++g) {
      if (bz >= bz_begin_per_gpu_[g] && bz < bz_end_per_gpu_[g]) {
        owner = g;
        break;
      }
    }
    for (int64_t by = 0; by < candidate_extent_.y; ++by) {
      for (int64_t bx = 0; bx < candidate_extent_.x; ++bx) {
        const int64_t origin_x = bx * MemX;
        const int64_t origin_y = by * MemY;
        const int64_t origin_z = bz * MemZ;
        BitMask mask{};
        mask.clear();
        bool any_active = false;
        int64_t local_active = 0;
        for (int iz = 0; iz < MemZ; ++iz) {
          const int64_t gz = origin_z + iz;
          if (gz >= domain_extent.z) {
            break;
          }
          for (int iy = 0; iy < MemY; ++iy) {
            const int64_t gy = origin_y + iy;
            if (gy >= domain_extent.y) {
              break;
            }
            for (int ix = 0; ix < MemX; ++ix) {
              const int64_t gx = origin_x + ix;
              if (gx >= domain_extent.x) {
                break;
              }
              if (is_active(int64_3d{gx, gy, gz})) {
                mask.set_active(ix, iy, iz);
                any_active = true;
                ++local_active;
              }
            }
          }
        }
        if (any_active) {
          const int64_t cand_idx = (bz * candidate_extent_.y + by) * candidate_extent_.x + bx;
          const uint32_t local_id = static_cast<uint32_t>(host_origins_per_gpu_[owner].size());
          ctx.candidate_to_local[static_cast<size_t>(cand_idx)] =
              typename BuildContext::CandidateMap{static_cast<uint32_t>(owner), local_id};
          host_origins_per_gpu_[owner].push_back(int32_3d{static_cast<int32_t>(origin_x),
                                                          static_cast<int32_t>(origin_y),
                                                          static_cast<int32_t>(origin_z)});
          host_bitmasks_per_gpu_[owner].push_back(mask);
          active_voxel_count_ += local_active;
        }
      }
    }
  }

  total_block_count_ = 0;
  for (size_t g = 0; g < ctx.n_gpus; ++g) {
    total_block_count_ += static_cast<uint32_t>(host_origins_per_gpu_[g].size());
  }
}

template <typename SBlock>
auto
Block<SBlock>::split_internal_boundary_(BuildContext& ctx) -> void {
  // Stable-partition each slab's owned bricks into [internal, boundary). A
  // brick is "boundary" iff its bz row touches a slab face that has a remote
  // neighbour — i.e. (bz == bz_begin[i] AND global slab index > 0) OR
  // (bz == bz_end[i] - 1 AND global slab index + 1 < global_slab_count_).
  // This matches the depth of the existing ghost slot reservation (one brick
  // row per face) and is the brick-level analogue of Dense's
  // halo_z-deep Internal / Boundary bands. After the partition we remap
  // candidate_to_local entries on this rank so the new local ids are in
  // effect when the connectivity rewrite + halo plan run below.
  n_internal_per_gpu_.assign(ctx.n_gpus, 0U);
  for (size_t i = 0; i < ctx.n_gpus; ++i) {
    const size_t n_owned = host_origins_per_gpu_[i].size();
    if (n_owned == 0U) {
      continue;
    }
    const size_t global_g = local_slab_offset_ + i;
    const bool has_lower_neighbour = global_g > 0U;
    const bool has_upper_neighbour = global_g + 1U < global_slab_count_;
    const int64_t bz_lo = bz_begin_per_gpu_[i];
    const int64_t bz_hi = bz_end_per_gpu_[i] - 1;
    auto is_boundary = [&](size_t local_id) -> bool {
      const int32_t origin_z = host_origins_per_gpu_[i][local_id].z;
      const int64_t bz_brick = static_cast<int64_t>(origin_z) / static_cast<int64_t>(MemZ);
      if (has_lower_neighbour && bz_brick == bz_lo) {
        return true;
      }
      if (has_upper_neighbour && bz_brick == bz_hi) {
        return true;
      }
      return false;
    };
    // Build a permutation old_id -> new_id by sweeping owned bricks in their
    // current (= candidate-row-major) order and emitting internal first then
    // boundary. Stable on each side so origins / bitmasks stay easy to debug.
    std::vector<uint32_t> permutation(n_owned, 0U);
    uint32_t internal_cursor = 0U;
    uint32_t boundary_cursor = 0U;
    for (size_t k = 0; k < n_owned; ++k) {
      if (!is_boundary(k)) {
        ++internal_cursor;
      } else {
        ++boundary_cursor;
      }
    }
    const uint32_t n_internal = internal_cursor;
    n_internal_per_gpu_[i] = n_internal;
    internal_cursor = 0U;
    boundary_cursor = n_internal;
    for (size_t k = 0; k < n_owned; ++k) {
      if (!is_boundary(k)) {
        permutation[k] = internal_cursor++;
      } else {
        permutation[k] = boundary_cursor++;
      }
    }
    // Apply permutation in place via a fresh scratch vector. Origins +
    // bitmasks are small structs; the copy is negligible vs. the candidate
    // sweep above.
    std::vector<int32_3d> new_origins(n_owned);
    std::vector<BitMask> new_bitmasks(n_owned);
    for (size_t k = 0; k < n_owned; ++k) {
      new_origins[permutation[k]] = host_origins_per_gpu_[i][k];
      new_bitmasks[permutation[k]] = host_bitmasks_per_gpu_[i][k];
    }
    host_origins_per_gpu_[i] = std::move(new_origins);
    host_bitmasks_per_gpu_[i] = std::move(new_bitmasks);
    // Remap every candidate_to_local entry on slab i to the new id. Entries
    // for other slabs / unretained candidates are untouched.
    for (size_t c = 0; c < ctx.candidate_to_local.size(); ++c) {
      auto& m = ctx.candidate_to_local[c];
      if (m.gpu_id == static_cast<uint32_t>(i) && m.local_id != detail::block::kInvalidBlockIdx) {
        m.local_id = permutation[m.local_id];
      }
    }
  }
}

template <typename SBlock>
auto
Block<SBlock>::reserve_ghost_slots_(const int64_3d& domain_extent,
                                    const ActiveCellPredicate& is_active,
                                    BuildContext& ctx) -> void {
  // Reserve ghost slots at the lower / upper z-edge of each local slab. Lower
  // ghosts mirror retained bricks at bz = bz_begin[i] - 1 owned by slab i-1
  // (which may be another local slab on this rank, or — in distributed mode —
  // a slab on the rank below us); upper ghosts mirror retained bricks at
  // bz = bz_end[i] owned by slab i+1 (likewise). We sweep the candidate
  // (bx, by) plane in row-major order so the resulting per-slab ghost-id
  // assignment is deterministic across ranks (NCCL pairs Sends and Recvs
  // positionally — both sides walk the same predicate over the same row in
  // the same order).
  //
  // A small helper evaluates the predicate on an arbitrary candidate brick.
  // For local-slab sources we also need the source's local_id on the
  // neighbouring slab (already recorded in candidate_to_local during the
  // main sweep above); for remote-rank sources we just need to know whether
  // the brick passes the predicate — its local_id on the peer is opaque to
  // this rank and pairing relies on positional ordering.
  auto eval_brick = [&](int64_t bx, int64_t by, int64_t bz) -> std::pair<bool, BitMask> {
    BitMask mask{};
    mask.clear();
    bool any_active = false;
    if (bz < 0 || bz >= candidate_extent_.z) {
      return {false, mask};
    }
    const int64_t origin_x = bx * MemX;
    const int64_t origin_y = by * MemY;
    const int64_t origin_z = bz * MemZ;
    for (int iz = 0; iz < MemZ; ++iz) {
      const int64_t gz = origin_z + iz;
      if (gz >= domain_extent.z) {
        break;
      }
      for (int iy = 0; iy < MemY; ++iy) {
        const int64_t gy = origin_y + iy;
        if (gy >= domain_extent.y) {
          break;
        }
        for (int ix = 0; ix < MemX; ++ix) {
          const int64_t gx = origin_x + ix;
          if (gx >= domain_extent.x) {
            break;
          }
          if (is_active(int64_3d{gx, gy, gz})) {
            mask.set_active(ix, iy, iz);
            any_active = true;
          }
        }
      }
    }
    return {any_active, mask};
  };
  // Returns the global slab id (= rank id under one-rank-per-GPU) that owns
  // brick z-row bz_row, or static_cast<size_t>(-1) if out of range.
  auto owner_of_bz = [&](int64_t bz_row) -> size_t {
    if (bz_row < 0 || bz_row >= candidate_extent_.z) {
      return static_cast<size_t>(-1);
    }
    for (size_t g = 0; g < global_slab_count_; ++g) {
      if (bz_row >= ctx.bz_begin_global[g] && bz_row < ctx.bz_end_global[g]) {
        return g;
      }
    }
    return static_cast<size_t>(-1);
  };

  ghost_lower_count_per_gpu_.assign(ctx.n_gpus, 0U);
  ghost_upper_count_per_gpu_.assign(ctx.n_gpus, 0U);
  // Per-ghost source descriptors (parallel to ghost slot reservation order).
  // peer_rank == -1 means the source is on another local slab; in that case
  // remote_owned holds the source brick's local_id on that slab. peer_rank
  // >= 0 means cross-rank source; remote_owned is unused. ghost_bitmasks
  // captures the source's bitmask for upload into device_bitmask_.
  ctx.ghost_lower_peer.assign(ctx.n_gpus, {});
  ctx.ghost_upper_peer.assign(ctx.n_gpus, {});
  ctx.ghost_lower_remote_owned.assign(ctx.n_gpus, {});
  ctx.ghost_upper_remote_owned.assign(ctx.n_gpus, {});
  ctx.ghost_lower_bitmasks.assign(ctx.n_gpus, {});
  ctx.ghost_upper_bitmasks.assign(ctx.n_gpus, {});
  ctx.candidate_to_ghost_local.assign(
      ctx.n_gpus, std::vector<uint32_t>(static_cast<size_t>(ctx.total_candidates),
                                        detail::block::kInvalidBlockIdx));
  for (size_t g = 0; g < ctx.n_gpus; ++g) {
    const uint32_t n_owned_g = static_cast<uint32_t>(host_origins_per_gpu_[g].size());
    auto reserve_row = [&](int64_t bz_row, bool is_lower) {
      if (bz_row < 0 || bz_row >= candidate_extent_.z) {
        return;
      }
      const size_t source_global = owner_of_bz(bz_row);
      if (source_global == static_cast<size_t>(-1)) {
        return;
      }
      const bool source_is_local =
          (source_global >= local_slab_offset_ && source_global < local_slab_offset_ + ctx.n_gpus);
      const size_t source_local_idx = source_is_local ? (source_global - local_slab_offset_) : 0;
      if (source_is_local && source_local_idx == g) {
        // Source slab is this very local slab — only possible in the (n_gpus,
        // global_slab_count_) == (1, 1) corner case, which has no halo at all.
        return;
      }
      for (int64_t by = 0; by < candidate_extent_.y; ++by) {
        for (int64_t bx = 0; bx < candidate_extent_.x; ++bx) {
          const int64_t cand_idx = (bz_row * candidate_extent_.y + by) * candidate_extent_.x + bx;
          BitMask mask{};
          uint32_t source_local_id = detail::block::kInvalidBlockIdx;
          if (source_is_local) {
            const auto neigh = ctx.candidate_to_local[static_cast<size_t>(cand_idx)];
            if (neigh.gpu_id == detail::block::kInvalidBlockIdx ||
                neigh.gpu_id != static_cast<uint32_t>(source_local_idx)) {
              continue;
            }
            mask = host_bitmasks_per_gpu_[source_local_idx][neigh.local_id];
            source_local_id = neigh.local_id;
          } else {
            // Remote source: evaluate the predicate locally — it's pure, so we
            // get the same answer the peer rank does.
            auto [any_active, m] = eval_brick(bx, by, bz_row);
            if (!any_active) {
              continue;
            }
            mask = m;
          }
          const uint32_t ghost_id =
              n_owned_g + ghost_lower_count_per_gpu_[g] + ghost_upper_count_per_gpu_[g];
          ctx.candidate_to_ghost_local[g][static_cast<size_t>(cand_idx)] = ghost_id;
          const int peer = source_is_local ? -1 : static_cast<int>(source_global);
          if (is_lower) {
            ++ghost_lower_count_per_gpu_[g];
            ctx.ghost_lower_peer[g].push_back(peer);
            ctx.ghost_lower_remote_owned[g].push_back(source_local_id);
            ctx.ghost_lower_bitmasks[g].push_back(mask);
          } else {
            ++ghost_upper_count_per_gpu_[g];
            ctx.ghost_upper_peer[g].push_back(peer);
            ctx.ghost_upper_remote_owned[g].push_back(source_local_id);
            ctx.ghost_upper_bitmasks[g].push_back(mask);
          }
        }
      }
    };
    if (global_slab_count_ > 1U) {
      reserve_row(bz_begin_per_gpu_[g] - 1, /*is_lower=*/true);
      reserve_row(bz_end_per_gpu_[g], /*is_lower=*/false);
    }
  }
}

template <typename SBlock>
auto
Block<SBlock>::build_connectivity_(const BuildContext& ctx) -> void {
  // Build per-GPU 27-way connectivity over owned bricks. Cross-GPU
  // neighbours route to ghost slots on this GPU when one was reserved by
  // @ref reserve_ghost_slots_ ; otherwise stay invalid (e.g. a diagonal at
  // bz_begin[g]-2 isn't a ghost since z-axis split + 27-way connectivity
  // only crosses the slab by ±1).
  for (size_t g = 0; g < ctx.n_gpus; ++g) {
    const uint32_t n_blocks_g = static_cast<uint32_t>(host_origins_per_gpu_[g].size());
    host_connectivity_per_gpu_[g].assign(static_cast<size_t>(n_blocks_g) * 27U,
                                         detail::block::kInvalidBlockIdx);
    for (uint32_t b = 0; b < n_blocks_g; ++b) {
      const auto origin = host_origins_per_gpu_[g][b];
      const int64_t bx = origin.x / MemX;
      const int64_t by = origin.y / MemY;
      const int64_t bz = origin.z / MemZ;
      for (int dz = -1; dz <= 1; ++dz) {
        const int64_t nbz = bz + dz;
        if (nbz < 0 || nbz >= candidate_extent_.z) {
          continue;
        }
        for (int dy = -1; dy <= 1; ++dy) {
          const int64_t nby = by + dy;
          if (nby < 0 || nby >= candidate_extent_.y) {
            continue;
          }
          for (int dx = -1; dx <= 1; ++dx) {
            const int64_t nbx = bx + dx;
            if (nbx < 0 || nbx >= candidate_extent_.x) {
              continue;
            }
            const int64_t cand_idx = (nbz * candidate_extent_.y + nby) * candidate_extent_.x + nbx;
            const auto neigh = ctx.candidate_to_local[static_cast<size_t>(cand_idx)];
            const uint32_t slot = detail::block::Span<SBlock>::connectivity_slot(dx, dy, dz);
            const size_t offset = static_cast<size_t>(b) * 27U + slot;
            if (neigh.gpu_id == static_cast<uint32_t>(g)) {
              host_connectivity_per_gpu_[g][offset] = neigh.local_id;
            } else {
              // Neighbour brick is either on another local GPU (then reserve_ghost_slots_
              // populated candidate_to_ghost_local[g][cand_idx] with the corresponding
              // ghost slot id) or on another MPI rank (same thing — the ghost reservation
              // pass evaluated the predicate locally and reserved a cross-rank ghost
              // slot that HaloBlock fills). Under MPI gpu_count == 1, so the
              // remote-rank case has neigh.gpu_id == kInvalidBlockIdx and the
              // previous `else if (neigh.gpu_id != kInvalidBlockIdx)` guard
              // dropped cross-rank connectivity on the floor — the halo bytes
              // arrived but Partition::neighbour could not see them.
              const uint32_t ghost_id =
                  ctx.candidate_to_ghost_local[g][static_cast<size_t>(cand_idx)];
              if (ghost_id != detail::block::kInvalidBlockIdx) {
                host_connectivity_per_gpu_[g][offset] = ghost_id;
              }
            }
          }
        }
      }
    }
  }
}

template <typename SBlock>
auto
Block<SBlock>::build_intra_halo_plan_(const BuildContext& ctx) -> void {
  // Halo plan: one upward + one downward vector per adjacency. For adjacency
  // adj = g (in [0, n_gpus - 1)):
  //   upward[g]   = pairs sending g's owned bricks at bz_end[g] - 1   → g+1's
  //                 lower-ghost slot.
  //   downward[g] = pairs sending g+1's owned bricks at bz_begin[g+1] → g's
  //                 upper-ghost slot.
  // Iteration order matches reserve_row's row-major (bx, by) sweep, so the kth
  // ghost_lower_remote_owned[g+1][k] is the kth source on g and the
  // corresponding lower-ghost id on g+1 is n_owned[g+1] + k. Same shape on the
  // downward side.
  halo_pairs_upward_.assign(ctx.n_gpus > 0 ? ctx.n_gpus - 1 : 0, {});
  halo_pairs_downward_.assign(ctx.n_gpus > 0 ? ctx.n_gpus - 1 : 0, {});
  for (size_t adj = 0; adj + 1U < ctx.n_gpus; ++adj) {
    // Lower-ghost slots on slab adj+1 whose source is local-slab adj. Slot ids
    // count from n_owned[adj+1] in reservation order; we step down_dst_id over
    // remote-source slots so the running index stays correct.
    uint32_t up_dst_id = static_cast<uint32_t>(host_origins_per_gpu_[adj + 1].size());
    for (size_t k = 0; k < ctx.ghost_lower_remote_owned[adj + 1].size(); ++k) {
      if (ctx.ghost_lower_peer[adj + 1][k] != -1) {
        ++up_dst_id;
        continue;
      }
      halo_pairs_upward_[adj].push_back(
          typename Block::HaloPair{ctx.ghost_lower_remote_owned[adj + 1][k], up_dst_id});
      ++up_dst_id;
    }
    // Upper-ghost slots on slab adj whose source is local-slab adj+1.
    uint32_t down_dst_id =
        static_cast<uint32_t>(host_origins_per_gpu_[adj].size()) + ghost_lower_count_per_gpu_[adj];
    for (size_t k = 0; k < ctx.ghost_upper_remote_owned[adj].size(); ++k) {
      if (ctx.ghost_upper_peer[adj][k] != -1) {
        ++down_dst_id;
        continue;
      }
      halo_pairs_downward_[adj].push_back(
          typename Block::HaloPair{ctx.ghost_upper_remote_owned[adj][k], down_dst_id});
      ++down_dst_id;
    }
  }
}

template <typename SBlock>
auto
Block<SBlock>::build_dist_halo_plan_(const BuildContext& ctx) -> void {
  // Cross-rank halo plan: only populated when there is at least one ghost slot
  // whose source is on a remote rank. Send entries are emitted by walking owned
  // bricks at this rank's local-slab edges that face a remote neighbour; Recv
  // entries are emitted by walking ghost slots with remote sources. NCCL pairing
  // across ranks is positional (FIFO), so both sides walk the same predicate
  // over the same (bx, by) row in row-major order and the resulting
  // Send/Recv counts and byte sizes match by construction.
  halo_dist_send_upward_.clear();
  halo_dist_recv_upward_.clear();
  halo_dist_send_downward_.clear();
  halo_dist_recv_downward_.clear();
  for (size_t i = 0; i < ctx.n_gpus; ++i) {
    const size_t g = local_slab_offset_ + i;
    if (g + 1U < global_slab_count_) {
      const size_t upper_owner = g + 1U;
      const bool upper_is_local =
          (upper_owner >= local_slab_offset_ && upper_owner < local_slab_offset_ + ctx.n_gpus);
      if (!upper_is_local) {
        const int64_t edge_bz = bz_end_per_gpu_[i] - 1;
        for (int64_t by = 0; by < candidate_extent_.y; ++by) {
          for (int64_t bx = 0; bx < candidate_extent_.x; ++bx) {
            const int64_t cand_idx =
                (edge_bz * candidate_extent_.y + by) * candidate_extent_.x + bx;
            const auto src = ctx.candidate_to_local[static_cast<size_t>(cand_idx)];
            if (src.gpu_id == detail::block::kInvalidBlockIdx ||
                src.gpu_id != static_cast<uint32_t>(i)) {
              continue;
            }
            halo_dist_send_upward_.push_back(
                typename Block::HaloDistEntry{src.local_id, static_cast<int>(upper_owner)});
          }
        }
      }
    }
    if (g > 0U) {
      const size_t lower_owner = g - 1U;
      const bool lower_is_local =
          (lower_owner >= local_slab_offset_ && lower_owner < local_slab_offset_ + ctx.n_gpus);
      if (!lower_is_local) {
        const int64_t edge_bz = bz_begin_per_gpu_[i];
        for (int64_t by = 0; by < candidate_extent_.y; ++by) {
          for (int64_t bx = 0; bx < candidate_extent_.x; ++bx) {
            const int64_t cand_idx =
                (edge_bz * candidate_extent_.y + by) * candidate_extent_.x + bx;
            const auto src = ctx.candidate_to_local[static_cast<size_t>(cand_idx)];
            if (src.gpu_id == detail::block::kInvalidBlockIdx ||
                src.gpu_id != static_cast<uint32_t>(i)) {
              continue;
            }
            halo_dist_send_downward_.push_back(
                typename Block::HaloDistEntry{src.local_id, static_cast<int>(lower_owner)});
          }
        }
      }
    }
    const uint32_t lower_base = static_cast<uint32_t>(host_origins_per_gpu_[i].size());
    for (size_t k = 0; k < ctx.ghost_lower_peer[i].size(); ++k) {
      if (ctx.ghost_lower_peer[i][k] != -1) {
        halo_dist_recv_upward_.push_back(typename Block::HaloDistEntry{
            lower_base + static_cast<uint32_t>(k), ctx.ghost_lower_peer[i][k]});
      }
    }
    const uint32_t upper_base = lower_base + ghost_lower_count_per_gpu_[i];
    for (size_t k = 0; k < ctx.ghost_upper_peer[i].size(); ++k) {
      if (ctx.ghost_upper_peer[i][k] != -1) {
        halo_dist_recv_downward_.push_back(typename Block::HaloDistEntry{
            upper_base + static_cast<uint32_t>(k), ctx.ghost_upper_peer[i][k]});
      }
    }
  }
}

template <typename SBlock>
auto
Block<SBlock>::upload_topology_and_build_spans_(const BuildContext& ctx) -> void {
  if (total_block_count_ == 0U) {
    spans_.assign(ctx.n_gpus, Span{});
    return;
  }

  // Per-GPU device buffer sizes. Bitmasks include ghost slots (so neighbour() can
  // walk across the slab boundary at iteration time); connectivity covers only
  // owned bricks (ghost slots are storage-only); origins also cover only owned
  // bricks (kernels iterate owned ids).
  std::vector<size_t> mask_counts(ctx.n_gpus, 0);
  std::vector<size_t> conn_counts(ctx.n_gpus, 0);
  std::vector<size_t> origin_counts(ctx.n_gpus, 0);
  for (size_t g = 0; g < ctx.n_gpus; ++g) {
    const size_t n_owned_g = host_origins_per_gpu_[g].size();
    mask_counts[g] = n_owned_g + static_cast<size_t>(ghost_lower_count_per_gpu_[g]) +
                     static_cast<size_t>(ghost_upper_count_per_gpu_[g]);
    conn_counts[g] = n_owned_g * 27U;
    origin_counts[g] = n_owned_g;
  }
  auto& gpus = system_.gpu_group();
  device_bitmask_ = gpus.template buffer<BitMask>(mask_counts);
  device_connectivity_ = gpus.template buffer<uint32_t>(conn_counts);
  device_origins_ = gpus.template buffer<int32_3d>(origin_counts);

  // Per-GPU async upload. Owned bitmasks copied from this GPU's host vector;
  // ghost bitmasks copied from the adjacent GPU's host vector (cheap, host-side).
  Stream stream = system_.stream();
  for (size_t g = 0; g < ctx.n_gpus; ++g) {
    const uint32_t n_owned_g = static_cast<uint32_t>(host_origins_per_gpu_[g].size());
    const size_t total_g = mask_counts[g];
    if (total_g == 0U) {
      continue;
    }
    auto host_mask_alloc = Cpu::buffer<BitMask>(total_g);
    for (size_t i = 0; i < static_cast<size_t>(n_owned_g); ++i) {
      host_mask_alloc.data()[i] = host_bitmasks_per_gpu_[g][i];
    }
    size_t ghost_cursor = static_cast<size_t>(n_owned_g);
    for (const auto& m : ctx.ghost_lower_bitmasks[g]) {
      host_mask_alloc.data()[ghost_cursor++] = m;
    }
    for (const auto& m : ctx.ghost_upper_bitmasks[g]) {
      host_mask_alloc.data()[ghost_cursor++] = m;
    }
    auto host_conn_alloc = Cpu::buffer<uint32_t>(static_cast<size_t>(n_owned_g) * 27U);
    auto host_orig_alloc = Cpu::buffer<int32_3d>(static_cast<size_t>(n_owned_g));
    std::copy(host_connectivity_per_gpu_[g].begin(), host_connectivity_per_gpu_[g].end(),
              host_conn_alloc.data());
    std::copy(host_origins_per_gpu_[g].begin(), host_origins_per_gpu_[g].end(),
              host_orig_alloc.data());
    system_.async_copy(host_mask_alloc, device_bitmask_[g], sizeof(BitMask) * total_g, stream);
    system_.async_copy(host_conn_alloc, device_connectivity_[g],
                       sizeof(uint32_t) * static_cast<size_t>(n_owned_g) * 27U, stream);
    system_.async_copy(host_orig_alloc, device_origins_[g],
                       sizeof(int32_3d) * static_cast<size_t>(n_owned_g), stream);
  }
  stream.wait();

  // Build per-GPU Spans. Even an empty slab still has a placeholder Span (null
  // bitmask, 0 block_count) so .spans()[g] is always indexable.
  spans_.clear();
  spans_.reserve(ctx.n_gpus);
  for (size_t g = 0; g < ctx.n_gpus; ++g) {
    const uint32_t n_blocks_g = static_cast<uint32_t>(host_origins_per_gpu_[g].size());
    if (n_blocks_g == 0U) {
      spans_.emplace_back();
      continue;
    }
    spans_.emplace_back(device_bitmask_[g].data(), n_blocks_g, n_internal_per_gpu_[g],
                        device_connectivity_[g].data(), device_origins_[g].data());
  }
}

template <typename SBlock>
auto
Block<SBlock>::build_locator_() const -> void {
  const size_t n_gpus = host_origins_per_gpu_.size();
  locator_storage_->geometry.assign(n_gpus, detail::block::LocatorGeometry{});
  if (n_gpus == 0U) {
    return;
  }

  // The table spans the bounding box of the bricks this GPU actually retained, not the
  // whole candidate lattice: a sparse grid over a large box would otherwise pay for a
  // table sized by the box. The box is grown by one brick on every side so the ghost ring
  // fits — see the ownership contract in detail/block/locator.h for why the ghosts belong
  // in the table even though the owned band alone defines what this GPU may integrate.
  std::vector<size_t> counts(n_gpus, 0U);
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& origins = host_origins_per_gpu_[g];
    auto& geom = locator_storage_->geometry[g];
    geom.voxel_extent = domain_extent_;
    geom.owned_bz_begin = bz_begin(g);
    geom.owned_bz_end = bz_end(g);
    if (origins.empty()) {
      continue;
    }

    int64_3d lo{origins[0].x / MemX, origins[0].y / MemY, origins[0].z / MemZ};
    int64_3d hi = lo;
    for (const int32_3d& o : origins) {
      const int64_3d b{o.x / MemX, o.y / MemY, o.z / MemZ};
      lo.x = std::min(lo.x, b.x);
      lo.y = std::min(lo.y, b.y);
      lo.z = std::min(lo.z, b.z);
      hi.x = std::max(hi.x, b.x);
      hi.y = std::max(hi.y, b.y);
      hi.z = std::max(hi.z, b.z);
    }
    geom.lattice_origin = {lo.x - 1, lo.y - 1, lo.z - 1};
    geom.lattice = {hi.x - lo.x + 3, hi.y - lo.y + 3, hi.z - lo.z + 3};
    counts[g] = static_cast<size_t>(geom.entry_count());
  }

  System& system = const_cast<System&>(system_);
  locator_storage_->table = system.gpu_group().template buffer<uint32_t>(counts);

  Stream stream = system.stream();
  for (size_t g = 0; g < n_gpus; ++g) {
    const size_t n_entries = counts[g];
    if (n_entries == 0U) {
      continue;
    }
    const auto& geom = locator_storage_->geometry[g];
    auto host_table = Cpu::buffer<uint32_t>(n_entries);
    std::fill_n(host_table.data(), n_entries, detail::block::kInvalidBlockIdx);
    const auto& origins = host_origins_per_gpu_[g];
    const auto entry_of = [&geom](int64_t bx, int64_t by, int64_t bz) -> int64_t {
      const int64_t lx = bx - geom.lattice_origin.x;
      const int64_t ly = by - geom.lattice_origin.y;
      const int64_t lz = bz - geom.lattice_origin.z;
      if (lx < 0 || ly < 0 || lz < 0 || lx >= geom.lattice.x || ly >= geom.lattice.y ||
          lz >= geom.lattice.z) {
        return -1;
      }
      return (lz * geom.lattice.y + ly) * geom.lattice.x + lx;
    };

    for (size_t b = 0; b < origins.size(); ++b) {
      const int64_t e = entry_of(static_cast<int64_t>(origins[b].x) / MemX,
                                 static_cast<int64_t>(origins[b].y) / MemY,
                                 static_cast<int64_t>(origins[b].z) / MemZ);
      host_table.data()[static_cast<size_t>(e)] = static_cast<uint32_t>(b);
    }

    // Ghost ring. A ghost slot's brick origin is not retained anywhere on the host, but the
    // 27-way connectivity table already routes each owned brick's cross-slab neighbours to
    // the matching ghost slot, so walking it recovers the (lattice position -> ghost slot)
    // map. Owned entries are never overwritten: a brick that is owned here is owned, and the
    // same lattice cell can be reached both as somebody's neighbour and as itself.
    const auto& conn = host_connectivity_per_gpu_[g];
    for (size_t b = 0; b < origins.size(); ++b) {
      const int64_t bx = static_cast<int64_t>(origins[b].x) / MemX;
      const int64_t by = static_cast<int64_t>(origins[b].y) / MemY;
      const int64_t bz = static_cast<int64_t>(origins[b].z) / MemZ;
      for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            const uint32_t slot = Span::connectivity_slot(dx, dy, dz);
            const uint32_t neigh = conn[b * 27U + slot];
            if (neigh == detail::block::kInvalidBlockIdx || neigh < origins.size()) {
              continue;
            }
            const int64_t e = entry_of(bx + dx, by + dy, bz + dz);
            if (e < 0 ||
                host_table.data()[static_cast<size_t>(e)] != detail::block::kInvalidBlockIdx) {
              continue;
            }
            host_table.data()[static_cast<size_t>(e)] = neigh;
          }
        }
      }
    }

    system_.async_copy(host_table, locator_storage_->table[g], sizeof(uint32_t) * n_entries,
                       stream);
  }
  stream.wait();
}

template <typename SBlock>
auto
Block<SBlock>::locator(size_t gpu_id,
                       float32_3d world_origin,
                       float world_spacing) const -> detail::block::Locator<SBlock> {
  std::call_once(locator_storage_->once, [this]() {
    this->build_locator_();
  });

  detail::block::Locator<SBlock> view{};
  view.world_origin = world_origin;
  view.world_spacing = world_spacing;
  if (gpu_id >= locator_storage_->geometry.size()) {
    return view;
  }

  const auto& geom = locator_storage_->geometry[gpu_id];
  view.lattice = geom.lattice;
  view.lattice_origin = geom.lattice_origin;
  view.voxel_extent = geom.voxel_extent;
  view.owned_z_lo = world_origin.z + static_cast<float>(geom.owned_bz_begin * MemZ) * world_spacing;
  view.owned_z_hi = world_origin.z + static_cast<float>(geom.owned_bz_end * MemZ) * world_spacing;
  if (geom.entry_count() > 0 && gpu_id < locator_storage_->table.size()) {
    view.table = locator_storage_->table[gpu_id].data();
  }
  return view;
}

template <typename SBlock>
template <typename T,
          int Components>
auto
Block<SBlock>::newField(MemorySide side,
                        std::string name) const -> Field<T,
                                                         Components> {
  const size_t n_gpus = system_.gpu_group().size();
  if (n_gpus == 0U || total_block_count_ == 0U) {
    return Field<T, Components>{};
  }

  // Per-GPU device buffer sizes — owned + ghost slots — so that ghost bricks
  // populated by HaloBlock land at slot ids in [n_owned, n_owned + n_ghost).
  std::vector<size_t> counts(n_gpus, 0);
  for (size_t g = 0; g < n_gpus; ++g) {
    const size_t n_total_g = static_cast<size_t>(host_origins_per_gpu_[g].size()) +
                             static_cast<size_t>(ghost_lower_count_per_gpu_[g]) +
                             static_cast<size_t>(ghost_upper_count_per_gpu_[g]);
    counts[g] = n_total_g * static_cast<size_t>(VoxelCount) * static_cast<size_t>(Components);
  }
  auto& gpus = const_cast<System&>(system_).gpu_group();
  Buffer<T> dev = gpus.template buffer<T>(counts);

  // Bitmask + 27-way connectivity are no longer threaded into Field — they live
  // on the per-GPU Span owned by this Block (see Span(...) ctor in the loop
  // above) and are read through the Span on every Partition::neighbour walk.
  // Field's lifetime no longer depends on the Block-borrowed device pointers;
  // the kernel-level "Block must outlive any kernel that captures its Span" is
  // the single remaining lifetime contract.

  // Host mirror is allocated eagerly for MemorySide::Host and
  // MemorySide::HostDevice — sized identically to the device buffer so
  // host_partitions[g] indexes the same brick layout as partitions[g].
  // Empty for MemorySide::Device, which feeds the device-only path
  // (host_partitions() stays empty so a host kernel reading this field
  // will throw a clean diagnostic from KernelLoader::lookup_partition_).
  if (side == MemorySide::Device) {
    Field<T, Components> field{std::move(dev), std::move(name)};
    // Wire the back-reference so the field can build a real HaloBlock
    // through make_halo_kernel() and so that kernels can fetch the per-GPU
    // Span (bitmask + connectivity) for Partition::neighbour walks. The
    // Block must outlive every Field it produced; see Field::set_block
    // for the lifetime contract. Repeated for the host-allocating branch
    // below.
    field.set_block(this);
    return field;
  }
  std::vector<detail::Allocation<T>> host_allocs;
  host_allocs.reserve(n_gpus);
  for (size_t g = 0; g < n_gpus; ++g) {
    host_allocs.emplace_back(Cpu::buffer<T>(counts[g]));
  }
  Buffer<T> host = Buffer<T>::from_vector(std::move(host_allocs));
  Field<T, Components> field{std::move(dev), std::move(host), std::move(name)};
  // Wire the back-reference (see Field::set_block for the lifetime contract).
  field.set_block(this);
  return field;
}

template <typename SBlock>
template <typename T,
          int Components>
auto
Block<SBlock>::download_to_host(const Field<T,
                                            Components>& field,
                                const ::carbon::Stream& stream) const
    -> std::vector<detail::Allocation<T>> {
  const size_t n_gpus = system_.gpu_group().size();
  std::vector<detail::Allocation<T>> result;
  result.reserve(n_gpus);
  if (n_gpus == 0U) {
    return result;
  }
  if (field.gpu_buffers().size() != n_gpus) {
    log::throw_with_log<std::invalid_argument>(
        "Block::download_to_host: field has {} per-GPU device buffers but Block has {} GPUs",
        field.gpu_buffers().size(), n_gpus);
  }
  for (size_t g = 0; g < n_gpus; ++g) {
    const uint32_t n_owned = block_count(g);
    const size_t owned_elements = static_cast<size_t>(n_owned) * static_cast<size_t>(VoxelCount) *
                                  static_cast<size_t>(Components);
    auto host_buf = Cpu::buffer<T>(owned_elements);
    if (owned_elements != 0U) {
      // Owned bricks live at the front of the per-GPU buffer (ghost slots are
      // at [n_owned, n_owned + n_ghost) by construction); copying the first
      // owned_elements scalars is exactly the owned-only payload.
      const_cast<System&>(system_).async_copy(field.gpu_buffers()[g], host_buf,
                                              owned_elements * sizeof(T), stream);
    }
    result.push_back(std::move(host_buf));
  }
  return result;
}

}  // namespace carbon
