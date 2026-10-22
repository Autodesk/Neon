#pragma once

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/distributed.h>
#endif

#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace carbon {

namespace detail::mgrid::impl {

/** Ceiling integer division for positive arguments (mirrors block::impl::ceil_div_pos). */
[[nodiscard]] inline auto
ceil_div_pos(int64_t a,
             int64_t b) noexcept -> int64_t {
  return (a + b - 1) / b;
}

/**
 * Even brick-row split of @p extent rows across @p n slabs: the canonical cut every
 * @c Block uses for its z-axis partition (first @c extent % n slabs take one extra row).
 * Pure + deterministic, so distributed ranks recompute the identical global cut without
 * communication. Returns per-slab @c [begin, end) brick-row ranges.
 */
[[nodiscard]] inline auto
even_split_z(int64_t extent,
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

/**
 * Weight-balanced contiguous split of @p row_weights across @p n slabs. Each slab receives a
 * contiguous z-brick-row range whose summed weight is as equal as possible (remainder distributed
 * like @ref even_split_z ). Falls back to @ref even_split_z when every weight is zero. Pure +
 * deterministic — distributed ranks recompute the identical cut without communication.
 */
[[nodiscard]] inline auto
weighted_split_z(const std::vector<int64_t>& row_weights,
                 size_t n) -> std::pair<std::vector<int64_t>,
                                        std::vector<int64_t>> {
  const int64_t num_rows = static_cast<int64_t>(row_weights.size());
  std::vector<int64_t> begin(n, 0);
  std::vector<int64_t> end(n, 0);
  if (n == 0U || num_rows <= 0) {
    return {std::move(begin), std::move(end)};
  }
  if (static_cast<size_t>(num_rows) < n) {
    return even_split_z(num_rows, n);
  }
  int64_t total = 0;
  for (int64_t w : row_weights) {
    total += w;
  }
  if (total <= 0) {
    return even_split_z(num_rows, n);
  }
  if (n == 2U) {
    int64_t best_cut = 1;
    int64_t best_diff = std::numeric_limits<int64_t>::max();
    int64_t prefix = 0;
    for (int64_t cut = 1; cut < num_rows; ++cut) {
      prefix += row_weights[static_cast<size_t>(cut - 1)];
      const int64_t diff = std::abs(prefix - (total - prefix));
      if (diff < best_diff) {
        best_diff = diff;
        best_cut = cut;
      }
    }
    begin[0] = 0;
    end[0] = best_cut;
    begin[1] = best_cut;
    end[1] = num_rows;
    return {std::move(begin), std::move(end)};
  }
  int64_t cursor = 0;
  int64_t remaining_weight = total;
  size_t remaining_slabs = n;
  for (size_t s = 0; s < n; ++s) {
    begin[s] = cursor;
    if (s + 1U == n) {
      end[s] = num_rows;
      break;
    }
    const int64_t target = (remaining_weight + static_cast<int64_t>(remaining_slabs) - 1) /
                           static_cast<int64_t>(remaining_slabs);
    int64_t taken = 0;
    while (cursor < num_rows) {
      taken += row_weights[static_cast<size_t>(cursor)];
      ++cursor;
      const int64_t rows_left = num_rows - cursor;
      const size_t slabs_left = remaining_slabs - 1U;
      if (taken >= target && cursor > begin[s]) {
        break;
      }
      if (static_cast<size_t>(rows_left) < slabs_left) {
        break;
      }
    }
    end[s] = cursor;
    remaining_weight -= taken;
    --remaining_slabs;
  }
  return {std::move(begin), std::move(end)};
}

}  // namespace detail::mgrid::impl

template <typename SBlock>
MultiResBlock<SBlock>::MultiResBlock(System system,
                                     int64_3d finest_extent,
                                     int depth,
                                     const ActiveCellPredicate& is_active,
                                     int ref_factor,
                                     std::string name)
    : system_(std::move(system)),
      finest_extent_(finest_extent),
      name_(std::move(name)) {
  if (depth < 1) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::MultiResBlock: depth must be >= 1 (got {}).", depth);
  }
  if (!(SBlock::UserX == SBlock::UserY && SBlock::UserY == SBlock::UserZ)) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::MultiResBlock: SBlock user block must be cubic (UserX==UserY==UserZ); got "
        "({}, {}, {}).",
        SBlock::UserX, SBlock::UserY, SBlock::UserZ);
  }
  if (ref_factor != SBlock::UserX) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::MultiResBlock: ref_factor ({}) must equal SBlock::UserX ({}) — Neon's "
        "refFactor==userBlockSize octree coupling.",
        ref_factor, SBlock::UserX);
  }
  if (finest_extent.x <= 0 || finest_extent.y <= 0 || finest_extent.z <= 0) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::MultiResBlock: finest_extent components must be positive (got {}, {}, {}).",
        finest_extent.x, finest_extent.y, finest_extent.z);
  }
  const size_t n_gpus = system_.gpu_group().size();
  if (n_gpus == 0U) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::MultiResBlock: system has no GPUs (gpu_group().size() == 0).");
  }

  descriptor_ = Descriptor(depth, ref_factor);
  const int64_t R = ref_factor;
  const int coarsest = depth - 1;

  // Resolve the number of @b global slabs the coarsest level is partitioned into: the local
  // GPU count single-process, or the MPI world size (one-rank-per-GPU) under NCCL. Mirrors
  // Block::initialise_partition_ so the projected cut and Block's own partition agree.
  size_t global_slabs = n_gpus;
#ifdef CARBON_HAVE_MPI_NCCL
  if (system_.distributed() != nullptr) {
    global_slabs = static_cast<size_t>(system_.distributed()->world_size());
  }
#endif

  // 0. Co-resident z-cut. With more than one slab we partition the coarsest level by even
  //    z-brick rows and project that single cut down to every finer level (the same
  //    world-space planes), so a parent brick and all of its descendants land on one
  //    GPU/rank — making the cross-level path Transfer-free (docs/design_mgrid.md §4). With
  //    a single slab we hand Block an empty partition (the plain even split == the original
  //    single-GPU layout). Recomputed identically on every rank (even_split_z is pure), so
  //    no communication is needed to agree on the cut.
  std::vector<typename BlockGrid::ZBrickPartition> level_cuts;
  if (global_slabs > 1U) {
    const int64_t spacing_coarsest = descriptor_.voxel_spacing(coarsest);  // R^(depth-1)
    if (finest_extent.z % (static_cast<int64_t>(MemZ) * spacing_coarsest) != 0) {
      log::throw_with_log<std::invalid_argument>(
          "carbon::MultiResBlock: multi-slab partition ({} slabs) requires finest_extent.z ({}) "
          "to be a multiple of MemZ*R^(depth-1) ({}) so the coarsest z-cut projects to exact "
          "brick-row boundaries on every level.",
          global_slabs, finest_extent.z, static_cast<int64_t>(MemZ) * spacing_coarsest);
    }
    const int64_t coarsest_cand_z = detail::mgrid::impl::ceil_div_pos(
        descriptor_.level_extent(finest_extent, coarsest).z, MemZ);
    if (static_cast<size_t>(coarsest_cand_z) < global_slabs) {
      log::throw_with_log<std::invalid_argument>(
          "carbon::MultiResBlock: coarsest brick z-extent ({}) is smaller than the slab count "
          "({}); every slab must own at least one coarsest brick row.",
          coarsest_cand_z, global_slabs);
    }
    std::vector<int64_t> coarse_row_weights(static_cast<size_t>(coarsest_cand_z), 0);
    for (int64_t cz = 0; cz < coarsest_cand_z; ++cz) {
      int64_t weight = 0;
      for (int l = 0; l < depth; ++l) {
        const int64_3d ext = descriptor_.level_extent(finest_extent, l);
        const int64_t cand_x = detail::mgrid::impl::ceil_div_pos(ext.x, MemX);
        const int64_t cand_y = detail::mgrid::impl::ceil_div_pos(ext.y, MemY);
        const int64_t cand_z = detail::mgrid::impl::ceil_div_pos(ext.z, MemZ);
        const int64_t factor = spacing_coarsest / descriptor_.voxel_spacing(l);
        const int64_t bz0 = cz * factor;
        const int64_t bz1 = std::min((cz + 1) * factor, cand_z);
        for (int64_t bz = bz0; bz < bz1; ++bz) {
          for (int64_t by = 0; by < cand_y; ++by) {
            for (int64_t bx = 0; bx < cand_x; ++bx) {
              const int64_t ox = bx * MemX;
              const int64_t oy = by * MemY;
              const int64_t oz = bz * MemZ;
              bool any = false;
              for (int lz = 0; lz < MemZ && !any; ++lz) {
                for (int ly = 0; ly < MemY && !any; ++ly) {
                  for (int lx = 0; lx < MemX && !any; ++lx) {
                    if (is_active(l, int64_3d{ox + lx, oy + ly, oz + lz})) {
                      any = true;
                    }
                  }
                }
              }
              if (any) {
                ++weight;
              }
            }
          }
        }
      }
      coarse_row_weights[static_cast<size_t>(cz)] = weight;
    }
    auto [cb_begin, cb_end] =
        detail::mgrid::impl::weighted_split_z(coarse_row_weights, global_slabs);
    level_cuts.resize(static_cast<size_t>(depth));
    for (int l = 0; l < depth; ++l) {
      // spacing(coarsest)/spacing(l) == R^(coarsest-l): coarsest brick rows are this many
      // level-l brick rows tall. The z-alignment precondition guarantees this projection
      // exactly tiles the level-l candidate z-extent.
      const int64_t factor = spacing_coarsest / descriptor_.voxel_spacing(l);
      typename BlockGrid::ZBrickPartition zp;
      zp.begin.resize(global_slabs);
      zp.end.resize(global_slabs);
      for (size_t s = 0; s < global_slabs; ++s) {
        zp.begin[s] = cb_begin[s] * factor;
        zp.end[s] = cb_end[s] * factor;
      }
      level_cuts[static_cast<size_t>(l)] = std::move(zp);
    }
  }

  // 1. Per-level Block tower (level 0 finest). Each level's Block reuses the full
  //    block-discovery / bitmask / connectivity / partitioning machinery verbatim; the
  //    projected cut (when multi-slab) keeps the levels co-resident.
  levels_.reserve(static_cast<size_t>(depth));
  for (int l = 0; l < depth; ++l) {
    const int64_3d ext = descriptor_.level_extent(finest_extent, l);
    auto level_pred = [&is_active, l](const int64_3d& v) -> bool {
      return is_active(l, v);
    };
    std::string level_name = name_.empty() ? std::string{} : (name_ + ":L" + std::to_string(l));
    if (level_cuts.empty()) {
      levels_.emplace_back(system_, ext, level_pred, std::move(level_name));
    } else {
      levels_.emplace_back(system_, ext, level_pred, std::move(level_cuts[static_cast<size_t>(l)]),
                           std::move(level_name));
    }
  }

  // 2. Per-(level, GPU) origin -> local block_idx map. Origins are brick-aligned, so o / Mem
  //    is the brick lattice index used as the lookup key. Co-residency guarantees a block's
  //    parent/child lives on the @b same GPU, so the cross-level lookups below only ever
  //    consult brick_map[other_level][same_gpu].
  using BrickKey = std::tuple<int64_t, int64_t, int64_t>;
  std::vector<std::vector<std::map<BrickKey, uint32_t>>> brick_map(
      static_cast<size_t>(depth), std::vector<std::map<BrickKey, uint32_t>>(n_gpus));
  for (int l = 0; l < depth; ++l) {
    for (size_t g = 0; g < n_gpus; ++g) {
      const auto& origins = levels_[static_cast<size_t>(l)].host_origins(g);
      for (uint32_t b = 0; b < origins.size(); ++b) {
        const int32_3d o = origins[b];
        brick_map[static_cast<size_t>(l)][g][BrickKey{o.x / MemX, o.y / MemY, o.z / MemZ}] = b;
      }
    }
  }

  // 3a. parent_block_id[l][g][b]: the level-(l+1) brick (on GPU g) whose sub-region holds
  //     block b's parent voxels. block b's origin maps up by O / R; the containing brick is
  //     (O/R)/Mem. Co-residency ensures that parent brick is on GPU g too.
  parent_block_id_host_.assign(static_cast<size_t>(depth),
                               std::vector<std::vector<uint32_t>>(n_gpus));
  for (int l = 0; l + 1 < depth; ++l) {
    for (size_t g = 0; g < n_gpus; ++g) {
      const auto& origins = levels_[static_cast<size_t>(l)].host_origins(g);
      std::vector<uint32_t>& table = parent_block_id_host_[static_cast<size_t>(l)][g];
      table.assign(origins.size(), detail::block::kInvalidBlockIdx);
      const auto& parent_map = brick_map[static_cast<size_t>(l + 1)][g];
      for (uint32_t b = 0; b < origins.size(); ++b) {
        const int32_3d o = origins[b];
        const int64_t pbx = (static_cast<int64_t>(o.x) / R) / MemX;
        const int64_t pby = (static_cast<int64_t>(o.y) / R) / MemY;
        const int64_t pbz = (static_cast<int64_t>(o.z) / R) / MemZ;
        const auto it = parent_map.find(BrickKey{pbx, pby, pbz});
        if (it != parent_map.end()) {
          table[b] = it->second;
        }
      }
    }
  }

  // 3b. child_block_id[l][g][b·VoxelCount + pitch]: per voxel of block b at level l (on GPU
  //     g), the level-(l-1) brick (on GPU g) containing that voxel's children. The voxel's
  //     global coord (level-l units) is O + intra; the child brick is (R·global)/Mem. Set for
  //     every voxel — correctness only matters for active voxels, and a non-refined voxel's
  //     lookup naturally misses (-> kInvalid).
  child_block_id_host_.assign(static_cast<size_t>(depth),
                              std::vector<std::vector<uint32_t>>(n_gpus));
  for (int l = 1; l < depth; ++l) {
    for (size_t g = 0; g < n_gpus; ++g) {
      const auto& origins = levels_[static_cast<size_t>(l)].host_origins(g);
      std::vector<uint32_t>& table = child_block_id_host_[static_cast<size_t>(l)][g];
      table.assign(static_cast<size_t>(origins.size()) * static_cast<size_t>(VoxelCount),
                   detail::block::kInvalidBlockIdx);
      const auto& child_map = brick_map[static_cast<size_t>(l - 1)][g];
      for (uint32_t b = 0; b < origins.size(); ++b) {
        const int32_3d o = origins[b];
        for (int iz = 0; iz < MemZ; ++iz) {
          for (int iy = 0; iy < MemY; ++iy) {
            for (int ix = 0; ix < MemX; ++ix) {
              const int64_t gx = static_cast<int64_t>(o.x) + ix;
              const int64_t gy = static_cast<int64_t>(o.y) + iy;
              const int64_t gz = static_cast<int64_t>(o.z) + iz;
              const int64_t cbx = (R * gx) / MemX;
              const int64_t cby = (R * gy) / MemY;
              const int64_t cbz = (R * gz) / MemZ;
              const auto it = child_map.find(BrickKey{cbx, cby, cbz});
              if (it != child_map.end()) {
                const int64_t pitch = static_cast<int64_t>(b) * static_cast<int64_t>(VoxelCount) +
                                      ix + static_cast<int64_t>(MemX) * iy +
                                      static_cast<int64_t>(MemX) * static_cast<int64_t>(MemY) * iz;
                table[static_cast<size_t>(pitch)] = it->second;
              }
            }
          }
        }
      }
    }
  }

  // 4. Upload tables to device, one allocation per GPU. Host staging buffers are kept alive
  //    in a vector until after stream.wait() so the async copies see live source memory. A
  //    level whose table is empty on every GPU keeps a default (size-0) Buffer.
  auto& gpus = system_.gpu_group();
  Stream stream = system_.stream();
  parent_block_id_dev_.assign(static_cast<size_t>(depth), Buffer<uint32_t>{});
  child_block_id_dev_.assign(static_cast<size_t>(depth), Buffer<uint32_t>{});
  std::vector<detail::Allocation<uint32_t>> staging;
  staging.reserve(static_cast<size_t>(2 * depth) * n_gpus);
  auto upload_level =
      [&](const std::vector<std::vector<uint32_t>>& host_per_gpu) -> Buffer<uint32_t> {
    std::vector<size_t> counts(n_gpus, 0U);
    bool any = false;
    for (size_t g = 0; g < n_gpus; ++g) {
      counts[g] = host_per_gpu[g].size();
      any = any || (counts[g] != 0U);
    }
    if (!any) {
      return Buffer<uint32_t>{};
    }
    Buffer<uint32_t> dev = gpus.template buffer<uint32_t>(counts);
    for (size_t g = 0; g < n_gpus; ++g) {
      if (host_per_gpu[g].empty()) {
        continue;
      }
      auto h = Cpu::buffer<uint32_t>(host_per_gpu[g].size());
      std::copy(host_per_gpu[g].begin(), host_per_gpu[g].end(), h.data());
      system_.async_copy(h, dev[g], sizeof(uint32_t) * host_per_gpu[g].size(), stream);
      staging.push_back(std::move(h));
    }
    return dev;
  };
  for (int l = 0; l < depth; ++l) {
    parent_block_id_dev_[static_cast<size_t>(l)] =
        upload_level(parent_block_id_host_[static_cast<size_t>(l)]);
    child_block_id_dev_[static_cast<size_t>(l)] =
        upload_level(child_block_id_host_[static_cast<size_t>(l)]);
  }
  stream.wait();
}

template <typename SBlock>
template <typename T,
          int Components>
auto
MultiResBlock<SBlock>::newField(MemorySide side,
                                std::string name) const -> Field<T,
                                                                 Components> {
  using LevelField = typename Field<T, Components>::LevelField;
  std::vector<LevelField> level_fields;
  level_fields.reserve(levels_.size());
  for (int l = 0; l < depth(); ++l) {
    std::string level_name = name.empty() ? std::string{} : (name + ":L" + std::to_string(l));
    level_fields.emplace_back(levels_[static_cast<size_t>(l)].template newField<T, Components>(
        side, std::move(level_name)));
  }
  return Field<T, Components>{std::move(level_fields), this, std::move(name)};
}

}  // namespace carbon

namespace carbon::detail::mgrid {

template <typename T,
          int Components,
          typename SBlock>
auto
Field<T,
      Components,
      SBlock>::make_partition(int level,
                              size_t gpu_id) const -> PartitionT {
  using SpanT = typename PartitionT::SpanT;
  const auto level_ptr = [](const LevelField& f, size_t g) -> T* {
    const auto& parts = f.partitions();
    return g < parts.size() ? parts[g].ptr() : nullptr;
  };

  T* own = level_ptr(level_field(level), gpu_id);
  const int ref_factor = owner_->ref_factor();

  T* parent_mem = nullptr;
  SpanT parent_span{};
  const uint32_t* parent_block_id = nullptr;
  if (level + 1 < owner_->depth()) {
    parent_mem = level_ptr(level_field(level + 1), gpu_id);
    parent_span = owner_->grid(level + 1).span(gpu_id);
    parent_block_id = owner_->parent_block_id_dev(level, gpu_id);
  }

  T* child_mem = nullptr;
  SpanT child_span{};
  const uint32_t* child_block_id = nullptr;
  if (level >= 1) {
    child_mem = level_ptr(level_field(level - 1), gpu_id);
    child_span = owner_->grid(level - 1).span(gpu_id);
    child_block_id = owner_->child_block_id_dev(level, gpu_id);
  }

  return PartitionT{own,       ref_factor, parent_mem,    parent_span, parent_block_id,
                    child_mem, child_span, child_block_id};
}

template <typename T,
          int Components,
          typename SBlock>
auto
Field<T,
      Components,
      SBlock>::make_host_partition(int level,
                                   size_t gpu_id) const -> PartitionT {
  using SpanT = typename PartitionT::SpanT;
  const auto host_level_ptr = [](const LevelField& f, size_t g) -> T* {
    const auto& parts = f.host_partitions();
    return g < parts.size() ? const_cast<T*>(parts[g].data()) : nullptr;
  };

  T* own = host_level_ptr(level_field(level), gpu_id);
  const int ref_factor = owner_->ref_factor();

  T* parent_mem = nullptr;
  SpanT parent_span{};
  const uint32_t* parent_block_id = nullptr;
  if (level + 1 < owner_->depth()) {
    parent_mem = host_level_ptr(level_field(level + 1), gpu_id);
    parent_span = owner_->grid(level + 1).host_spans()[gpu_id];
    parent_block_id = owner_->parent_block_id_host(level, gpu_id).data();
  }

  T* child_mem = nullptr;
  SpanT child_span{};
  const uint32_t* child_block_id = nullptr;
  if (level >= 1) {
    child_mem = host_level_ptr(level_field(level - 1), gpu_id);
    child_span = owner_->grid(level - 1).host_spans()[gpu_id];
    child_block_id = owner_->child_block_id_host(level, gpu_id).data();
  }

  return PartitionT{own,       ref_factor, parent_mem,    parent_span, parent_block_id,
                    child_mem, child_span, child_block_id};
}

template <typename T,
          int Components,
          typename SBlock>
auto
Field<T,
      Components,
      SBlock>::level_view(int level) const -> LevelView {
  const size_t n_gpus = owner_->system().gpu_group().size();
  std::vector<PartitionT> parts;
  parts.reserve(n_gpus);
  for (size_t g = 0; g < n_gpus; ++g) {
    parts.push_back(make_partition(level, g));
  }
  std::vector<PartitionT> host_parts;
  if (!level_field(level).cpu_buffers().empty()) {
    host_parts.reserve(n_gpus);
    for (size_t g = 0; g < n_gpus; ++g) {
      host_parts.push_back(make_host_partition(level, g));
    }
  }
  std::string view_name = name_ + ":L" + std::to_string(level);
  return LevelView{&level_field(level), std::move(parts), std::move(host_parts),
                   std::move(view_name)};
}

}  // namespace carbon::detail::mgrid
