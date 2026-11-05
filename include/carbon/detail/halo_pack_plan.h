#pragma once

/**
 * @file halo_pack_plan.h
 * @brief Host-side planning and shared index math for the *packed* multi-axis halo exchange.
 *
 * Why this exists
 * ---------------
 * The run-based multi-axis builder in @c halo_distributed_build.h describes every halo face as a list of
 * contiguous runs. In the x-fastest layout that is one run per (y, z) row for an X face
 * (@c haloed.y * @c haloed.z runs of @c halo.x elements) and one run per z-plane for a Y face.
 * Each run becomes one @c ncclSend / @c ncclRecv (cross-rank) or one async copy (intra). At a
 * 256^3 local block that is ~66k transfers per face per component: the exchange is dominated by
 * per-message overhead, not bandwidth.
 *
 * The packed path instead gathers a face into one contiguous staging region, moves it with ONE
 * message per (axis, side), and scatters it into the halo on the other side:
 *
 *     gather (kernel) -> [intra copies] -> NCCL group (1 Send + 1 Recv per side) -> scatter (kernel)
 *
 * X and Y faces are always packed. A Z face is one contiguous run PER COMPONENT, so it only needs
 * packing when it carries several components (a D3Q19 population field moves 5 populations per
 * face with the lattice filter, 19 without): without packing that is one NCCL message per
 * component per direction, and the messages that cross a node boundary are issued back to back
 * through the network proxy. The caller decides (see HaloDistributed::build_multi_axis_); the plan builder
 * handles Z like any other axis.
 *
 * This header is host-only and has no backend dependency, so the whole plan (extents, component
 * lists, peers, message ordering) and the per-thread index math are unit-testable with a plain
 * C++ compiler (see tests/test_halo_pack_plan.cpp). The CUDA/HIP kernels in src/halo_pack.cu are
 * thin wrappers that forward their launch coordinates to @ref pack_gather_thread /
 * @ref pack_scatter_thread, so the tested code IS the device code.
 *
 * Face extents (design: staged X -> Y -> Z)
 * -----------------------------------------
 *  - X pass : halo.x  x  OWNED y  x  OWNED z
 *  - Y pass : filled-x  x  halo.y  x  OWNED z      ("filled-x" = x range already written by the
 *                                                   X pass; exterior x halos on open axes are
 *                                                   excluded, so they are never overwritten)
 *  - Z pass : filled-x  x  filled-y  x  halo.z      (same rule for y: the X and Y passes have
 *                                                   already filled every halo cell that has a
 *                                                   neighbour, corners included)
 * The run-based path sends the full haloed y/z (resp. z) range instead; those extra rows carry
 * not-yet-filled halo cells (wasted bandwidth) and can overwrite exterior halo cells on end ranks
 * of open axes.
 */

#include <carbon/boundary_mode_distributed.h>
#include <carbon/cuda_annotation.h>
#include <carbon/detail/halo_distributed_build.h>
#include <carbon/process_grid.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#if defined(__GNUC__) || defined(__CUDACC__) || defined(__HIPCC__)
#define CARBON_PACK_RESTRICT __restrict__
#else
#define CARBON_PACK_RESTRICT
#endif

namespace carbon::detail::halo_pack {

using halo_distributed_build::AxisGeometry;
using halo_distributed_build::AxisSide;

/** Maximum components packed by one kernel launch (kernel argument struct is fixed-size). */
inline constexpr uint32_t kMaxPackComps = 32U;
/** Threads per block along x for pack/unpack kernels. */
inline constexpr uint32_t kPackBlockX = 256U;

/** Axis-aligned box in *field buffer* coordinates (x fastest, includes halos). */
struct PackBox {
  int64_t lo_x{};
  int64_t lo_y{};
  int64_t lo_z{};
  int64_t ext_x{};
  int64_t ext_y{};
  int64_t ext_z{};

  [[nodiscard]] constexpr auto
  volume() const noexcept -> size_t {
    return static_cast<size_t>(ext_x) * static_cast<size_t>(ext_y) * static_cast<size_t>(ext_z);
  }
};

/**
 * Everything a pack/unpack kernel needs, as a POD passed by value.
 *
 * Field addressing (matches DenseDistributed): element (c, x, y, z) lives at
 *   c * comp_stride + z * pitch + y * haloed_x + x.
 * Staging addressing: plane k (= the k-th packed component) is contiguous and x-fastest:
 *   k * stage_comp_stride + (iz * ext_y + iy) * ext_x + ix,   stage_comp_stride = ext_x*ext_y*ext_z.
 */
struct PackKernelArgs {
  int64_t lo_x{};
  int64_t lo_y{};
  int64_t lo_z{};
  int64_t haloed_x{};
  int64_t pitch{};
  int64_t comp_stride{};
  int64_t stage_comp_stride{};
  uint32_t ext_x{};
  uint32_t ext_y{};
  uint32_t ext_z{};
  uint32_t row{};  ///< ext_x * ext_y (threads along the flattened x/y plane of the box)
  uint32_t num_comps{};
  int32_t comps[kMaxPackComps]{};
};

struct PackLaunchShape {
  uint32_t grid_x{};
  uint32_t grid_y{};
  uint32_t grid_z{};
  uint32_t block_x{};
};

/** Launch geometry: x = flattened (x,y) row, y = grid-stride over z, z = packed component. */
[[nodiscard]] inline auto
pack_launch_shape(const PackKernelArgs& a) noexcept -> PackLaunchShape {
  PackLaunchShape s;
  s.block_x = kPackBlockX;
  s.grid_x = (a.row + kPackBlockX - 1U) / kPackBlockX;
  s.grid_y = std::min<uint32_t>(a.ext_z, 65535U);
  s.grid_z = a.num_comps;
  return s;
}

/**
 * One device thread of the gather kernel (field -> staging). Host+device so the same code runs
 * in the CUDA/HIP kernel and in the host unit test.
 */
template <typename W>
CARBON_HD inline void
pack_gather_thread(const W* CARBON_PACK_RESTRICT field,
                   W* CARBON_PACK_RESTRICT stage,
                   const PackKernelArgs& a,
                   uint32_t bx,
                   uint32_t by,
                   uint32_t bz,
                   uint32_t tx,
                   uint32_t bdx,
                   uint32_t gdy) {
  const uint32_t j = bx * bdx + tx;
  if (j >= a.row || bz >= a.num_comps) {
    return;
  }
  const uint32_t iy = j / a.ext_x;
  const uint32_t ix = j - iy * a.ext_x;
  const int64_t fbase = static_cast<int64_t>(a.comps[bz]) * a.comp_stride +
                        (a.lo_y + static_cast<int64_t>(iy)) * a.haloed_x + a.lo_x +
                        static_cast<int64_t>(ix);
  const int64_t sbase = static_cast<int64_t>(bz) * a.stage_comp_stride + static_cast<int64_t>(j);
  for (uint32_t iz = by; iz < a.ext_z; iz += gdy) {
    stage[sbase + static_cast<int64_t>(iz) * static_cast<int64_t>(a.row)] =
        field[fbase + (a.lo_z + static_cast<int64_t>(iz)) * a.pitch];
  }
}

/** One device thread of the scatter kernel (staging -> field). */
template <typename W>
CARBON_HD inline void
pack_scatter_thread(const W* CARBON_PACK_RESTRICT stage,
                    W* CARBON_PACK_RESTRICT field,
                    const PackKernelArgs& a,
                    uint32_t bx,
                    uint32_t by,
                    uint32_t bz,
                    uint32_t tx,
                    uint32_t bdx,
                    uint32_t gdy) {
  const uint32_t j = bx * bdx + tx;
  if (j >= a.row || bz >= a.num_comps) {
    return;
  }
  const uint32_t iy = j / a.ext_x;
  const uint32_t ix = j - iy * a.ext_x;
  const int64_t fbase = static_cast<int64_t>(a.comps[bz]) * a.comp_stride +
                        (a.lo_y + static_cast<int64_t>(iy)) * a.haloed_x + a.lo_x +
                        static_cast<int64_t>(ix);
  const int64_t sbase = static_cast<int64_t>(bz) * a.stage_comp_stride + static_cast<int64_t>(j);
  for (uint32_t iz = by; iz < a.ext_z; iz += gdy) {
    field[fbase + (a.lo_z + static_cast<int64_t>(iz)) * a.pitch] =
        stage[sbase + static_cast<int64_t>(iz) * static_cast<int64_t>(a.row)];
  }
}

/** Host reference of a full gather launch (used by tests and as executable documentation). */
template <typename W>
inline auto
host_gather(const W* field,
            W* stage,
            const PackKernelArgs& a) -> void {
  const PackLaunchShape s = pack_launch_shape(a);
  for (uint32_t bz = 0; bz < s.grid_z; ++bz) {
    for (uint32_t by = 0; by < s.grid_y; ++by) {
      for (uint32_t bx = 0; bx < s.grid_x; ++bx) {
        for (uint32_t tx = 0; tx < s.block_x; ++tx) {
          pack_gather_thread<W>(field, stage, a, bx, by, bz, tx, s.block_x, s.grid_y);
        }
      }
    }
  }
}

/** Host reference of a full scatter launch. */
template <typename W>
inline auto
host_scatter(const W* stage,
             W* field,
             const PackKernelArgs& a) -> void {
  const PackLaunchShape s = pack_launch_shape(a);
  for (uint32_t bz = 0; bz < s.grid_z; ++bz) {
    for (uint32_t by = 0; by < s.grid_y; ++by) {
      for (uint32_t bx = 0; bx < s.grid_x; ++bx) {
        for (uint32_t tx = 0; tx < s.block_x; ++tx) {
          pack_scatter_thread<W>(stage, field, a, bx, by, bz, tx, s.block_x, s.grid_y);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

enum class MsgOp : std::uint8_t {
  Send = 0,
  Recv = 1,
};

/** One kernel launch (<= kMaxPackComps components) plus where its staging region starts. */
struct PackStep {
  PackKernelArgs args{};
  size_t stage_offset{};  ///< element offset into the HaloDistributed's staging buffer
};

/** One cross-rank message over a contiguous staging region. */
struct PackMsg {
  int peer{};
  MsgOp op{MsgOp::Send};
  size_t first{};  ///< element offset into the staging buffer
  size_t count{};  ///< elements
  /// (axis, src_rank, src_side, dst_rank, dst_side): identical on both ends of a message.
  std::tuple<int, int, int, int, int> key{};
};

struct AxisPackPlan {
  std::vector<PackStep> gathers{};   ///< run BEFORE the exchange (cross-rank sends + self wraps)
  std::vector<PackStep> scatters{};  ///< run AFTER the exchange (cross-rank recvs + self wraps)
  std::vector<PackMsg> msgs{};       ///< cross-rank messages (sort with @ref sort_msgs first)
  size_t stage_elems{};              ///< staging elements consumed by this plan
};

/**
 * NCCL point-to-point has no tags, so both ends of every peer pair must issue their operations in
 * the same order: all sends before all recvs per peer, ordered by a key identical on both ends.
 * With only two ranks along a periodic axis the same peer appears on both sides, which is exactly
 * why the key (not insertion order) must decide.
 */
inline auto
sort_msgs(std::vector<PackMsg>& msgs) -> void {
  std::sort(msgs.begin(), msgs.end(), [](const PackMsg& a, const PackMsg& b) {
    return std::tie(a.peer, a.op, a.key) < std::tie(b.peer, b.op, b.key);
  });
}

/**
 * x and y ranges (buffer coordinates) already filled by earlier passes, i.e. the part of a face
 * that carries valid data. Exterior halos on open axes are excluded so user boundary data there
 * is never overwritten. Both ends of a message compute the same range because their neighbours
 * along the passed axis share the other two process coordinates.
 */
struct FilledRange {
  int64_t x_lo{};
  int64_t x_hi{};
  int64_t y_lo{};
  int64_t y_hi{};
};

/** Owned face of @p side: the cells this rank sends to its neighbour on @p side. */
[[nodiscard]] inline auto
pack_face_box(const AxisGeometry& g,
              Axis axis,
              AxisSide side,
              const FilledRange& r) -> PackBox {
  const bool lower = side == AxisSide::Lower;
  switch (axis) {
    case Axis::X:
      return PackBox{lower ? g.halo.x : g.owned.x, g.halo.y, g.halo.z,
                     g.halo.x,                     g.owned.y, g.owned.z};
    case Axis::Y:
      return PackBox{r.x_lo,        lower ? g.halo.y : g.owned.y, g.halo.z,
                     r.x_hi - r.x_lo, g.halo.y,                   g.owned.z};
    case Axis::Z:
      return PackBox{r.x_lo,          r.y_lo,          lower ? g.halo.z : g.owned.z,
                     r.x_hi - r.x_lo, r.y_hi - r.y_lo, g.halo.z};
  }
  throw std::invalid_argument("pack_face_box: unknown axis");
}

/** Halo region of @p side: the cells this rank receives from its neighbour on @p side. */
[[nodiscard]] inline auto
pack_halo_box(const AxisGeometry& g,
              Axis axis,
              AxisSide side,
              const FilledRange& r) -> PackBox {
  const bool lower = side == AxisSide::Lower;
  switch (axis) {
    case Axis::X:
      return PackBox{lower ? 0 : g.halo.x + g.owned.x, g.halo.y, g.halo.z,
                     g.halo.x,                         g.owned.y, g.owned.z};
    case Axis::Y:
      return PackBox{r.x_lo,          lower ? 0 : g.halo.y + g.owned.y, g.halo.z,
                     r.x_hi - r.x_lo, g.halo.y,                         g.owned.z};
    case Axis::Z:
      return PackBox{r.x_lo,          r.y_lo,          lower ? 0 : g.halo.z + g.owned.z,
                     r.x_hi - r.x_lo, r.y_hi - r.y_lo, g.halo.z};
  }
  throw std::invalid_argument("pack_halo_box: unknown axis");
}

/** Convenience overloads keeping the X/Y-only call shape (y range = whole haloed y). */
[[nodiscard]] inline auto
pack_face_box(const AxisGeometry& g,
              Axis axis,
              AxisSide side,
              int64_t x_lo,
              int64_t x_hi) -> PackBox {
  return pack_face_box(g, axis, side, FilledRange{x_lo, x_hi, 0, g.haloed.y});
}

[[nodiscard]] inline auto
pack_halo_box(const AxisGeometry& g,
              Axis axis,
              AxisSide side,
              int64_t x_lo,
              int64_t x_hi) -> PackBox {
  return pack_halo_box(g, axis, side, FilledRange{x_lo, x_hi, 0, g.haloed.y});
}

namespace detail_plan {

/** Append one launch per chunk of <= kMaxPackComps components for @p box / @p comps. */
inline auto
append_steps(std::vector<PackStep>& out,
             const PackBox& box,
             const std::vector<int>& comps,
             const AxisGeometry& g,
             int64_t comp_stride,
             size_t region_offset) -> void {
  if (box.volume() == 0U || comps.empty()) {
    return;
  }
  if (box.ext_x * box.ext_y > static_cast<int64_t>(UINT32_MAX) ||
      box.ext_z > static_cast<int64_t>(UINT32_MAX)) {
    throw std::invalid_argument("halo pack: face extent exceeds 32-bit launch range");
  }
  const int64_t per_comp = static_cast<int64_t>(box.volume());
  for (size_t first = 0; first < comps.size(); first += kMaxPackComps) {
    const size_t n = std::min<size_t>(kMaxPackComps, comps.size() - first);
    PackStep step;
    PackKernelArgs& a = step.args;
    a.lo_x = box.lo_x;
    a.lo_y = box.lo_y;
    a.lo_z = box.lo_z;
    a.haloed_x = g.haloed.x;
    a.pitch = g.pitch;
    a.comp_stride = comp_stride;
    a.stage_comp_stride = per_comp;
    a.ext_x = static_cast<uint32_t>(box.ext_x);
    a.ext_y = static_cast<uint32_t>(box.ext_y);
    a.ext_z = static_cast<uint32_t>(box.ext_z);
    a.row = a.ext_x * a.ext_y;
    a.num_comps = static_cast<uint32_t>(n);
    for (size_t k = 0; k < n; ++k) {
      a.comps[k] = comps[first + k];
    }
    step.stage_offset = region_offset + first * static_cast<size_t>(per_comp);
    out.push_back(step);
  }
}

}  // namespace detail_plan

/**
 * Build the packed exchange plan for one X or Y pass of one rank.
 *
 * @param stage_base element offset at which this plan's staging regions start (so several passes
 *                   can share one staging buffer)
 * @param active_send / active_recv  (component, axis, side) -> bool; same predicates the
 *                   run-based builder uses, so lattice direction filtering carries over.
 *
 * Semantics per side, identical to the run-based builder:
 *  - cross-rank: send OUR owned face at `side` to the neighbour on `side`; receive INTO OUR halo at
 *    `side` from that neighbour's opposite face.
 *  - self wrap (periodic axis owned by a single rank): halo at `side` <- OUR owned face on the
 *    opposite side, staged through the same buffer (no message).
 */
template <typename SendActive,
          typename RecvActive>
[[nodiscard]] inline auto
build_axis_pack_plan(const ProcessGrid& pg,
                     BoundaryModeDistributed boundary,
                     int32_3d coords,
                     int32_t local_rank,
                     const AxisGeometry& g,
                     Axis axis,
                     int components,
                     int64_t comp_stride,
                     SendActive&& active_send,
                     RecvActive&& active_recv,
                     size_t stage_base) -> AxisPackPlan {
  AxisPackPlan plan;
  const int64_t owned = axis == Axis::X ? g.owned.x : (axis == Axis::Y ? g.owned.y : g.owned.z);
  const int64_t halo = axis == Axis::X ? g.halo.x : (axis == Axis::Y ? g.halo.y : g.halo.z);
  if (halo <= 0) {
    return plan;
  }
  if (owned < halo) {
    throw std::invalid_argument("halo pack: owned extent is smaller than the halo");
  }

  // Ranges already filled when this pass starts (see file comment). Neighbours along this axis
  // share our other two coordinates, so both ends of a message compute the same ranges.
  FilledRange fill{0, g.haloed.x, 0, g.haloed.y};
  if (axis != Axis::X && g.halo.x > 0) {
    const bool has_lo =
        halo_distributed_build::axis_neighbour(pg, coords, Axis::X, boundary, AxisSide::Lower).present;
    const bool has_hi =
        halo_distributed_build::axis_neighbour(pg, coords, Axis::X, boundary, AxisSide::Upper).present;
    fill.x_lo = has_lo ? 0 : g.halo.x;
    fill.x_hi = g.haloed.x - (has_hi ? 0 : g.halo.x);
  }
  if (axis == Axis::Z && g.halo.y > 0) {
    const bool has_lo =
        halo_distributed_build::axis_neighbour(pg, coords, Axis::Y, boundary, AxisSide::Lower).present;
    const bool has_hi =
        halo_distributed_build::axis_neighbour(pg, coords, Axis::Y, boundary, AxisSide::Upper).present;
    fill.y_lo = has_lo ? 0 : g.halo.y;
    fill.y_hi = g.haloed.y - (has_hi ? 0 : g.halo.y);
  }

  size_t cursor = stage_base;
  const int axis_id = static_cast<int>(axis);

  for (const AxisSide side : {AxisSide::Lower, AxisSide::Upper}) {
    const auto nb = halo_distributed_build::axis_neighbour(pg, coords, axis, boundary, side);
    if (!nb.present) {
      continue;
    }
    const AxisSide opposite = side == AxisSide::Lower ? AxisSide::Upper : AxisSide::Lower;
    const int local_side = static_cast<int>(side);
    const int peer_side = 1 - local_side;

    std::vector<int> send_comps;
    std::vector<int> recv_comps;
    for (int c = 0; c < components; ++c) {
      if (active_send(c, axis, side)) {
        send_comps.push_back(c);
      }
      if (active_recv(c, axis, side)) {
        recv_comps.push_back(c);
      }
    }

    const PackBox halo_box = pack_halo_box(g, axis, side, fill);

    if (nb.rank == local_rank) {
      // Self wrap: gather our `opposite` owned face, scatter into our `side` halo.
      const PackBox src_box = pack_face_box(g, axis, opposite, fill);
      const size_t region = cursor;
      detail_plan::append_steps(plan.gathers, src_box, recv_comps, g, comp_stride, region);
      detail_plan::append_steps(plan.scatters, halo_box, recv_comps, g, comp_stride, region);
      cursor += recv_comps.size() * halo_box.volume();
      continue;
    }

    // Cross-rank: one send region + one recv region per side.
    const PackBox send_box = pack_face_box(g, axis, side, fill);
    if (!send_comps.empty() && send_box.volume() > 0U) {
      const size_t region = cursor;
      detail_plan::append_steps(plan.gathers, send_box, send_comps, g, comp_stride, region);
      const size_t count = send_comps.size() * send_box.volume();
      plan.msgs.push_back({nb.rank,
                           MsgOp::Send,
                           region,
                           count,
                           {axis_id, static_cast<int>(local_rank), local_side,
                            static_cast<int>(nb.rank), peer_side}});
      cursor += count;
    }
    if (!recv_comps.empty() && halo_box.volume() > 0U) {
      const size_t region = cursor;
      detail_plan::append_steps(plan.scatters, halo_box, recv_comps, g, comp_stride, region);
      const size_t count = recv_comps.size() * halo_box.volume();
      plan.msgs.push_back({nb.rank,
                           MsgOp::Recv,
                           region,
                           count,
                           {axis_id, static_cast<int>(nb.rank), peer_side,
                            static_cast<int>(local_rank), local_side}});
      cursor += count;
    }
  }

  plan.stage_elems = cursor - stage_base;
  return plan;
}

}  // namespace carbon::detail::halo_pack
