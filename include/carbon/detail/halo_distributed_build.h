#pragma once

/**
 * @file halo_distributed_build.h
 * @brief Per-axis geometry helpers for @ref HaloDistributed.
 *
 * A distributed rank exchanges halo faces with up to two neighbours on each of the three axes. The
 * helpers here describe, for one rank's storage geometry, which contiguous runs of the field make
 * up the halo face and the owned face on a given axis and side, and who the neighbouring rank is
 * (with periodic wrap). They are pure host-side index arithmetic: no device state, no MPI.
 *
 * Storage addressing (matches @c DenseDistributed::Span): element @c (x, y, z) of one component
 * lives at @c z * pitch + y * haloed.x + x, where @c pitch is the number of elements per XY plane.
 *
 * Kept apart from @c halo_build.h so the Z-slab @ref Halo keeps its original helper set untouched.
 */

#include <carbon/boundary_mode_distributed.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/process_grid.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace carbon::detail::halo_distributed_build {

enum class AxisSide : std::uint8_t {
  Lower = 0,
  Upper = 1,
};

struct AxisNeighbour {
  bool present{false};
  bool self{false};
  int32_3d coords{};
  int32_t rank{-1};
};

struct AxisGeometry {
  int64_3d owned{};
  int64_3d halo{};
  int64_3d haloed{};
  int64_t pitch{};
};

struct AxisRun {
  size_t offset{};
  size_t count{};
};

template <typename V>
[[nodiscard]] constexpr auto
axis_value(V v,
           Axis axis) noexcept -> int64_t {
  switch (axis) {
    case Axis::X:
      return static_cast<int64_t>(v.x);
    case Axis::Y:
      return static_cast<int64_t>(v.y);
    case Axis::Z:
      return static_cast<int64_t>(v.z);
  }
  return 0;
}

[[nodiscard]] inline auto
axis_neighbour(const ProcessGrid& grid,
               int32_3d coords,
               Axis axis,
               BoundaryModeDistributed boundary,
               AxisSide side) -> AxisNeighbour {
  const int32_t n = static_cast<int32_t>(axis_value(grid.dims(), axis));
  if (n <= 0) {
    return {};
  }

  const int32_t current = static_cast<int32_t>(axis_value(coords, axis));
  int32_t next = current + (side == AxisSide::Lower ? -1 : 1);

  if (next < 0 || next >= n) {
    if (!is_periodic(boundary, axis)) {
      return {};
    }
    next = (next + n) % n;
  }

  int32_3d nb = coords;
  switch (axis) {
    case Axis::X:
      nb.x = next;
      break;
    case Axis::Y:
      nb.y = next;
      break;
    case Axis::Z:
      nb.z = next;
      break;
  }

  return AxisNeighbour{
      true,
      next == current,
      nb,
      grid.rank_of(nb),
  };
}

[[nodiscard]] inline auto
axis_halo_runs(const AxisGeometry& g,
               Axis axis,
               AxisSide side) -> std::vector<AxisRun> {
  std::vector<AxisRun> runs;

  const int64_t owned = axis_value(g.owned, axis);
  const int64_t halo = axis_value(g.halo, axis);
  if (halo <= 0) {
    return runs;
  }
  if (owned < halo) {
    log::throw_with_log<std::invalid_argument>(
        "axis_halo_runs: owned extent {} is smaller than halo {} on axis {}", owned, halo,
        static_cast<int>(axis));
  }

  switch (axis) {
    case Axis::X: {
      const int64_t start = side == AxisSide::Lower ? 0 : g.halo.x + g.owned.x;
      const size_t count = static_cast<size_t>(g.halo.x);
      runs.reserve(static_cast<size_t>(g.haloed.y * g.haloed.z));
      for (int64_t z = 0; z < g.haloed.z; ++z) {
        for (int64_t y = 0; y < g.haloed.y; ++y) {
          runs.push_back({
              static_cast<size_t>(z * g.pitch + y * g.haloed.x) + static_cast<size_t>(start),
              count,
          });
        }
      }
      break;
    }

    case Axis::Y: {
      const int64_t start = (side == AxisSide::Lower ? 0 : g.halo.y + g.owned.y) * g.haloed.x;
      const size_t count = static_cast<size_t>(g.halo.y * g.haloed.x);
      runs.reserve(static_cast<size_t>(g.haloed.z));
      for (int64_t z = 0; z < g.haloed.z; ++z) {
        runs.push_back({
            static_cast<size_t>(z * g.pitch) + static_cast<size_t>(start),
            count,
        });
      }
      break;
    }

    case Axis::Z: {
      const int64_t start = (side == AxisSide::Lower ? 0 : g.halo.z + g.owned.z) * g.pitch;
      const size_t count = static_cast<size_t>(g.halo.z * g.pitch);
      runs.push_back({
          static_cast<size_t>(start),
          count,
      });
      break;
    }
  }

  return runs;
}

[[nodiscard]] inline auto
axis_owned_face_runs(const AxisGeometry& g,
                     Axis axis,
                     AxisSide side) -> std::vector<AxisRun> {
  std::vector<AxisRun> runs;

  const int64_t owned = axis_value(g.owned, axis);
  const int64_t halo = axis_value(g.halo, axis);
  if (halo <= 0) {
    return runs;
  }
  if (owned < halo) {
    log::throw_with_log<std::invalid_argument>(
        "axis_owned_face_runs: owned extent {} is smaller than halo {} on axis {}", owned, halo,
        static_cast<int>(axis));
  }

  switch (axis) {
    case Axis::X: {
      const int64_t start = side == AxisSide::Lower ? g.halo.x : g.owned.x;
      const size_t count = static_cast<size_t>(g.halo.x);
      runs.reserve(static_cast<size_t>(g.haloed.y * g.haloed.z));
      for (int64_t z = 0; z < g.haloed.z; ++z) {
        for (int64_t y = 0; y < g.haloed.y; ++y) {
          runs.push_back({
              static_cast<size_t>(z * g.pitch + y * g.haloed.x) + static_cast<size_t>(start),
              count,
          });
        }
      }
      break;
    }

    case Axis::Y: {
      const int64_t start = (side == AxisSide::Lower ? g.halo.y : g.owned.y) * g.haloed.x;
      const size_t count = static_cast<size_t>(g.halo.y * g.haloed.x);
      runs.reserve(static_cast<size_t>(g.haloed.z));
      for (int64_t z = 0; z < g.haloed.z; ++z) {
        runs.push_back({
            static_cast<size_t>(z * g.pitch) + static_cast<size_t>(start),
            count,
        });
      }
      break;
    }

    case Axis::Z: {
      const int64_t start = (side == AxisSide::Lower ? g.halo.z : g.owned.z) * g.pitch;
      const size_t count = static_cast<size_t>(g.halo.z * g.pitch);
      runs.push_back({
          static_cast<size_t>(start),
          count,
      });
      break;
    }
  }

  return runs;
}

inline auto
validate_boundary_mode_bits(BoundaryModeDistributed boundary,
                            const char* family_tag) -> void {
  const auto bits = static_cast<std::uint8_t>(boundary);
  const auto known =
      static_cast<std::uint8_t>(static_cast<std::uint8_t>(BoundaryModeDistributed::PeriodicX) |
                                static_cast<std::uint8_t>(BoundaryModeDistributed::PeriodicY) |
                                static_cast<std::uint8_t>(BoundaryModeDistributed::PeriodicZ));

  if ((bits & static_cast<std::uint8_t>(~known)) != 0U) {
    log::throw_with_log<std::invalid_argument>(
        "{}: unrecognised BoundaryModeDistributed bits (got {})", family_tag,
        static_cast<int>(bits));
  }
}

}  // namespace carbon::detail::halo_distributed_build
