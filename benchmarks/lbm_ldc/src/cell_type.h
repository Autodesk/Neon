#pragma once

// Per-cell classification for the LBM lid-driven cavity benchmark.
//
// One int32 per cell, encoded as a packed (classification, wall_neighbour_bits)
// pair — same scheme Neon uses (Neon/.../src/CellType.h) but flat-packed into
// a single int32 so it can ride a carbon::Dense::Field<int32_t, 1> without
// adding a custom POD type.
//
// Bit layout (LSB to MSB):
//   bits  0..3   : classification (Bulk = 0, BounceBack = 1, MovingWall = 2)
//   bits  4..30  : up to 27-bit wall_neighbour_bits — bit k set iff the
//                  kth direction's source neighbour (the cell at -c[k]) is a
//                  wall (any non-Bulk cell). Used by the iteration kernel
//                  to choose the bounce-back path without a second Field
//                  load + branch on classification.
//
// 4 bits is plenty for the 3-value classification enum and gives 28 bits
// for the wall mask — enough for any lattice up to D3Q27 with one bit to
// spare. Older revisions of this benchmark used an 8-bit class field /
// 19-bit wall mask (D3Q19-only); the layout was widened when D3Q27 + KBC
// landed.

#include <carbon/cuda_annotation.h>

#include <stdint.h>

namespace carbon::bench::lbm {

enum class Classification : int {
  Bulk = 0,
  BounceBack = 1,
  MovingWall = 2,
};

inline constexpr int kClassMask = 0xF;
inline constexpr int kWallBitsShift = 4;

[[nodiscard]] CARBON_HD constexpr auto
encode_cell(Classification cls,
            uint32_t wall_bits = 0U) noexcept -> int32_t {
  // wall_bits is a Q-bit mask (Q in {19, 27} for the lattices we support);
  // 28 bits available so D3Q27's 27-bit mask fits with one bit to spare.
  // Cast through uint32_t so the shift does not turn into UB on negative
  // ints.
  return static_cast<int32_t>(static_cast<uint32_t>(cls) |
                              (wall_bits << static_cast<uint32_t>(kWallBitsShift)));
}

[[nodiscard]] CARBON_HD constexpr auto
classification_of(int32_t packed) noexcept -> Classification {
  return static_cast<Classification>(packed & kClassMask);
}

[[nodiscard]] CARBON_HD constexpr auto
wall_bits_of(int32_t packed) noexcept -> uint32_t {
  return static_cast<uint32_t>(packed) >> static_cast<uint32_t>(kWallBitsShift);
}

}  // namespace carbon::bench::lbm
