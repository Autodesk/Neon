// Host-only smoke test for the carbon::Block primitives that don't require a GPU:
// StaticBlock<Wx, Wy, Wz>::BitMask layout, Span<SBlock> pitch math, Partition<T, C,
// SBlock>::connectivity_slot encoding. Locks the storage formula
//
//   linear = (block_idx * Components + component_id) * VoxelCount +
//            (x + MemX * y + MemX * MemY * z)
//
// the device kernels in test_block_cuda use, so any future re-tuning of the layout
// (e.g. swapping x/z fastest, packing the bitmask differently) trips this test before
// it silently scrambles a field's contents.

#include <carbon/detail/block/idx.h>
#include <carbon/detail/block/partition.h>
#include <carbon/detail/block/span.h>
#include <carbon/detail/block/static_block.h>

#include <cassert>
#include <cstdint>
#include <iostream>

namespace {

using SB = carbon::detail::block::StaticBlock<8, 8, 8>;
using BitMask = SB::BitMask;
using Span = carbon::detail::block::Span<SB>;
using Idx = carbon::detail::block::Idx;
using MicroIndex = carbon::detail::block::MicroIndex;
using Partition32_2 = carbon::detail::block::Partition<int32_t, 2, SB>;

// Multi-resolution brick: an 8x8x8 memory brick subdivided into 2x2x2 user blocks,
// i.e. BlockRatio 4 per axis (64 user-block trays per brick) — the per-level shape a
// future mGrid octree uses.
using SBMultiRes = carbon::detail::block::StaticBlock<8, 8, 8, 2, 2, 2, true>;
using SpanMultiRes = carbon::detail::block::Span<SBMultiRes>;

[[nodiscard]] auto
test_static_block_constants() -> int {
  static_assert(SB::MemX == 8 && SB::MemY == 8 && SB::MemZ == 8);
  static_assert(SB::VoxelCount == 512);
  static_assert(SB::WordCount == 16);
  static_assert(SB::BitsPerWord == 32);
  static_assert(SB::PitchY == 8);
  static_assert(SB::PitchZ == 64);
  return 0;
}

[[nodiscard]] auto
test_static_block_user_block_defaults() -> int {
  // Single-resolution default: user block == memory brick, all ratios 1, not multi-res.
  static_assert(SB::UserX == 8 && SB::UserY == 8 && SB::UserZ == 8);
  static_assert(SB::BlockRatioX == 1 && SB::BlockRatioY == 1 && SB::BlockRatioZ == 1);
  static_assert(!SB::IsMultiResMode);
  // Multi-res brick: 2x2x2 user blocks → ratio 4 per axis, multi-res tag set.
  static_assert(SBMultiRes::UserX == 2 && SBMultiRes::UserY == 2 && SBMultiRes::UserZ == 2);
  static_assert(SBMultiRes::BlockRatioX == 4 && SBMultiRes::BlockRatioY == 4 &&
                SBMultiRes::BlockRatioZ == 4);
  static_assert(SBMultiRes::IsMultiResMode);
  return 0;
}

[[nodiscard]] auto
test_micro_index_single_res_identity() -> int {
  // With User == Mem (ratio 1) the tray view is the identity of the data-block view:
  // tray_block_idx == block_idx and in_tray == intra, and the round-trip is exact.
  Span span{nullptr, /*block_count=*/4U};
  for (uint32_t b = 0; b < 4U; ++b) {
    for (int z = 0; z < 8; z += 3) {
      for (int y = 0; y < 8; y += 3) {
        for (int x = 0; x < 8; x += 3) {
          const Idx idx{b, static_cast<int8_t>(x), static_cast<int8_t>(y), static_cast<int8_t>(z)};
          const MicroIndex micro = span.to_micro_index(idx);
          if (micro.tray_block_idx() != static_cast<int32_t>(b) || micro.in_tray().x != x ||
              micro.in_tray().y != y || micro.in_tray().z != z) {
            std::cerr << "test_micro_index_single_res_identity: not identity at b=" << b << " ("
                      << x << "," << y << "," << z << ")\n";
            return 1;
          }
          const Idx back = span.from_micro_index(micro);
          if (back.block_idx() != b || back.intra().x != x || back.intra().y != y ||
              back.intra().z != z) {
            std::cerr << "test_micro_index_single_res_identity: round-trip mismatch\n";
            return 1;
          }
        }
      }
    }
  }
  return 0;
}

[[nodiscard]] auto
test_micro_index_multi_res_roundtrip() -> int {
  // 2x2x2 user blocks in an 8^3 brick: 64 trays per brick. Verify a couple of known
  // mappings, the fan-out, and an exhaustive round-trip over a few blocks.
  SpanMultiRes span{nullptr, /*block_count=*/3U};

  // block 0, voxel (2,0,0): tray_x = 2/2 = 1 → tray index 1; in_tray (0,0,0).
  {
    const Idx idx{0U, 2, 0, 0};
    const MicroIndex m = span.to_micro_index(idx);
    if (m.tray_block_idx() != 1 || m.in_tray().x != 0 || m.in_tray().y != 0 || m.in_tray().z != 0) {
      std::cerr << "test_micro_index_multi_res_roundtrip: (2,0,0) mapping wrong tray="
                << m.tray_block_idx() << "\n";
      return 1;
    }
  }
  // block 0, voxel (7,7,7): tray (3,3,3) → 3 + 3*4 + 3*16 = 63; in_tray (1,1,1).
  {
    const Idx idx{0U, 7, 7, 7};
    const MicroIndex m = span.to_micro_index(idx);
    if (m.tray_block_idx() != 63 || m.in_tray().x != 1 || m.in_tray().y != 1 ||
        m.in_tray().z != 1) {
      std::cerr << "test_micro_index_multi_res_roundtrip: (7,7,7) mapping wrong tray="
                << m.tray_block_idx() << "\n";
      return 1;
    }
  }
  // block 1, voxel (0,0,0): block fan-out 64 → tray 64; in_tray (0,0,0).
  {
    const Idx idx{1U, 0, 0, 0};
    const MicroIndex m = span.to_micro_index(idx);
    if (m.tray_block_idx() != 64) {
      std::cerr << "test_micro_index_multi_res_roundtrip: block-1 fan-out wrong tray="
                << m.tray_block_idx() << "\n";
      return 1;
    }
  }
  // Exhaustive round-trip.
  for (uint32_t b = 0; b < 3U; ++b) {
    for (int z = 0; z < 8; ++z) {
      for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
          const Idx idx{b, static_cast<int8_t>(x), static_cast<int8_t>(y), static_cast<int8_t>(z)};
          const Idx back = span.from_micro_index(span.to_micro_index(idx));
          if (back.block_idx() != b || back.intra().x != x || back.intra().y != y ||
              back.intra().z != z) {
            std::cerr << "test_micro_index_multi_res_roundtrip: round-trip mismatch at b=" << b
                      << " (" << x << "," << y << "," << z << ")\n";
            return 1;
          }
        }
      }
    }
  }
  return 0;
}

[[nodiscard]] auto
test_bitmask_set_and_test() -> int {
  BitMask m{};
  m.clear();
  for (int i = 0; i < SB::WordCount; ++i) {
    if (m.words[i] != 0U) {
      std::cerr << "test_bitmask_set_and_test: word " << i << " not cleared\n";
      return 1;
    }
  }
  for (int z = 0; z < 8; ++z) {
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        if (m.is_active(x, y, z)) {
          std::cerr << "test_bitmask_set_and_test: post-clear (" << x << "," << y << "," << z
                    << ") was active\n";
          return 1;
        }
      }
    }
  }
  m.set_active(0, 0, 0);
  m.set_active(7, 7, 7);
  m.set_active(3, 5, 1);
  if (!m.is_active(0, 0, 0) || !m.is_active(7, 7, 7) || !m.is_active(3, 5, 1)) {
    std::cerr << "test_bitmask_set_and_test: set/test mismatch\n";
    return 1;
  }
  if (m.is_active(0, 0, 1) || m.is_active(1, 0, 0)) {
    std::cerr << "test_bitmask_set_and_test: spurious activation\n";
    return 1;
  }
  return 0;
}

[[nodiscard]] auto
test_bitmask_full_sweep() -> int {
  BitMask m{};
  m.clear();
  for (int z = 0; z < 8; ++z) {
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        if ((x + y + z) % 2 == 0) {
          m.set_active(x, y, z);
        }
      }
    }
  }
  for (int z = 0; z < 8; ++z) {
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        const bool exp = ((x + y + z) % 2 == 0);
        if (m.is_active(x, y, z) != exp) {
          std::cerr << "test_bitmask_full_sweep: (" << x << "," << y << "," << z << ")\n";
          return 1;
        }
      }
    }
  }
  return 0;
}

[[nodiscard]] auto
test_span_linear_offset() -> int {
  // Two-block, 2-component field. The pitch formula must give:
  //   linear(b, c, x, y, z) = (b * 2 + c) * 512 + x + 8 * y + 64 * z
  // and span->per_card_elements() must be 2 * 512 = 1024.
  Span span{nullptr, /*block_count=*/2U};
  if (span.per_card_elements() != 1024) {
    std::cerr << "test_span_linear_offset: per_card_elements\n";
    return 1;
  }
  if (span.linear_storage_elements(2) != 2048) {
    std::cerr << "test_span_linear_offset: linear_storage_elements\n";
    return 1;
  }
  for (uint32_t b = 0; b < 2; ++b) {
    for (int c = 0; c < 2; ++c) {
      for (int z = 0; z < 8; ++z) {
        for (int y = 0; y < 8; ++y) {
          for (int x = 0; x < 8; ++x) {
            const Idx idx{b, static_cast<int8_t>(x), static_cast<int8_t>(y),
                          static_cast<int8_t>(z)};
            const int64_t got = span.linear_offset(idx, /*components=*/2, c);
            const int64_t expected = (static_cast<int64_t>(b) * 2 + c) * 512 + x + 8 * y + 64 * z;
            if (got != expected) {
              std::cerr << "test_span_linear_offset: b=" << b << " c=" << c << " (" << x << "," << y
                        << "," << z << ") got " << got << " want " << expected << "\n";
              return 1;
            }
          }
        }
      }
    }
  }
  return 0;
}

[[nodiscard]] auto
test_connectivity_slot_encoding() -> int {
  // The dispatcher relies on ((dz+1)*9 + (dy+1)*3 + (dx+1)) being a unique [0, 27)
  // index for each (dx, dy, dz) triple in {-1, 0, 1}^3. Confirm + lock the (0,0,0)
  // slot at 13 (which is what the 27-way connectivity uses for self).
  bool seen[27] = {false};
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const auto slot = Partition32_2::connectivity_slot(dx, dy, dz);
        if (slot >= 27U || seen[slot]) {
          std::cerr << "test_connectivity_slot_encoding: collision at (" << dx << "," << dy << ","
                    << dz << ") -> " << slot << "\n";
          return 1;
        }
        seen[slot] = true;
      }
    }
  }
  if (Partition32_2::connectivity_slot(0, 0, 0) != 13U) {
    std::cerr << "test_connectivity_slot_encoding: self slot not 13\n";
    return 1;
  }
  return 0;
}

[[nodiscard]] auto
test_span_level_spacing_metadata() -> int {
  // Single-level default: level 0, spacing {1,1,1} on both the empty placeholder and a
  // populated span built through the existing 5-arg ctor (no level/spacing passed).
  Span empty{};
  if (empty.level() != 0 || empty.spacing().x != 1 || empty.spacing().y != 1 ||
      empty.spacing().z != 1) {
    std::cerr << "test_span_level_spacing_metadata: empty span default metadata wrong\n";
    return 1;
  }
  Span single{nullptr, /*block_count=*/2U, /*n_internal=*/2U, /*connectivity=*/nullptr,
              /*origins=*/nullptr};
  if (single.level() != 0 || single.spacing().x != 1 || single.spacing().y != 1 ||
      single.spacing().z != 1) {
    std::cerr << "test_span_level_spacing_metadata: 5-arg ctor default metadata wrong\n";
    return 1;
  }
  // Explicit coarse-level metadata is carried through.
  Span coarse{nullptr,
              /*block_count=*/2U,
              /*n_internal=*/2U,
              /*connectivity=*/nullptr,
              /*origins=*/nullptr,
              /*level=*/2,
              carbon::int32_3d{4, 4, 4}};
  if (coarse.level() != 2 || coarse.spacing().x != 4 || coarse.spacing().y != 4 ||
      coarse.spacing().z != 4) {
    std::cerr << "test_span_level_spacing_metadata: explicit level/spacing not carried\n";
    return 1;
  }
  return 0;
}

[[nodiscard]] auto
test_span_set_and_validate() -> int {
  // Build a 1-block span backed by a hand-rolled bitmask. Confirm the validator
  // returns false outside [0, MemX) etc., false on cleared bits, true on set bits.
  BitMask m{};
  m.clear();
  m.set_active(2, 3, 4);
  Span span{&m, 1U};
  Idx idx{};
  if (span.set_and_validate(0, -1, 0, 0, idx)) {
    std::cerr << "test_span_set_and_validate: negative ix accepted\n";
    return 1;
  }
  if (span.set_and_validate(0, 8, 0, 0, idx)) {
    std::cerr << "test_span_set_and_validate: ix=MemX accepted\n";
    return 1;
  }
  if (span.set_and_validate(0, 0, 0, 0, idx)) {
    std::cerr << "test_span_set_and_validate: cleared bit accepted\n";
    return 1;
  }
  if (!span.set_and_validate(0, 2, 3, 4, idx)) {
    std::cerr << "test_span_set_and_validate: set bit rejected\n";
    return 1;
  }
  if (idx.block_idx() != 0U || idx.intra().x != 2 || idx.intra().y != 3 || idx.intra().z != 4) {
    std::cerr << "test_span_set_and_validate: idx not populated\n";
    return 1;
  }
  if (span.set_and_validate(/*block_idx=*/2U, 2, 3, 4, idx)) {
    std::cerr << "test_span_set_and_validate: out-of-range block accepted\n";
    return 1;
  }
  return 0;
}

}  // namespace

int
main() {
  if (test_static_block_constants() != 0)
    return 1;
  if (test_static_block_user_block_defaults() != 0)
    return 1;
  if (test_micro_index_single_res_identity() != 0)
    return 1;
  if (test_micro_index_multi_res_roundtrip() != 0)
    return 1;
  if (test_bitmask_set_and_test() != 0)
    return 1;
  if (test_bitmask_full_sweep() != 0)
    return 1;
  if (test_span_linear_offset() != 0)
    return 1;
  if (test_connectivity_slot_encoding() != 0)
    return 1;
  if (test_span_level_spacing_metadata() != 0)
    return 1;
  if (test_span_set_and_validate() != 0)
    return 1;
  std::cout << "test_block_static: OK\n";
  return 0;
}
