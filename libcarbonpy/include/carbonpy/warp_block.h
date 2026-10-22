#pragma once

#include <stdint.h>
#include <string.h>

#include "carbonpy/warp_partition_types.h"
#include "carbonpy/warp_dense.h"

#if defined(__CUDACC__)
#define CARBON_WARP_HD __device__ __host__ __forceinline__
#else
#define CARBON_WARP_HD inline
#endif

namespace wp {

static constexpr int kBlockMemX = 8;
static constexpr int kBlockMemY = 8;
static constexpr int kBlockMemZ = 8;
static constexpr int kBlockVoxelCount = kBlockMemX * kBlockMemY * kBlockMemZ;

struct CarbonBlockSpan {
  uint64_t bitmask_ptr;
  uint64_t connectivity_ptr;
  uint64_t origins_ptr;
  uint32_t block_count;
  uint32_t n_internal;
  uint32_t iter_block_count;
  int32_t mem_x;
  int32_t mem_y;
  int32_t mem_z;

  CARBON_WARP_HD CarbonBlockSpan() = default;

  CARBON_WARP_HD CarbonBlockSpan(const char* bytes,
                                 size_t n) {
    if (n == sizeof(*this)) {
      memcpy(this, bytes, n);
    }
  }
};

CARBON_WARP_HD auto
carbon_block_voxel_pitch(int x,
                         int y,
                         int z) -> int {
  return x + kBlockMemX * y + kBlockMemX * kBlockMemY * z;
}

CARBON_WARP_HD auto
carbon_block_is_active(const CarbonBlockSpan& span,
                       uint32_t block_idx,
                       int x,
                       int y,
                       int z) -> bool {
  if (span.bitmask_ptr == 0U) {
    return false;
  }
  struct BitMaskWords {
    uint32_t words[16];
  };
  const auto* masks = reinterpret_cast<const BitMaskWords*>(static_cast<uintptr_t>(span.bitmask_ptr));
  const int pitch = carbon_block_voxel_pitch(x, y, z);
  const uint32_t word = masks[block_idx].words[pitch / 32];
  return (word & (1U << (pitch % 32))) != 0U;
}

CARBON_WARP_HD auto
carbon_block_view_block_idx(const CarbonBlockSpan& span,
                            uint32_t iter_idx) -> uint32_t {
  (void)span;
  return iter_idx;
}

CARBON_WARP_HD auto
carbon_block_global_x(const CarbonBlockSpan& span,
                      uint32_t block_idx,
                      int x,
                      int y,
                      int z) -> int64_t {
  (void)y;
  (void)z;
  if (span.origins_ptr == 0U || block_idx >= span.block_count) {
    return 0;
  }
  struct Origin3 {
    int32_t x;
    int32_t y;
    int32_t z;
  };
  const auto* origins = reinterpret_cast<const Origin3*>(static_cast<uintptr_t>(span.origins_ptr));
  return static_cast<int64_t>(origins[block_idx].x) + static_cast<int64_t>(x);
}

CARBON_WARP_HD auto
carbon_block_global_y(const CarbonBlockSpan& span,
                      uint32_t block_idx,
                      int x,
                      int y,
                      int z) -> int64_t {
  (void)x;
  (void)z;
  if (span.origins_ptr == 0U || block_idx >= span.block_count) {
    return 0;
  }
  struct Origin3 {
    int32_t x;
    int32_t y;
    int32_t z;
  };
  const auto* origins = reinterpret_cast<const Origin3*>(static_cast<uintptr_t>(span.origins_ptr));
  return static_cast<int64_t>(origins[block_idx].y) + static_cast<int64_t>(y);
}

CARBON_WARP_HD auto
carbon_block_global_z(const CarbonBlockSpan& span,
                      uint32_t block_idx,
                      int x,
                      int y,
                      int z) -> int64_t {
  (void)x;
  (void)y;
  if (span.origins_ptr == 0U || block_idx >= span.block_count) {
    return 0;
  }
  struct Origin3 {
    int32_t x;
    int32_t y;
    int32_t z;
  };
  const auto* origins = reinterpret_cast<const Origin3*>(static_cast<uintptr_t>(span.origins_ptr));
  return static_cast<int64_t>(origins[block_idx].z) + static_cast<int64_t>(z);
}

CARBON_WARP_HD auto
carbon_block_linear_offset([[maybe_unused]] const CarbonBlockSpan& span,
                           uint32_t block_idx,
                           int x,
                           int y,
                           int z,
                           int components,
                           int component_id) -> int64_t {
  const int64_t in_block = static_cast<int64_t>(x) +
                           static_cast<int64_t>(kBlockMemX) * static_cast<int64_t>(y) +
                           static_cast<int64_t>(kBlockMemX) * static_cast<int64_t>(kBlockMemY) *
                               static_cast<int64_t>(z);
  const int64_t block_card_pitch = static_cast<int64_t>(block_idx) * static_cast<int64_t>(components) +
                                   static_cast<int64_t>(component_id);
  return block_card_pitch * static_cast<int64_t>(kBlockVoxelCount) + in_block;
}

CARBON_WARP_HD auto
carbon_block_partition_read_f32(const CarbonBlockPartitionF32& part,
                                const CarbonBlockSpan& span,
                                uint32_t block_idx,
                                int x,
                                int y,
                                int z,
                                int component_id) -> float {
  auto* mem = reinterpret_cast<float*>(static_cast<uintptr_t>(part.data_ptr));
  return mem[carbon_block_linear_offset(span, block_idx, x, y, z, part.components, component_id)];
}

CARBON_WARP_HD void
carbon_block_partition_write_f32(const CarbonBlockPartitionF32& part,
                                 const CarbonBlockSpan& span,
                                 uint32_t block_idx,
                                 int x,
                                 int y,
                                 int z,
                                 int component_id,
                                 float value) {
  auto* mem = reinterpret_cast<float*>(static_cast<uintptr_t>(part.data_ptr));
  mem[carbon_block_linear_offset(span, block_idx, x, y, z, part.components, component_id)] = value;
}

CARBON_WARP_HD auto
carbon_block_partition_read_f64(const CarbonBlockPartitionF64& part,
                                const CarbonBlockSpan& span,
                                uint32_t block_idx,
                                int x,
                                int y,
                                int z,
                                int component_id) -> double {
  auto* mem = reinterpret_cast<double*>(static_cast<uintptr_t>(part.data_ptr));
  return mem[carbon_block_linear_offset(span, block_idx, x, y, z, part.components, component_id)];
}

CARBON_WARP_HD void
carbon_block_partition_write_f64(const CarbonBlockPartitionF64& part,
                                 const CarbonBlockSpan& span,
                                 uint32_t block_idx,
                                 int x,
                                 int y,
                                 int z,
                                 int component_id,
                                 double value) {
  auto* mem = reinterpret_cast<double*>(static_cast<uintptr_t>(part.data_ptr));
  mem[carbon_block_linear_offset(span, block_idx, x, y, z, part.components, component_id)] = value;
}

CARBON_WARP_HD auto
carbon_block_partition_read_i32(const CarbonBlockPartitionI32& part,
                                const CarbonBlockSpan& span,
                                uint32_t block_idx,
                                int x,
                                int y,
                                int z,
                                int component_id) -> int32_t {
  auto* mem = reinterpret_cast<int32_t*>(static_cast<uintptr_t>(part.data_ptr));
  return mem[carbon_block_linear_offset(span, block_idx, x, y, z, part.components, component_id)];
}

CARBON_WARP_HD void
carbon_block_partition_write_i32(const CarbonBlockPartitionI32& part,
                                 const CarbonBlockSpan& span,
                                 uint32_t block_idx,
                                 int x,
                                 int y,
                                 int z,
                                 int component_id,
                                 int32_t value) {
  auto* mem = reinterpret_cast<int32_t*>(static_cast<uintptr_t>(part.data_ptr));
  mem[carbon_block_linear_offset(span, block_idx, x, y, z, part.components, component_id)] = value;
}

CARBON_WARP_HD auto
carbon_block_partition_read_i64(const CarbonBlockPartitionI64& part,
                                const CarbonBlockSpan& span,
                                uint32_t block_idx,
                                int x,
                                int y,
                                int z,
                                int component_id) -> int64_t {
  auto* mem = reinterpret_cast<int64_t*>(static_cast<uintptr_t>(part.data_ptr));
  return mem[carbon_block_linear_offset(span, block_idx, x, y, z, part.components, component_id)];
}

CARBON_WARP_HD void
carbon_block_partition_write_i64(const CarbonBlockPartitionI64& part,
                                 const CarbonBlockSpan& span,
                                 uint32_t block_idx,
                                 int x,
                                 int y,
                                 int z,
                                 int component_id,
                                 int64_t value) {
  auto* mem = reinterpret_cast<int64_t*>(static_cast<uintptr_t>(part.data_ptr));
  mem[carbon_block_linear_offset(span, block_idx, x, y, z, part.components, component_id)] = value;
}

struct CarbonBlockIdx {
  uint32_t block_idx;
  int32_t x;
  int32_t y;
  int32_t z;

  CARBON_WARP_HD CarbonBlockIdx() = default;

  CARBON_WARP_HD CarbonBlockIdx(const char* bytes,
                                size_t n) {
    if (n == sizeof(*this)) {
      memcpy(this, bytes, n);
    }
  }
};

CARBON_WARP_HD auto
carbon_block_set(const CarbonBlockSpan& span,
                 uint32_t iter_idx,
                 int x,
                 int y,
                 int z) -> CarbonBlockIdx {
  CarbonBlockIdx idx{};
  idx.block_idx = carbon_block_view_block_idx(span, iter_idx);
  idx.x = x;
  idx.y = y;
  idx.z = z;
  return idx;
}

CARBON_WARP_HD auto
carbon_block_idx_is_valid(const CarbonBlockSpan& span,
                          const CarbonBlockIdx& idx) -> bool {
  return carbon_block_is_active(span, idx.block_idx, idx.x, idx.y, idx.z);
}

CARBON_WARP_HD auto
carbon_block_partition_read_f32(const CarbonBlockPartitionF32& part,
                                const CarbonBlockSpan& span,
                                const CarbonBlockIdx& idx,
                                int component_id) -> float {
  return carbon_block_partition_read_f32(part, span, idx.block_idx, idx.x, idx.y, idx.z,
                                         component_id);
}

CARBON_WARP_HD auto
carbon_block_partition_read_f64(const CarbonBlockPartitionF64& part,
                                const CarbonBlockSpan& span,
                                const CarbonBlockIdx& idx,
                                int component_id) -> double {
  return carbon_block_partition_read_f64(part, span, idx.block_idx, idx.x, idx.y, idx.z,
                                         component_id);
}

CARBON_WARP_HD auto
carbon_block_partition_read_i32(const CarbonBlockPartitionI32& part,
                                const CarbonBlockSpan& span,
                                const CarbonBlockIdx& idx,
                                int component_id) -> int32_t {
  return carbon_block_partition_read_i32(part, span, idx.block_idx, idx.x, idx.y, idx.z,
                                         component_id);
}

CARBON_WARP_HD auto
carbon_block_partition_read_i64(const CarbonBlockPartitionI64& part,
                                const CarbonBlockSpan& span,
                                const CarbonBlockIdx& idx,
                                int component_id) -> int64_t {
  return carbon_block_partition_read_i64(part, span, idx.block_idx, idx.x, idx.y, idx.z,
                                         component_id);
}

CARBON_WARP_HD void
carbon_block_partition_write_f32(const CarbonBlockPartitionF32& part,
                                   const CarbonBlockSpan& span,
                                   const CarbonBlockIdx& idx,
                                   int component_id,
                                   float value) {
  carbon_block_partition_write_f32(part, span, idx.block_idx, idx.x, idx.y, idx.z, component_id,
                                   value);
}

CARBON_WARP_HD void
carbon_block_partition_write_f64(const CarbonBlockPartitionF64& part,
                                   const CarbonBlockSpan& span,
                                   const CarbonBlockIdx& idx,
                                   int component_id,
                                   double value) {
  carbon_block_partition_write_f64(part, span, idx.block_idx, idx.x, idx.y, idx.z, component_id,
                                   value);
}

CARBON_WARP_HD void
carbon_block_partition_write_i32(const CarbonBlockPartitionI32& part,
                                   const CarbonBlockSpan& span,
                                   const CarbonBlockIdx& idx,
                                   int component_id,
                                   int32_t value) {
  carbon_block_partition_write_i32(part, span, idx.block_idx, idx.x, idx.y, idx.z, component_id,
                                   value);
}

CARBON_WARP_HD void
carbon_block_partition_write_i64(const CarbonBlockPartitionI64& part,
                                   const CarbonBlockSpan& span,
                                   const CarbonBlockIdx& idx,
                                   int component_id,
                                   int64_t value) {
  carbon_block_partition_write_i64(part, span, idx.block_idx, idx.x, idx.y, idx.z, component_id,
                                   value);
}

static constexpr uint32_t kCarbonInvalidBlockIdx = 0xFFFFFFFFU;
static constexpr uint32_t kCarbonSlotsPerBlock = 27U;

CARBON_WARP_HD auto
carbon_block_connectivity_slot(int dx,
                               int dy,
                               int dz) -> uint32_t {
  return static_cast<uint32_t>((dz + 1) * 9 + (dy + 1) * 3 + (dx + 1));
}

CARBON_WARP_HD auto
carbon_block_neighbour_resolve(const CarbonBlockSpan& span,
                               uint32_t block_idx,
                               int x,
                               int y,
                               int z,
                               int ox,
                               int oy,
                               int oz,
                               uint32_t& out_block,
                               int& out_x,
                               int& out_y,
                               int& out_z) -> bool {
  if (span.bitmask_ptr == 0U) {
    return false;
  }
  int tx = x + ox;
  int ty = y + oy;
  int tz = z + oz;
  int dx = 0;
  int dy = 0;
  int dz = 0;
  if (tx < 0) {
    dx = -1;
    tx += kBlockMemX;
  } else if (tx >= kBlockMemX) {
    dx = +1;
    tx -= kBlockMemX;
  }
  if (ty < 0) {
    dy = -1;
    ty += kBlockMemY;
  } else if (ty >= kBlockMemY) {
    dy = +1;
    ty -= kBlockMemY;
  }
  if (tz < 0) {
    dz = -1;
    tz += kBlockMemZ;
  } else if (tz >= kBlockMemZ) {
    dz = +1;
    tz -= kBlockMemZ;
  }
  if (tx < 0 || tx >= kBlockMemX || ty < 0 || ty >= kBlockMemY || tz < 0 || tz >= kBlockMemZ) {
    return false;
  }
  uint32_t target_block = block_idx;
  if (dx != 0 || dy != 0 || dz != 0) {
    if (span.connectivity_ptr == 0U) {
      return false;
    }
    const auto* connectivity =
        reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(span.connectivity_ptr));
    const uint32_t slot = carbon_block_connectivity_slot(dx, dy, dz);
    target_block = connectivity[block_idx * kCarbonSlotsPerBlock + slot];
    if (target_block == kCarbonInvalidBlockIdx) {
      return false;
    }
  }
  if (!carbon_block_is_active(span, target_block, tx, ty, tz)) {
    return false;
  }
  out_block = target_block;
  out_x = tx;
  out_y = ty;
  out_z = tz;
  return true;
}

#define CARBON_DEFINE_BLOCK_NEIGHBOUR(TypeName, ValueType, ReadFn)                       \
  CARBON_WARP_HD auto carbon_block_neighbour_##TypeName(                               \
      const CarbonBlockPartition##TypeName& part,                                      \
      const CarbonBlockSpan& span,                                                     \
      const CarbonBlockIdx& idx,                                                       \
      int ox,                                                                          \
      int oy,                                                                          \
      int oz,                                                                          \
      int component_id) -> CarbonNeighbourValue##TypeName {                            \
    CarbonNeighbourValue##TypeName out{};                                              \
    uint32_t target_block = 0U;                                                        \
    int tx = 0;                                                                        \
    int ty = 0;                                                                        \
    int tz = 0;                                                                        \
    if (!carbon_block_neighbour_resolve(span, idx.block_idx, idx.x, idx.y, idx.z, ox, oy, oz, \
                                       target_block, tx, ty, tz)) {                    \
      return out;                                                                      \
    }                                                                                  \
    out.value = ReadFn(part, span, target_block, tx, ty, tz, component_id);            \
    out.valid = 1;                                                                     \
    return out;                                                                        \
  }                                                                                    \
  CARBON_WARP_HD auto carbon_block_read_ngh_##TypeName(                                \
      const CarbonBlockPartition##TypeName& part,                                      \
      const CarbonBlockSpan& span,                                                     \
      const CarbonBlockIdx& idx,                                                       \
      int ox,                                                                          \
      int oy,                                                                          \
      int oz,                                                                          \
      int component_id,                                                                \
      ValueType alternative) -> ValueType {                                            \
    return carbon_neighbour_value_or_##TypeName(                                       \
        carbon_block_neighbour_##TypeName(part, span, idx, ox, oy, oz, component_id),  \
        alternative);                                                                  \
  }

CARBON_DEFINE_BLOCK_NEIGHBOUR(F32, float, carbon_block_partition_read_f32)
CARBON_DEFINE_BLOCK_NEIGHBOUR(F64, double, carbon_block_partition_read_f64)
CARBON_DEFINE_BLOCK_NEIGHBOUR(I32, int32_t, carbon_block_partition_read_i32)
CARBON_DEFINE_BLOCK_NEIGHBOUR(I64, int64_t, carbon_block_partition_read_i64)

#undef CARBON_DEFINE_BLOCK_NEIGHBOUR

}  // namespace wp

#if !defined(__CUDA_ARCH__)
#include "carbonpy/pod_layout_checks.h"
#endif
