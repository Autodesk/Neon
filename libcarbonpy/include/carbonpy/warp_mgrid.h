#pragma once

#include "carbonpy/warp_block.h"

#if defined(__CUDACC__)
#define CARBON_MGRID_WARP_HD __device__ __host__ __forceinline__
#else
#define CARBON_MGRID_WARP_HD inline
#endif

namespace wp {

static constexpr uint32_t kMgridInvalidBlockIdx = 0xFFFFFFFFU;

#define CARBON_DEFINE_MGRID_PARTITION(TypeName)                                        \
  struct TypeName {                                                                    \
    uint64_t data_ptr;                                                                 \
    int32_t ref_factor;                                                                \
    int32_t components;                                                                \
    CarbonBlockSpan own_span;                                                          \
    uint64_t parent_data_ptr;                                                          \
    uint64_t parent_block_id_ptr;                                                      \
    CarbonBlockSpan parent_span;                                                       \
    uint64_t child_data_ptr;                                                           \
    uint64_t child_block_id_ptr;                                                       \
    CarbonBlockSpan child_span;                                                        \
    CARBON_MGRID_WARP_HD TypeName() = default;                                         \
    CARBON_MGRID_WARP_HD TypeName(const char* bytes,                                    \
                                  size_t n) {                                          \
      if (n == sizeof(*this)) {                                                        \
        memcpy(this, bytes, n);                                                        \
      }                                                                                \
    }                                                                                  \
  };

CARBON_DEFINE_MGRID_PARTITION(CarbonMgridPartitionF32)
CARBON_DEFINE_MGRID_PARTITION(CarbonMgridPartitionF64)
CARBON_DEFINE_MGRID_PARTITION(CarbonMgridPartitionI32)
CARBON_DEFINE_MGRID_PARTITION(CarbonMgridPartitionI64)

using CarbonMgridPartition = CarbonMgridPartitionF32;

#undef CARBON_DEFINE_MGRID_PARTITION

CARBON_MGRID_WARP_HD auto
carbon_mgrid_child_pitch(uint32_t block_idx,
                         int x,
                         int y,
                         int z) -> int64_t {
  const int64_t voxel = static_cast<int64_t>(x) +
                        static_cast<int64_t>(kBlockMemX) * static_cast<int64_t>(y) +
                        static_cast<int64_t>(kBlockMemX) * static_cast<int64_t>(kBlockMemY) *
                            static_cast<int64_t>(z);
  return static_cast<int64_t>(block_idx) * static_cast<int64_t>(kBlockVoxelCount) + voxel;
}

template <typename Part>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_has_parent(const Part& part,
                        uint32_t block_idx) -> bool {
  if (part.parent_data_ptr == 0U || part.parent_block_id_ptr == 0U ||
      block_idx == kMgridInvalidBlockIdx) {
    return false;
  }
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.parent_block_id_ptr));
  return ids[block_idx] != kMgridInvalidBlockIdx;
}

template <typename Part>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_has_children(const Part& part,
                          uint32_t block_idx,
                          int x,
                          int y,
                          int z) -> bool {
  if (part.child_data_ptr == 0U || part.child_block_id_ptr == 0U ||
      block_idx == kMgridInvalidBlockIdx) {
    return false;
  }
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.child_block_id_ptr));
  return ids[carbon_mgrid_child_pitch(block_idx, x, y, z)] != kMgridInvalidBlockIdx;
}

template <typename Part>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_parent_valid(const Part& part,
                          const CarbonBlockSpan& own_span,
                          uint32_t block_idx,
                          int x,
                          int y,
                          int z) -> bool {
  if (!carbon_mgrid_has_parent(part, block_idx)) {
    return false;
  }
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.parent_block_id_ptr));
  const uint32_t parent_block = ids[block_idx];
  const int64_t gx = carbon_block_global_x(own_span, block_idx, x, y, z);
  const int64_t gy = carbon_block_global_y(own_span, block_idx, x, y, z);
  const int64_t gz = carbon_block_global_z(own_span, block_idx, x, y, z);
  const int px = static_cast<int>((gx / part.ref_factor) % kBlockMemX);
  const int py = static_cast<int>((gy / part.ref_factor) % kBlockMemY);
  const int pz = static_cast<int>((gz / part.ref_factor) % kBlockMemZ);
  return carbon_block_is_active(part.parent_span, parent_block, px, py, pz);
}

CARBON_MGRID_WARP_HD auto
carbon_mgrid_parent_read_f32(const CarbonMgridPartitionF32& part,
                             const CarbonBlockSpan& own_span,
                             uint32_t block_idx,
                             int x,
                             int y,
                             int z,
                             int component_id) -> float {
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.parent_block_id_ptr));
  const uint32_t parent_block = ids[block_idx];
  const int64_t gx = carbon_block_global_x(own_span, block_idx, x, y, z);
  const int64_t gy = carbon_block_global_y(own_span, block_idx, x, y, z);
  const int64_t gz = carbon_block_global_z(own_span, block_idx, x, y, z);
  const int px = static_cast<int>((gx / part.ref_factor) % kBlockMemX);
  const int py = static_cast<int>((gy / part.ref_factor) % kBlockMemY);
  const int pz = static_cast<int>((gz / part.ref_factor) % kBlockMemZ);
  auto* mem = reinterpret_cast<float*>(static_cast<uintptr_t>(part.parent_data_ptr));
  return mem[carbon_block_linear_offset(part.parent_span, parent_block, px, py, pz, part.components,
                                        component_id)];
}

CARBON_MGRID_WARP_HD auto
carbon_mgrid_parent_read_f64(const CarbonMgridPartitionF64& part,
                             const CarbonBlockSpan& own_span,
                             uint32_t block_idx,
                             int x,
                             int y,
                             int z,
                             int component_id) -> double {
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.parent_block_id_ptr));
  const uint32_t parent_block = ids[block_idx];
  const int64_t gx = carbon_block_global_x(own_span, block_idx, x, y, z);
  const int64_t gy = carbon_block_global_y(own_span, block_idx, x, y, z);
  const int64_t gz = carbon_block_global_z(own_span, block_idx, x, y, z);
  const int px = static_cast<int>((gx / part.ref_factor) % kBlockMemX);
  const int py = static_cast<int>((gy / part.ref_factor) % kBlockMemY);
  const int pz = static_cast<int>((gz / part.ref_factor) % kBlockMemZ);
  auto* mem = reinterpret_cast<double*>(static_cast<uintptr_t>(part.parent_data_ptr));
  return mem[carbon_block_linear_offset(part.parent_span, parent_block, px, py, pz, part.components,
                                        component_id)];
}

CARBON_MGRID_WARP_HD auto
carbon_mgrid_parent_read_i32(const CarbonMgridPartitionI32& part,
                             const CarbonBlockSpan& own_span,
                             uint32_t block_idx,
                             int x,
                             int y,
                             int z,
                             int component_id) -> int32_t {
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.parent_block_id_ptr));
  const uint32_t parent_block = ids[block_idx];
  const int64_t gx = carbon_block_global_x(own_span, block_idx, x, y, z);
  const int64_t gy = carbon_block_global_y(own_span, block_idx, x, y, z);
  const int64_t gz = carbon_block_global_z(own_span, block_idx, x, y, z);
  const int px = static_cast<int>((gx / part.ref_factor) % kBlockMemX);
  const int py = static_cast<int>((gy / part.ref_factor) % kBlockMemY);
  const int pz = static_cast<int>((gz / part.ref_factor) % kBlockMemZ);
  auto* mem = reinterpret_cast<int32_t*>(static_cast<uintptr_t>(part.parent_data_ptr));
  return mem[carbon_block_linear_offset(part.parent_span, parent_block, px, py, pz, part.components,
                                        component_id)];
}

CARBON_MGRID_WARP_HD auto
carbon_mgrid_parent_read_i64(const CarbonMgridPartitionI64& part,
                             const CarbonBlockSpan& own_span,
                             uint32_t block_idx,
                             int x,
                             int y,
                             int z,
                             int component_id) -> int64_t {
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.parent_block_id_ptr));
  const uint32_t parent_block = ids[block_idx];
  const int64_t gx = carbon_block_global_x(own_span, block_idx, x, y, z);
  const int64_t gy = carbon_block_global_y(own_span, block_idx, x, y, z);
  const int64_t gz = carbon_block_global_z(own_span, block_idx, x, y, z);
  const int px = static_cast<int>((gx / part.ref_factor) % kBlockMemX);
  const int py = static_cast<int>((gy / part.ref_factor) % kBlockMemY);
  const int pz = static_cast<int>((gz / part.ref_factor) % kBlockMemZ);
  auto* mem = reinterpret_cast<int64_t*>(static_cast<uintptr_t>(part.parent_data_ptr));
  return mem[carbon_block_linear_offset(part.parent_span, parent_block, px, py, pz, part.components,
                                        component_id)];
}

template <typename Part>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_child_valid(const Part& part,
                         uint32_t block_idx,
                         int x,
                         int y,
                         int z,
                         int ox,
                         int oy,
                         int oz) -> bool {
  if (!carbon_mgrid_has_children(part, block_idx, x, y, z)) {
    return false;
  }
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.child_block_id_ptr));
  const uint32_t child_block = ids[carbon_mgrid_child_pitch(block_idx, x, y, z)];
  const int cx = (part.ref_factor * x + ox) % kBlockMemX;
  const int cy = (part.ref_factor * y + oy) % kBlockMemY;
  const int cz = (part.ref_factor * z + oz) % kBlockMemZ;
  return carbon_block_is_active(part.child_span, child_block, cx, cy, cz);
}

CARBON_MGRID_WARP_HD auto
carbon_mgrid_child_read_f32(const CarbonMgridPartitionF32& part,
                            uint32_t block_idx,
                            int x,
                            int y,
                            int z,
                            int ox,
                            int oy,
                            int oz,
                            int component_id) -> float {
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.child_block_id_ptr));
  const uint32_t child_block = ids[carbon_mgrid_child_pitch(block_idx, x, y, z)];
  const int cx = (part.ref_factor * x + ox) % kBlockMemX;
  const int cy = (part.ref_factor * y + oy) % kBlockMemY;
  const int cz = (part.ref_factor * z + oz) % kBlockMemZ;
  auto* mem = reinterpret_cast<float*>(static_cast<uintptr_t>(part.child_data_ptr));
  return mem[carbon_block_linear_offset(part.child_span, child_block, cx, cy, cz, part.components,
                                        component_id)];
}

CARBON_MGRID_WARP_HD auto
carbon_mgrid_child_read_f64(const CarbonMgridPartitionF64& part,
                            uint32_t block_idx,
                            int x,
                            int y,
                            int z,
                            int ox,
                            int oy,
                            int oz,
                            int component_id) -> double {
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.child_block_id_ptr));
  const uint32_t child_block = ids[carbon_mgrid_child_pitch(block_idx, x, y, z)];
  const int cx = (part.ref_factor * x + ox) % kBlockMemX;
  const int cy = (part.ref_factor * y + oy) % kBlockMemY;
  const int cz = (part.ref_factor * z + oz) % kBlockMemZ;
  auto* mem = reinterpret_cast<double*>(static_cast<uintptr_t>(part.child_data_ptr));
  return mem[carbon_block_linear_offset(part.child_span, child_block, cx, cy, cz, part.components,
                                        component_id)];
}

CARBON_MGRID_WARP_HD auto
carbon_mgrid_child_read_i32(const CarbonMgridPartitionI32& part,
                            uint32_t block_idx,
                            int x,
                            int y,
                            int z,
                            int ox,
                            int oy,
                            int oz,
                            int component_id) -> int32_t {
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.child_block_id_ptr));
  const uint32_t child_block = ids[carbon_mgrid_child_pitch(block_idx, x, y, z)];
  const int cx = (part.ref_factor * x + ox) % kBlockMemX;
  const int cy = (part.ref_factor * y + oy) % kBlockMemY;
  const int cz = (part.ref_factor * z + oz) % kBlockMemZ;
  auto* mem = reinterpret_cast<int32_t*>(static_cast<uintptr_t>(part.child_data_ptr));
  return mem[carbon_block_linear_offset(part.child_span, child_block, cx, cy, cz, part.components,
                                        component_id)];
}

CARBON_MGRID_WARP_HD auto
carbon_mgrid_child_read_i64(const CarbonMgridPartitionI64& part,
                            uint32_t block_idx,
                            int x,
                            int y,
                            int z,
                            int ox,
                            int oy,
                            int oz,
                            int component_id) -> int64_t {
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.child_block_id_ptr));
  const uint32_t child_block = ids[carbon_mgrid_child_pitch(block_idx, x, y, z)];
  const int cx = (part.ref_factor * x + ox) % kBlockMemX;
  const int cy = (part.ref_factor * y + oy) % kBlockMemY;
  const int cz = (part.ref_factor * z + oz) % kBlockMemZ;
  auto* mem = reinterpret_cast<int64_t*>(static_cast<uintptr_t>(part.child_data_ptr));
  return mem[carbon_block_linear_offset(part.child_span, child_block, cx, cy, cz, part.components,
                                        component_id)];
}

template <typename Part>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_parent_global_x(const Part& part,
                             const CarbonBlockSpan& own_span,
                             uint32_t block_idx,
                             int x,
                             int y,
                             int z) -> int64_t {
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.parent_block_id_ptr));
  const uint32_t parent_block = ids[block_idx];
  const int64_t gx = carbon_block_global_x(own_span, block_idx, x, y, z);
  const int px = static_cast<int>((gx / part.ref_factor) % kBlockMemX);
  const int py = static_cast<int>((carbon_block_global_y(own_span, block_idx, x, y, z) /
                                   part.ref_factor) %
                                  kBlockMemY);
  const int pz = static_cast<int>((carbon_block_global_z(own_span, block_idx, x, y, z) /
                                   part.ref_factor) %
                                  kBlockMemZ);
  return carbon_block_global_x(part.parent_span, parent_block, px, py, pz);
}

template <typename Part>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_parent_global_y(const Part& part,
                             const CarbonBlockSpan& own_span,
                             uint32_t block_idx,
                             int x,
                             int y,
                             int z) -> int64_t {
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.parent_block_id_ptr));
  const uint32_t parent_block = ids[block_idx];
  const int64_t gy = carbon_block_global_y(own_span, block_idx, x, y, z);
  const int px = static_cast<int>((carbon_block_global_x(own_span, block_idx, x, y, z) /
                                   part.ref_factor) %
                                  kBlockMemX);
  const int py = static_cast<int>((gy / part.ref_factor) % kBlockMemY);
  const int pz = static_cast<int>((carbon_block_global_z(own_span, block_idx, x, y, z) /
                                   part.ref_factor) %
                                  kBlockMemZ);
  return carbon_block_global_y(part.parent_span, parent_block, px, py, pz);
}

template <typename Part>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_parent_global_z(const Part& part,
                             const CarbonBlockSpan& own_span,
                             uint32_t block_idx,
                             int x,
                             int y,
                             int z) -> int64_t {
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.parent_block_id_ptr));
  const uint32_t parent_block = ids[block_idx];
  const int64_t gz = carbon_block_global_z(own_span, block_idx, x, y, z);
  const int px = static_cast<int>((carbon_block_global_x(own_span, block_idx, x, y, z) /
                                   part.ref_factor) %
                                  kBlockMemX);
  const int py = static_cast<int>((carbon_block_global_y(own_span, block_idx, x, y, z) /
                                   part.ref_factor) %
                                  kBlockMemY);
  const int pz = static_cast<int>((gz / part.ref_factor) % kBlockMemZ);
  return carbon_block_global_z(part.parent_span, parent_block, px, py, pz);
}

template <typename Part>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_child_global_x(const Part& part,
                            uint32_t block_idx,
                            int x,
                            int y,
                            int z,
                            int ox,
                            int oy,
                            int oz) -> int64_t {
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.child_block_id_ptr));
  const uint32_t child_block = ids[carbon_mgrid_child_pitch(block_idx, x, y, z)];
  const int cx = (part.ref_factor * x + ox) % kBlockMemX;
  const int cy = (part.ref_factor * y + oy) % kBlockMemY;
  const int cz = (part.ref_factor * z + oz) % kBlockMemZ;
  return carbon_block_global_x(part.child_span, child_block, cx, cy, cz);
}

template <typename Part>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_child_global_y(const Part& part,
                            uint32_t block_idx,
                            int x,
                            int y,
                            int z,
                            int ox,
                            int oy,
                            int oz) -> int64_t {
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.child_block_id_ptr));
  const uint32_t child_block = ids[carbon_mgrid_child_pitch(block_idx, x, y, z)];
  const int cx = (part.ref_factor * x + ox) % kBlockMemX;
  const int cy = (part.ref_factor * y + oy) % kBlockMemY;
  const int cz = (part.ref_factor * z + oz) % kBlockMemZ;
  return carbon_block_global_y(part.child_span, child_block, cx, cy, cz);
}

template <typename Part>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_child_global_z(const Part& part,
                            uint32_t block_idx,
                            int x,
                            int y,
                            int z,
                            int ox,
                            int oy,
                            int oz) -> int64_t {
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.child_block_id_ptr));
  const uint32_t child_block = ids[carbon_mgrid_child_pitch(block_idx, x, y, z)];
  const int cx = (part.ref_factor * x + ox) % kBlockMemX;
  const int cy = (part.ref_factor * y + oy) % kBlockMemY;
  const int cz = (part.ref_factor * z + oz) % kBlockMemZ;
  return carbon_block_global_z(part.child_span, child_block, cx, cy, cz);
}

#define CARBON_DEFINE_MGRID_OWN_READ(Suffix, ValueType)                                  \
  CARBON_MGRID_WARP_HD auto carbon_mgrid_own_read_##Suffix(                            \
      const CarbonMgridPartition##Suffix& part,                                        \
      const CarbonBlockSpan& span,                                                     \
      uint32_t block_idx,                                                              \
      int x,                                                                           \
      int y,                                                                           \
      int z,                                                                           \
      int component_id) -> ValueType {                                                 \
    auto* mem = reinterpret_cast<ValueType*>(static_cast<uintptr_t>(part.data_ptr));   \
    return mem[carbon_block_linear_offset(span, block_idx, x, y, z, part.components,   \
                                         component_id)];                               \
  }

CARBON_DEFINE_MGRID_OWN_READ(F32, float)
CARBON_DEFINE_MGRID_OWN_READ(F64, double)
CARBON_DEFINE_MGRID_OWN_READ(I32, int32_t)
CARBON_DEFINE_MGRID_OWN_READ(I64, int64_t)

#undef CARBON_DEFINE_MGRID_OWN_READ

#define CARBON_DEFINE_MGRID_OWN_WRITE(Suffix, ValueType)                                 \
  CARBON_MGRID_WARP_HD auto carbon_mgrid_own_write_##Suffix(                           \
      const CarbonMgridPartition##Suffix& part,                                        \
      const CarbonBlockSpan& span,                                                     \
      uint32_t block_idx,                                                              \
      int x,                                                                           \
      int y,                                                                           \
      int z,                                                                           \
      int component_id,                                                                \
      ValueType value) -> void {                                                       \
    auto* mem = reinterpret_cast<ValueType*>(static_cast<uintptr_t>(part.data_ptr));   \
    mem[carbon_block_linear_offset(span, block_idx, x, y, z, part.components,          \
                                   component_id)] = value;                             \
  }

CARBON_DEFINE_MGRID_OWN_WRITE(F32, float)
CARBON_DEFINE_MGRID_OWN_WRITE(F64, double)
CARBON_DEFINE_MGRID_OWN_WRITE(I32, int32_t)
CARBON_DEFINE_MGRID_OWN_WRITE(I64, int64_t)

#undef CARBON_DEFINE_MGRID_OWN_WRITE

#define CARBON_DEFINE_MGRID_NEIGHBOUR(Suffix, ValueType, ReadFn)                         \
  CARBON_MGRID_WARP_HD auto carbon_mgrid_neighbour_##Suffix(                           \
      const CarbonMgridPartition##Suffix& part,                                        \
      const CarbonBlockSpan& span,                                                     \
      const CarbonBlockIdx& idx,                                                       \
      int ox,                                                                          \
      int oy,                                                                          \
      int oz,                                                                          \
      int component_id) -> CarbonNeighbourValue##Suffix {                               \
    CarbonNeighbourValue##Suffix out{};                                                \
    uint32_t target_block = 0U;                                                      \
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
  CARBON_MGRID_WARP_HD auto carbon_mgrid_read_ngh_##Suffix(                            \
      const CarbonMgridPartition##Suffix& part,                                        \
      const CarbonBlockSpan& span,                                                     \
      const CarbonBlockIdx& idx,                                                       \
      int ox,                                                                          \
      int oy,                                                                          \
      int oz,                                                                          \
      int component_id,                                                                \
      ValueType alternative) -> ValueType {                                            \
    return carbon_neighbour_value_or_##Suffix(                                         \
        carbon_mgrid_neighbour_##Suffix(part, span, idx, ox, oy, oz, component_id),    \
        alternative);                                                                  \
  }

CARBON_DEFINE_MGRID_NEIGHBOUR(F32, float, carbon_mgrid_own_read_F32)
CARBON_DEFINE_MGRID_NEIGHBOUR(F64, double, carbon_mgrid_own_read_F64)
CARBON_DEFINE_MGRID_NEIGHBOUR(I32, int32_t, carbon_mgrid_own_read_I32)
CARBON_DEFINE_MGRID_NEIGHBOUR(I64, int64_t, carbon_mgrid_own_read_I64)

#undef CARBON_DEFINE_MGRID_NEIGHBOUR

// ---------------------------------------------------------------------------
// Multires LBM primitives (Neon parity: mPartition explosion / coalescence).
// ---------------------------------------------------------------------------

// hasChildren(cell): this cell is a coarse placeholder with a finer child block.
template <typename Part>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_has_child_idx(const Part& part,
                           const CarbonBlockIdx& idx) -> bool {
  return carbon_mgrid_has_children(part, idx.block_idx, idx.x, idx.y, idx.z);
}

// hasParent(cell): the parent block of this cell is active.
template <typename Part>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_has_parent_idx(const Part& part,
                            const CarbonBlockIdx& idx) -> bool {
  return carbon_mgrid_has_parent(part, idx.block_idx);
}

// hasChildren(cell, nghDir): the active same-level neighbour in nghDir is itself refined.
template <typename Part>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_has_finer_ngh(const Part& part,
                           const CarbonBlockIdx& idx,
                           int ox,
                           int oy,
                           int oz) -> bool {
  uint32_t nb = 0U;
  int nx = 0;
  int ny = 0;
  int nz = 0;
  if (!carbon_block_neighbour_resolve(part.own_span, idx.block_idx, idx.x, idx.y, idx.z, ox, oy,
                                      oz, nb, nx, ny, nz)) {
    return false;
  }
  return carbon_mgrid_has_children(part, nb, nx, ny, nz);
}

// Neon uncleOffset scalar: off(i, j) = (i + j <= 0) ? i + j : i + j - 1.
CARBON_MGRID_WARP_HD auto
carbon_mgrid_uncle_off(int i,
                       int j) -> int {
  const int s = i + j;
  return (s <= 0) ? s : s - 1;
}

// Parent voxel (block + in-block x,y,z) of a fine cell; false if no active parent block.
template <typename Part>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_parent_voxel(const Part& part,
                          uint32_t block_idx,
                          int x,
                          int y,
                          int z,
                          uint32_t& pblock,
                          int& px,
                          int& py,
                          int& pz) -> bool {
  if (!carbon_mgrid_has_parent(part, block_idx)) {
    return false;
  }
  const auto* ids =
      reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(part.parent_block_id_ptr));
  pblock = ids[block_idx];
  const int64_t gx = carbon_block_global_x(part.own_span, block_idx, x, y, z);
  const int64_t gy = carbon_block_global_y(part.own_span, block_idx, x, y, z);
  const int64_t gz = carbon_block_global_z(part.own_span, block_idx, x, y, z);
  px = static_cast<int>((gx / part.ref_factor) % kBlockMemX);
  py = static_cast<int>((gy / part.ref_factor) % kBlockMemY);
  pz = static_cast<int>((gz / part.ref_factor) % kBlockMemZ);
  return true;
}

// getUncle(cell, dir): parent of cell, then its active neighbour in dir on the parent span.
// dir == (0,0,0) resolves to the direct parent. Returns false if unreachable/inactive.
template <typename Part>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_get_uncle(const Part& part,
                       uint32_t block_idx,
                       int x,
                       int y,
                       int z,
                       int dx,
                       int dy,
                       int dz,
                       uint32_t& ublock,
                       int& ux,
                       int& uy,
                       int& uz) -> bool {
  uint32_t pblock = 0U;
  int px = 0;
  int py = 0;
  int pz = 0;
  if (!carbon_mgrid_parent_voxel(part, block_idx, x, y, z, pblock, px, py, pz)) {
    return false;
  }
  return carbon_block_neighbour_resolve(part.parent_span, pblock, px, py, pz, dx, dy, dz, ublock,
                                        ux, uy, uz);
}

// Device/host atomic accumulate into parent-level storage (coalescence fan-in).
#if defined(__CUDA_ARCH__)
template <typename T>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_atomic_add(T* ptr,
                        T value) -> void {
  atomicAdd(ptr, value);
}
CARBON_MGRID_WARP_HD auto
carbon_mgrid_atomic_add(int64_t* ptr,
                        int64_t value) -> void {
  atomicAdd(reinterpret_cast<unsigned long long*>(ptr), static_cast<unsigned long long>(value));
}
#else
template <typename T>
CARBON_MGRID_WARP_HD auto
carbon_mgrid_atomic_add(T* ptr,
                        T value) -> void {
  *ptr += value;
}
#endif

#define CARBON_DEFINE_MGRID_LBM(Suffix, ValueType)                                       \
  /* Explosion read: raw parent-level uncle population toward pull dir (ngh). */          \
  CARBON_MGRID_WARP_HD auto carbon_mgrid_lbm_read_coarser_ngh_##Suffix(                 \
      const CarbonMgridPartition##Suffix& part,                                          \
      const CarbonBlockIdx& idx,                                                         \
      const CarbonIndex3& ngh,                                                           \
      int card,                                                                          \
      ValueType alternative,                                                             \
      bool& valid) -> ValueType {                                                        \
    const int ox = carbon_mgrid_uncle_off(idx.x % part.ref_factor, static_cast<int>(ngh.x)); \
    const int oy = carbon_mgrid_uncle_off(idx.y % part.ref_factor, static_cast<int>(ngh.y)); \
    const int oz = carbon_mgrid_uncle_off(idx.z % part.ref_factor, static_cast<int>(ngh.z)); \
    uint32_t ub = 0U;                                                                    \
    int ux = 0;                                                                          \
    int uy = 0;                                                                          \
    int uz = 0;                                                                          \
    if (!carbon_mgrid_get_uncle(part, idx.block_idx, idx.x, idx.y, idx.z, ox, oy, oz, ub, \
                                ux, uy, uz)) {                                            \
      valid = false;                                                                     \
      return alternative;                                                                \
    }                                                                                    \
    valid = true;                                                                        \
    auto* mem = reinterpret_cast<ValueType*>(static_cast<uintptr_t>(part.parent_data_ptr)); \
    return mem[carbon_block_linear_offset(part.parent_span, ub, ux, uy, uz,              \
                                         part.components, card)];                        \
  }                                                                                      \
  /* Coalescence store: atomic fan-in of a fine population into the coarse uncle slot. */ \
  CARBON_MGRID_WARP_HD auto carbon_mgrid_mres_lbm_store_op_##Suffix(                    \
      const CarbonMgridPartition##Suffix& part,                                          \
      const CarbonBlockIdx& idx,                                                         \
      int card,                                                                          \
      const CarbonIndex3& ngh,                                                           \
      ValueType value) -> void {                                                         \
    const int qx = static_cast<int>(ngh.x);                                              \
    const int qy = static_cast<int>(ngh.y);                                              \
    const int qz = static_cast<int>(ngh.z);                                              \
    if (qx == 0 && qy == 0 && qz == 0) {                                                 \
      return;                                                                            \
    }                                                                                    \
    const int ux = carbon_mgrid_uncle_off(idx.x % part.ref_factor, qx);                  \
    const int uy = carbon_mgrid_uncle_off(idx.y % part.ref_factor, qy);                  \
    const int uz = carbon_mgrid_uncle_off(idx.z % part.ref_factor, qz);                  \
    uint32_t nb = 0U;                                                                    \
    int nx = 0;                                                                          \
    int ny = 0;                                                                          \
    int nz = 0;                                                                          \
    /* If the same-level neighbour in uncleDir is active, this is an intra-level face. */ \
    if (carbon_block_neighbour_resolve(part.own_span, idx.block_idx, idx.x, idx.y, idx.z, \
                                       ux, uy, uz, nb, nx, ny, nz)) {                     \
      return;                                                                            \
    }                                                                                    \
    uint32_t ub = 0U;                                                                    \
    int uxv = 0;                                                                         \
    int uyv = 0;                                                                         \
    int uzv = 0;                                                                         \
    if (!carbon_mgrid_get_uncle(part, idx.block_idx, idx.x, idx.y, idx.z, ux, uy, uz, ub, \
                                uxv, uyv, uzv)) {                                         \
      return;                                                                            \
    }                                                                                    \
    const int csx = ux - qx;                                                             \
    const int csy = uy - qy;                                                             \
    const int csz = uz - qz;                                                             \
    uint32_t cb = 0U;                                                                    \
    int cx = 0;                                                                          \
    int cy = 0;                                                                          \
    int cz = 0;                                                                          \
    const bool cs_active = carbon_mgrid_get_uncle(part, idx.block_idx, idx.x, idx.y,      \
                                                  idx.z, csx, csy, csz, cb, cx, cy, cz);  \
    uint32_t sb = 0U;                                                                    \
    int sx = 0;                                                                          \
    int sy = 0;                                                                          \
    int sz = 0;                                                                          \
    const bool cs_child = carbon_block_neighbour_resolve(part.own_span, idx.block_idx,    \
                                                         idx.x, idx.y, idx.z, csx, csy,   \
                                                         csz, sb, sx, sy, sz);            \
    if (cs_active && cs_child) {                                                          \
      auto* mem =                                                                        \
          reinterpret_cast<ValueType*>(static_cast<uintptr_t>(part.parent_data_ptr));    \
      const int64_t off = carbon_block_linear_offset(part.parent_span, cb, cx, cy, cz,   \
                                                     part.components, card);             \
      carbon_mgrid_atomic_add(&mem[off], value);                                         \
    }                                                                                    \
  }

CARBON_DEFINE_MGRID_LBM(F32, float)
CARBON_DEFINE_MGRID_LBM(F64, double)
CARBON_DEFINE_MGRID_LBM(I32, int32_t)
CARBON_DEFINE_MGRID_LBM(I64, int64_t)

#undef CARBON_DEFINE_MGRID_LBM

}  // namespace wp

#if !defined(__CUDA_ARCH__)
#include "carbonpy/pod_layout_checks.h"
#endif
