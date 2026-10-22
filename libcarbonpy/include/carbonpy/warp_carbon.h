#pragma once

#include "carbonpy/warp_block.h"
#include "carbonpy/warp_dense.h"
#include "carbonpy/warp_mgrid.h"

namespace wp {
namespace carbon {

CARBON_WARP_HD auto
read(const CarbonMgridPartitionF32& part,
     const CarbonBlockSpan& own_span,
     const CarbonBlockIdx& idx,
     int component_id) -> float {
  return carbon_mgrid_parent_read_f32(part, own_span, idx.block_idx, idx.x, idx.y, idx.z,
                                      component_id);
}

CARBON_WARP_HD auto
read(const CarbonMgridPartitionF64& part,
     const CarbonBlockSpan& own_span,
     const CarbonBlockIdx& idx,
     int component_id) -> double {
  return carbon_mgrid_parent_read_f64(part, own_span, idx.block_idx, idx.x, idx.y, idx.z,
                                      component_id);
}

CARBON_WARP_HD auto
read(const CarbonMgridPartitionI32& part,
     const CarbonBlockSpan& own_span,
     const CarbonBlockIdx& idx,
     int component_id) -> int32_t {
  return carbon_mgrid_parent_read_i32(part, own_span, idx.block_idx, idx.x, idx.y, idx.z,
                                      component_id);
}

CARBON_WARP_HD auto
read(const CarbonMgridPartitionI64& part,
     const CarbonBlockSpan& own_span,
     const CarbonBlockIdx& idx,
     int component_id) -> int64_t {
  return carbon_mgrid_parent_read_i64(part, own_span, idx.block_idx, idx.x, idx.y, idx.z,
                                      component_id);
}

CARBON_WARP_HD auto
read_child(const CarbonMgridPartitionF32& part,
           const CarbonBlockIdx& idx,
           int ox,
           int oy,
           int oz,
           int component_id) -> float {
  return carbon_mgrid_child_read_f32(part, idx.block_idx, idx.x, idx.y, idx.z, ox, oy, oz,
                                     component_id);
}

CARBON_WARP_HD auto
read_child(const CarbonMgridPartitionF64& part,
           const CarbonBlockIdx& idx,
           int ox,
           int oy,
           int oz,
           int component_id) -> double {
  return carbon_mgrid_child_read_f64(part, idx.block_idx, idx.x, idx.y, idx.z, ox, oy, oz,
                                     component_id);
}

CARBON_WARP_HD auto
read_child(const CarbonMgridPartitionI32& part,
           const CarbonBlockIdx& idx,
           int ox,
           int oy,
           int oz,
           int component_id) -> int32_t {
  return carbon_mgrid_child_read_i32(part, idx.block_idx, idx.x, idx.y, idx.z, ox, oy, oz,
                                     component_id);
}

CARBON_WARP_HD auto
read_child(const CarbonMgridPartitionI64& part,
           const CarbonBlockIdx& idx,
           int ox,
           int oy,
           int oz,
           int component_id) -> int64_t {
  return carbon_mgrid_child_read_i64(part, idx.block_idx, idx.x, idx.y, idx.z, ox, oy, oz,
                                     component_id);
}

CARBON_WARP_HD auto
view_block_idx(const CarbonBlockSpan& span,
               uint32_t iter_idx) -> uint32_t {
  return carbon_block_view_block_idx(span, iter_idx);
}

CARBON_WARP_HD auto
is_active(const CarbonBlockSpan& span,
          uint32_t block_idx,
          int x,
          int y,
          int z) -> bool {
  return carbon_block_is_active(span, block_idx, x, y, z);
}

CARBON_WARP_HD auto
global_x(const CarbonBlockSpan& span,
         uint32_t block_idx,
         int x,
         int y,
         int z) -> int64_t {
  return carbon_block_global_x(span, block_idx, x, y, z);
}

CARBON_WARP_HD auto
global_y(const CarbonBlockSpan& span,
         uint32_t block_idx,
         int x,
         int y,
         int z) -> int64_t {
  return carbon_block_global_y(span, block_idx, x, y, z);
}

CARBON_WARP_HD auto
global_z(const CarbonBlockSpan& span,
         uint32_t block_idx,
         int x,
         int y,
         int z) -> int64_t {
  return carbon_block_global_z(span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
has_parent(const CarbonMgridPartitionF32& part,
           uint32_t block_idx) -> bool {
  return carbon_mgrid_has_parent(part, block_idx);
}

CARBON_MGRID_WARP_HD auto
has_parent(const CarbonMgridPartitionF64& part,
           uint32_t block_idx) -> bool {
  return carbon_mgrid_has_parent(part, block_idx);
}

CARBON_MGRID_WARP_HD auto
has_parent(const CarbonMgridPartitionI32& part,
           uint32_t block_idx) -> bool {
  return carbon_mgrid_has_parent(part, block_idx);
}

CARBON_MGRID_WARP_HD auto
has_parent(const CarbonMgridPartitionI64& part,
           uint32_t block_idx) -> bool {
  return carbon_mgrid_has_parent(part, block_idx);
}

CARBON_MGRID_WARP_HD auto
has_children(const CarbonMgridPartitionF32& part,
             uint32_t block_idx,
             int x,
             int y,
             int z) -> bool {
  return carbon_mgrid_has_children(part, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
has_children(const CarbonMgridPartitionF64& part,
             uint32_t block_idx,
             int x,
             int y,
             int z) -> bool {
  return carbon_mgrid_has_children(part, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
has_children(const CarbonMgridPartitionI32& part,
             uint32_t block_idx,
             int x,
             int y,
             int z) -> bool {
  return carbon_mgrid_has_children(part, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
has_children(const CarbonMgridPartitionI64& part,
             uint32_t block_idx,
             int x,
             int y,
             int z) -> bool {
  return carbon_mgrid_has_children(part, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
parent_valid(const CarbonMgridPartitionF32& part,
             const CarbonBlockSpan& own_span,
             uint32_t block_idx,
             int x,
             int y,
             int z) -> bool {
  return carbon_mgrid_parent_valid(part, own_span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
parent_valid(const CarbonMgridPartitionF64& part,
             const CarbonBlockSpan& own_span,
             uint32_t block_idx,
             int x,
             int y,
             int z) -> bool {
  return carbon_mgrid_parent_valid(part, own_span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
parent_valid(const CarbonMgridPartitionI32& part,
             const CarbonBlockSpan& own_span,
             uint32_t block_idx,
             int x,
             int y,
             int z) -> bool {
  return carbon_mgrid_parent_valid(part, own_span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
parent_valid(const CarbonMgridPartitionI64& part,
             const CarbonBlockSpan& own_span,
             uint32_t block_idx,
             int x,
             int y,
             int z) -> bool {
  return carbon_mgrid_parent_valid(part, own_span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
child_valid(const CarbonMgridPartitionF32& part,
            uint32_t block_idx,
            int x,
            int y,
            int z,
            int ox,
            int oy,
            int oz) -> bool {
  return carbon_mgrid_child_valid(part, block_idx, x, y, z, ox, oy, oz);
}

CARBON_MGRID_WARP_HD auto
child_valid(const CarbonMgridPartitionF64& part,
            uint32_t block_idx,
            int x,
            int y,
            int z,
            int ox,
            int oy,
            int oz) -> bool {
  return carbon_mgrid_child_valid(part, block_idx, x, y, z, ox, oy, oz);
}

CARBON_MGRID_WARP_HD auto
child_valid(const CarbonMgridPartitionI32& part,
            uint32_t block_idx,
            int x,
            int y,
            int z,
            int ox,
            int oy,
            int oz) -> bool {
  return carbon_mgrid_child_valid(part, block_idx, x, y, z, ox, oy, oz);
}

CARBON_MGRID_WARP_HD auto
child_valid(const CarbonMgridPartitionI64& part,
            uint32_t block_idx,
            int x,
            int y,
            int z,
            int ox,
            int oy,
            int oz) -> bool {
  return carbon_mgrid_child_valid(part, block_idx, x, y, z, ox, oy, oz);
}

CARBON_MGRID_WARP_HD auto
parent_global_x(const CarbonMgridPartitionF32& part,
                const CarbonBlockSpan& own_span,
                uint32_t block_idx,
                int x,
                int y,
                int z) -> int64_t {
  return carbon_mgrid_parent_global_x(part, own_span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
parent_global_x(const CarbonMgridPartitionF64& part,
                const CarbonBlockSpan& own_span,
                uint32_t block_idx,
                int x,
                int y,
                int z) -> int64_t {
  return carbon_mgrid_parent_global_x(part, own_span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
parent_global_x(const CarbonMgridPartitionI32& part,
                const CarbonBlockSpan& own_span,
                uint32_t block_idx,
                int x,
                int y,
                int z) -> int64_t {
  return carbon_mgrid_parent_global_x(part, own_span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
parent_global_x(const CarbonMgridPartitionI64& part,
                const CarbonBlockSpan& own_span,
                uint32_t block_idx,
                int x,
                int y,
                int z) -> int64_t {
  return carbon_mgrid_parent_global_x(part, own_span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
parent_global_y(const CarbonMgridPartitionF32& part,
                const CarbonBlockSpan& own_span,
                uint32_t block_idx,
                int x,
                int y,
                int z) -> int64_t {
  return carbon_mgrid_parent_global_y(part, own_span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
parent_global_y(const CarbonMgridPartitionF64& part,
                const CarbonBlockSpan& own_span,
                uint32_t block_idx,
                int x,
                int y,
                int z) -> int64_t {
  return carbon_mgrid_parent_global_y(part, own_span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
parent_global_y(const CarbonMgridPartitionI32& part,
                const CarbonBlockSpan& own_span,
                uint32_t block_idx,
                int x,
                int y,
                int z) -> int64_t {
  return carbon_mgrid_parent_global_y(part, own_span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
parent_global_y(const CarbonMgridPartitionI64& part,
                const CarbonBlockSpan& own_span,
                uint32_t block_idx,
                int x,
                int y,
                int z) -> int64_t {
  return carbon_mgrid_parent_global_y(part, own_span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
parent_global_z(const CarbonMgridPartitionF32& part,
                const CarbonBlockSpan& own_span,
                uint32_t block_idx,
                int x,
                int y,
                int z) -> int64_t {
  return carbon_mgrid_parent_global_z(part, own_span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
parent_global_z(const CarbonMgridPartitionF64& part,
                const CarbonBlockSpan& own_span,
                uint32_t block_idx,
                int x,
                int y,
                int z) -> int64_t {
  return carbon_mgrid_parent_global_z(part, own_span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
parent_global_z(const CarbonMgridPartitionI32& part,
                const CarbonBlockSpan& own_span,
                uint32_t block_idx,
                int x,
                int y,
                int z) -> int64_t {
  return carbon_mgrid_parent_global_z(part, own_span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
parent_global_z(const CarbonMgridPartitionI64& part,
                const CarbonBlockSpan& own_span,
                uint32_t block_idx,
                int x,
                int y,
                int z) -> int64_t {
  return carbon_mgrid_parent_global_z(part, own_span, block_idx, x, y, z);
}

CARBON_MGRID_WARP_HD auto
child_global_x(const CarbonMgridPartitionF32& part,
               uint32_t block_idx,
               int x,
               int y,
               int z,
               int ox,
               int oy,
               int oz) -> int64_t {
  return carbon_mgrid_child_global_x(part, block_idx, x, y, z, ox, oy, oz);
}

CARBON_MGRID_WARP_HD auto
child_global_x(const CarbonMgridPartitionF64& part,
               uint32_t block_idx,
               int x,
               int y,
               int z,
               int ox,
               int oy,
               int oz) -> int64_t {
  return carbon_mgrid_child_global_x(part, block_idx, x, y, z, ox, oy, oz);
}

CARBON_MGRID_WARP_HD auto
child_global_x(const CarbonMgridPartitionI32& part,
               uint32_t block_idx,
               int x,
               int y,
               int z,
               int ox,
               int oy,
               int oz) -> int64_t {
  return carbon_mgrid_child_global_x(part, block_idx, x, y, z, ox, oy, oz);
}

CARBON_MGRID_WARP_HD auto
child_global_x(const CarbonMgridPartitionI64& part,
               uint32_t block_idx,
               int x,
               int y,
               int z,
               int ox,
               int oy,
               int oz) -> int64_t {
  return carbon_mgrid_child_global_x(part, block_idx, x, y, z, ox, oy, oz);
}

CARBON_MGRID_WARP_HD auto
child_global_y(const CarbonMgridPartitionF32& part,
               uint32_t block_idx,
               int x,
               int y,
               int z,
               int ox,
               int oy,
               int oz) -> int64_t {
  return carbon_mgrid_child_global_y(part, block_idx, x, y, z, ox, oy, oz);
}

CARBON_MGRID_WARP_HD auto
child_global_y(const CarbonMgridPartitionF64& part,
               uint32_t block_idx,
               int x,
               int y,
               int z,
               int ox,
               int oy,
               int oz) -> int64_t {
  return carbon_mgrid_child_global_y(part, block_idx, x, y, z, ox, oy, oz);
}

CARBON_MGRID_WARP_HD auto
child_global_y(const CarbonMgridPartitionI32& part,
               uint32_t block_idx,
               int x,
               int y,
               int z,
               int ox,
               int oy,
               int oz) -> int64_t {
  return carbon_mgrid_child_global_y(part, block_idx, x, y, z, ox, oy, oz);
}

CARBON_MGRID_WARP_HD auto
child_global_y(const CarbonMgridPartitionI64& part,
               uint32_t block_idx,
               int x,
               int y,
               int z,
               int ox,
               int oy,
               int oz) -> int64_t {
  return carbon_mgrid_child_global_y(part, block_idx, x, y, z, ox, oy, oz);
}

CARBON_MGRID_WARP_HD auto
child_global_z(const CarbonMgridPartitionF32& part,
               uint32_t block_idx,
               int x,
               int y,
               int z,
               int ox,
               int oy,
               int oz) -> int64_t {
  return carbon_mgrid_child_global_z(part, block_idx, x, y, z, ox, oy, oz);
}

CARBON_MGRID_WARP_HD auto
child_global_z(const CarbonMgridPartitionF64& part,
               uint32_t block_idx,
               int x,
               int y,
               int z,
               int ox,
               int oy,
               int oz) -> int64_t {
  return carbon_mgrid_child_global_z(part, block_idx, x, y, z, ox, oy, oz);
}

CARBON_MGRID_WARP_HD auto
child_global_z(const CarbonMgridPartitionI32& part,
               uint32_t block_idx,
               int x,
               int y,
               int z,
               int ox,
               int oy,
               int oz) -> int64_t {
  return carbon_mgrid_child_global_z(part, block_idx, x, y, z, ox, oy, oz);
}

CARBON_MGRID_WARP_HD auto
child_global_z(const CarbonMgridPartitionI64& part,
               uint32_t block_idx,
               int x,
               int y,
               int z,
               int ox,
               int oy,
               int oz) -> int64_t {
  return carbon_mgrid_child_global_z(part, block_idx, x, y, z, ox, oy, oz);
}

CARBON_WARP_HD auto
set(const CarbonDenseSpan& span,
    int64_t ix,
    int64_t iy,
    int64_t iz) -> CarbonDenseIdx {
  return carbon_dense_set(span, ix, iy, iz);
}

CARBON_WARP_HD auto
set(const CarbonBlockSpan& span,
    uint32_t iter_idx,
    int x,
    int y,
    int z) -> CarbonBlockIdx {
  return carbon_block_set(span, iter_idx, x, y, z);
}

CARBON_WARP_HD auto
is_valid(const CarbonBlockSpan& span,
         const CarbonBlockIdx& idx) -> bool {
  return carbon_block_idx_is_valid(span, idx);
}

CARBON_WARP_HD auto
global_idx(const CarbonDenseSpan& span,
           const CarbonDenseIdx& idx) -> CarbonIndex3 {
  return carbon_dense_global_idx(span, idx);
}

CARBON_WARP_HD auto
global_idx(const CarbonBlockSpan& span,
           const CarbonBlockIdx& idx) -> CarbonIndex3 {
  CarbonIndex3 out{};
  out.x = carbon_block_global_x(span, idx.block_idx, idx.x, idx.y, idx.z);
  out.y = carbon_block_global_y(span, idx.block_idx, idx.x, idx.y, idx.z);
  out.z = carbon_block_global_z(span, idx.block_idx, idx.x, idx.y, idx.z);
  return out;
}

// Neon parity: global-coordinate getters return int32 (Neon's Index_3d
// components are int), so `wp.vec3i(get_x, get_y, get_z)` type-checks in the
// XLB masker helpers. The backing CarbonIndex3 stays int64 for POD layout.
CARBON_WARP_HD auto
get_x(const CarbonIndex3& idx) -> int32_t {
  return static_cast<int32_t>(carbon_index3_get_x(idx));
}

CARBON_WARP_HD auto
get_y(const CarbonIndex3& idx) -> int32_t {
  return static_cast<int32_t>(carbon_index3_get_y(idx));
}

CARBON_WARP_HD auto
get_z(const CarbonIndex3& idx) -> int32_t {
  return static_cast<int32_t>(carbon_index3_get_z(idx));
}

CARBON_WARP_HD auto
read(const CarbonDensePartitionF32& part,
     const CarbonDenseSpan& span,
     const CarbonDenseIdx& idx,
     int component_id) -> float {
  return carbon_dense_partition_read_f32(part, span, idx, component_id);
}

CARBON_WARP_HD auto
read(const CarbonDensePartitionF64& part,
     const CarbonDenseSpan& span,
     const CarbonDenseIdx& idx,
     int component_id) -> double {
  return carbon_dense_partition_read_f64(part, span, idx, component_id);
}

CARBON_WARP_HD auto
read(const CarbonDensePartitionI32& part,
     const CarbonDenseSpan& span,
     const CarbonDenseIdx& idx,
     int component_id) -> int32_t {
  return carbon_dense_partition_read_i32(part, span, idx, component_id);
}

CARBON_WARP_HD auto
read(const CarbonDensePartitionI64& part,
     const CarbonDenseSpan& span,
     const CarbonDenseIdx& idx,
     int component_id) -> int64_t {
  return carbon_dense_partition_read_i64(part, span, idx, component_id);
}

CARBON_WARP_HD void
write(const CarbonDensePartitionF32& part,
      const CarbonDenseSpan& span,
      const CarbonDenseIdx& idx,
      int component_id,
      float value) {
  carbon_dense_partition_write_f32(part, span, idx, component_id, value);
}

CARBON_WARP_HD void
write(const CarbonDensePartitionF64& part,
      const CarbonDenseSpan& span,
      const CarbonDenseIdx& idx,
      int component_id,
      double value) {
  carbon_dense_partition_write_f64(part, span, idx, component_id, value);
}

CARBON_WARP_HD void
write(const CarbonDensePartitionI32& part,
      const CarbonDenseSpan& span,
      const CarbonDenseIdx& idx,
      int component_id,
      int32_t value) {
  carbon_dense_partition_write_i32(part, span, idx, component_id, value);
}

CARBON_WARP_HD void
write(const CarbonDensePartitionI64& part,
      const CarbonDenseSpan& span,
      const CarbonDenseIdx& idx,
      int component_id,
      int64_t value) {
  carbon_dense_partition_write_i64(part, span, idx, component_id, value);
}

CARBON_WARP_HD auto
read(const CarbonDensePartitionU8& part,
     const CarbonDenseSpan& span,
     const CarbonDenseIdx& idx,
     int component_id) -> uint8_t {
  return carbon_dense_partition_read_u8(part, span, idx, component_id);
}

CARBON_WARP_HD void
write(const CarbonDensePartitionU8& part,
      const CarbonDenseSpan& span,
      const CarbonDenseIdx& idx,
      int component_id,
      uint8_t value) {
  carbon_dense_partition_write_u8(part, span, idx, component_id, value);
}

CARBON_WARP_HD auto
read(const CarbonBlockPartitionF32& part,
     const CarbonBlockSpan& span,
     const CarbonBlockIdx& idx,
     int component_id) -> float {
  return carbon_block_partition_read_f32(part, span, idx, component_id);
}

CARBON_WARP_HD auto
read(const CarbonBlockPartitionF64& part,
     const CarbonBlockSpan& span,
     const CarbonBlockIdx& idx,
     int component_id) -> double {
  return carbon_block_partition_read_f64(part, span, idx, component_id);
}

CARBON_WARP_HD auto
read(const CarbonBlockPartitionI32& part,
     const CarbonBlockSpan& span,
     const CarbonBlockIdx& idx,
     int component_id) -> int32_t {
  return carbon_block_partition_read_i32(part, span, idx, component_id);
}

CARBON_WARP_HD auto
read(const CarbonBlockPartitionI64& part,
     const CarbonBlockSpan& span,
     const CarbonBlockIdx& idx,
     int component_id) -> int64_t {
  return carbon_block_partition_read_i64(part, span, idx, component_id);
}

CARBON_WARP_HD void
write(const CarbonBlockPartitionF32& part,
      const CarbonBlockSpan& span,
      const CarbonBlockIdx& idx,
      int component_id,
      float value) {
  carbon_block_partition_write_f32(part, span, idx, component_id, value);
}

CARBON_WARP_HD void
write(const CarbonBlockPartitionF64& part,
      const CarbonBlockSpan& span,
      const CarbonBlockIdx& idx,
      int component_id,
      double value) {
  carbon_block_partition_write_f64(part, span, idx, component_id, value);
}

CARBON_WARP_HD void
write(const CarbonBlockPartitionI32& part,
      const CarbonBlockSpan& span,
      const CarbonBlockIdx& idx,
      int component_id,
      int32_t value) {
  carbon_block_partition_write_i32(part, span, idx, component_id, value);
}

CARBON_WARP_HD void
write(const CarbonBlockPartitionI64& part,
      const CarbonBlockSpan& span,
      const CarbonBlockIdx& idx,
      int component_id,
      int64_t value) {
  carbon_block_partition_write_i64(part, span, idx, component_id, value);
}

#define CARBON_WARP_NEIGHBOUR_DENSE(TypeName, ValueType)                                 \
  CARBON_WARP_HD auto                                                                    \
  neighbour(const CarbonDensePartition##TypeName& part,                                  \
            const CarbonDenseSpan& span,                                                 \
            const CarbonDenseIdx& idx,                                                   \
            int ox,                                                                      \
            int oy,                                                                      \
            int oz,                                                                      \
            int component_id) -> CarbonNeighbourValue##TypeName {                        \
    return carbon_dense_neighbour_##TypeName(part, span, idx, ox, oy, oz, component_id); \
  }                                                                                      \
  CARBON_WARP_HD auto                                                                    \
  read_ngh(const CarbonDensePartition##TypeName& part,                                   \
           const CarbonDenseSpan& span,                                                  \
           const CarbonDenseIdx& idx,                                                    \
           int ox,                                                                       \
           int oy,                                                                       \
           int oz,                                                                       \
           int component_id,                                                             \
           ValueType alternative) -> ValueType {                                         \
    return carbon_dense_read_ngh_##TypeName(part, span, idx, ox, oy, oz, component_id,  \
                                            alternative);                                \
  }                                                                                      \
  CARBON_WARP_HD auto                                                                    \
  value_or(const CarbonNeighbourValue##TypeName& nv,                                     \
           ValueType alternative) -> ValueType {                                         \
    return carbon_neighbour_value_or_##TypeName(nv, alternative);                        \
  }                                                                                      \
  CARBON_WARP_HD auto                                                                    \
  value(const CarbonNeighbourValue##TypeName& nv) -> ValueType {                       \
    return carbon_neighbour_value_##TypeName(nv);                                        \
  }                                                                                      \
  CARBON_WARP_HD auto                                                                    \
  valid(const CarbonNeighbourValue##TypeName& nv) -> bool {                            \
    return carbon_neighbour_valid_##TypeName(nv);                                        \
  }

CARBON_WARP_NEIGHBOUR_DENSE(F32, float)
CARBON_WARP_NEIGHBOUR_DENSE(F64, double)
CARBON_WARP_NEIGHBOUR_DENSE(I32, int32_t)
CARBON_WARP_NEIGHBOUR_DENSE(I64, int64_t)
CARBON_WARP_NEIGHBOUR_DENSE(U8, uint8_t)

#undef CARBON_WARP_NEIGHBOUR_DENSE

// Spanless dense overloads (Neon parity): address memory from the span embedded in the
// partition, so kernels call read/write/set/global_idx/neighbour/read_ngh with the
// partition alone. Each forwards to the span-carrying overload above via part.span.
#define CARBON_WARP_DENSE_SPANLESS(TypeName, ValueType)                                  \
  CARBON_WARP_HD auto set(const CarbonDensePartition##TypeName& part, int64_t ix,        \
                          int64_t iy, int64_t iz) -> CarbonDenseIdx {                    \
    return set(part.span, ix, iy, iz);                                                   \
  }                                                                                      \
  CARBON_WARP_HD auto global_idx(const CarbonDensePartition##TypeName& part,             \
                                 const CarbonDenseIdx& idx) -> CarbonIndex3 {            \
    return global_idx(part.span, idx);                                                   \
  }                                                                                      \
  CARBON_WARP_HD auto read(const CarbonDensePartition##TypeName& part,                   \
                           const CarbonDenseIdx& idx, int component_id) -> ValueType {  \
    return read(part, part.span, idx, component_id);                                     \
  }                                                                                      \
  CARBON_WARP_HD void write(const CarbonDensePartition##TypeName& part,                  \
                            const CarbonDenseIdx& idx, int component_id,                 \
                            ValueType value) {                                           \
    write(part, part.span, idx, component_id, value);                                    \
  }                                                                                      \
  CARBON_WARP_HD auto neighbour(const CarbonDensePartition##TypeName& part,              \
                                const CarbonDenseIdx& idx, int ox, int oy, int oz,       \
                                int component_id) -> CarbonNeighbourValue##TypeName {    \
    return neighbour(part, part.span, idx, ox, oy, oz, component_id);                    \
  }                                                                                      \
  CARBON_WARP_HD auto read_ngh(const CarbonDensePartition##TypeName& part,               \
                               const CarbonDenseIdx& idx, int ox, int oy, int oz,        \
                               int component_id, ValueType alternative) -> ValueType {  \
    return read_ngh(part, part.span, idx, ox, oy, oz, component_id, alternative);        \
  }

CARBON_WARP_DENSE_SPANLESS(F32, float)
CARBON_WARP_DENSE_SPANLESS(F64, double)
CARBON_WARP_DENSE_SPANLESS(I32, int32_t)
CARBON_WARP_DENSE_SPANLESS(I64, int64_t)
CARBON_WARP_DENSE_SPANLESS(U8, uint8_t)

#undef CARBON_WARP_DENSE_SPANLESS

// Neon-parity offset helper: pack a stencil direction (dx, dy, dz) into a CarbonIndex3
// consumed by the offset form of read_ngh below. Mirrors Neon's wp.neon_ngh_idx.
CARBON_WARP_HD auto
ngh_idx(int8_t dx,
        int8_t dy,
        int8_t dz) -> CarbonIndex3 {
  CarbonIndex3 o{};
  o.x = static_cast<int64_t>(dx);
  o.y = static_cast<int64_t>(dy);
  o.z = static_cast<int64_t>(dz);
  return o;
}

// read_ngh taking a packed offset + trailing is_valid (Neon's wp.neon_read_ngh shape).
// is_valid is currently ignored (single-res callers pass an unused flag); the multires
// pass will wire it to the neighbour's validity.
#define CARBON_WARP_DENSE_READ_NGH_OFFSET(TypeName, ValueType)                           \
  CARBON_WARP_HD auto read_ngh(const CarbonDensePartition##TypeName& part,               \
                               const CarbonDenseIdx& idx, const CarbonIndex3& ngh,       \
                               int component_id, ValueType alternative, bool is_valid)   \
      -> ValueType {                                                                     \
    (void)is_valid;                                                                      \
    return read_ngh(part, idx, static_cast<int>(ngh.x), static_cast<int>(ngh.y),         \
                    static_cast<int>(ngh.z), component_id, alternative);                 \
  }

CARBON_WARP_DENSE_READ_NGH_OFFSET(F32, float)
CARBON_WARP_DENSE_READ_NGH_OFFSET(F64, double)
CARBON_WARP_DENSE_READ_NGH_OFFSET(I32, int32_t)
CARBON_WARP_DENSE_READ_NGH_OFFSET(I64, int64_t)
CARBON_WARP_DENSE_READ_NGH_OFFSET(U8, uint8_t)

#undef CARBON_WARP_DENSE_READ_NGH_OFFSET

#define CARBON_WARP_NEIGHBOUR_BLOCK(TypeName, ValueType)                                 \
  CARBON_WARP_HD auto                                                                    \
  neighbour(const CarbonBlockPartition##TypeName& part,                                  \
            const CarbonBlockSpan& span,                                                 \
            const CarbonBlockIdx& idx,                                                   \
            int ox,                                                                      \
            int oy,                                                                      \
            int oz,                                                                      \
            int component_id) -> CarbonNeighbourValue##TypeName {                        \
    return carbon_block_neighbour_##TypeName(part, span, idx, ox, oy, oz, component_id); \
  }                                                                                      \
  CARBON_WARP_HD auto                                                                    \
  read_ngh(const CarbonBlockPartition##TypeName& part,                                   \
           const CarbonBlockSpan& span,                                                  \
           const CarbonBlockIdx& idx,                                                    \
           int ox,                                                                       \
           int oy,                                                                       \
           int oz,                                                                       \
           int component_id,                                                             \
           ValueType alternative) -> ValueType {                                         \
    return carbon_block_read_ngh_##TypeName(part, span, idx, ox, oy, oz, component_id,  \
                                            alternative);                                \
  }

CARBON_WARP_NEIGHBOUR_BLOCK(F32, float)
CARBON_WARP_NEIGHBOUR_BLOCK(F64, double)
CARBON_WARP_NEIGHBOUR_BLOCK(I32, int32_t)
CARBON_WARP_NEIGHBOUR_BLOCK(I64, int64_t)

#undef CARBON_WARP_NEIGHBOUR_BLOCK

#define CARBON_WARP_NEIGHBOUR_MGRID(TypeName, ValueType)                                 \
  CARBON_WARP_HD auto                                                                    \
  neighbour(const CarbonMgridPartition##TypeName& part,                                  \
            const CarbonBlockSpan& span,                                                 \
            const CarbonBlockIdx& idx,                                                   \
            int ox,                                                                      \
            int oy,                                                                      \
            int oz,                                                                      \
            int component_id) -> CarbonNeighbourValue##TypeName {                        \
    return carbon_mgrid_neighbour_##TypeName(part, span, idx, ox, oy, oz, component_id); \
  }                                                                                      \
  CARBON_WARP_HD auto                                                                    \
  read_ngh(const CarbonMgridPartition##TypeName& part,                                   \
           const CarbonBlockSpan& span,                                                  \
           const CarbonBlockIdx& idx,                                                    \
           int ox,                                                                       \
           int oy,                                                                       \
           int oz,                                                                       \
           int component_id,                                                             \
           ValueType alternative) -> ValueType {                                         \
    return carbon_mgrid_read_ngh_##TypeName(part, span, idx, ox, oy, oz, component_id,  \
                                            alternative);                                \
  }

CARBON_WARP_NEIGHBOUR_MGRID(F32, float)
CARBON_WARP_NEIGHBOUR_MGRID(F64, double)
CARBON_WARP_NEIGHBOUR_MGRID(I32, int32_t)
CARBON_WARP_NEIGHBOUR_MGRID(I64, int64_t)

#undef CARBON_WARP_NEIGHBOUR_MGRID

// Spanless mgrid own-level overloads (Neon parity): the fat cross partition embeds its
// own-level span, so kernels call set/global_idx/read/write/neighbour/read_ngh with the
// partition alone. Each forwards to the span-carrying own-level op via part.own_span.
// (The 3-arg read(part, span, idx, c) overload above stays the *parent* read.)
#define CARBON_WARP_MGRID_SPANLESS(TypeName, ValueType)                                  \
  CARBON_WARP_HD auto set(const CarbonMgridPartition##TypeName& part, uint32_t iter_idx,  \
                          int x, int y, int z) -> CarbonBlockIdx {                        \
    return carbon_block_set(part.own_span, iter_idx, x, y, z);                            \
  }                                                                                      \
  CARBON_WARP_HD auto global_idx(const CarbonMgridPartition##TypeName& part,             \
                                 const CarbonBlockIdx& idx) -> CarbonIndex3 {            \
    return global_idx(part.own_span, idx);                                               \
  }                                                                                      \
  CARBON_WARP_HD auto read(const CarbonMgridPartition##TypeName& part,                   \
                           const CarbonBlockIdx& idx, int component_id) -> ValueType {  \
    return carbon_mgrid_own_read_##TypeName(part, part.own_span, idx.block_idx, idx.x,   \
                                            idx.y, idx.z, component_id);                  \
  }                                                                                      \
  CARBON_WARP_HD void write(const CarbonMgridPartition##TypeName& part,                  \
                            const CarbonBlockIdx& idx, int component_id,                 \
                            ValueType value) {                                           \
    carbon_mgrid_own_write_##TypeName(part, part.own_span, idx.block_idx, idx.x, idx.y,  \
                                      idx.z, component_id, value);                        \
  }                                                                                      \
  CARBON_WARP_HD auto neighbour(const CarbonMgridPartition##TypeName& part,              \
                                const CarbonBlockIdx& idx, int ox, int oy, int oz,       \
                                int component_id) -> CarbonNeighbourValue##TypeName {    \
    return carbon_mgrid_neighbour_##TypeName(part, part.own_span, idx, ox, oy, oz,       \
                                             component_id);                              \
  }                                                                                      \
  CARBON_WARP_HD auto read_ngh(const CarbonMgridPartition##TypeName& part,               \
                               const CarbonBlockIdx& idx, int ox, int oy, int oz,        \
                               int component_id, ValueType alternative) -> ValueType {  \
    return carbon_mgrid_read_ngh_##TypeName(part, part.own_span, idx, ox, oy, oz,        \
                                            component_id, alternative);                  \
  }                                                                                      \
  /* Offset form (Neon wp.neon_read_ngh shape): packed dir + trailing is_valid out-ref, \
     wired to own-level neighbour validity so multires stream steps can branch on it. */ \
  CARBON_WARP_HD auto read_ngh(const CarbonMgridPartition##TypeName& part,               \
                               const CarbonBlockIdx& idx, const CarbonIndex3& ngh,       \
                               int component_id, ValueType alternative, bool& is_valid)  \
      -> ValueType {                                                                     \
    const auto nv = carbon_mgrid_neighbour_##TypeName(                                   \
        part, part.own_span, idx, static_cast<int>(ngh.x), static_cast<int>(ngh.y),      \
        static_cast<int>(ngh.z), component_id);                                          \
    is_valid = carbon_neighbour_valid_##TypeName(nv);                                    \
    return carbon_neighbour_value_or_##TypeName(nv, alternative);                        \
  }

CARBON_WARP_MGRID_SPANLESS(F32, float)
CARBON_WARP_MGRID_SPANLESS(F64, double)
CARBON_WARP_MGRID_SPANLESS(I32, int32_t)
CARBON_WARP_MGRID_SPANLESS(I64, int64_t)

#undef CARBON_WARP_MGRID_SPANLESS

// Multires LBM builtins (Neon parity). has_child / has_parent take a CarbonBlockIdx (the
// spanless own-level index); has_finer_ngh / lbm_read_coarser_ngh / mres_lbm_store_op take a
// packed CarbonIndex3 direction (from ngh_idx). These are the wp.neon_* multires surface.
#define CARBON_WARP_MGRID_LBM(TypeName, ValueType)                                       \
  CARBON_WARP_HD auto has_child(const CarbonMgridPartition##TypeName& part,              \
                                const CarbonBlockIdx& idx) -> bool {                     \
    return carbon_mgrid_has_child_idx(part, idx);                                        \
  }                                                                                      \
  CARBON_WARP_HD auto has_parent(const CarbonMgridPartition##TypeName& part,             \
                                 const CarbonBlockIdx& idx) -> bool {                    \
    return carbon_mgrid_has_parent_idx(part, idx);                                       \
  }                                                                                      \
  CARBON_WARP_HD auto has_finer_ngh(const CarbonMgridPartition##TypeName& part,          \
                                    const CarbonBlockIdx& idx, const CarbonIndex3& ngh)  \
      -> bool {                                                                          \
    return carbon_mgrid_has_finer_ngh(part, idx, static_cast<int>(ngh.x),                \
                                      static_cast<int>(ngh.y), static_cast<int>(ngh.z)); \
  }                                                                                      \
  CARBON_WARP_HD auto lbm_read_coarser_ngh(const CarbonMgridPartition##TypeName& part,   \
                                           const CarbonBlockIdx& idx,                    \
                                           const CarbonIndex3& ngh, int card,            \
                                           ValueType alternative, bool& valid)           \
      -> ValueType {                                                                     \
    return carbon_mgrid_lbm_read_coarser_ngh_##TypeName(part, idx, ngh, card, alternative, \
                                                        valid);                          \
  }                                                                                      \
  CARBON_WARP_HD void mres_lbm_store_op(const CarbonMgridPartition##TypeName& part,      \
                                        const CarbonBlockIdx& idx, int card,             \
                                        const CarbonIndex3& ngh, ValueType value) {      \
    carbon_mgrid_mres_lbm_store_op_##TypeName(part, idx, card, ngh, value);              \
  }

CARBON_WARP_MGRID_LBM(F32, float)
CARBON_WARP_MGRID_LBM(F64, double)
CARBON_WARP_MGRID_LBM(I32, int32_t)
CARBON_WARP_MGRID_LBM(I64, int64_t)

#undef CARBON_WARP_MGRID_LBM

}  // namespace carbon
}  // namespace wp
