#pragma once

#include <carbonpy/carbonpy.h>
#include <carbonpy/warp_block.h>
#include <carbonpy/warp_dense.h>
#include <carbonpy/warp_mgrid.h>
#include <carbonpy/warp_partition_types.h>

#include <cstddef>

namespace carbonpy::detail {

#define CARBONPY_CHECK_FIELD(WpType, CType, field)                                       \
  static_assert(offsetof(WpType, field) == offsetof(CType, field),                       \
                "POD layout drift: " #WpType "::" #field " vs " #CType "::" #field)

#define CARBONPY_CHECK_SIZE(WpType, CType)                                               \
  static_assert(sizeof(WpType) == sizeof(CType),                                         \
                "POD layout drift: sizeof " #WpType " vs " #CType)

inline auto
check_dense_span_layout() -> void {
  CARBONPY_CHECK_SIZE(wp::CarbonDenseSpan, CarbonDenseSpan);
  CARBONPY_CHECK_FIELD(wp::CarbonDenseSpan, CarbonDenseSpan, iter_x);
  CARBONPY_CHECK_FIELD(wp::CarbonDenseSpan, CarbonDenseSpan, iter_y);
  CARBONPY_CHECK_FIELD(wp::CarbonDenseSpan, CarbonDenseSpan, iter_z);
  CARBONPY_CHECK_FIELD(wp::CarbonDenseSpan, CarbonDenseSpan, logic_x);
  CARBONPY_CHECK_FIELD(wp::CarbonDenseSpan, CarbonDenseSpan, logic_y);
  CARBONPY_CHECK_FIELD(wp::CarbonDenseSpan, CarbonDenseSpan, logic_z);
  CARBONPY_CHECK_FIELD(wp::CarbonDenseSpan, CarbonDenseSpan, pitch_xy);
  CARBONPY_CHECK_FIELD(wp::CarbonDenseSpan, CarbonDenseSpan, z_offset);
  CARBONPY_CHECK_FIELD(wp::CarbonDenseSpan, CarbonDenseSpan, halo_x);
  CARBONPY_CHECK_FIELD(wp::CarbonDenseSpan, CarbonDenseSpan, halo_y);
  CARBONPY_CHECK_FIELD(wp::CarbonDenseSpan, CarbonDenseSpan, halo_z);
  CARBONPY_CHECK_FIELD(wp::CarbonDenseSpan, CarbonDenseSpan, global_x);
  CARBONPY_CHECK_FIELD(wp::CarbonDenseSpan, CarbonDenseSpan, global_y);
  CARBONPY_CHECK_FIELD(wp::CarbonDenseSpan, CarbonDenseSpan, global_z);
  CARBONPY_CHECK_FIELD(wp::CarbonDenseSpan, CarbonDenseSpan, boundary_mode);
  CARBONPY_CHECK_FIELD(wp::CarbonDenseSpan, CarbonDenseSpan, data_view);
}

inline auto
check_dense_partition_layout() -> void {
  CARBONPY_CHECK_SIZE(wp::CarbonDensePartitionF32, CarbonDensePartition);
  CARBONPY_CHECK_FIELD(wp::CarbonDensePartitionF32, CarbonDensePartition, data_ptr);
  CARBONPY_CHECK_FIELD(wp::CarbonDensePartitionF32, CarbonDensePartition, span);
  CARBONPY_CHECK_SIZE(wp::CarbonDensePartitionU8, CarbonDensePartition);
  CARBONPY_CHECK_FIELD(wp::CarbonDensePartitionU8, CarbonDensePartition, data_ptr);
  CARBONPY_CHECK_FIELD(wp::CarbonDensePartitionU8, CarbonDensePartition, span);
}

inline auto
check_block_span_layout() -> void {
  CARBONPY_CHECK_SIZE(wp::CarbonBlockSpan, CarbonBlockSpan);
  CARBONPY_CHECK_FIELD(wp::CarbonBlockSpan, CarbonBlockSpan, bitmask_ptr);
  CARBONPY_CHECK_FIELD(wp::CarbonBlockSpan, CarbonBlockSpan, connectivity_ptr);
  CARBONPY_CHECK_FIELD(wp::CarbonBlockSpan, CarbonBlockSpan, origins_ptr);
  CARBONPY_CHECK_FIELD(wp::CarbonBlockSpan, CarbonBlockSpan, block_count);
  CARBONPY_CHECK_FIELD(wp::CarbonBlockSpan, CarbonBlockSpan, n_internal);
  CARBONPY_CHECK_FIELD(wp::CarbonBlockSpan, CarbonBlockSpan, iter_block_count);
  CARBONPY_CHECK_FIELD(wp::CarbonBlockSpan, CarbonBlockSpan, mem_x);
  CARBONPY_CHECK_FIELD(wp::CarbonBlockSpan, CarbonBlockSpan, mem_y);
  CARBONPY_CHECK_FIELD(wp::CarbonBlockSpan, CarbonBlockSpan, mem_z);
}

inline auto
check_block_partition_layout() -> void {
  CARBONPY_CHECK_SIZE(wp::CarbonBlockPartitionF32, CarbonBlockPartition);
  CARBONPY_CHECK_FIELD(wp::CarbonBlockPartitionF32, CarbonBlockPartition, data_ptr);
  CARBONPY_CHECK_FIELD(wp::CarbonBlockPartitionF32, CarbonBlockPartition, components);
}

inline auto
check_mgrid_partition_layout() -> void {
  CARBONPY_CHECK_SIZE(wp::CarbonMgridPartitionF32, CarbonMgridPartition);
  CARBONPY_CHECK_FIELD(wp::CarbonMgridPartitionF32, CarbonMgridPartition, data_ptr);
  CARBONPY_CHECK_FIELD(wp::CarbonMgridPartitionF32, CarbonMgridPartition, ref_factor);
  CARBONPY_CHECK_FIELD(wp::CarbonMgridPartitionF32, CarbonMgridPartition, components);
  CARBONPY_CHECK_FIELD(wp::CarbonMgridPartitionF32, CarbonMgridPartition, own_span);
  CARBONPY_CHECK_FIELD(wp::CarbonMgridPartitionF32, CarbonMgridPartition, parent_data_ptr);
  CARBONPY_CHECK_FIELD(wp::CarbonMgridPartitionF32, CarbonMgridPartition, parent_block_id_ptr);
  CARBONPY_CHECK_FIELD(wp::CarbonMgridPartitionF32, CarbonMgridPartition, parent_span);
  CARBONPY_CHECK_FIELD(wp::CarbonMgridPartitionF32, CarbonMgridPartition, child_data_ptr);
  CARBONPY_CHECK_FIELD(wp::CarbonMgridPartitionF32, CarbonMgridPartition, child_block_id_ptr);
  CARBONPY_CHECK_FIELD(wp::CarbonMgridPartitionF32, CarbonMgridPartition, child_span);
}

inline auto
check_all_pod_layouts() -> void {
  check_dense_span_layout();
  check_dense_partition_layout();
  check_block_span_layout();
  check_block_partition_layout();
  check_mgrid_partition_layout();
}

#undef CARBONPY_CHECK_FIELD
#undef CARBONPY_CHECK_SIZE

}  // namespace carbonpy::detail
