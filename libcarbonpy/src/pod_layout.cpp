#include <carbonpy/carbonpy.h>
#include <carbonpy/pod_layout_checks.h>

#include <cstddef>

namespace {

const auto k_layout_checked = []() {
  carbonpy::detail::check_all_pod_layouts();
  return 0;
}();

}  // namespace

extern "C" {

CARBONPY_API auto
carbonpy_sizeof_dense_span() -> int {
  return static_cast<int>(sizeof(CarbonDenseSpan));
}

CARBONPY_API auto
carbonpy_offsetof_dense_span_iter_x() -> int {
  return static_cast<int>(offsetof(CarbonDenseSpan, iter_x));
}
CARBONPY_API auto
carbonpy_offsetof_dense_span_iter_y() -> int {
  return static_cast<int>(offsetof(CarbonDenseSpan, iter_y));
}
CARBONPY_API auto
carbonpy_offsetof_dense_span_iter_z() -> int {
  return static_cast<int>(offsetof(CarbonDenseSpan, iter_z));
}
CARBONPY_API auto
carbonpy_offsetof_dense_span_logic_x() -> int {
  return static_cast<int>(offsetof(CarbonDenseSpan, logic_x));
}
CARBONPY_API auto
carbonpy_offsetof_dense_span_logic_y() -> int {
  return static_cast<int>(offsetof(CarbonDenseSpan, logic_y));
}
CARBONPY_API auto
carbonpy_offsetof_dense_span_logic_z() -> int {
  return static_cast<int>(offsetof(CarbonDenseSpan, logic_z));
}
CARBONPY_API auto
carbonpy_offsetof_dense_span_pitch_xy() -> int {
  return static_cast<int>(offsetof(CarbonDenseSpan, pitch_xy));
}
CARBONPY_API auto
carbonpy_offsetof_dense_span_z_offset() -> int {
  return static_cast<int>(offsetof(CarbonDenseSpan, z_offset));
}
CARBONPY_API auto
carbonpy_offsetof_dense_span_halo_x() -> int {
  return static_cast<int>(offsetof(CarbonDenseSpan, halo_x));
}
CARBONPY_API auto
carbonpy_offsetof_dense_span_halo_y() -> int {
  return static_cast<int>(offsetof(CarbonDenseSpan, halo_y));
}
CARBONPY_API auto
carbonpy_offsetof_dense_span_halo_z() -> int {
  return static_cast<int>(offsetof(CarbonDenseSpan, halo_z));
}
CARBONPY_API auto
carbonpy_offsetof_dense_span_global_x() -> int {
  return static_cast<int>(offsetof(CarbonDenseSpan, global_x));
}
CARBONPY_API auto
carbonpy_offsetof_dense_span_global_y() -> int {
  return static_cast<int>(offsetof(CarbonDenseSpan, global_y));
}
CARBONPY_API auto
carbonpy_offsetof_dense_span_global_z() -> int {
  return static_cast<int>(offsetof(CarbonDenseSpan, global_z));
}
CARBONPY_API auto
carbonpy_offsetof_dense_span_boundary_mode() -> int {
  return static_cast<int>(offsetof(CarbonDenseSpan, boundary_mode));
}
CARBONPY_API auto
carbonpy_offsetof_dense_span_data_view() -> int {
  return static_cast<int>(offsetof(CarbonDenseSpan, data_view));
}

CARBONPY_API auto
carbonpy_sizeof_block_span() -> int {
  return static_cast<int>(sizeof(CarbonBlockSpan));
}

CARBONPY_API auto
carbonpy_offsetof_block_span_bitmask_ptr() -> int {
  return static_cast<int>(offsetof(CarbonBlockSpan, bitmask_ptr));
}
CARBONPY_API auto
carbonpy_offsetof_block_span_connectivity_ptr() -> int {
  return static_cast<int>(offsetof(CarbonBlockSpan, connectivity_ptr));
}
CARBONPY_API auto
carbonpy_offsetof_block_span_origins_ptr() -> int {
  return static_cast<int>(offsetof(CarbonBlockSpan, origins_ptr));
}
CARBONPY_API auto
carbonpy_offsetof_block_span_block_count() -> int {
  return static_cast<int>(offsetof(CarbonBlockSpan, block_count));
}
CARBONPY_API auto
carbonpy_offsetof_block_span_n_internal() -> int {
  return static_cast<int>(offsetof(CarbonBlockSpan, n_internal));
}
CARBONPY_API auto
carbonpy_offsetof_block_span_iter_block_count() -> int {
  return static_cast<int>(offsetof(CarbonBlockSpan, iter_block_count));
}
CARBONPY_API auto
carbonpy_offsetof_block_span_mem_x() -> int {
  return static_cast<int>(offsetof(CarbonBlockSpan, mem_x));
}
CARBONPY_API auto
carbonpy_offsetof_block_span_mem_y() -> int {
  return static_cast<int>(offsetof(CarbonBlockSpan, mem_y));
}
CARBONPY_API auto
carbonpy_offsetof_block_span_mem_z() -> int {
  return static_cast<int>(offsetof(CarbonBlockSpan, mem_z));
}
CARBONPY_API auto
carbonpy_sizeof_mgrid_partition() -> int {
  return static_cast<int>(sizeof(CarbonMgridPartition));
}

CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_data_ptr() -> int {
  return static_cast<int>(offsetof(CarbonMgridPartition, data_ptr));
}
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_ref_factor() -> int {
  return static_cast<int>(offsetof(CarbonMgridPartition, ref_factor));
}
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_components() -> int {
  return static_cast<int>(offsetof(CarbonMgridPartition, components));
}
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_own_span() -> int {
  return static_cast<int>(offsetof(CarbonMgridPartition, own_span));
}
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_parent_data_ptr() -> int {
  return static_cast<int>(offsetof(CarbonMgridPartition, parent_data_ptr));
}
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_parent_block_id_ptr() -> int {
  return static_cast<int>(offsetof(CarbonMgridPartition, parent_block_id_ptr));
}
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_parent_span() -> int {
  return static_cast<int>(offsetof(CarbonMgridPartition, parent_span));
}
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_child_data_ptr() -> int {
  return static_cast<int>(offsetof(CarbonMgridPartition, child_data_ptr));
}
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_child_block_id_ptr() -> int {
  return static_cast<int>(offsetof(CarbonMgridPartition, child_block_id_ptr));
}
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_child_span() -> int {
  return static_cast<int>(offsetof(CarbonMgridPartition, child_span));
}

}  // extern "C"
