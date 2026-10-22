#pragma once

#include <stdint.h>

#if defined(_WIN32)
#if defined(CARBONPY_EXPORTS)
#define CARBONPY_API __declspec(dllexport)
#else
#define CARBONPY_API __declspec(dllimport)
#endif
#else
#define CARBONPY_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque library lifetime — idempotent. Returns 0 on success. */
CARBONPY_API auto
carbonpy_init() -> int;

CARBONPY_API auto
carbonpy_shutdown() -> int;

CARBONPY_API auto
carbonpy_device_count() -> int;

/** POD blittable Dense span snapshot for Warp native types. */
struct CarbonDenseSpan {
  int64_t iter_x;
  int64_t iter_y;
  int64_t iter_z;
  int64_t logic_x;
  int64_t logic_y;
  int64_t logic_z;
  int64_t pitch_xy;
  int64_t z_offset;
  int32_t halo_x;
  int32_t halo_y;
  int32_t halo_z;
  int64_t global_x;
  int64_t global_y;
  int64_t global_z;
  int32_t boundary_mode;
  int32_t data_view;
};

/** POD blittable Dense partition snapshot (device pointer as uint64).
 *
 * The partition embeds the field's per-(device, data_view) @ref CarbonDenseSpan so that
 * Warp kernels can address memory from the partition alone (Neon-style spanless
 * @c read / @c write). @c get_partition fills @c span for the requested data view. The
 * span-carrying builtins keep working unchanged. */
struct CarbonDensePartition {
  uint64_t data_ptr;
  struct CarbonDenseSpan span;
};

/** POD blittable Block span snapshot (StaticBlock<8,8,8> only in this pass). */
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
};

/** POD blittable Block partition snapshot. */
struct CarbonBlockPartition {
  uint64_t data_ptr;
  int32_t components;
};

/** POD blittable mGrid cross-level partition snapshot (StaticBlock<8,8,8> octree). */
struct CarbonMgridPartition {
  uint64_t data_ptr;
  int32_t ref_factor;
  int32_t components;
  struct CarbonBlockSpan own_span;
  uint64_t parent_data_ptr;
  uint64_t parent_block_id_ptr;
  struct CarbonBlockSpan parent_span;
  uint64_t child_data_ptr;
  uint64_t child_block_id_ptr;
  struct CarbonBlockSpan child_span;
};

CARBONPY_API auto
carbonpy_system_new(void** handle) -> int;

CARBONPY_API auto
carbonpy_system_delete(void** handle) -> int;

CARBONPY_API auto
carbonpy_dense_new(void** handle,
                   void* system_handle,
                   int64_t extent_x,
                   int64_t extent_y,
                   int64_t extent_z,
                   int halo_radius) -> int;

CARBONPY_API auto
carbonpy_dense_delete(void** handle) -> int;

/** @p dtype is a carbonpy::detail::CarbonDtype (0=f32, 1=f64, 2=i32, 3=i64). */
CARBONPY_API auto
carbonpy_dense_field_new(void** handle,
                         void* dense_handle,
                         int components,
                         int dtype) -> int;

CARBONPY_API auto
carbonpy_dense_field_delete(void** handle) -> int;

CARBONPY_API auto
carbonpy_dense_get_span(void* dense_handle,
                        int device,
                        int data_view,
                        struct CarbonDenseSpan* out_span) -> int;

/** Dispatches on the field's stored dtype; the partition POD is type-erased. */
CARBONPY_API auto
carbonpy_dense_field_get_partition(void* field_handle,
                                   int device,
                                   int data_view,
                                   struct CarbonDensePartition* out_part) -> int;

CARBONPY_API auto
carbonpy_dense_halo_new(void** handle,
                        void* field_handle) -> int;

CARBONPY_API auto
carbonpy_dense_halo_run(void* halo_handle,
                        void* system_handle) -> int;

CARBONPY_API auto
carbonpy_dense_halo_delete(void** handle) -> int;

CARBONPY_API auto
carbonpy_block_new(void** handle,
                   void* system_handle,
                   int64_t extent_x,
                   int64_t extent_y,
                   int64_t extent_z) -> int;

CARBONPY_API auto
carbonpy_block_delete(void** handle) -> int;

/** @p dtype is a carbonpy::detail::CarbonDtype (0=f32, 1=f64, 2=i32, 3=i64). */
CARBONPY_API auto
carbonpy_block_field_new(void** handle,
                         void* block_handle,
                         int components,
                         int dtype) -> int;

CARBONPY_API auto
carbonpy_block_field_delete(void** handle) -> int;

CARBONPY_API auto
carbonpy_block_get_span(void* block_handle,
                        int device,
                        int data_view,
                        struct CarbonBlockSpan* out_span) -> int;

/** Dispatches on the field's stored dtype; the partition POD is type-erased. */
CARBONPY_API auto
carbonpy_block_field_get_partition(void* field_handle,
                                   int device,
                                   int data_view,
                                   struct CarbonBlockPartition* out_part) -> int;

/** POD layout introspection for ctypes / Warp drift guards. */
CARBONPY_API auto
carbonpy_sizeof_dense_span() -> int;
CARBONPY_API auto
carbonpy_offsetof_dense_span_iter_x() -> int;
CARBONPY_API auto
carbonpy_offsetof_dense_span_iter_y() -> int;
CARBONPY_API auto
carbonpy_offsetof_dense_span_iter_z() -> int;
CARBONPY_API auto
carbonpy_offsetof_dense_span_logic_x() -> int;
CARBONPY_API auto
carbonpy_offsetof_dense_span_logic_y() -> int;
CARBONPY_API auto
carbonpy_offsetof_dense_span_logic_z() -> int;
CARBONPY_API auto
carbonpy_offsetof_dense_span_pitch_xy() -> int;
CARBONPY_API auto
carbonpy_offsetof_dense_span_z_offset() -> int;
CARBONPY_API auto
carbonpy_offsetof_dense_span_halo_x() -> int;
CARBONPY_API auto
carbonpy_offsetof_dense_span_halo_y() -> int;
CARBONPY_API auto
carbonpy_offsetof_dense_span_halo_z() -> int;
CARBONPY_API auto
carbonpy_offsetof_dense_span_global_x() -> int;
CARBONPY_API auto
carbonpy_offsetof_dense_span_global_y() -> int;
CARBONPY_API auto
carbonpy_offsetof_dense_span_global_z() -> int;
CARBONPY_API auto
carbonpy_offsetof_dense_span_boundary_mode() -> int;
CARBONPY_API auto
carbonpy_offsetof_dense_span_data_view() -> int;

CARBONPY_API auto
carbonpy_sizeof_block_span() -> int;
CARBONPY_API auto
carbonpy_offsetof_block_span_bitmask_ptr() -> int;
CARBONPY_API auto
carbonpy_offsetof_block_span_connectivity_ptr() -> int;
CARBONPY_API auto
carbonpy_offsetof_block_span_origins_ptr() -> int;
CARBONPY_API auto
carbonpy_offsetof_block_span_block_count() -> int;
CARBONPY_API auto
carbonpy_offsetof_block_span_n_internal() -> int;
CARBONPY_API auto
carbonpy_offsetof_block_span_iter_block_count() -> int;
CARBONPY_API auto
carbonpy_offsetof_block_span_mem_x() -> int;
CARBONPY_API auto
carbonpy_offsetof_block_span_mem_y() -> int;
CARBONPY_API auto
carbonpy_offsetof_block_span_mem_z() -> int;

CARBONPY_API auto
carbonpy_sizeof_mgrid_partition() -> int;
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_data_ptr() -> int;
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_ref_factor() -> int;
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_components() -> int;
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_own_span() -> int;
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_parent_data_ptr() -> int;
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_parent_block_id_ptr() -> int;
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_parent_span() -> int;
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_child_data_ptr() -> int;
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_child_block_id_ptr() -> int;
CARBONPY_API auto
carbonpy_offsetof_mgrid_partition_child_span() -> int;

#ifdef __cplusplus
}
#endif
