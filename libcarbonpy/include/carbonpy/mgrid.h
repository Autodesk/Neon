#pragma once

#include <stdint.h>

#include "carbonpy/carbonpy.h"

#ifdef __cplusplus
extern "C" {
#endif

CARBONPY_API auto
carbonpy_mgrid_new(void** handle,
                   void* system_handle,
                   int64_t finest_x,
                   int64_t finest_y,
                   int64_t finest_z,
                   int depth,
                   int ref_factor) -> int;

/**
 * Build a *sparse* mGrid from per-level dense activity masks + origins, mirroring
 * Neon's pyNeon `mGrid(sparsity_pattern_list, sparsity_pattern_origins)` interface.
 *
 * The C++ `MultiResBlock` core already accepts an `is_active(level, voxel)` predicate; this
 * entrypoint marshals the masks across the C ABI and rebuilds that predicate on the C++ side —
 * no core change. All coordinates are **level-local** (level 0 = finest):
 *
 *   - @p masks_flat  : every level's mask concatenated in level order (finest first). Each level
 *                      is a dense int32 array of `shape.x*shape.y*shape.z` elements in C-order
 *                      (`idx = (x*sy + y)*sz + z`); non-zero == active. May be null iff every
 *                      level's mask element count is zero (degenerate).
 *   - @p mask_shapes : `depth` triples `(sx, sy, sz)` — each level's mask extent (level-local).
 *   - @p mask_origins: `depth` triples `(ox, oy, oz)` — level-local offset where each mask's
 *                      `(0,0,0)` sits; a voxel `v` is active iff `mask[v - origin]` is in-bounds
 *                      and non-zero.
 *
 * The masks are copied into the predicate closure, so the caller may free them immediately.
 */
CARBONPY_API auto
carbonpy_mgrid_new_sparse(void** handle,
                          void* system_handle,
                          int64_t finest_x,
                          int64_t finest_y,
                          int64_t finest_z,
                          int depth,
                          int ref_factor,
                          const int32_t* masks_flat,
                          const int64_t* mask_shapes,
                          const int64_t* mask_origins) -> int;

CARBONPY_API auto
carbonpy_mgrid_delete(void** handle) -> int;

CARBONPY_API auto
carbonpy_mgrid_depth(void* mgrid_handle) -> int;

/** @p dtype is a carbonpy::detail::CarbonDtype (0=f32, 1=f64, 2=i32, 3=i64). */
CARBONPY_API auto
carbonpy_mgrid_field_new(void** handle,
                         void* mgrid_handle,
                         int components,
                         int dtype) -> int;

CARBONPY_API auto
carbonpy_mgrid_field_delete(void** handle) -> int;

CARBONPY_API auto
carbonpy_mgrid_get_span(void* mgrid_handle,
                        int level,
                        int device,
                        int data_view,
                        struct CarbonBlockSpan* out_span) -> int;

/** Dispatches on the field's stored dtype; the partition POD is type-erased. */
CARBONPY_API auto
carbonpy_mgrid_field_get_partition(void* field_handle,
                                   int level,
                                   int device,
                                   int /*data_view*/,
                                   struct CarbonBlockPartition* out_part) -> int;

CARBONPY_API auto
carbonpy_mgrid_field_get_cross_partition(void* field_handle,
                                         int level,
                                         int device,
                                         int /*data_view*/,
                                         struct CarbonMgridPartition* out_part) -> int;

#ifdef __cplusplus
}
#endif
