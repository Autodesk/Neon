#pragma once

#include "carbonpy/carbonpy.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Ordered sequence of Warp Driver containers (Dense and/or Block). */
CARBONPY_API auto
carbonpy_skeleton_new(void** handle,
                      void* system_handle) -> int;

CARBONPY_API auto
carbonpy_skeleton_delete(void** handle) -> int;

CARBONPY_API auto
carbonpy_skeleton_clear(void* skeleton_handle) -> int;

CARBONPY_API auto
carbonpy_skeleton_add_dense_container(void* skeleton_handle,
                                      void* container_handle) -> int;

CARBONPY_API auto
carbonpy_skeleton_add_block_container(void* skeleton_handle,
                                      void* container_handle) -> int;

CARBONPY_API auto
carbonpy_skeleton_add_mgrid_container(void* skeleton_handle,
                                      void* container_handle) -> int;

CARBONPY_API auto
carbonpy_skeleton_compile(void* skeleton_handle,
                          int occ_mode) -> int;

CARBONPY_API auto
carbonpy_skeleton_inserted_halo_count(void* skeleton_handle) -> int;

CARBONPY_API auto
carbonpy_skeleton_export_dot(void* skeleton_handle,
                             const char* path,
                             const char* graph_name) -> int;

CARBONPY_API auto
carbonpy_skeleton_run(void* skeleton_handle,
                        int stream_idx,
                        int data_view) -> int;

CARBONPY_API auto
carbonpy_skeleton_kernel_stream_count(void* skeleton_handle) -> int;

CARBONPY_API auto
carbonpy_skeleton_kernel_stream_at(void* skeleton_handle,
                                   int index) -> int;

CARBONPY_API auto
carbonpy_skeleton_num_streams(void* skeleton_handle) -> int;

#ifdef __cplusplus
}
#endif
