#pragma once

#include <stddef.h>
#include <stdint.h>

#include "carbonpy/carbonpy.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Must match Warp @c LAUNCH_MAX_DIMS / @c wp.launch_bounds_t layout. */
struct CarbonLaunchBounds {
  int32_t shape[4];
  int32_t ndim;
  size_t size;
};

CARBONPY_API auto
carbonpy_cuda_driver_new(void** handle,
                         void* system_handle) -> int;

CARBONPY_API auto
carbonpy_cuda_driver_delete(void** handle) -> int;

/** Launch a Warp-compiled @c CUfunction using Warp's 1-D grid/block policy. */
CARBONPY_API auto
carbonpy_cuda_driver_launch(void* driver_handle,
                            int gpu_id,
                            void* cuda_stream,
                            void* cu_function,
                            struct CarbonLaunchBounds bounds,
                            int block_dim,
                            int shared_memory_bytes,
                            void** kernel_params,
                            int param_count) -> int;

CARBONPY_API auto
carbonpy_dense_container_new(void** handle,
                             void* driver_handle,
                             void* dense_handle,
                             void** kernel_hooks,
                             int hook_count,
                             int field_count,
                             int capture) -> int;

CARBONPY_API auto
carbonpy_dense_container_delete(void** handle) -> int;

/** Snapshot the field partition + dense span used at launch for @p device. */
CARBONPY_API auto
carbonpy_dense_container_bind_field(void* container_handle,
                                    void* field_handle,
                                    int device,
                                    int field_slot) -> int;

CARBONPY_API auto
carbonpy_dense_container_run(void* container_handle,
                             int stream_idx,
                             int data_view) -> int;

CARBONPY_API auto
carbonpy_block_container_new(void** handle,
                             void* driver_handle,
                             void* block_handle,
                             void** kernel_hooks,
                             int hook_count) -> int;

CARBONPY_API auto
carbonpy_block_container_delete(void** handle) -> int;

CARBONPY_API auto
carbonpy_block_container_bind_field(void* container_handle,
                                    void* field_handle,
                                    int device) -> int;

CARBONPY_API auto
carbonpy_block_container_run(void* container_handle,
                             int stream_idx,
                             int data_view) -> int;

CARBONPY_API auto
carbonpy_mgrid_container_new(void** handle,
                             void* driver_handle,
                             void* mgrid_handle,
                             int level,
                             void** kernel_hooks,
                             int hook_count) -> int;

CARBONPY_API auto
carbonpy_mgrid_container_delete(void** handle) -> int;

CARBONPY_API auto
carbonpy_mgrid_container_bind_field(void* container_handle,
                                    void* field_handle,
                                    int device) -> int;

CARBONPY_API auto
carbonpy_mgrid_container_run(void* container_handle,
                             int stream_idx,
                             int data_view) -> int;

CARBONPY_API auto
carbonpy_warp_container_run_scheduled(void* container_handle,
                                      int kind,
                                      int stream_idx,
                                      int data_view,
                                      int sync_at_end) -> int;

CARBONPY_API auto
carbonpy_warp_container_token_count(void* container_handle,
                                    int kind) -> int;

CARBONPY_API auto
carbonpy_warp_container_token_at(void* container_handle,
                                 int kind,
                                 int index,
                                 uint64_t* out_uid,
                                 int* out_mode,
                                 int* out_pattern) -> int;

CARBONPY_API auto
carbonpy_warp_container_driver(void* container_handle,
                               int kind,
                               void** out_driver) -> int;

CARBONPY_API auto
carbonpy_warp_driver_ensure_streams(void* driver_handle,
                                    int stream_count) -> int;

CARBONPY_API auto
carbonpy_warp_driver_set_device(void* driver_handle,
                                int gpu_id) -> int;

CARBONPY_API auto
carbonpy_warp_driver_sync_streams(void* driver_handle,
                                  int stream_count) -> int;

CARBONPY_API auto
carbonpy_warp_driver_stream_at(void* driver_handle,
                               int stream_idx,
                               int gpu_id,
                               void** out_stream) -> int;

#ifdef __cplusplus
}
#endif
