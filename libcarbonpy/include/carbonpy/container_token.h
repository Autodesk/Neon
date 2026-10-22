#pragma once

#include <stddef.h>
#include <stdint.h>

#include "carbonpy/carbonpy.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Mirrors @c carbon::AccessMode (Read=0, Write=1, ReadWrite=2). */
enum CarbonpyAccessMode {
  CarbonpyAccessRead = 0,
  CarbonpyAccessWrite = 1,
  CarbonpyAccessReadWrite = 2,
};

/** Mirrors @c carbon::AccessPattern (Map=0, Stencil=1). */
enum CarbonpyAccessPattern {
  CarbonpyAccessMap = 0,
  CarbonpyAccessStencil = 1,
};

CARBONPY_API auto
carbonpy_dense_field_get_uid(void* field_handle,
                             uint64_t* out_uid) -> int;

CARBONPY_API auto
carbonpy_block_field_get_uid(void* field_handle,
                             uint64_t* out_uid) -> int;

CARBONPY_API auto
carbonpy_mgrid_field_get_uid(void* field_handle,
                             uint64_t* out_uid) -> int;

CARBONPY_API auto
carbonpy_dense_container_add_token(void* container_handle,
                                   void* field_handle,
                                   int access_mode,
                                   int access_pattern) -> int;

CARBONPY_API auto
carbonpy_block_container_add_token(void* container_handle,
                                   void* field_handle,
                                   int access_mode,
                                   int access_pattern) -> int;

CARBONPY_API auto
carbonpy_mgrid_container_add_token(void* container_handle,
                                   void* field_handle,
                                   int access_mode,
                                   int access_pattern) -> int;

#ifdef __cplusplus
}
#endif
