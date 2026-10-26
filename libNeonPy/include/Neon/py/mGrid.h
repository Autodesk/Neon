#pragma once
#include <cstdint>
#include "Neon/domain/Grids.h"

extern "C" auto mGrid_save_topology(
    void*       gridHandle,
    const char* path)
    -> int;

extern "C" auto mGrid_new_from_topology(
    void**      handle,
    void*       backendPtr,
    const char* path)
    -> int;

extern "C" auto mGrid_mField_checkpoint_save_int8(void* fieldHandle, const char* basePath, const char* name) -> int;
extern "C" auto mGrid_mField_checkpoint_save_uint8(void* fieldHandle, const char* basePath, const char* name) -> int;
extern "C" auto mGrid_mField_checkpoint_save_bool(void* fieldHandle, const char* basePath, const char* name) -> int;
extern "C" auto mGrid_mField_checkpoint_save_int32(void* fieldHandle, const char* basePath, const char* name) -> int;
extern "C" auto mGrid_mField_checkpoint_save_uint32(void* fieldHandle, const char* basePath, const char* name) -> int;
extern "C" auto mGrid_mField_checkpoint_save_int64(void* fieldHandle, const char* basePath, const char* name) -> int;
extern "C" auto mGrid_mField_checkpoint_save_uint64(void* fieldHandle, const char* basePath, const char* name) -> int;
extern "C" auto mGrid_mField_checkpoint_save_float32(void* fieldHandle, const char* basePath, const char* name) -> int;
extern "C" auto mGrid_mField_checkpoint_save_float64(void* fieldHandle, const char* basePath, const char* name) -> int;
extern "C" auto mGrid_mField_checkpoint_save_float16(void* fieldHandle, const char* basePath, const char* name) -> int;

extern "C" auto mGrid_mField_checkpoint_load_int8(void* fieldHandle, const char* basePath) -> int;
extern "C" auto mGrid_mField_checkpoint_load_uint8(void* fieldHandle, const char* basePath) -> int;
extern "C" auto mGrid_mField_checkpoint_load_bool(void* fieldHandle, const char* basePath) -> int;
extern "C" auto mGrid_mField_checkpoint_load_int32(void* fieldHandle, const char* basePath) -> int;
extern "C" auto mGrid_mField_checkpoint_load_uint32(void* fieldHandle, const char* basePath) -> int;
extern "C" auto mGrid_mField_checkpoint_load_int64(void* fieldHandle, const char* basePath) -> int;
extern "C" auto mGrid_mField_checkpoint_load_uint64(void* fieldHandle, const char* basePath) -> int;
extern "C" auto mGrid_mField_checkpoint_load_float32(void* fieldHandle, const char* basePath) -> int;
extern "C" auto mGrid_mField_checkpoint_load_float64(void* fieldHandle, const char* basePath) -> int;
extern "C" auto mGrid_mField_checkpoint_load_float16(void* fieldHandle, const char* basePath) -> int;
