#pragma once
#include <cstdint>
#include "Neon/set/Backend.h"

/**
 *
 */
extern "C" auto dBackend_new(
    uint64_t& handle,
    int runtime /*! Type of runtime to use */,
    int numDecices /*! Number of devices */,
    const int* devIds /*!  Vectors of device ids. There are CUDA device ids */)
    -> int;

/**
 * Delete a backend object on the heap.
 */
extern "C" auto dBackend_delete(
    uint64_t& handle)
    -> int;

extern "C" auto dBackend_get_string(uint64_t& handle) -> const char*;

extern "C" auto dBackend_sync(uint64_t& handle) -> int;
