#pragma once
#include <cstdint>
#include "Neon/set/Backend.h"

/**
 *
 */
extern "C" auto backend_new(
    uint64_t& handle,
    int runtime /*! Type of runtime to use */,
    int numDecices /*! Number of devices */,
    const int* devIds /*!  Vectors of device ids. There are CUDA device ids */)
    -> int;

/**
 * Delete a backend object on the heap.
 */
extern "C" auto backend_delete(
    uint64_t& handle)
    -> int;

extern "C" auto backend_get_string(uint64_t& handle) -> const char*;

extern "C" auto backend_sync(uint64_t& handle) -> int;
