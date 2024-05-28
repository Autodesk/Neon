#pragma once
#include <cstdint>
#include "Neon/set/Backend.h"


/**
 * Empty constructor
 */
extern "C" auto dBackend_new1(
    uint64_t& handle)
    -> int;

/**
 * Creating a Backend object with the first nGpus devices.
 */
extern "C" auto dBackend_new2(
    uint64_t& handle,
    int nGpus /*!   Number of devices. The devices are selected in the order specifies by CUDA */,
    int runtime /*! Type of runtime to use */)
    -> int;

/**
 *
 */
extern "C" auto dBackend_new3(
    uint64_t& handle,
    const int* devIds /*!  Vectors of device ids. There are CUDA device ids */,
    int runtime /*! Type of runtime to use */)
    -> int;





/**
 * Delete a backend object on the heap.
 */
extern "C" auto dBackend_delete(
    uint64_t& handle)
    -> int;


extern "C" auto dBackend_get_string(uint64_t& handle) -> const char*;

