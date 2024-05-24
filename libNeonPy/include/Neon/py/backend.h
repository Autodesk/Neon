#pragma once
#include <cstdint>
#include "Neon/set/Backend.h"


/**
 * Empty constructor
 */
extern "C" auto dBackend_new(
    uint64_t& handle)
    -> int;

/**
 * Creating a Backend object with the first nGpus devices.
 */
extern "C" auto dBackend_new(
    uint64_t& handle,
    int nGpus /*!   Number of devices. The devices are selected in the order specifies by CUDA */,
    Neon::Runtime runtime /*! Type of runtime to use */)
    -> int;

/**
 *
 */
extern "C" auto dBackend_new(
    uint64_t& handle,
    const std::vector<int>& devIds /*!  Vectors of device ids. There are CUDA device ids */,
    Neon::Runtime runtime /*! Type of runtime to use */)
    -> int;

/**
 *
 */
extern "C" auto dBackend_new(
    uint64_t& handle,
    const Neon::set::DevSet& devSet,
    Neon::Runtime runtime /*! Type of runtime to use */)
    -> int;

/**
 *
 */
extern "C" auto dBackend_new(
    uint64_t& handle,
    const std::vector<int>&     devIds,
    const Neon::set::StreamSet& streamSet)
    -> int;

/**
 *
 */
extern "C" auto dBackend_new(
    uint64_t& handle,
    const Neon::set::DevSet&    devSet,
    const Neon::set::StreamSet& streamSet)
    -> int;

/**
 * Delete a backend object on the heap.
 */
extern "C" auto dBackend_delete(
    uint64_t& handle)
    -> int;
