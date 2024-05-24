#pragma once
#include <cstdint>
#include "Neon/domain/Grids.h"

/**
 * Initialize a new grid object on the heap.
 * NOTE: some parameters are still not exposed
 */
extern "C" auto dGrid_new(
    uint64_t& handle)
    -> int;

/**
 * Delete a grid object on the heap.
 */
extern "C" auto dGrid_delete(
    uint64_t& handle)
    -> int;

/**
 * Generates a new field object on the heap.
 */
extern "C" auto dGrid_dField_new(
    uint64_t& handle,
    uint64_t& grid)
    -> int;

/**
 * Delete a field object on the heap.
 */
extern "C" auto dGrid_dField_delete(
    uint64_t& handle)
    -> int;

extern "C" auto dGrid_dField_get_partition(
    uint64_t&                       field_handle,
    Neon::dGrid::Partition<int, 0>* partition_handle,
    Neon::Execution                 execution,
    int                             device,
    Neon::DataView                  data_view)
    -> int;

extern "C" auto dGrid_get_span(
    uint64_t&          gridHandle,
    Neon::dGrid::Span* spanRes,
    int                execution,
    int                device,
    int                data_view)
    -> int;

extern "C" auto dGrid_span_size(
    Neon::dGrid::Span* spanRes)
    -> int;

extern "C" auto dGrid_dField_partition_size(
    Neon::dGrid::Partition<int, 0>* partitionPtr)
    -> int;