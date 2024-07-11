#pragma once
#include <cstdint>
#include "Neon/domain/Grids.h"

/**
 * Initialize a new grid object on the heap.
 * NOTE: some parameters are still not exposed
 */
extern "C" auto dGrid_new(
    uint64_t& handle,
    uint64_t& backendPtr,
    const Neon::int32_3d* dim,
    int* sparsity_pattern)
    -> int;

/**
 * Delete a grid object on the heap.
 */
extern "C" auto dGrid_delete(
    uint64_t& handle)
    -> int;

extern "C" auto dGrid_get_dimensions(
    uint64_t& gridHandle,
    Neon::index_3d* dim)
    -> int;

/**
 * Generates a new field object on the heap.
 */
extern "C" auto dGrid_dField_new(
    uint64_t& handle,
    uint64_t& grid,
    int cardinality)
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

extern "C" auto dGrid_get_properties( 
    uint64_t& gridHandle,
    const Neon::index_3d* const idx) 
    -> int;

extern "C" auto dGrid_is_inside_domain(
    uint64_t& gridHandle,
    const Neon::index_3d* const idx)
    -> bool;

extern "C" auto dGrid_dField_read(
    uint64_t& fieldHandle,
    const Neon::index_3d* idx,
    int cardinality)
    -> int;

extern "C" auto dGrid_dField_write(
    uint64_t& fieldHandle,
    const Neon::index_3d* idx,
    int cardinality,
    int newValue)
    -> int;

extern "C" auto dGrid_dField_update_host_data(
    uint64_t& fieldHandle,
    int streamSetId)
    -> int;

extern "C" auto dGrid_dField_update_device_data(
    uint64_t& fieldHandle,
    int streamSetId)
    -> int;

extern "C" auto dGrid_dSpan_get_member_field_offsets(size_t* offsets, size_t* length)
    -> void;

extern "C" auto dGrid_dField_dPartition_get_member_field_offsets(size_t* offsets, size_t* length)
    -> void;
