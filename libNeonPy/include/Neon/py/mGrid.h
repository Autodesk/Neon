#pragma once
#include <cstdint>
#include "Neon/domain/Grids.h"

/**
 * Initialize a new grid object on the heap.
 * NOTE: some parameters are still not exposed
 */ /* TODOMATT fix the constructor to have correct arguments */
extern "C" auto mGrid_new(
    uint64_t& handle,
    uint64_t& backendPtr,
    const Neon::index_3d* dim,
    int* sparsity_pattern,
    uint32_t depth)
    -> int;


/**
 * Delete a grid object on the heap.
 */
extern "C" auto mGrid_delete(
    uint64_t& handle)
    -> int;

extern "C" auto dGrid_get_dimensions(
    uint64_t& gridHandle,
    Neon::index_3d* dim)
    -> int;

/**
 * Generates a new field object on the heap.
 */
extern "C" auto mGrid_mField_new(
    uint64_t& handle,
    uint64_t& grid,
    int cardinality)
    -> int;

/**
 * Delete a field object on the heap.
 */
extern "C" auto mGrid_mField_delete(
    uint64_t& handle)
    -> int;

extern "C" auto mGrid_mField_get_partition(
    uint64_t&                       field_handle,
    Neon::domain::mGrid::Partition<int, 0>* partition_handle,
    uint64_t                        field_level,
    Neon::Execution                 execution,
    int                             device,
    Neon::DataView                  data_view)
    -> int;

extern "C" auto mGrid_get_span(
    uint64_t&          gridHandle,
    uint64_t           grid_level,
    Neon::domain::mGrid::Span* spanRes,
    int                execution,
    int                device,
    int                data_view)
    -> int;

extern "C" auto mGrid_span_size(
    Neon::domain::mGrid::Span* spanRes)
    -> int;

extern "C" auto mGrid_mField_partition_size(
    Neon::domain::mGrid::Partition<int, 0>* partitionPtr)
    -> int;

extern "C" auto mGrid_get_properties( /* TODOMATT verify what the return of this method should be */
    uint64_t& gridHandle,
    uint64_t  grid_level,
    const Neon::index_3d* idx) 
    -> int;

extern "C" auto mGrid_is_inside_domain(
    uint64_t& gridHandle,
    uint64_t  grid_level,
    const Neon::index_3d* idx) 
    -> bool;

extern "C" auto mGrid_mField_read(
    uint64_t& fieldHandle,
    uint64_t  field_level,
    const Neon::index_3d* idx,
    const int cardinality)
    -> int;

extern "C" auto mGrid_mField_write(
    uint64_t& fieldHandle,
    uint64_t  field_level,
    const Neon::index_3d* idx,
    const int cardinality,
    int newValue)
    -> int;

extern "C" auto mGrid_mField_update_host_data(
    uint64_t& fieldHandle,
    int streamSetId)
    -> int;

extern "C" auto mGrid_mField_update_device_data(
    uint64_t& fieldHandle,
    int streamSetId)
    -> int;

extern "C" auto mGrid_mField_mPartition_get_member_field_offsets(
    size_t* offsets,
    size_t* length)
    -> void;