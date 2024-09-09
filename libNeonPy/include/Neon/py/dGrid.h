#pragma once
#include <cstdint>
#include "Neon/domain/Grids.h"

/**
 * Initialize a new grid object on the heap.
 * NOTE: some parameters are still not exposed
 */
extern "C" auto dGrid_new(
    void**                handle,
    void*                 backendPtr,
    const Neon::index_3d* dim,
    int const*            sparsity_pattern,
    int                   numStencilPoints,
    int const*            stencilPointFlatArray)
    -> int;

/**
 * Delete a grid object on the heap.
 */
extern "C" auto dGrid_delete(
    void** handle)
    -> int;

extern "C" auto dGrid_get_dimensions(
    void*           gridHandle,
    Neon::index_3d* dim)
    -> int;

/**
 * Generates a new field object on the heap.
 */
extern "C" auto dGrid_dField_new(
    void** handle,
    void*  grid,
    int    cardinality)
    -> int;

/**
 * Delete a field object on the heap.
 */
extern "C" auto dGrid_dField_delete(
    void** handle)
    -> int;

extern "C" auto dGrid_dField_get_partition(
    void*                           field_handle,
    Neon::dGrid::Partition<int, 0>* partition_handle,
    Neon::Execution                 execution,
    int                             device,
    Neon::DataView                  data_view)
    -> int;

extern "C" auto dGrid_get_span(
    void*              gridHandle,
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
    void*                 gridHandle,
    Neon::index_3d const* idx)
    -> int;

extern "C" auto dGrid_is_inside_domain(
    void*                       gridHandle,
    const Neon::index_3d* const idx)
    -> bool;

extern "C" auto dGrid_dField_read(
    void*                 fieldHandle,
    const Neon::index_3d* idx,
    int                   cardinality)
    -> int;

extern "C" auto dGrid_dField_write(
    void*                 fieldHandle,
    const Neon::index_3d* idx,
    int                   cardinality,
    int                   newValue)
    -> int;

extern "C" auto dGrid_dField_update_host_data(
    void* fieldHandle,
    int   streamSetId)
    -> int;

extern "C" auto dGrid_dField_update_device_data(
    void* fieldHandle,
    int   streamSetId)
    -> int;

extern "C" auto dGrid_dSpan_get_member_field_offsets(
    size_t* offsets,
    size_t* length)
    -> void;

extern "C" auto dGrid_dField_dPartition_get_member_field_offsets(
    size_t* offsets,
    size_t* length)
    -> void;
