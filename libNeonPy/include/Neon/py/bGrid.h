#pragma once
#include <cstdint>
#include "Neon/domain/Grids.h"

/**
 * Initialize a new grid object on the heap.
 * NOTE: some parameters are still not exposed
 */ /* TODOMATT fix the constructor to have correct arguments */
extern "C" auto bGrid_new(
    uint64_t& handle,
    Neon::Backend* backendPtr,
    Neon::index_3d dim)
    -> int;

/**
 * Delete a grid object on the heap.
 */
extern "C" auto bGrid_delete(
    uint64_t& handle)
    -> int;

/**
 * Generates a new field object on the heap.
 */
extern "C" auto bGrid_bField_new(
    uint64_t& handle,
    uint64_t& grid)
    -> int;

/**
 * Delete a field object on the heap.
 */
extern "C" auto bGrid_bField_delete(
    uint64_t& handle)
    -> int;

extern "C" auto bGrid_bField_get_partition(
    uint64_t&                       field_handle,
    Neon::bGrid::Partition<int, 0>* partition_handle,
    Neon::Execution                 execution,
    int                             device,
    Neon::DataView                  data_view)
    -> int;

extern "C" auto bGrid_get_span(
    uint64_t&          gridHandle,
    Neon::bGrid::Span* spanRes,
    int                execution,
    int                device,
    int                data_view)
    -> int;

extern "C" auto bGrid_span_size(
    Neon::bGrid::Span* spanRes)
    -> int;

extern "C" auto bGrid_bField_partition_size(
    Neon::bGrid::Partition<int, 0>* partitionPtr)
    -> int;

extern "C" auto bGrid_get_properties( /* TODOMATT verify what the return of this method should be */
    uint64_t& gridHandle,
    const Neon::index_3d& idx) 
    -> int;

extern "C" auto bGrid_is_inside_domain(
    uint64_t& gridHandle,
    const Neon::index_3d& idx) 
    -> bool;

extern "C" auto bGrid_bField_read(
    uint64_t& fieldHandle,
    const Neon::index_3d& idx,
    const int& cardinality)
    -> int;

extern "C" auto bGrid_bField_write(
    uint64_t& fieldHandle,
    const Neon::index_3d& idx,
    const int& cardinality,
    int newValue)
    -> int;

extern "C" auto bGrid_bField_update_host_data(
    uint64_t& fieldHandle,
    int streamSetId)
    -> int;

extern "C" auto bGrid_bField_update_device_data(
    uint64_t& fieldHandle,
    int streamSetId)
    -> int;

extern "C" auto dGrid_dSpan_get_member_field_offsets(std::size_t* length)
    -> std::size_t*;