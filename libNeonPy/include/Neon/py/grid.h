#pragma once
#include <cstdint>
#include "Neon/domain/Grids.h"

/**
 * Initialize a new grid object on the heap.
 * NOTE: some parameters are still not exposed
 */
extern "C" auto dGrid_new(uint64_t& handle) -> int;

/**
 * De allocate a new grid object on the heap.
 * NOTE: some parameters are still not exposed
 */
extern "C" auto dGrid_delete(uint64_t& handle) -> int;

extern "C" auto dGrid_dField_new(uint64_t& handle,
                                 uint64_t& grid) -> int;

extern "C" auto dGrid_dField_delete(uint64_t& handle) -> int;

extern "C" auto dGrid_get_span(uint64_t&          gridHandle,
                               Neon::dGrid::Span* spanRes,
                               int                execution,
                               int                device,
                               int                data_view) -> int;