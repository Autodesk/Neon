#pragma once
#include <cstdint>

/**
* Initialize a new grid object on the heap.
* NOTE: some parameters are still not exposed
*/
extern "C" auto grid_new(uint64_t& handle)-> int;

/**
* De allocate a new grid object on the heap.
* NOTE: some parameters are still not exposed
*/
extern "C" auto grid_delete(uint64_t& handle)-> int;

extern "C" auto field_new(uint64_t& handle, uint64_t& grid)-> int;

extern "C" auto field_delete(uint64_t& handle)-> int;