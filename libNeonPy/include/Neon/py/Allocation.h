#pragma once
#include <cstdint>
#include "AllocationCounter.h"

/**
 * Get Counter
 */
extern "C" auto get_allocation_counter()
    -> int;