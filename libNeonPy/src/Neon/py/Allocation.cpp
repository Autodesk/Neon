#include "Neon/py/Allocation.h"
#include "Neon/py/AllocationCounter.h"

auto get_allocation_counter()
    -> int
{
    return AllocationCounter::GetAllocationCount();
}