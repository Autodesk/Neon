#include "Neon/py/AllocationCounter.h"

// Initialize the static member variable
int AllocationCounter::allocationCount = 0;

// Initialize the static mutex
std::mutex AllocationCounter::allocationMutex;

// Static method to manage allocations
void AllocationCounter::Allocation() {
    std::lock_guard<std::mutex> lock(allocationMutex);
    allocationCount++;
}

// Static method to manage deallocations
void AllocationCounter::Deallocation() {
    std::lock_guard<std::mutex> lock(allocationMutex);
    allocationCount--;
}

// Static method to get the current allocation count
int AllocationCounter::GetAllocationCount() {
    std::lock_guard<std::mutex> lock(allocationMutex);
    return allocationCount;
}

// Private static method to get the singleton instance
AllocationCounter& AllocationCounter::GetInstance() {
    static AllocationCounter instance;
    return instance;
}