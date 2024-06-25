#pragma once
#include <mutex>

class AllocationCounter {
public:
    // Static methods to manage allocations and deallocations
    static void Allocation();
    static void Deallocation();
    
    // Static method to get the current allocation count
    static int GetAllocationCount();

private:
    // Private constructor to prevent instantiation
    AllocationCounter() = default;

    // Private static method to get the singleton instance
    static AllocationCounter& GetInstance();

    // Private static variable to hold the allocation count
    static int allocationCount;

    // Mutex to protect access to allocationCount
    static std::mutex allocationMutex;
};

