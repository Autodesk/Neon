#pragma once

#include "Neon/sys/devices/DevInterface.h"
#include "Neon/core/types/Allocator.h"


namespace Neon {
namespace sys {

class CpuDev : public DeviceInterface
{
   public:
    CpuDev();
    virtual ~CpuDev() = default;

    /**
     * Returns ratio of used memory to total memory
     * @return ratio of used memory to total memory
     */
    double usage() const override;

    /**
     * Returns the size of available virtual memory
     */
    int64_t virtMemory() const override;

    /**
     * Returns the size of available physical memory
     */
    int64_t physMemory() const override;

    /**
     * Returns the size of used virtual memory
     */
    int64_t usedVirtMemory() const override;

    /**
     * Returns the size of used physical memory
     */
    int64_t usedPhysMemory() const override;

    int64_t processUsedPhysMemory() const override;

    struct memory_t
    {
        /**
         * Allocating memory with the standard malloc method.
         */
        static void* mallocByte(size_t size);
        /**
         * Frees memory allocated with the standard malloc method.
         */
        static void free(void* pointer);
        /**
         * Allocating memory with the cuda host allocator.
         * This is pinned memory.
         */
        static void* mallocCudaHostByte(size_t size);
        /**
         * Frees memory allocated with the cuda host allocator.
         * This is pinned memory.
         */
        static void freeCudaHostByte(void* pointer);
        /**
         * Allocating memory with the cuda managed allocator (Unified memory).
         * This is pinned memory.
         */
        static void* mallocCudaManagedByte(size_t size);
        /**
         * Frees memory allocated with the cuda managed allocator (Unified memory).
         * This is pinned memory.
         */
        static void freeCudaManagedByte(void* pointer);
    };

    memory_t memory;

   private:
    static int64_t parseLineGetByte(char* line);
};

}  // namespace sys
}  // End of namespace Neon
