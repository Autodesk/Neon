#pragma once
#include <cuda_runtime.h>
#include "Neon/core/types/Macros.h"

#ifdef NEON_PLACE_CUDA_DEVICE
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#endif

namespace Neon::sys {

extern __shared__ char SHMEM_START[];

#ifdef NEON_PLACE_CUDA_DEVICE
/**
 * Load data in shared memory using memcpu_async API with optional sync  
 * @param block cooperative group block 
 * @param in input buffer in global memory 
 * @param size number of elements to load
 * @param out output buffer in shared memory
 * @param with_wait whether to wait after loading in shared memory or not
*/
template <typename T, typename SizeT, typename CGType>
NEON_CUDA_DEVICE_ONLY __inline__ void loadSharedMemAsync(
    CGType&     block,
    const T*    in,
    const SizeT size,
    T*          out,
    bool        with_wait)
{

    cooperative_groups::memcpy_async(block, out, in, sizeof(T) * size);

    if (with_wait) {
        cooperative_groups::wait(block);
    }
}
#endif

/**
 * Shared memory allocator that should make it easy to allocate different
 * segments of the shared memory of different types.
 */
struct ShmemAllocator
{
    NEON_CUDA_HOST_DEVICE ShmemAllocator()
        : m_ptr(nullptr)
    {
#ifdef NEON_PLACE_CUDA_DEVICE
        m_ptr = SHMEM_START;
#endif
    }

    /**
     * Allocate num_bytes and return a pointer to the start of the
     * allocation. The return pointer is aligned to bytes_alignment.
     * This function could be called by all threads if ShmemAllocator is in the
     * register. If ShmemAllocator is declared as __shared__, only one thread per
     * block should call this function.
     * @param num_bytes to allocate
     * @param byteAlignment alignment size
     */
    NEON_CUDA_HOST_DEVICE __forceinline__ char* alloc(size_t numBytes,
                                                      size_t byteAlignment = 8)
    {
        align(byteAlignment, m_ptr);

        char* ret = m_ptr;

        m_ptr = m_ptr + numBytes;

        assert(getAllocatedSizeBytes() <= getMaxSizeBytes());

        return ret;
    }

    /**
     * a typed version of alloc() where the input number of elements (not
     * number of bytes).
     * @tparam T type of the pointer
     * @param count number of elements to be allocated
     * @param byteAlignment alignment size
     */
    template <typename T>
    NEON_CUDA_HOST_DEVICE __forceinline__ T* alloc(size_t count,
                                                   size_t byteAlignment = sizeof(T))
    {
        return reinterpret_cast<T*>(alloc(count * sizeof(T), byteAlignment));
    }

    /**
     * return the maximum allocation size which is the same as the number
     * of bytes passed during the kernel launch
     */
    NEON_CUDA_HOST_DEVICE __forceinline__ uint32_t getMaxSizeBytes()
    {
        uint32_t ret = 0;
#ifdef NEON_PLACE_CUDA_DEVICE
        asm("mov.u32 %0, %dynamic_smem_size;"
            : "=r"(ret));
#endif
        return ret;
    }

    /**
     * return the number of bytes that has been allocated
     */
    NEON_CUDA_HOST_DEVICE __forceinline__ uint32_t getAllocatedSizeBytes()
    {
#ifdef NEON_PLACE_CUDA_DEVICE
        return static_cast<uint32_t>(m_ptr - SHMEM_START);
#else
        return 0;
#endif
    }

   private:
    /**
     * given a pointer, this function returns a pointer to the first
     * location at the boundary of a given alignment size. This what std:align
     * does but it does not work with CUDA so this a stripped down version of
     * it.
     * @tparam T type of the pointer
     * @param byteAlignment number of bytes to get the pointer to be aligned to
     * @param ptr input/output pointer pointing at first usable location. On
     * return, it will be properly aligned to the beginning of the first element
     * that is aligned to alignment
     */
    template <typename T>
    NEON_CUDA_HOST_DEVICE __host__ __inline__ void align(const std::size_t byteAlignment,
                                                         T*&               ptr) noexcept
    {
        const uint64_t intptr = reinterpret_cast<uint64_t>(ptr);
        const uint64_t remainder = intptr % byteAlignment;
        if (remainder == 0) {
            return;
        }
        const uint64_t aligned = intptr + byteAlignment - remainder;
        ptr = reinterpret_cast<T*>(aligned);
    }
    char* m_ptr;
};

}  // namespace Neon::sys
