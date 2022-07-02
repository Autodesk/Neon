#pragma once
#include <iostream>
#include "Neon/core/types/DeviceType.h"

namespace Neon {

/**
 * Enumeration for supported allocators
 */
enum struct Allocator
{
    MEM_ERROR = 0,        /**< Used for default initialization */
    CUDA_MEM_UNIFIED = 1, /**< CUDA unified memory allocation  */
    CUDA_MEM_DEVICE = 2,  /**< CUDA device memory allocation   */
    CUDA_MEM_HOST = 3,    /**< CUDA host memory allocation     */
    HWLOC_MEM = 4,        /**< HWLOC allocations (CPU side)    */
    MALLOC = 5,           /**< C++ malloc allocator            */
    NULL_MEM = 6,         /**< Allocation of a null pointer    */
    MANAGED = 7,          /**< Memory that the system does not need to garbage collect */
    MIXED_MEM = 8         /**< Used to described aggregated memory containers (like mirror) where potentially different memory types can coexsist*/
};

/**
 * Utility class for managing Allocator enum values.
 */
struct AllocatorUtils
{
    /**
     * Returns a string for the selected allocator
     *
     * @param allocator
     * @return
     */
    static auto toString(Allocator allocator) -> const char*;

    /**
     * Check is the selected device and allocator are compatible
     *
     * @param devEt
     * @param type
     * @return
     */
    static auto compatible(Neon::DeviceType devEt, Allocator type) -> bool;

    static auto getDefault(Neon::DeviceType devEt) -> Allocator;
};

/**
 * operator<<
 *
 * @param os
 * @param m
 * @return
 */
std::ostream& operator<<(std::ostream& os, Allocator const& m);

}  // End of namespace Neon
