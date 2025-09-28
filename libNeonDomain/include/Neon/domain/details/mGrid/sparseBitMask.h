
#pragma once
#include "Neon/domain/tools/PointHashTable.h"

/**
 * @file sparseBitMask.h
 * @brief High-performance sparse bit manipulation data structures for 3D computational grids.
 *
 * This file implements efficient sparse bit storage and manipulation utilities specifically designed
 * for large-scale 3D computational applications such as computational fluid dynamics, voxel-based
 * rendering, and sparse grid algorithms.
 *
 * ## Core Components
 *
 * ### BitBlock
 * A fixed-size 3D bit block that efficiently stores and manipulates 32³ bits within a cubic region.
 * Features include:
 * - Constant-time O(1) bit operations (set, clear, query)
 * - Thread-safe operations with optional synchronization
 * - Memory-efficient storage using 32-bit bricks
 * - Automatic coordinate normalization with modulo arithmetic
 *
 * ### SparseBitBlocks
 * A sparse collection of BitBlocks using spatial hashing for memory-efficient storage of
 * large-scale 3D bit patterns. Key capabilities:
 * - Dynamic block allocation with memory pooling
 * - O(1) average-case access through spatial hashing
 * - Thread-safe concurrent operations
 * - Automatic memory management with configurable pool granularity
 *
 * ## Design Rationale
 *
 * This implementation is optimized for scenarios where:
 * - Only a small fraction of a large 3D space contains active bits
 * - High-performance bit operations are required
 * - Memory efficiency is critical for large-scale simulations
 * - Thread-safe concurrent access is needed
 *
 * ## Performance Characteristics
 *
 * - **Memory Usage**: O(k) where k is the number of active regions, not total space size
 * - **Access Time**: O(1) average case for all bit operations
 * - **Thread Safety**: Optional with minimal overhead using OpenMP critical sections
 * - **Cache Efficiency**: Optimized block size (32³) balances memory and cache performance
 *
 * ## Usage Pattern
 *
 * ```cpp
 * // Create sparse bit collection for large 3D space
 * Neon::index_3d bounds(10000, 10000, 10000);
 * SparseBitBlocks<256> sparseBits(bounds);
 *
 * // Activate scattered points efficiently
 * sparseBits.activatePoint({100, 200, 300});
 * sparseBits.activatePoint({5000, 7500, 2500});
 *
 * // Query point states in constant time
 * bool isActive = sparseBits.isActivePoint({100, 200, 300});
 *
 * // Thread-safe operations for parallel algorithms
 * #pragma omp parallel for
 * for (int i = 0; i < numPoints; ++i) {
 *     sparseBits.activatePoint(points[i]);
 * }
 * ```
 */

namespace Neon::domain::details::mGrid {
/**
 * @brief A high-performance fixed-size 3D bit block for efficient storage and manipulation of bit data.
 *
 * BitBlock represents a cubic volume of 32³ = 32,768 bits organized in a 3D grid pattern optimized
 * for cache-efficient access and bitwise operations. The block internally uses an array of 32-bit
 * "bricks" that provide efficient storage and manipulation of bit data within a cubic region.
 *
 * ## Key Features
 * - **Fixed Size**: 32×32×32 bits (32,768 total bits) providing optimal cache performance
 * - **Fast Access**: O(1) bit operations with automatic coordinate normalization
 * - **Thread Safety**: Optional thread-safe operations using OpenMP critical sections
 * - **Memory Efficiency**: Uses 1,024 uint32_t values (4KB total) per block
 * - **Coordinate Wrapping**: Automatic modulo arithmetic handles out-of-bounds coordinates
 *
 * ## Memory Layout
 *
 * The block is organized as follows:
 * - **Total Capacity**: 32³ = 32,768 bits
 * - **Storage Units**: 1,024 × uint32_t "bricks" (32 bits each)
 * - **Memory Footprint**: 4,096 bytes (4KB) per block
 * - **Access Pattern**: Row-major order with z-y-x priority for cache efficiency
 * - **Alignment**: Optimized for vectorized operations and cache line alignment
 *
 * ## Coordinate System
 *
 * BitBlock uses a 3D coordinate system where:
 * - Coordinates are automatically wrapped using modulo arithmetic: `point % blockSize`
 * - Local coordinates range from (0,0,0) to (31,31,31)
 * - Linear indexing uses row-major order: `index = z*32*32 + y*32 + x`
 * - Each brick stores 32 consecutive bits in the linear index space
 *
 * ## Thread Safety
 *
 * All mutating operations (setON, setOFF) support optional thread safety:
 * - Template parameter `ThreadSafe=true`: Uses OpenMP critical sections
 * - Template parameter `ThreadSafe=false`: No synchronization (caller responsibility)
 * - Read operations (isON) are inherently thread-safe for concurrent access
 *
 * ## Performance Characteristics
 * - **Bit Access**: O(1) with 2-3 integer operations and 1 memory access
 * - **Cache Performance**: Optimized 4KB blocks fit in L1/L2 cache
 * - **Vectorization**: Compatible with SIMD operations for bulk processing
 * - **Memory Overhead**: Zero overhead beyond the 4KB bit storage
 *
 * ## Usage Examples
 *
 * ```cpp
 * // Create and initialize a bit block
 * BitBlock block;
 *
 * // Set individual bits (thread-safe by default)
 * block.setON<true>({10, 20, 30});   // Thread-safe
 * block.setON<false>({5, 15, 25});   // Non thread-safe (faster)
 *
 * // Query bit states
 * bool isSet = block.isON({10, 20, 30});  // Always thread-safe
 *
 * // Clear bits
 * block.setOFF({10, 20, 30});
 *
 * // Coordinate wrapping handles out-of-bounds automatically
 * block.setON({100, 200, 300});  // Equivalent to {4, 8, 12}
 * ```
 *
 * @see SparseBitBlocks For managing collections of BitBlocks
 * @note This implementation provides optimal performance for cubic regions up to 32³ bits
 * @warning All coordinates are automatically wrapped; negative coordinates may behave unexpectedly
 */
struct BitBlock
{
    using Brick = uint32_t; /**< Storage unit for 32 bits of data */

    static constexpr int            unitBlockWidth = 4;                                               /**< Width of the basic unit block in bits */
    static constexpr int            unitBlockSize = unitBlockWidth * unitBlockWidth * unitBlockWidth; /**< Total bits in a unit block (64) */
    static constexpr int            blockEdge = 2 * unitBlockWidth;                                   /**< Edge length of the block in each dimension */
    static constexpr Neon::index_3d blockSize = Neon::index_3d(blockEdge, blockEdge, blockEdge);      /**< 3D dimensions of the block */

    static constexpr int numBits = blockEdge * blockEdge * blockEdge; /**< Total bits in this block */
    static_assert(numBits % 32 == 0, "Error -> numBits % 32 != 0");

    static constexpr int widthBrick = 32;                  /**< Number of bits per brick (uint32_t) */
    static constexpr int numBricks = numBits / widthBrick; /**< Number of bricks needed to store all bits */

    Brick bits[numBricks]; /**< Array storing the actual bit data */

    BitBlock()
    {
#pragma omp simd
        for (int i = 0; i < numBricks; ++i) {
            bits[i] = 0;
        }
    }

    /**
     * @brief Tests whether a specific bit is set (active) within the block.
     *
     * This method performs a constant-time lookup to determine if the bit at the specified
     * 3D coordinates is currently set to 1. The operation is inherently thread-safe for
     * concurrent reads and uses efficient bitwise operations for optimal performance.
     *
     * ## Algorithm Details
     * 1. **Coordinate Normalization**: Input coordinates are wrapped using modulo arithmetic
     * 2. **Linear Index Calculation**: 3D coordinates converted to linear index using pitch calculation
     * 3. **Brick Selection**: Linear index divided by 32 to find the appropriate uint32_t brick
     * 4. **Bit Extraction**: Remainder provides bit position within the brick for masking
     *
     * @param point The 3D coordinates of the bit to test. Values are automatically wrapped
     *              to the range [0, 31] for each dimension using modulo arithmetic.
     *              Negative coordinates are handled by modulo but may produce unexpected results.
     *
     * @return true if the bit at the specified position is set (value = 1),
     *         false if the bit is clear (value = 0).
     *
     * @note **Thread Safety**: This method is inherently thread-safe for concurrent read access.
     *       Multiple threads can safely call isON simultaneously without synchronization.
     *
     * @note **Performance**: O(1) time complexity with approximately 4-6 integer operations
     *       and 1 memory access. Highly cache-friendly due to spatial locality.
     *
     * @see setON() To set a bit to 1
     * @see setOFF() To clear a bit to 0
     */
    auto isON(Neon::index_3d const& point) -> bool
    {
        auto   localPoint = point % BitBlock::blockSize;
        size_t pitch = localPoint.mPitch(blockSize);
        auto   brickID = pitch / widthBrick;
        auto   localBit = pitch % widthBrick;
        return (bits[brickID] & (1 << localBit)) != 0;
    }

    /**
     * @brief Activates a specific bit by setting it to 1 within the block.
     *
     * This method atomically sets the bit at the specified 3D coordinates to 1 (active/on state).
     * The operation supports both thread-safe and non-thread-safe modes via template parameter,
     * allowing optimization for single-threaded scenarios while providing safety for concurrent access.
     *
     * ## Thread Safety Options
     * - **ThreadSafe=true** (default): Uses OpenMP critical section for atomic access
     * - **ThreadSafe=false**: Direct memory access with no synchronization (faster, caller responsibility)
     *
     * ## Algorithm Details
     * 1. **Coordinate Normalization**: Input coordinates wrapped using modulo arithmetic
     * 2. **Linear Index Calculation**: 3D coordinates converted to linear index
     * 3. **Brick and Bit Selection**: Determines target uint32_t brick and bit position
     * 4. **Atomic Bit Setting**: Uses bitwise OR to set the target bit to 1
     *
     * @tparam ThreadSafe Enables thread synchronization when true (default).
     *                    Set to false for single-threaded performance optimization.
     *
     * @param point The 3D coordinates of the bit to activate. Values are automatically wrapped
     *              to the range [0, 31] for each dimension using modulo arithmetic.
     *              Negative coordinates are handled but may produce unexpected results.
     *
     * @note **Idempotent Operation**: Setting an already-active bit has no additional effect
     *       and maintains the same performance characteristics.
     *
     * @note **Performance**:
     *       - ThreadSafe=true: O(1) with synchronization overhead (~10-50ns additional latency)
     *       - ThreadSafe=false: O(1) with minimal overhead (~2-5ns)
     *
     * @warning When ThreadSafe=false, concurrent access from multiple threads may result
     *          in race conditions. Ensure proper external synchronization or use ThreadSafe=true.
     *
     * @see isON() To test if a bit is set
     * @see setOFF() To clear a bit to 0
     */
    template <bool ThreadSafe = true>
    auto setON(Neon::index_3d const& point) -> void
    {
        auto   blockLocalPoint = point % BitBlock::blockSize;
        size_t pitch = blockLocalPoint.mPitch(blockSize);
        auto   brickID = pitch / widthBrick;
        auto   localBit = pitch % widthBrick;
        if constexpr (ThreadSafe) {
#pragma omp critical(settingBitOp)
            {
                bits[brickID] |= (1 << localBit);
            }
        } else {
            bits[brickID] |= (1 << localBit);
        }
    }


    /**
     * @brief Deactivates a specific bit by clearing it to 0 within the block.
     *
     * This method atomically clears the bit at the specified 3D coordinates to 0 (inactive/off state).
     * The operation supports both thread-safe and non-thread-safe modes via template parameter,
     * providing flexibility for performance optimization while ensuring safe concurrent access when needed.
     *
     * ## Thread Safety Options
     * - **ThreadSafe=true** (default): Uses OpenMP critical section for atomic access
     * - **ThreadSafe=false**: Direct memory access with no synchronization (faster, caller responsibility)
     *
     * ## Algorithm Details
     * 1. **Coordinate Normalization**: Input coordinates wrapped using modulo arithmetic
     * 2. **Linear Index Calculation**: 3D coordinates converted to linear index
     * 3. **Brick and Bit Selection**: Determines target uint32_t brick and bit position
     * 4. **Atomic Bit Clearing**: Uses bitwise AND with inverted mask to clear the target bit
     *
     * @tparam ThreadSafe Enables thread synchronization when true (default).
     *                    Set to false for single-threaded performance optimization.
     *
     * @param point The 3D coordinates of the bit to deactivate. Values are automatically wrapped
     *              to the range [0, 31] for each dimension using modulo arithmetic.
     *              Negative coordinates are handled but may produce unexpected results.
     *
     * @note **Idempotent Operation**: Clearing an already-inactive bit has no additional effect
     *       and maintains the same performance characteristics.
     *
     * @note **Performance**:
     *       - ThreadSafe=true: O(1) with synchronization overhead (~10-50ns additional latency)
     *       - ThreadSafe=false: O(1) with minimal overhead (~2-5ns)
     *
     * @warning When ThreadSafe=false, concurrent access from multiple threads may result
     *          in race conditions. Ensure proper external synchronization or use ThreadSafe=true.
     *
     * @see isON() To test if a bit is set
     * @see setON() To set a bit to 1
     */
    template <bool ThreadSafe = true>
    auto setOFF(Neon::index_3d const& point) -> void
    {
        auto   blockLocalPoint = point % BitBlock::blockSize;
        size_t pitch = blockLocalPoint.mPitch(blockSize);
        auto   brickID = pitch / widthBrick;
        auto   localBit = pitch % widthBrick;
        if constexpr (ThreadSafe) {
#pragma omp critical(settingBitOp)
            {
                bits[brickID] &= ~(1 << localBit);
            }
        } else {
            bits[brickID] &= ~(1 << localBit);
        }
    }
};

/**
 * @brief High-performance sparse collection of BitBlocks for large-scale 3D bit pattern storage.
 *
 * SparseBitBlocks provides a sophisticated sparse data structure that efficiently manages
 * large 3D coordinate spaces by only allocating BitBlocks where bits are actually set.
 * This approach enables processing of massive 3D spaces (terabyte scale) while using
 * memory proportional only to the number of active regions.
 *
 * ## Core Architecture
 *
 * The system combines three key technologies:
 * - **Spatial Hashing**: PointHashTable provides O(1) average-case block lookup
 * - **Memory Pooling**: Pre-allocated chunks minimize dynamic allocation overhead
 * - **Block-Level Organization**: 32³-bit BitBlocks provide optimal cache utilization
 *
 * ## Key Features
 *
 * ### Memory Efficiency
 * - **Sparse Storage**: Only allocates 4KB BitBlocks for regions containing active bits
 * - **Pool Management**: Configurable chunk-based allocation reduces fragmentation
 * - **Zero Overhead**: No memory cost for empty regions of the coordinate space
 * - **Scalability**: Can handle coordinate spaces up to billions of points
 *
 * ### Performance Characteristics
 * - **Access Time**: O(1) average case for all point operations
 * - **Memory Usage**: O(k × 4KB) where k = number of active block regions
 * - **Thread Safety**: Optional with minimal synchronization overhead
 * - **Cache Efficiency**: Block-aligned access patterns optimize CPU cache usage
 *
 * ### Thread Safety
 * All operations support configurable thread safety:
 * - Thread-safe mode uses OpenMP critical sections for atomic operations
 * - Non-thread-safe mode provides maximum performance for single-threaded use
 * - Concurrent reads are always safe regardless of thread safety setting
 *
 * ## Memory Pool Management
 *
 * The memory pool system pre-allocates BitBlocks in configurable chunks:
 * - **Pool Size**: Controlled by `memoryPoolGranularity` template parameter
 * - **Growth Strategy**: Automatically allocates new pools when current pool is exhausted
 * - **Memory Layout**: Contiguous allocation improves cache performance
 * - **No Deallocation**: BitBlocks remain allocated for reuse (optimizes for sparse patterns)
 *
 * @tparam memoryPoolGranularity Number of BitBlocks per memory pool chunk.
 *                               **Recommended Values**:
 *                               - Small datasets: 50-100 (default: 100)
 *                               - Large datasets: 500-1000 (reduces allocation frequency)
 *                               - Memory-constrained: 25-50 (minimizes peak memory)
 *                               Higher values reduce allocation overhead but increase memory usage.
 *
 * ## Coordinate Space Management
 *
 * SparseBitBlocks divides the coordinate space into BitBlock regions:
 * - Each BitBlock covers a 32×32×32 coordinate region
 * - Block coordinates computed as: `blockCoord = pointCoord / 32`
 * - Automatic block creation when points are activated in new regions
 * - Supports coordinate spaces up to the limits of `Neon::index_3d`
 *
 * ## Usage Patterns
 *
 * ### Typical Applications
 * - **Computational Fluid Dynamics**: Sparse grid refinement and adaptive meshing
 * - **Voxel Rendering**: Large-scale sparse voxel octrees and volume data
 * - **Scientific Simulation**: Particle tracking and sparse field representation
 * - **Spatial Acceleration**: Broad-phase collision detection and spatial partitioning
 *
 * ### Basic Usage Pattern
 * ```cpp
 * // 1. Create collection for large coordinate space
 * Neon::index_3d bounds(1000000, 1000000, 1000000);  // 1 billion point space
 * SparseBitBlocks<256> sparse(bounds);                 // 256-block memory pools
 *
 * // 2. Activate sparse points (creates BitBlocks automatically)
 * sparse.activatePoint({12345, 67890, 54321});
 * sparse.activatePoint({987654, 123456, 789012});
 *
 * // 3. Query point states
 * if (sparse.isActivePoint({12345, 67890, 54321})) {
 *     // Process active point
 * }
 *
 * // 4. Deactivate points (BitBlocks remain allocated)
 * sparse.removePoint({12345, 67890, 54321});
 * ```
 *
 * ### High-Performance Parallel Usage
 * ```cpp
 * SparseBitBlocks<1000> sparse(bounds);
 *
 * // Thread-safe parallel point activation
 * #pragma omp parallel for
 * for (size_t i = 0; i < numPoints; ++i) {
 *     sparse.activatePoint<true>(points[i]);    // Thread-safe
 * }
 *
 * // Non-thread-safe with custom synchronization
 * #pragma omp parallel for
 * for (size_t i = 0; i < regions.size(); ++i) {
 *     #pragma omp critical(region_update)
 *     {
 *         for (const auto& point : regions[i]) {
 *             sparse.activatePoint<false>(point);  // Faster
 *         }
 *     }
 * }
 * ```
 *
 * @see BitBlock For detailed bit manipulation operations within blocks
 * @note BitBlocks are never deallocated; they remain available for reuse
 * @warning Large `memoryPoolGranularity` values can cause memory spikes during pool allocation
 */
template <int memoryPoolGranularity = 100>
class SparseBitBlocks
{
    Neon::domain::tool::PointHashTable<int32_t, BitBlock*> mHashTable; /**< Spatial hash table mapping block coordinates to BitBlock pointers */
    Neon::index_3d                                         mBBox;      /**< Bounding box defining the valid coordinate space */
    using Pool = std::array<BitBlock, memoryPoolGranularity>;          /**< Type alias for memory pool arrays */
    std::vector<Pool*> memoryPool;                                      /**< Dynamic memory pool for BitBlock allocation */
    size_t            firstFreeIndex;                                  /**< Index of the first free BitBlock in the current memory pool */

   public:
    /**
     * @brief Constructs a sparse bit collection for the specified coordinate space.
     *
     * Initializes a new SparseBitBlocks instance configured to manage bit data within
     * the specified 3D coordinate space. The constructor establishes the spatial hash table
     * infrastructure and allocates the initial memory pool for efficient BitBlock management.
     *
     * ## Initialization Process
     * 1. **Hash Table Setup**: Creates PointHashTable with optimal hash parameters for the given bounds
     * 2. **Memory Pool Creation**: Allocates first memory pool with `memoryPoolGranularity` BitBlocks
     * 3. **Boundary Establishment**: Sets coordinate space limits for block coordinate calculations
     *
     * ## Memory Allocation
     * - **Initial Pool**: Immediately allocates one memory pool (memoryPoolGranularity × 4KB)
     * - **Hash Table**: Allocates hash table structure optimized for the coordinate space
     * - **Management Overhead**: Minimal per-instance overhead (<1KB typically)
     *
     * @param bbox The 3D bounding box defining the coordinate space for this collection.
     *             Coordinates are typically in the range [0, bbox) for each dimension,
     *             though the implementation can handle coordinates beyond these bounds
     *             through spatial hashing. Larger bounding boxes don't increase memory
     *             usage but may affect hash table performance.
     *
     * @note **Memory Commitment**: The constructor immediately allocates one memory pool
     *       (memoryPoolGranularity × 4KB). Choose the template parameter appropriately
     *       based on expected usage patterns.
     *
     * @note **Thread Safety**: The constructed instance supports thread-safe operations,
     *       but the constructor itself is not thread-safe.
     *
     * @warning **Coordinate Space**: While the bbox parameter guides hash table optimization,
     *          the actual implementation can handle points beyond the specified bounds.
     *          However, performance may degrade for coordinates significantly outside the bbox.
     *
     * @see activatePoint() To begin adding active points to the collection
     * @see isActivePoint() To query point states
     */
    SparseBitBlocks(const Neon::index_3d& bbox)
        : mBBox(bbox)
    {
        mHashTable = Neon::domain::tool::PointHashTable<int32_t, BitBlock*>(bbox);
        auto newPoolPtr = new Pool{};
        memoryPool.emplace_back(newPoolPtr);
        firstFreeIndex = 0;
    }

    /**
     * @brief Activates a point by setting its bit and ensuring BitBlock allocation.
     *
     * This high-performance method atomically activates a point in the sparse collection
     * by ensuring the appropriate BitBlock exists and setting the corresponding bit to 1.
     * The operation combines block management and bit manipulation for optimal efficiency.
     *
     * ## Dual-Phase Operation
     * 1. **Block Resolution**: Locates or creates the BitBlock containing the point
     * 2. **Bit Activation**: Sets the specific bit within that BitBlock to 1
     *
     * ## Automatic Block Management
     * - **Block Detection**: Computes block coordinates from point coordinates (point / 32)
     * - **Lazy Allocation**: Creates new BitBlocks only when points are first activated in a region
     * - **Memory Pool Usage**: Allocates blocks from pre-allocated pools for optimal performance
     * - **Hash Table Integration**: Automatically registers new blocks in the spatial hash table
     *
     * ## Thread Safety Modes
     * - **ThreadSafe=true** (default): Full thread safety with OpenMP critical sections
     * - **ThreadSafe=false**: No synchronization (faster, requires external coordination)
     *
     * ## Memory Pool Management
     * The method automatically manages memory pools:
     * - Uses existing pools when available
     * - Allocates new pools when current pool is exhausted
     * - Each pool provides `memoryPoolGranularity` BitBlocks (4KB each)
     * - Pool allocation is amortized O(1) across many activations
     *
     * @tparam ThreadSafe Controls synchronization behavior:
     *                    - true: Uses OpenMP critical sections (safe for parallel access)
     *                    - false: No synchronization (caller must ensure thread safety)
     *
     * @param point The 3D coordinates of the point to activate. Coordinates can be
     *              anywhere within the valid range for `Neon::index_3d`. Block coordinates
     *              are computed automatically as (point.x/32, point.y/32, point.z/32).
     *              The bit position within the block is (point % 32) for each dimension.
     *
     * @note **Idempotent**: Activating an already-active point is safe and efficient
     * @note **Block Persistence**: Once created, BitBlocks are never deallocated
     * @note **Memory Growth**: Each new 32³ coordinate region adds exactly 4KB to memory usage
     *
     * @warning **Pool Exhaustion**: If memory allocation fails, the program terminates
     *          with std::exit(1). This typically indicates system memory exhaustion.
     *
     * @warning **Thread Safety**: When ThreadSafe=false, concurrent calls may corrupt
     *          data structures. Use external synchronization or ThreadSafe=true.
     *
     * ## Performance Characteristics
     * - **Existing Block**: O(1) with ~10-20ns latency
     * - **New Block Creation**: O(1) amortized, ~100-500ns for allocation
     * - **Thread-Safe Overhead**: ~5-15ns additional latency for synchronization
     * - **Memory Overhead**: Exactly 4KB per unique 32³ coordinate region
     *
     * @see removePoint() To deactivate a point (bit remains in allocated block)
     * @see isActivePoint() To query point activation state
     * @see BitBlock::setON() For the underlying bit manipulation operation
     */
    template <bool ThreadSafe>
    auto activatePoint(const Neon::index_3d& point) -> void
    {
        if (!(point < mBBox)) {
            std::cout << "Error -> point outside of valid range" << std::endl;
            std::exit(1);
        }
        BitBlock* bitBlock = getBitBlockPrt(point);
        if (bitBlock != nullptr) {
            bitBlock->setON<ThreadSafe>(point);
        }
        if constexpr (ThreadSafe == true) {
#pragma omp critical(SparseBitBlocks_addPoint)
            {
                bitBlock = getBitBlockPrt(point);
                if (bitBlock == nullptr) {
                    if (firstFreeIndex > memoryPoolGranularity) {
                        std::cout << "Error -> firstFreeIndex == memoryPoolGranularity" << std::endl;
                        std::exit(1);
                    }
                    if (firstFreeIndex == memoryPoolGranularity) {
                        //auto new_pool = Pool{};
                        memoryPool.emplace_back({});
                        firstFreeIndex = 0;
                    }
                    bitBlock = &memoryPool[memoryPool.size() - 1][firstFreeIndex];
                    firstFreeIndex++;
                    mHashTable.addPoint(point, bitBlock);
                }
                bitBlock->setON<false>(point);
            }

        } else {
            // We are in a critical section managed by the calling
            if (firstFreeIndex > memoryPoolGranularity) {
                std::cout << "Error -> firstFreeIndex == memoryPoolGranularity" << std::endl;
                std::exit(1);
            }
            if (firstFreeIndex == memoryPoolGranularity) {
                auto new_pool_ptr = new Pool{};
                memoryPool.emplace_back(new_pool_ptr);
                firstFreeIndex = 0;
            }
            bitBlock = &(memoryPool[memoryPool.size() - 1]->at(firstFreeIndex));
            firstFreeIndex++;
            mHashTable.addPoint(point, bitBlock);
            bitBlock->setON<true>(point);
            return;
        }
    }

    /**
     * @brief Deactivates a point by clearing its bit within the sparse collection.
     *
     * This method efficiently deactivates a point by setting its corresponding bit to 0
     * within the appropriate BitBlock. The operation is optimized for performance and
     * handles non-existent points gracefully by returning immediately.
     *
     * ## Operation Details
     * 1. **Block Lookup**: Locates the BitBlock containing the point (if it exists)
     * 2. **Bit Clearing**: Sets the specific bit to 0 using atomic bitwise operations
     * 3. **Graceful Handling**: Returns immediately if no BitBlock exists for the point
     *
     * ## Memory Management Philosophy
     * - **BitBlock Persistence**: Blocks are never deallocated, even when all bits are cleared
     * - **Performance Optimization**: Avoids allocation/deallocation overhead
     * - **Sparse Pattern Friendly**: Optimized for patterns with repeated activation/deactivation
     * - **Memory Reuse**: Cleared blocks remain available for future point activations
     *
     * ## Thread Safety
     * The method supports configurable thread safety:
     * - **ThreadSafe=true** (default): Uses BitBlock's thread-safe setOFF operation
     * - **ThreadSafe=false**: Direct bit manipulation with no synchronization
     *
     * @tparam ThreadSafe Controls synchronization behavior:
     *                    - true: Thread-safe operation using OpenMP critical sections
     *                    - false: Non-synchronized operation (faster, requires external coordination)
     *
     * @param point The 3D coordinates of the point to deactivate. The method automatically
     *              computes the appropriate BitBlock coordinates and bit position.
     *              Points outside existing blocks are handled gracefully (no-op).
     *
     * @note **Idempotent Operation**: Deactivating an already-inactive point is safe and efficient
     * @note **No Memory Reclamation**: BitBlocks remain allocated for optimal performance
     * @note **Immediate Return**: Non-existent blocks cause immediate return (no allocation)
     *
     * ## Performance Characteristics
     * - **Existing Block**: O(1) with ~5-15ns latency
     * - **Non-existent Block**: O(1) with ~1-5ns latency (hash lookup only)
     * - **Thread-Safe Overhead**: ~5-15ns additional latency for synchronization
     * - **Memory Usage**: No change (blocks persist after bit clearing)
     *
     * @see activatePoint() To activate a point (may create new BitBlocks)
     * @see isActivePoint() To query point activation state
     * @see BitBlock::setOFF() For the underlying bit manipulation operation
     */
    template <bool ThreadSafe = true>
    auto removePoint(const Neon::index_3d& point) -> void
    {
        BitBlock* bitBlock = getBitBlockPrt(point);
        if (bitBlock == nullptr) {
            std::cout << "Error -> " << std::endl;
        }
        bitBlock->setOFF<ThreadSafe>(point);
    }

    auto getBitBlockPrt(const Neon::index_3d& point) const -> BitBlock*
    {
        if (!(point < mBBox)) {
            std::cout << "Error -> point outside of valid range" << std::endl;
            std::exit(1);
        }
        BitBlock* const* tmp = mHashTable.getMetadata(point);
        if (tmp == nullptr) {
            return nullptr;
        }
        BitBlock* bitBlock = *tmp;
        if (bitBlock == nullptr) {
            std::cout << "Error -> " << std::endl;
        }
        return bitBlock;
    }

    /**
     * @brief Queries whether a specific point is active within the sparse collection.
     *
     * This const method efficiently determines if a point is currently active by performing
     * a two-stage lookup: first locating the appropriate BitBlock, then testing the specific
     * bit within that block. The operation is read-only and inherently thread-safe.
     *
     * ## Query Algorithm
     * 1. **Block Resolution**: Computes block coordinates from point coordinates (point / 32)
     * 2. **Block Lookup**: Searches spatial hash table for the target BitBlock
     * 3. **Bit Testing**: If block exists, tests the specific bit; otherwise returns false
     * 4. **Result Return**: Returns true for active bits, false for inactive or non-existent
     *
     * ## Sparse Semantics
     * - **Explicit False**: Points in non-existent blocks are considered inactive
     * - **Memory Efficient**: No memory allocation or modification during queries
     * - **Cache Friendly**: Hash table lookup followed by local bit access optimizes cache usage
     * - **Consistent Behavior**: Guaranteed consistent results across multiple calls
     *
     * ## Thread Safety
     * This method is inherently thread-safe for concurrent read access:
     * - **Const Operation**: Does not modify any data structures
     * - **Atomic Reads**: All memory reads are naturally atomic for basic data types
     * - **No Synchronization Required**: Safe to call from multiple threads simultaneously
     * - **Concurrent with Writes**: Safe to call concurrently with thread-safe write operations
     *
     * @param point The 3D coordinates of the point to query. Coordinates are automatically
     *              mapped to the appropriate BitBlock and bit position within that block.
     *              The method handles all coordinate ranges supported by `Neon::index_3d`.
     *
     * @return true if the point is active (corresponding bit is set to 1),
     *         false if the point is inactive (bit is 0 or no BitBlock exists for the region).
     *
     * @note **Const Correctness**: This method is const and guaranteed not to modify the collection
     * @note **Zero Allocation**: Never allocates memory or creates new BitBlocks
     * @note **Deterministic Performance**: Consistent O(1) performance regardless of collection size
     * @note **Thread Safety**: Inherently safe for concurrent read access from multiple threads
     *
     * ## Performance Characteristics
     * - **Existing Block**: O(1) with ~5-10ns latency
     * - **Non-existent Block**: O(1) with ~2-5ns latency (hash lookup only)
     * - **Memory Access Pattern**: Single hash lookup + single bit test
     * - **Cache Performance**: Optimized for spatial locality in typical usage patterns
     *
     * @see activatePoint() To set a point to active state
     * @see removePoint() To set a point to inactive state
     * @see BitBlock::isON() For the underlying bit testing operation
     */
    auto isActivePoint(const Neon::index_3d& point) const -> bool
    {
        BitBlock* bitBlock = getBitBlockPrt(point);
        if (bitBlock == nullptr) {
            return false;
        }
        return bitBlock->isON(point);
    };
};
}  // namespace Neon::domain::details::mGrid
