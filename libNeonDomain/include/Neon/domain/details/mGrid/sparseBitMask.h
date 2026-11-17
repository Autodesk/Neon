#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "Neon/core/core.h"

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
 * A fixed-size 3D bit block that efficiently stores and manipulates 8³ = 512 bits within a cubic region.
 * Features include:
 * - Constant-time O(1) bit operations (set, clear, query)
 * - Thread-safe operations with optional synchronization
 * - Memory-efficient storage using 32-bit bricks
 * - Automatic coordinate normalization with modulo arithmetic
 *
 * ### SparseBitBlocks
 * A sparse collection of BitBlocks, indexed by a two-level table of atomic pointers, for memory-efficient storage of
 * large-scale 3D bit patterns. Key capabilities:
 * - Dynamic block allocation with memory pooling
 * - O(1) lock-free lookup: two dependent loads, no hashing
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
 * - **Thread Safety**: Lock-free reads and atomic bit writes; only block allocation takes a lock
 * - **Cache Efficiency**: A block (8³ bits) is exactly one 64-byte cache line
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
 * BitBlock represents a cubic volume of 8³ = 512 bits organized in a 3D grid pattern optimized
 * for cache-efficient access and bitwise operations. The block internally uses an array of 32-bit
 * "bricks" that provide efficient storage and manipulation of bit data within a cubic region.
 *
 * ## Key Features
 * - **Fixed Size**: 8×8×8 bits (512 total bits), exactly one 64-byte cache line
 * - **Fast Access**: O(1) bit operations with automatic coordinate normalization
 * - **Thread Safety**: Optional atomic bit updates (`omp atomic`)
 * - **Memory Efficiency**: Uses 16 uint32_t values (64 bytes total) per block
 * - **Coordinate Wrapping**: Automatic modulo arithmetic handles out-of-bounds coordinates
 *
 * ## Memory Layout
 *
 * The block is organized as follows:
 * - **Total Capacity**: 8³ = 512 bits
 * - **Storage Units**: 16 × uint32_t "bricks" (32 bits each)
 * - **Memory Footprint**: 64 bytes per block
 * - **Access Pattern**: Row-major order with z-y-x priority for cache efficiency
 * - **Alignment**: Optimized for vectorized operations and cache line alignment
 *
 * ## Coordinate System
 *
 * BitBlock uses a 3D coordinate system where:
 * - Coordinates are automatically wrapped using modulo arithmetic: `point % blockSize`
 * - Local coordinates range from (0,0,0) to (7,7,7)
 * - Linear indexing uses row-major order: `index = z*8*8 + y*8 + x`
 * - Each brick stores 32 consecutive bits in the linear index space
 *
 * ## Thread Safety
 *
 * All mutating operations (setON, setOFF) support optional thread safety:
 * - Template parameter `ThreadSafe=true`: `omp atomic update` on the affected brick
 * - Template parameter `ThreadSafe=false`: No synchronization (caller responsibility)
 * - isON reads the brick with `omp atomic read`, so it may run concurrently with ThreadSafe writers
 *
 * ## Performance Characteristics
 * - **Bit Access**: O(1) with 2-3 integer operations and 1 memory access
 * - **Cache Performance**: A block is a single cache line
 * - **Vectorization**: Compatible with SIMD operations for bulk processing
 * - **Memory Overhead**: Zero overhead beyond the 64-byte bit storage
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
 * block.setON({100, 200, 300});  // Equivalent to {4, 0, 4}
 * ```
 *
 * @see SparseBitBlocks For managing collections of BitBlocks
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
     *              to the range [0, 7] for each dimension using modulo arithmetic.
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
    auto isON(Neon::index_3d const& point) const -> bool
    {
        auto   localPoint = point % BitBlock::blockSize;
        size_t pitch = localPoint.mPitch(blockSize);
        auto   brickID = pitch / widthBrick;
        auto   localBit = pitch % widthBrick;
        return (loadBrick(brickID) & (Brick(1) << localBit)) != 0;
    }

    /**
     * @brief Reads one brick without tearing or racing a concurrent setON/setOFF.
     *
     * Writers update bricks with `omp atomic update`; a plain load racing with them is a
     * data race in the C++ memory model even though it is a single aligned 32-bit move.
     * A relaxed atomic read compiles to the same instruction on x86 and AArch64.
     */
    auto loadBrick(size_t brickID) const -> Brick
    {
        Brick brick;
#pragma omp atomic read
        brick = bits[brickID];
        return brick;
    }

    /**
     * @brief Activates a specific bit by setting it to 1 within the block.
     *
     * This method atomically sets the bit at the specified 3D coordinates to 1 (active/on state).
     * The operation supports both thread-safe and non-thread-safe modes via template parameter,
     * allowing optimization for single-threaded scenarios while providing safety for concurrent access.
     *
     * ## Thread Safety Options
     * - **ThreadSafe=true** (default): Uses an `omp atomic update` on the brick
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
     *              to the range [0, 7] for each dimension using modulo arithmetic.
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
        auto        blockLocalPoint = point % BitBlock::blockSize;
        size_t      pitch = blockLocalPoint.mPitch(blockSize);
        auto        brickID = pitch / widthBrick;
        auto        localBit = pitch % widthBrick;
        Brick const mask = Brick(1) << localBit;
        if constexpr (ThreadSafe) {
            // An atomic read-modify-write on a single brick. This replaces a named
            // critical section, which was a *global* lock shared by every BitBlock in
            // the process and serialized all bitmask writes during mGrid construction.
#pragma omp atomic update
            bits[brickID] |= mask;
        } else {
            bits[brickID] |= mask;
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
     * - **ThreadSafe=true** (default): Uses an `omp atomic update` on the brick
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
     *              to the range [0, 7] for each dimension using modulo arithmetic.
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
        auto        blockLocalPoint = point % BitBlock::blockSize;
        size_t      pitch = blockLocalPoint.mPitch(blockSize);
        auto        brickID = pitch / widthBrick;
        auto        localBit = pitch % widthBrick;
        Brick const mask = ~(Brick(1) << localBit);
        if constexpr (ThreadSafe) {
            // See setON: an atomic update instead of a process-wide critical section.
#pragma omp atomic update
            bits[brickID] &= mask;
        } else {
            bits[brickID] &= mask;
        }
    }
};

/**
 * @brief High-performance sparse collection of BitBlocks for large-scale 3D bit pattern storage.
 *
 * SparseBitBlocks provides a sophisticated sparse data structure that efficiently manages
 * large 3D coordinate spaces by only allocating BitBlocks where bits are actually set.
 * Memory is proportional to the number of active regions, plus a small directory that
 * scales with the bounding box (8 bytes per 128³ points).
 *
 * ## Core Architecture
 *
 * The system combines three pieces:
 * - **Two-level lookup**: a fixed directory of atomic Tile pointers, each Tile holding
 *   16³ atomic BitBlock pointers; a lookup is two dependent loads and never locks
 * - **Memory Pooling**: Pre-allocated chunks minimize dynamic allocation overhead
 * - **Block-Level Organization**: 8³-bit BitBlocks, one cache line each
 *
 * ## Key Features
 *
 * ### Memory Efficiency
 * - **Sparse Storage**: Only allocates BitBlocks and Tiles for regions containing active bits
 * - **Pool Management**: Configurable chunk-based allocation reduces fragmentation
 * - **Zero Overhead**: No memory cost for empty regions of the coordinate space
 * - **Scalability**: Can handle coordinate spaces up to billions of points
 *
 * ### Performance Characteristics
 * - **Access Time**: O(1) worst case for all point operations
 * - **Memory Usage**: O(k × 64 bytes) where k = number of active block regions, plus 32KB per touched tile
 * - **Thread Safety**: Lock-free lookups; a lock is only taken to allocate a new block
 * - **Cache Efficiency**: Block-aligned access patterns optimize CPU cache usage
 *
 * ### Thread Safety
 * All operations support configurable thread safety:
 * - Thread-safe mode updates bits with `omp atomic` and allocates blocks under the
 *   SparseBitBlocks_addPoint critical section; lookups never lock
 * - Non-thread-safe mode provides maximum performance for single-threaded use
 * - Reads may run concurrently with thread-safe writes
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
 * - Each BitBlock covers an 8×8×8 coordinate region
 * - Block coordinates computed as: `blockCoord = pointCoord / 8`
 * - Automatic block creation when points are activated in new regions
 * - Supports coordinate spaces up to directoryMaxEntries tiles of 128³ points
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
    /**
     * The block-coordinate -> BitBlock* lookup is a two-level table: a directory with one
     * entry per tileEdge^3 block slots, each pointing to a lazily allocated Tile of slots.
     *
     * Readers never take a lock, so every location a reader can load from must stay put
     * for the lifetime of the collection: the directory is sized once in the constructor,
     * Tiles and BitBlocks are never moved or freed, and each directory entry and tile slot
     * changes at most once, from nullptr to its final value, under the
     * SparseBitBlocks_addPoint critical section. Publication is a release store and every
     * lookup an acquire load, so a reader that sees a pointer also sees what it points to.
     */
    static constexpr int    tileShift = 4;
    static constexpr int    tileEdge = 1 << tileShift;
    static constexpr int    tileMask = tileEdge - 1;
    static constexpr size_t tileSlots = size_t(tileEdge) * size_t(tileEdge) * size_t(tileEdge);

    /** Refuse coordinate spaces whose directory alone would take more than 8GB. */
    static constexpr size_t directoryMaxEntries = 1024ull * 1024ull * 1024ull;

    struct Tile
    {
        std::atomic<BitBlock*> slot[tileSlots]{};
    };

    using Pool = std::array<BitBlock, memoryPoolGranularity>; /**< Type alias for memory pool arrays */

    Neon::index_3d                        mBBox;            /**< Valid block-coordinate space */
    Neon::index_3d                        mTileGrid;        /**< Directory extent: mBBox rounded up to whole tiles */
    std::unique_ptr<std::atomic<Tile*>[]> mDirectory;       /**< One entry per tile, never reallocated */
    std::vector<std::unique_ptr<Tile>>    mTiles;           /**< Owners of the allocated tiles */
    std::vector<std::unique_ptr<Pool>>    memoryPool;       /**< Dynamic memory pool for BitBlock allocation */
    size_t                                firstFreeIndex;   /**< Index of the first free BitBlock in the current memory pool */
    std::vector<Neon::index_3d>           mAllocatedBlocks; /**< Coordinate of every BitBlock handed out, in allocation order */

    auto tileIndex(const Neon::index_3d& blockCoord) const -> size_t
    {
        return Neon::index_3d(blockCoord.x >> tileShift,
                              blockCoord.y >> tileShift,
                              blockCoord.z >> tileShift)
            .mPitch(mTileGrid);
    }

    static auto slotIndex(const Neon::index_3d& blockCoord) -> size_t
    {
        return Neon::index_3d(blockCoord.x & tileMask,
                              blockCoord.y & tileMask,
                              blockCoord.z & tileMask)
            .mPitch(tileEdge, tileEdge);
    }

   public:
    /**
     * @brief Constructs a sparse bit collection for the specified coordinate space.
     *
     * Sizes the lookup directory for the given coordinate space and allocates the first
     * memory pool. Tiles of the lookup table are only allocated when a block inside them is
     * first activated.
     *
     * ## Memory Allocation
     * - **Directory**: 8 bytes per 128³ points of the bounding box (one entry per tile of
     *   16³ blocks), allocated here and never resized
     * - **Tiles**: 32KB each, allocated lazily, so proportional to the active region
     * - **Initial Pool**: one memory pool of memoryPoolGranularity × 64 bytes
     *
     * @param bbox The 3D bounding box defining the coordinate space for this collection.
     *             Valid points have coordinates in [0, bbox) in every dimension; activating
     *             a point outside it terminates the process.
     *
     * @throws NeonException if the directory would exceed directoryMaxEntries entries.
     *
     * @note **Thread Safety**: The constructed instance supports thread-safe operations,
     *       but the constructor itself is not thread-safe.
     *
     * @see activatePoint() To begin adding active points to the collection
     * @see isActivePoint() To query point states
     */
    SparseBitBlocks(const Neon::index_3d& bbox)
        : mBBox( (bbox.x + BitBlock::blockSize.x - 1) / BitBlock::blockSize.x,
                 (bbox.y + BitBlock::blockSize.y - 1) / BitBlock::blockSize.y,
                 (bbox.z + BitBlock::blockSize.z - 1) / BitBlock::blockSize.z ),
          mTileGrid((mBBox.x + tileEdge - 1) / tileEdge,
                    (mBBox.y + tileEdge - 1) / tileEdge,
                    (mBBox.z + tileEdge - 1) / tileEdge)
    {
        const size_t tilesXY = static_cast<size_t>(mTileGrid.x) * static_cast<size_t>(mTileGrid.y);
        const size_t tilesZ = static_cast<size_t>(mTileGrid.z);
        // Compared by division: tilesXY * tilesZ can overflow size_t for huge bounding boxes
        if (tilesZ != 0 && tilesXY > directoryMaxEntries / tilesZ) {
            NeonException exp("SparseBitBlocks");
            exp << "Coordinate space " << bbox << " needs a directory of " << mTileGrid
                << " tiles, more than the supported " << directoryMaxEntries;
            NEON_THROW(exp);
        }
        const size_t numTiles = tilesXY * tilesZ;
        // make_unique<T[]> value-initializes, so every entry starts as nullptr
        mDirectory = std::make_unique<std::atomic<Tile*>[]>(numTiles);

        memoryPool.emplace_back(std::make_unique<Pool>());
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
     * - **Block Detection**: Computes block coordinates from point coordinates (point / 8)
     * - **Lazy Allocation**: Creates new BitBlocks only when points are first activated in a region
     * - **Memory Pool Usage**: Allocates blocks from pre-allocated pools for optimal performance
     * - **Publication**: Registers new blocks in the lookup table with a release store
     *
     * ## Thread Safety Modes
     * - **ThreadSafe=true**: Lock-free lookup and atomic bit update; a miss allocates the block under
     *   the SparseBitBlocks_addPoint critical section
     * - **ThreadSafe=false**: No synchronization (faster, requires external coordination)
     *
     * ## Memory Pool Management
     * The method automatically manages memory pools:
     * - Uses existing pools when available
     * - Allocates new pools when current pool is exhausted
     * - Each pool provides `memoryPoolGranularity` BitBlocks (64 bytes each)
     * - Pool allocation is amortized O(1) across many activations
     *
     * @tparam ThreadSafe Controls synchronization behavior:
     *                    - true: safe for parallel access; locks only to allocate a block
     *                    - false: No synchronization (caller must ensure thread safety)
     *
     * @param point The 3D coordinates of the point to activate. Coordinates can be
     *              anywhere within the valid range for `Neon::index_3d`. Block coordinates
     *              are computed automatically as (point.x/8, point.y/8, point.z/8).
     *              The bit position within the block is (point % 8) for each dimension.
     *
     * @note **Idempotent**: Activating an already-active point is safe and efficient
     * @note **Block Persistence**: Once created, BitBlocks are never deallocated
     * @note **Memory Growth**: Each new 8³ coordinate region adds 64 bytes, plus 32KB for the
     *       first block in a new 128³ tile
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
     * - **Memory Overhead**: 64 bytes per unique 8³ coordinate region
     *
     * @see removePoint() To deactivate a point (bit remains in allocated block)
     * @see isActivePoint() To query point activation state
     * @see BitBlock::setON() For the underlying bit manipulation operation
     */
    template <bool ThreadSafe>
    auto activatePoint(const Neon::index_3d& point) -> void
    {
        Neon::index_3d block_coord(point.x / BitBlock::blockSize.x,
                                   point.y / BitBlock::blockSize.y,
                                   point.z / BitBlock::blockSize.z);
        Neon::index_3d local(point.x % BitBlock::blockSize.x,
                             point.y % BitBlock::blockSize.y,
                             point.z % BitBlock::blockSize.z);

        if (!(block_coord < mBBox)) {
            std::cout << "Error -> block outside of valid range" << std::endl;
            std::exit(1);
        }

        BitBlock* bitBlock = getBitBlockPrt(block_coord);
        if (bitBlock != nullptr) {
            bitBlock->setON<ThreadSafe>(local);
            return;
        }

        if constexpr (ThreadSafe == true) {
#pragma omp critical(SparseBitBlocks_addPoint)
            {
                bitBlock = getBitBlockPrt(block_coord);
                if (bitBlock == nullptr) {
                    if (firstFreeIndex == memoryPoolGranularity) {
                        memoryPool.emplace_back(std::make_unique<Pool>());
                        firstFreeIndex = 0;
                    }
                    bitBlock = &((*memoryPool.back())[firstFreeIndex]);
                    firstFreeIndex++;
                    // Set the bit *before* publishing the pointer. Readers do not take
                    // this lock, so a block must never become reachable in a state where
                    // its first bit is still being written non-atomically.
                    bitBlock->setON<false>(local);
                    publishBitBlockPtr(block_coord, bitBlock);
                } else {
                    bitBlock->setON<true>(local);
                }
            }

        } else {
            // We are in a critical section managed by the calling
            if (firstFreeIndex == memoryPoolGranularity) {
                memoryPool.emplace_back(std::make_unique<Pool>());
                firstFreeIndex = 0;
            }
            bitBlock = &((*memoryPool.back())[firstFreeIndex]);
            firstFreeIndex++;
            bitBlock->setON<ThreadSafe>(local);
            publishBitBlockPtr(block_coord, bitBlock);
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
     *                    - true: Thread-safe operation using `omp atomic update`
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
     * - **Non-existent Block**: O(1) with ~1-5ns latency (lookup only)
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
        Neon::index_3d block_coord(point.x / BitBlock::blockSize.x,
                                   point.y / BitBlock::blockSize.y,
                                   point.z / BitBlock::blockSize.z);
        Neon::index_3d local(point.x % BitBlock::blockSize.x,
                             point.y % BitBlock::blockSize.y,
                             point.z % BitBlock::blockSize.z);

        BitBlock* bitBlock = getBitBlockPrt(block_coord);
        if (bitBlock == nullptr) {
            return;
        }
        bitBlock->setOFF<ThreadSafe>(local);
    }

    /**
     * @brief Makes a freshly allocated BitBlock reachable from the given block coordinate.
     *
     * Must only be called while holding the SparseBitBlocks_addPoint critical section
     * (or, for activatePoint<false>, the caller's equivalent), and only for a coordinate
     * that has no block yet. bitBlock must be fully initialized: once the release store
     * below lands, lock-free readers may use it.
     */
    auto publishBitBlockPtr(const Neon::index_3d& coord, BitBlock* bitBlock) -> void
    {
        mAllocatedBlocks.push_back(coord);

        std::atomic<Tile*>& entry = mDirectory[tileIndex(coord)];
        Tile*               tile = entry.load(std::memory_order_relaxed);
        if (tile == nullptr) {
            mTiles.push_back(std::make_unique<Tile>());
            tile = mTiles.back().get();
            entry.store(tile, std::memory_order_release);
        }
        tile->slot[slotIndex(coord)].store(bitBlock, std::memory_order_release);
    }

    /**
     * @brief Coordinates of every BitBlock that has been allocated.
     *
     * That is, every 8^3 region that has held at least one active point at some point.
     * A caller that needs to visit the active set can iterate these instead of sweeping
     * the whole coordinate space: for a sparse domain that is orders of magnitude less
     * work, and any point outside an allocated block is inactive by construction.
     *
     * Blocks are never freed, so a block whose bits have all since been cleared still
     * appears here. Returned by value because callers commonly mutate the collection
     * while iterating.
     */
    auto getAllocatedBlockCoords() const -> std::vector<Neon::index_3d>
    {
        return mAllocatedBlocks;
    }

    /**
     * @brief Invokes f(point) for every active point inside one allocated BitBlock.
     *
     * The point handed to the lambda is in this collection's coordinate space, not
     * block-local. Bricks holding nothing are skipped, so an allocated block that is
     * mostly empty costs far less than its 512 bits suggest.
     *
     * Bits are read as the traversal proceeds. A concurrent update may or may not be
     * observed, so a caller that mutates while iterating must re-check the points it
     * cares about with isActivePoint().
     */
    template <typename UserLambda>
    auto forEachActivePointInBlock(const Neon::index_3d& blockCoord,
                                   const UserLambda&     f) const -> void
    {
        BitBlock* bitBlock = getBitBlockPrt(blockCoord);
        if (bitBlock == nullptr) {
            return;
        }

        constexpr int edge = BitBlock::blockEdge;
        const int     baseX = blockCoord.x * edge;
        const int     baseY = blockCoord.y * edge;
        const int     baseZ = blockCoord.z * edge;

        for (int brickID = 0; brickID < BitBlock::numBricks; ++brickID) {
            const BitBlock::Brick brick = bitBlock->loadBrick(brickID);
            if (brick == 0) {
                continue;
            }
            for (int bit = 0; bit < BitBlock::widthBrick; ++bit) {
                if ((brick & (BitBlock::Brick(1) << bit)) == 0) {
                    continue;
                }
                // Matches Integer_3d::mPitch: pitch = x + y * edge + z * edge * edge
                const int pitch = brickID * BitBlock::widthBrick + bit;
                f(Neon::index_3d(baseX + pitch % edge,
                                 baseY + (pitch / edge) % edge,
                                 baseZ + pitch / (edge * edge)));
            }
        }
    }

    auto getBitBlockPrt(const Neon::index_3d& coord) const -> BitBlock*
    {
        if (!(coord < mBBox)) {
            std::cout << "Error -> coord outside of valid range" << std::endl;
            std::exit(1);
        }
        if (coord.x < 0 || coord.y < 0 || coord.z < 0) {
            return nullptr;
        }
        Tile const* tile = mDirectory[tileIndex(coord)].load(std::memory_order_acquire);
        if (tile == nullptr) {
            return nullptr;
        }
        return tile->slot[slotIndex(coord)].load(std::memory_order_acquire);
    }

    /**
     * @brief Queries whether a specific point is active within the sparse collection.
     *
     * This const method efficiently determines if a point is currently active by performing
     * a two-stage lookup: first locating the appropriate BitBlock, then testing the specific
     * bit within that block. The operation is read-only and inherently thread-safe.
     *
     * ## Query Algorithm
     * 1. **Block Resolution**: Computes block coordinates from point coordinates (point / 8)
     * 2. **Block Lookup**: Two acquire loads through the directory and the tile
     * 3. **Bit Testing**: If block exists, tests the specific bit; otherwise returns false
     * 4. **Result Return**: Returns true for active bits, false for inactive or non-existent
     *
     * ## Sparse Semantics
     * - **Explicit False**: Points in non-existent blocks are considered inactive
     * - **Memory Efficient**: No memory allocation or modification during queries
     * - **Cache Friendly**: The directory stays cache resident; consecutive blocks share a tile
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
     * - **Non-existent Block**: O(1) with ~2-5ns latency (lookup only)
     * - **Memory Access Pattern**: Two table loads + single bit test
     * - **Cache Performance**: Optimized for spatial locality in typical usage patterns
     *
     * @see activatePoint() To set a point to active state
     * @see removePoint() To set a point to inactive state
     * @see BitBlock::isON() For the underlying bit testing operation
     */
    auto isActivePoint(const Neon::index_3d& point) const -> bool
    {
        Neon::index_3d block_coord(point.x / BitBlock::blockSize.x,
                                   point.y / BitBlock::blockSize.y,
                                   point.z / BitBlock::blockSize.z);
        Neon::index_3d local(point.x % BitBlock::blockSize.x,
                             point.y % BitBlock::blockSize.y,
                             point.z % BitBlock::blockSize.z);

        BitBlock* bitBlock = getBitBlockPrt(block_coord);
        if (bitBlock == nullptr) {
            return false;
        }
        return bitBlock->isON(local);
    };
};
}  // namespace Neon::domain::details::mGrid
