
/**
 * @file mGrid.cpp
 * @brief Implementation of the multi-resolution grid (mGrid) data structure for adaptive mesh refinement
 *
 * This file implements a hierarchical multi-resolution grid system that stacks multiple block sparse grids
 * at different resolution levels to enable adaptive mesh refinement (AMR) computations. The mGrid is designed
 * for high-performance computing on both CPU and GPU platforms with seamless integration between resolution levels.
 *
 * ## Architecture Overview
 *
 * The mGrid consists of multiple resolution levels (0 = finest, N-1 = coarsest) where each level is an
 * independent block sparse grid with its own resolution and spacing. Levels are connected via parent-child
 * relationships to form a unified hierarchical data structure.
 *
 * ### Key Components:
 * - **Bitmask System**: Efficient tracking of active voxels using 32-bit bitmasks per block
 * - **Hierarchical Linking**: Parent-child relationships between resolution levels
 * - **Overlap Culling**: Removes redundant coarse cells that are fully covered by fine cells
 * - **Strong Balancing**: Ensures smooth resolution transitions (max 1 level difference between neighbors)
 * - **NVTX Integration**: Performance profiling with automatic range tracking
 *
 * ### Memory Layout:
 * - Each level maintains its own block sparse grid with independent memory allocation
 * - Bitmasks are stored contiguously for cache-efficient access patterns
 * - Parent-child relationships are stored in device-accessible memory sets
 *
 * ### Performance Characteristics:
 * - Construction: O(N * log(L)) where N = domain size, L = number of levels
 * - Memory: O(A * R^3) where A = active blocks, R = refinement factor
 * - Access: O(1) for same-level operations, O(log(L)) for cross-level operations
 *
 * ## Supported Operations:
 * - Multi-level field storage and access
 * - Hierarchical traversal between resolution levels
 * - Adaptive refinement and coarsening
 * - Parallel computation kernels across all levels
 * - Reduction operations with automatic level aggregation
 *
 * ## Thread Safety:
 * - Construction: Not thread-safe (single-threaded initialization required)
 * - Access: Thread-safe for read operations, requires synchronization for writes
 * - GPU Operations: Fully thread-safe within CUDA kernels
 *
 * @note This implementation is optimized for octree structures (2x2x2 refinement factor)
 * @see mGrid.h for the complete class interface and template parameters
 */

#include "Neon/domain/details//mGrid/mGrid.h"
#include "Neon/domain/details/mGrid/mPartition.h"
#include "Neon/domain/details/mGrid/sparseBitMask.h"


namespace Neon::domain::details::mGrid {

/**
 * @brief Constructs a multi-resolution grid by stacking multiple block sparse grids.
 *
 * Creates a hierarchical multi-resolution grid system where each level represents a different
 * resolution/spacing. The levels are connected via parent-child relationships to form a unified
 * data structure for multi-scale computations suitable for adaptive mesh refinement (AMR).
 *
 * ## Construction Algorithm (6 Phases):
 *
 * ### Phase 1: Parameter Validation and Setup
 * - Validates backend compatibility (single GPU only)
 * - Ensures octree structure (2x2x2 refinement)
 * - Validates refinement factor consistency
 * - Calculates top-level spacing and domain compatibility
 * - Initializes bitmask storage for all levels
 *
 * ### Phase 2: Bitmask Creation for Active Cells
 * - Uses user-provided lambda functions to determine cell activity
 * - Implements two-pass algorithm per block:
 *   1. Check which voxels should be active based on lambda
 *   2. If block contains active voxels, activate all voxels (block filling)
 * - Propagates activation hierarchically to parent levels
 *
 * ### Phase 3: Overlap Culling (Optional)
 * - Removes coarse voxels that are fully covered by fine voxels
 * - Only removes voxels where ALL 26 neighbors are also refined
 * - Prevents redundant multi-level representation
 *
 * ### Phase 4: Strong Balancing (Optional)
 * - Ensures smooth resolution transitions (max 1 level difference)
 * - Iteratively activates intermediate levels until constraint satisfied
 * - Prevents numerical issues from sudden resolution jumps
 *
 * ### Phase 5: Internal Block Sparse Grid Creation
 * - Creates independent block sparse grid for each level
 * - Each grid has its own memory allocation and indexing
 * - Enables parallel operations within and across levels
 *
 * ### Phase 6: Hierarchical Linking
 * - Establishes parent-child relationships between levels
 * - Creates device-accessible memory sets for GPU operations
 * - Enables seamless traversal between resolution levels
 *
 *
 * ## Performance Considerations:
 * - Parallel construction using OpenMP for Phase 2-4
 * - Memory-efficient bitmask representation
 * - Cache-friendly sequential access patterns
 *
 * @param backend Computational backend (CPU/CUDA) - must be single GPU
 * @param domainSize 3D dimensions of the computational domain (must be >= top-level spacing)
 * @param activeCellLambda Functions (one per level) determining which cells are active at each resolution
 * @param stencil Computational stencil pattern (parameter preserved for interface compatibility)
 * @param descriptor Refinement structure defining depth, refinement factors, and spacing per level
 * @param isStrongBalanced Enable strong balancing for smooth resolution transitions (recommended: true)
 * @param isCullOverlaps Enable overlap culling to remove redundant coarse cells (recommended: true)
 * @param spacingData Physical spacing information (parameter preserved for interface compatibility)
 * @param origin Physical origin of the domain (parameter preserved for interface compatibility)
 *
 * @throws NeonException if backend has multiple GPUs
 * @throws NeonException if refinement factors don't match block size requirements
 * @throws NeonException if domain size is incompatible with top-level spacing
 * @throws NeonException if refinement factors decrease from fine to coarse levels
 *
 * @note Timer integration provides detailed profiling of each construction phase
 */
template <typename SBlock>
mGrid<SBlock>::mGrid(
    const Neon::Backend&                                    backend,
    const Neon::int32_3d&                                   domainSize,
    std::vector<std::function<bool(const Neon::index_3d&)>> activeCellLambda,
    [[maybe_unused]] const Neon::domain::Stencil&           stencil,
    const Descriptor                                        descriptor,
    bool                                                    isStrongBalanced,
    bool                                                    isCullOverlaps,
    [[maybe_unused]] const double_3d&                       spacingData,
    [[maybe_unused]] const double_3d&                       origin)
{
    Neon::TimerManagerSec timeTracker("mGrid");
    timeTracker.start_with_info("initialization");


    // Debug code for process identification - commented out

    // ==============================================
    // PHASE 1: Parameter Validation and Setup
    // ==============================================
    if (backend.devSet().numDevs() > 1) {
        NeonException exp("mGrid");
        exp << "mGrid only supported on a single GPU";
        NEON_THROW(exp);
    }

    // Ensure we're using octree structure (2x2x2 refinement)
    static_assert(SBlock::userBlockSizeX == 2 && SBlock::userBlockSizeY == 2 && SBlock::userBlockSizeZ == 2, "mGird only supports octree!");

    // Validate that descriptor refinement factors match the block size requirements
    for (int l = 0; l < descriptor.getDepth(); ++l) {
        if (descriptor.getRefFactor(l) != SBlock::userBlockSizeX ||
            descriptor.getRefFactor(l) != SBlock::userBlockSizeY ||
            descriptor.getRefFactor(l) != SBlock::userBlockSizeZ) {
            NeonException exp("mGrid");
            exp << "Mismatch between the grid descriptor and the userBlockSize";
            exp << "Level = " << l << " refinement factor = " << descriptor.getRefFactor(l) << " userBlockSize= " << SBlock::userBlockSizeX << ", " << SBlock::userBlockSizeY << ", " << SBlock::userBlockSizeZ;
            NEON_THROW(exp);
        }
    }

    // Initialize shared data structure and store configuration
    mData = std::make_shared<Data>();
    mData->backend = backend;
    mData->domainSize = domainSize;
    mData->mStrongBalanced = isStrongBalanced;
    mData->mCullOverlaps = isCullOverlaps;
    mData->mDescriptor = descriptor;

    // Calculate top-level spacing (coarsest grid resolution)
    int top_level_spacing = 1;
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        if (l > 0) {
            top_level_spacing *= mData->mDescriptor.getRefFactor(l);
            // Validate non-decreasing refinement factors from fine to coarse levels
            if (mData->mDescriptor.getRefFactor(l) < mData->mDescriptor.getRefFactor(l - 1)) {
                NeonException exp("mGrid::mGrid");
                exp << "The grid refinement factor should only go up from one level to another starting with Level 0 the leaf/finest level\n";
                exp << "Level " << l - 1 << " refinement factor= " << mData->mDescriptor.getRefFactor(l - 1) << "\n";
                exp << "Level " << l << " refinement factor= " << mData->mDescriptor.getRefFactor(l) << "\n";
                NEON_THROW(exp);
            }
        }
    }

    // Validate domain size compatibility with coarsest level spacing
    if (domainSize.x < top_level_spacing || domainSize.y < top_level_spacing || domainSize.z < top_level_spacing) {
        NeonException exp("mGrid::mGrid");
        exp << "The spacing of the top level of the multi-resolution grid is bigger than the domain size";
        exp << " This may create problems. Please consider increasing the domain size or decrease the branching factor or depth of the grid\n";
        exp << "DomainSize= " << domainSize << "\n";
        exp << "Top level spacing= " << top_level_spacing << "\n";
        NEON_THROW(exp);
    }

    // Initialize block count arrays and bitmasks for each level
    mData->mTotalNumBlocks.resize(mData->mDescriptor.getDepth());
    // constexpr uint32_t MaskSize = 32;  // Size of each bitmask element (32-bit integers)
    for (int i = 0; i < mData->mDescriptor.getDepth(); ++i) {
        const int refFactor = mData->mDescriptor.getRefFactor(i);
        const int spacing = mData->mDescriptor.getSpacing(i);

        // Calculate total blocks needed per dimension
        mData->mTotalNumBlocks[i].set(NEON_DIVIDE_UP(domainSize.x, spacing),
                                      NEON_DIVIDE_UP(domainSize.y, spacing),
                                      NEON_DIVIDE_UP(domainSize.z, spacing));

        // // Create bitmask for tracking active voxels (refFactor^3 voxels per block)
        // std::vector<uint32_t> msk(NEON_DIVIDE_UP(static_cast<int64_t>(refFactor) *
        //                                              static_cast<int64_t>(refFactor) *
        //                                              static_cast<int64_t>(refFactor) *
        //                                              mData->mTotalNumBlocks[i].template rMulTyped<int64_t>(),
        //                                          static_cast<int64_t>(MaskSize)),
        //                           0);
        auto const bbox = mData->mTotalNumBlocks[i] * refFactor;
        mData->sparseLevelsBitmask.emplace_back(bbox);
    }

    // ==============================================
    // PHASE 2: Bitmask Creation for Each Resolution Level
    // ==============================================
    timeTracker.start_with_trace("Bitmask creation");

    // For each resolution level, determine which voxels are active based on the user-provided lambda functions.
    // Each level operates as an independent block sparse grid with its own resolution and spacing.
    // If a block contains any active voxels, the entire block is marked as active.
    // Parent-child relationships between levels are established to connect the stacked grids.
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        const int refFactor = mData->mDescriptor.getRefFactor(l);
        // Process all blocks at current level in parallel
        // Two-pass algorithm:
        // 1st pass: Check which voxels should be active based on lambda functions
        // 2nd pass: If block contains active voxels, activate all voxels in block (fill block)
#pragma omp parallel for collapse(3) schedule(static)
        for (size_t bzUint64 = 0; bzUint64 < static_cast<size_t>(mData->mTotalNumBlocks[l].z); bzUint64++) {
            for (size_t byUint64 = 0; byUint64 < static_cast<size_t>(mData->mTotalNumBlocks[l].y); byUint64++) {
                for (size_t bxUint64 = 0; bxUint64 < static_cast<size_t>(mData->mTotalNumBlocks[l].x); bxUint64++) {
                    int const bz = static_cast<int>(bzUint64);
                    int const by = static_cast<int>(byUint64);
                    int const bx = static_cast<int>(bxUint64);

                    // Convert block indices to base index space coordinates
                    Neon::index_3d blockOrigin = mData->mDescriptor.toBaseIndexSpace({bx, by, bz}, l + 1);

                    // PASS 1: Check if any voxel in this block should be active
                    bool containVoxels = false;
                    for (int z = 0; z < refFactor; z++) {
                        for (int y = 0; y < refFactor; y++) {
                            for (int x = 0; x < refFactor; x++) {

                                // Convert local block coordinates to global voxel coordinates
                                const Neon::int32_3d voxel = mData->mDescriptor.parentToChild(blockOrigin, l, {x, y, z});

                                if (voxel < domainSize) {
                                    // Check if voxel is already active or should be activated by lambda
                                    // if (levelBitMaskIsSet(l, {bx, by, bz}, {x, y, z})) {
                                    //     containVoxels = true;
                                    // } else {
                                        if (activeCellLambda[l](voxel)) {
                                            containVoxels = true;
#pragma omp critical
                                            {
                                                // Set the bitmask for this voxel if it is active
                                                setLevelBitMask(l, {bx, by, bz}, {x, y, z});
                                            }
                                        }
                                    //}
                                }
                            }
                        }
                    }

                    // PASS 2: If block contains any active voxels, activate all voxels in the block
                    if (containVoxels) {
                        for (int z = 0; z < refFactor; z++) {
                            for (int y = 0; y < refFactor; y++) {
                                for (int x = 0; x < refFactor; x++) {

                                    const Neon::int32_3d voxel = mData->mDescriptor.parentToChild(blockOrigin, l, {x, y, z});

                                    if (voxel < domainSize) {
#pragma omp critical
                                        {
                                            setLevelBitMask(l, {bx, by, bz}, {x, y, z});
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // HIERARCHICAL ACTIVATION: Activate parent block if this block contains voxels
                    if (containVoxels) {
                        // Propagate activation to the next coarser level
                        // This block becomes a voxel in its parent block at the next level
                        if (l < mData->mDescriptor.getDepth() - 1) {
                            // Find which parent block this block belongs to
                            Neon::int32_3d parentBlock = mData->mDescriptor.childToParent(blockOrigin, l + 1);

                            // Find local position within the parent block
                            Neon::int32_3d indexInParentBlock = mData->mDescriptor.toLocalIndex(blockOrigin, l + 1);
#pragma omp critical
                            {
                                // Activate the corresponding voxel in the parent block
                                setLevelBitMask(l + 1, parentBlock, indexInParentBlock);
                            }
                        }
                    }
                }
            }
        }
    }
    timeTracker.stop_with_trace("Bitmask creation");

    // ==============================================
    // PHASE 3: Overlap Culling (Optional)
    // ==============================================
    timeTracker.start_with_trace("Cull Overlaps");

    /**
     * ## Overlap Culling Algorithm
     *
     * Overlap culling eliminates redundant coarse voxels that are fully covered by fine voxels,
     * reducing memory usage and preventing duplicate computations across resolution levels.
     *
     * ### Culling Criteria:
     * A coarse voxel is removed if and only if:
     * 1. **It is refined**: Has active children at the next finer level
     * 2. **All neighbors are refined**: ALL 26 neighbors (3x3x3 - center) are also refined
     *
     *
     * ### Conservative Approach:
     * Only removes voxels when ALL neighbors are refined, ensuring:
     * - Interface cells between levels are always preserved
     * - Interpolation and restriction operations remain well-defined
     * - No orphaned fine cells (every fine cell has a coarse parent available)
     *
     */
    if (mData->mCullOverlaps) {

        /**
         * @brief Check if a voxel at a given level is refined (has active children).
         *
         * Determines whether a coarse voxel has any active children at the next finer level.
         * This is used to identify candidates for overlap culling.
         *
         * @param level Resolution level of the voxel (must be > 0)
         * @param voxel 3D coordinates of the voxel to check
         * @return true if voxel has any active children, false otherwise
         *
         * ### Algorithm:
         * 1. Maps the coarse voxel to its fine-level child region
         * 2. Iterates through all possible child positions (refFactor^3)
         * 3. Checks if any child is active in the bitmask
         * 4. Returns true on first active child found (early termination)
         */
        auto isRefined = [&](int level, const Neon::int32_3d& voxel) {
            if (level < 1) {
                NeonException exp("mGrid::mGrid");
                exp << "isRefined only work with level > 0. Input level =" << level;
                NEON_THROW(exp);
            }

            // Check if this voxel has any active children at the finer level (level-1)
            const int refFactor = mData->mDescriptor.getRefFactor(level);
            const int spacing = mData->mDescriptor.getSpacing(level - 1);

            // Iterate through all possible child positions within this voxel
            for (int z = 0; z < refFactor; z++) {
                for (int y = 0; y < refFactor; y++) {
                    for (int x = 0; x < refFactor; x++) {

                        const Neon::int32_3d childLocal(x, y, z);

                        // Calculate global position of this child
                        const Neon::int32_3d child = mData->mDescriptor.neighbourBlock(voxel, level - 1, childLocal);

                        if (child < domainSize) {
                            // Find which block this child belongs to at the finer level
                            const Neon::int32_3d childBlock(child.x / spacing,
                                                            child.y / spacing,
                                                            child.z / spacing);
                            // If any child is active, this voxel is refined
                            if (levelBitMaskIsSet(level - 1, childBlock, childLocal)) {
                                return true;
                            }
                        }
                    }
                }
            }

            return false;
        };

        // Process levels from coarsest to finest (skip level 0 which has no children)
        for (int l = mData->mDescriptor.getDepth() - 1; l > 0; --l) {
            const int refFactor = mData->mDescriptor.getRefFactor(l);

            // Process all blocks at this level in parallel
#pragma omp parallel for collapse(3) schedule(static)
            for (size_t bzUint64 = 0; bzUint64 < static_cast<size_t>(mData->mTotalNumBlocks[l].z); bzUint64++) {
                for (size_t byUint64 = 0; byUint64 < static_cast<size_t>(mData->mTotalNumBlocks[l].y); byUint64++) {
                    for (size_t bxUint64 = 0; bxUint64 < static_cast<size_t>(mData->mTotalNumBlocks[l].x); bxUint64++) {
                        int const bz = static_cast<int>(bzUint64);
                        int const by = static_cast<int>(byUint64);
                        int const bx = static_cast<int>(bxUint64);

                        const Neon::index_3d blockOrigin = mData->mDescriptor.toBaseIndexSpace({bx, by, bz}, l + 1);

                        // Check each voxel in this block for potential culling
                        for (int z = 0; z < refFactor; z++) {
                            for (int y = 0; y < refFactor; y++) {
                                for (int x = 0; x < refFactor; x++) {

                                    // Only consider active voxels
                                    if (levelBitMaskIsSet(l, {bx, by, bz}, {x, y, z})) {

                                        const Neon::int32_3d voxel = mData->mDescriptor.parentToChild(blockOrigin, l, {x, y, z});

                                        // Only cull if voxel is within domain and is refined
                                        if (voxel < domainSize) {
                                            if (isRefined(l, voxel)) {

                                                // Check all 26 neighbors in 3D
                                                // Deactivate only if ALL neighbors are also refined
                                                bool deactivate = true;
                                                for (int k = -1; k < 2; k++) {
                                                    for (int j = -1; j < 2; j++) {
                                                        for (int i = -1; i < 2; i++) {
                                                            if (i == 0 && j == 0 && k == 0) {
                                                                continue;  // Skip center voxel
                                                            }

                                                            const Neon::int32_3d neighborVoxel = mData->mDescriptor.neighbourBlock(voxel, l, {i, j, k});

                                                            // Check if neighbor is within domain bounds
                                                            if (neighborVoxel.x >= 0 && neighborVoxel.y >= 0 && neighborVoxel.z >= 0 && neighborVoxel < domainSize) {
                                                                // If any neighbor is not refined, don't deactivate
                                                                if (!isRefined(l, neighborVoxel)) {
                                                                    deactivate = false;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }

                                                // Deactivate voxel if it and all neighbors are refined
                                                if (deactivate) {
#pragma omp critical
                                                    {
                                                        clearLevelBitMask(l, {bx, by, bz}, {x, y, z});
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    timeTracker.stop_with_trace("Cull Overlaps");

    // ==============================================
    // PHASE 4: Strong Balancing Between Resolution Levels (Optional)
    // ==============================================
    timeTracker.start_with_trace("Strong Balance");

    /**
     * ## Strong Balancing Algorithm
     *
     * Strong balancing enforces smooth resolution transitions by ensuring that adjacent cells
     * differ by at most one resolution level. This constraint is critical for:
     * - Numerical stability in multi-scale computations
     * - Well-conditioned interpolation/restriction operators
     * - Preventing artificial discontinuities at level interfaces
     *
     * ### Balancing Constraint:
     * For any active voxel at level L, all 26 neighbors must exist at levels:
     * - L (same level) - always acceptable
     * - L+1 (one level coarser) - acceptable
     * - L-1 (one level finer) - acceptable
     * - L+2 or higher (multiple levels coarser) - **VIOLATION** → activate level L+1
     *
     * ### Iterative Algorithm:
     * 1. **Scan Phase**: Check all active voxels for constraint violations
     * 2. **Activation Phase**: Activate intermediate levels to fix violations
     * 3. **Repeat**: Continue until no new activations occur (convergence)
     *
     * ### Algorithm Properties:
     * - **Convergence**: Guaranteed in finite iterations (typically 2-3)
     * - **Consistency**: Preserves user-specified finest-level refinement
     *
     *
     * ### Numerical Benefits:
     * - Smooth interpolation between levels (no high-frequency artifacts)
     * - Stable restriction/prolongation operators
     * - Predictable convergence rates for iterative solvers
     * - Reduced aliasing in multi-scale computations
     */
    if (mData->mStrongBalanced) {
        // Iteratively refine grid until strong balance condition is satisfied
        bool again = true;
        while (again) {
            again = false;

            // Check all levels for balance violations
            for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
                const int refFactor = mData->mDescriptor.getRefFactor(l);
                const int childSpacing = mData->mDescriptor.getSpacing(l - 1);

#pragma omp parallel for collapse(3) schedule(static)
                for (size_t bzUint64 = 0; bzUint64 < static_cast<size_t>(mData->mTotalNumBlocks[l].z); bzUint64++) {
                    for (size_t byUint64 = 0; byUint64 < static_cast<size_t>(mData->mTotalNumBlocks[l].y); byUint64++) {
                        for (size_t bxUint64 = 0; bxUint64 < static_cast<size_t>(mData->mTotalNumBlocks[l].x); bxUint64++) {
                            int const bz = static_cast<int>(bzUint64);
                            int const by = static_cast<int>(byUint64);
                            int const bx = static_cast<int>(bxUint64);

                            // Check each voxel in the current block
                            for (int z = 0; z < refFactor; z++) {
                                for (int y = 0; y < refFactor; y++) {
                                    for (int x = 0; x < refFactor; x++) {

                                        // Only process active voxels
                                        if (levelBitMaskIsSet(l, {bx, by, bz}, {x, y, z})) {

                                            // Calculate global position of this voxel
                                            const Neon::int32_3d voxel(bx * refFactor + x,
                                                                       by * refFactor + y,
                                                                       bz * refFactor + z);

                                            // Check all 26 neighbors for balance violations
                                            for (int k = -1; k < 2; k++) {
                                                for (int j = -1; j < 2; j++) {
                                                    for (int i = -1; i < 2; i++) {
                                                        if (i == 0 && j == 0 && k == 0) {
                                                            continue;  // Skip center voxel
                                                        }

                                                        // Calculate neighbor position
                                                        Neon::int32_3d proxyVoxel(voxel.x + i,
                                                                                  voxel.y + j,
                                                                                  voxel.z + k);

                                                        // Convert to physical coordinates
                                                        const Neon::int32_3d proxyVoxelLocation(proxyVoxel.x * childSpacing,
                                                                                                proxyVoxel.y * childSpacing,
                                                                                                proxyVoxel.z * childSpacing);

                                                        if (proxyVoxelLocation < domainSize && proxyVoxelLocation >= 0) {

                                                            // Store previous level information for potential activation
                                                            Neon::int32_3d prv_nVoxelBlockOrigin(0), prv_nVoxelLocalID(0);

                                                            // Search through all coarser levels to find neighbor
                                                            for (int l_n = l; l_n < mData->mDescriptor.getDepth(); ++l_n) {
                                                                const int l_n_ref_factor = mData->mDescriptor.getRefFactor(l_n);

                                                                // Calculate block and local indices at level l_n
                                                                const Neon::int32_3d nVoxelBlockOrigin(proxyVoxel.x / l_n_ref_factor,
                                                                                                       proxyVoxel.y / l_n_ref_factor,
                                                                                                       proxyVoxel.z / l_n_ref_factor);

                                                                const Neon::int32_3d nVoxelLocalID(proxyVoxel.x % l_n_ref_factor,
                                                                                                   proxyVoxel.y % l_n_ref_factor,
                                                                                                   proxyVoxel.z % l_n_ref_factor);

                                                                // Check if neighbor exists at this level
                                                                if (levelBitMaskIsSet(l_n, nVoxelBlockOrigin, nVoxelLocalID)) {

                                                                    // Strong balance: neighbors can differ by at most 1 level
                                                                    if (l_n == l || l_n == l + 1) {
                                                                        break;  // Balance satisfied
                                                                    } else {
#pragma omp critical
                                                                        {
                                                                            // Balance violation: activate intermediate level
                                                                            setLevelBitMask(l_n - 1, prv_nVoxelBlockOrigin, prv_nVoxelLocalID);
                                                                            again = true;  // Need another iteration
                                                                        }
                                                                    }
                                                                }

                                                                // Move to next coarser level
                                                                proxyVoxel = nVoxelBlockOrigin;

                                                                // Cache current level info for potential activation
                                                                prv_nVoxelBlockOrigin = nVoxelBlockOrigin;
                                                                prv_nVoxelLocalID = nVoxelLocalID;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    timeTracker.stop_with_trace("Strong Balance");
    // ==============================================
    // PHASE 5: Internal Block Sparse Grid Creation
    // ==============================================
    timeTracker.start_with_trace("bGrid initialization");

    // Create individual block sparse grids for each resolution level
    // Each grid operates independently but they are connected via parent-child relationships
    mData->grids.resize(mData->mDescriptor.getDepth());
    int const num_levels = mData->mDescriptor.getDepth();
    for (int l = 0; l < num_levels; ++l) {

        int blockSize = mData->mDescriptor.getRefFactor(l);
        int voxelSpacing = mData->mDescriptor.getSpacing(l - 1);

        Neon::int32_3d levelDomainSize(mData->mTotalNumBlocks[l].x * blockSize,
                                       mData->mTotalNumBlocks[l].y * blockSize,
                                       mData->mTotalNumBlocks[l].z * blockSize);
        mData->grids[l] =
            InternalGrid(
                backend,
                levelDomainSize,
                [&](Neon::int32_3d id) {
                    if (id < domainSize) {
                        Neon::index_3d blockID = mData->mDescriptor.childToParent(id, l);
                        Neon::index_3d localID = mData->mDescriptor.toLocalIndex(id, l);
                        return levelBitMaskIsSet(l, blockID, localID);
                    } else {
                        return false;
                    }
                },
                stencil,
                voxelSpacing,
                spacingData,
                origin);
    }

    Neon::MemoryOptions memOptionsAoS(Neon::DeviceType::CPU,
                                      Neon::Allocator::MALLOC,
                                      Neon::DeviceType::CUDA,
                                      ((backend.devType() == Neon::DeviceType::CUDA) ? Neon::Allocator::CUDA_MEM_DEVICE : Neon::Allocator::NULL_MEM),
                                      Neon::MemoryLayout::arrayOfStructs);
    Neon::MemoryOptions memOptionsSoA(Neon::DeviceType::CPU,
                                      Neon::Allocator::MALLOC,
                                      Neon::DeviceType::CUDA,
                                      ((backend.devType() == Neon::DeviceType::CUDA) ? Neon::Allocator::CUDA_MEM_DEVICE : Neon::Allocator::NULL_MEM),
                                      Neon::MemoryLayout::structOfArrays);

    timeTracker.stop_with_trace("bGrid initialization");

    // ==============================================
    // PHASE 6: Hierarchical Linking Between Resolution Levels
    // ==============================================
    timeTracker.start_with_trace("Linking bGrids");

    /**
     * ## Hierarchical Linking Algorithm
     *
     * Establishes bidirectional parent-child relationships between resolution levels,
     * creating a unified hierarchical data structure that enables seamless traversal
     * and communication between different resolution grids.
     *
     * ### Data Structures Created:
     * - **Parent Block IDs**: For each block, stores reference to parent at coarser level
     * - **Child Block IDs**: For each voxel, stores references to children at finer level
     * - **Refinement Factors**: Device-accessible array of refinement factors per level
     * - **Spacing Arrays**: Device-accessible array of spacing values per level
     *
     * ### Memory Layout Optimization:
     * - **Structure of Arrays (SoA)**: Child references for cache-efficient access
     * - **Array of Structures (AoS)**: Parent references for spatial locality
     * - **Device Memory**: All data structures are GPU-accessible
     * - **Host-Device Sync**: Automatic synchronization for CUDA backends
     *
     * ### Linking Algorithm:
     * 1. **Memory Allocation**: Size calculation based on active block counts
     * 2. **Parent Mapping**: Each block finds its parent in the coarser level
     * 3. **Child Mapping**: Each voxel maps to its children in the finer level
     * 4. **Invalid References**: Use max value to indicate non-existent relationships
     * 5. **GPU Transfer**: Upload all relationships to device memory
     *
     * ### Performance Considerations:
     * - **Block-aligned Access**: Memory layout optimized for block-wise operations
     * - **Coalesced Reads**: GPU memory access patterns optimized for throughput
     * - **Minimal Indirection**: Direct indexing without pointer chasing
     * - **Cache Efficiency**: Related data stored contiguously
     *
     * ### Use Cases Enabled:
     * - **Interpolation**: Fine→Coarse data transfer using parent relationships
     * - **Restriction**: Coarse→Fine data transfer using child relationships
     * - **Ghost Exchange**: Communication between adjacent levels
     * - **Adaptive Refinement**: Dynamic level activation/deactivation
     * - **Parallel Traversal**: Concurrent operations across levels
     */

    // Initialize parent block ID storage for each level (except the coarsest)
    // Each fine block stores a reference to its parent block at the next coarser level
    mData->mParentBlockID.resize(mData->mDescriptor.getDepth() - 1);
    for (int l = 0; l < mData->mDescriptor.getDepth() - 1; ++l) {
        mData->mParentBlockID[l] = backend.devSet().template newMemSet<typename Idx::DataBlockIdx>({Neon::DataUse::HOST_DEVICE},
                                                                                                   1,
                                                                                                   memOptionsAoS,
                                                                                                   mData->grids[l].getBlockViewGrid().getNumActiveCellsPerPartition());
    }

    // Initialize child block ID storage to link each level to its finer resolution children
    std::vector<Neon::set::DataSet<uint64_t>> childAllocSize(mData->mDescriptor.getDepth());
    for (int l = 0; l < descriptor.getDepth(); ++l) {
        childAllocSize[l] = backend.devSet().template newDataSet<uint64_t>();
        for (int64_t i = 0; i < childAllocSize[l].size(); ++i) {
            if (l > 0) {
                // Calculate memory needed for child references at this level
                childAllocSize[l][i] = mData->grids[l].helpGetPartitioner1D().getStandardCount()[0] *
                                       SBlock::memBlockSizeX * SBlock::memBlockSizeY * SBlock::memBlockSizeZ;
            } else {
                // Level 0 (finest) has no children, so minimal allocation
                childAllocSize[l][i] = 1;
            }
        }
    }

    mData->mChildBlockID.resize(mData->mDescriptor.getDepth());
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        mData->mChildBlockID[l] = backend.devSet().template newMemSet<typename Idx::DataBlockIdx>({Neon::DataUse::HOST_DEVICE},
                                                                                                  1,
                                                                                                  memOptionsSoA,
                                                                                                  childAllocSize[l]);
        for (int32_t c = 0; c < childAllocSize[l].cardinality(); ++c) {
            SetIdx devID(c);
            for (size_t i = 0; i < childAllocSize[l][c]; ++i) {
                mData->mChildBlockID[l].eRef(devID, i) = std::numeric_limits<typename Idx::DataBlockIdx>::max();
            }
        }
    }


    // descriptor
    auto descriptorSize = backend.devSet().template newDataSet<uint64_t>();
    for (int32_t c = 0; c < descriptorSize.cardinality(); ++c) {
        descriptorSize[c] = mData->mDescriptor.getDepth();
    }
    mData->mRefFactors = backend.devSet().template newMemSet<int>({Neon::DataUse::HOST_DEVICE},
                                                                  1,
                                                                  memOptionsAoS,
                                                                  descriptorSize);
    for (int32_t c = 0; c < mData->mRefFactors.cardinality(); ++c) {
        SetIdx devID(c);
        for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
            mData->mRefFactors.eRef(c, l) = mData->mDescriptor.getRefFactor(l);
        }
    }

    mData->mSpacing = backend.devSet().template newMemSet<int>({Neon::DataUse::HOST_DEVICE},
                                                               1,
                                                               memOptionsAoS,
                                                               descriptorSize);
    for (int32_t c = 0; c < mData->mSpacing.cardinality(); ++c) {
        SetIdx devID(c);
        for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
            mData->mSpacing.eRef(c, l) = mData->mDescriptor.getSpacing(l);
        }
    }

    SetIdx devID(0);

    // Populate the actual parent-child relationships between resolution levels
    // This creates the connectivity that allows seamless traversal between stacked grids
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        const int refFactor = mData->mDescriptor.getRefFactor(l);
        const int spacing = mData->mDescriptor.getSpacing(l);
        int       voxelSpacing = mData->mDescriptor.getSpacing(l - 1);


        mData->grids[l].helpGetPartitioner1D().forEachPar(devID, [&](int blockIdx, Neon::index_3d memBlockOrigin, auto /*byPartition*/) {
            Neon::index_3d blockOrigin = memBlockOrigin;
            blockOrigin.x *= SBlock::memBlockSizeX * voxelSpacing;
            blockOrigin.y *= SBlock::memBlockSizeY * voxelSpacing;
            blockOrigin.z *= SBlock::memBlockSizeZ * voxelSpacing;

            if (l > 0) {
                // loop over user block
                for (uint32_t k = 0; k < SBlock::userBlockPerMemBlockZ; ++k) {
                    for (uint32_t j = 0; j < SBlock::userBlockPerMemBlockY; ++j) {
                        for (uint32_t i = 0; i < SBlock::userBlockPerMemBlockX; ++i) {

                            const Neon::index_3d userBlockOrigin(i * SBlock::userBlockSizeX * voxelSpacing + blockOrigin.x,
                                                                 j * SBlock::userBlockSizeY * voxelSpacing + blockOrigin.y,
                                                                 k * SBlock::userBlockSizeZ * voxelSpacing + blockOrigin.z);

                            const Neon::int32_3d block3DIndex = userBlockOrigin / spacing;

                            // loop over each voxel in the user block
                            for (int32_t z = 0; z < refFactor; z++) {
                                for (int32_t y = 0; y < refFactor; y++) {
                                    for (int32_t x = 0; x < refFactor; x++) {
                                        Neon::index_3d localChild(x, y, z);

                                        const Neon::index_3d voxelGlobalID(x * voxelSpacing + userBlockOrigin.x,
                                                                           y * voxelSpacing + userBlockOrigin.y,
                                                                           z * voxelSpacing + userBlockOrigin.z);
                                        if (voxelGlobalID.x >= domainSize.x || voxelGlobalID.y >= domainSize.y || voxelGlobalID.z >= domainSize.z) {
                                            continue;
                                        }


                                        // set child ID
                                        if (levelBitMaskIsSet(l, block3DIndex, localChild)) {

                                            Neon::index_3d childBlock3DIndex(block3DIndex.x * refFactor + x,
                                                                             block3DIndex.y * refFactor + y,
                                                                             block3DIndex.z * refFactor + z);

                                            bool childExist = false;
                                            for (int32_t cz = 0; cz < refFactor; cz++) {
                                                for (int32_t cy = 0; cy < refFactor; cy++) {
                                                    for (int32_t cx = 0; cx < refFactor; cx++) {
                                                        Neon::index_3d cc(cx, cy, cz);
                                                        childExist = childExist || levelBitMaskIsSet(l - 1, childBlock3DIndex, cc);
                                                    }
                                                }
                                            }

                                            uint32_t pitch = blockIdx * SBlock::memBlockSizeX * SBlock::memBlockSizeY * SBlock::memBlockSizeZ +
                                                             (i * SBlock::userBlockSizeX + x) +
                                                             (j * SBlock::userBlockSizeY + y) * SBlock::memBlockSizeY +
                                                             (k * SBlock::userBlockSizeZ + z) * SBlock::memBlockSizeY * SBlock::memBlockSizeZ;

                                            if (childExist) {

                                                Neon::index_3d childId = mData->mDescriptor.parentToChild(userBlockOrigin, l, localChild);

                                                auto [setIdx, childBlockID] = mData->grids[l - 1].helpGetSetIdxAndGridIdx(childId);

                                                if (setIdx.idx() == -1) {
                                                    NeonException exp("mGrid::mGrid");
                                                    exp << "Can not find the child";
                                                    NEON_THROW(exp);
                                                }
                                                mData->mChildBlockID[l].eRef(devID, pitch) = childBlockID.getDataBlockIdx();
                                            } else {
                                                mData->mChildBlockID[l].eRef(devID, pitch) = std::numeric_limits<typename Idx::DataBlockIdx>::max();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // set the parent info
            if (l < mData->mDescriptor.getDepth() - 1) {
                Neon::index_3d parentOrigin = mData->mDescriptor.toBaseIndexSpace(mData->mDescriptor.childToParent(blockOrigin, l + 1), l + 2);

                auto [setIdx, parentID] = mData->grids[l + 1].helpGetSetIdxAndGridIdx(parentOrigin);

                if (setIdx.idx() == -1) {
                    mData->mParentBlockID[l].eRef(devID, blockIdx) = std::numeric_limits<typename Idx::DataBlockIdx>::max();
                } else {
                    mData->mParentBlockID[l].eRef(devID, blockIdx) = parentID.getDataBlockIdx();
                }
            }
        });
    }

    if (backend.devType() == Neon::DeviceType::CUDA) {
        for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
            if (l < mData->mDescriptor.getDepth() - 1) {
                mData->mParentBlockID[l].updateDeviceData(backend, 0);
            }
            if (l > 0) {
                mData->mChildBlockID[l].updateDeviceData(backend, 0);
            }
        }
        mData->mRefFactors.updateDeviceData(backend, 0);
        mData->mSpacing.updateDeviceData(backend, 0);
    }
    timeTracker.stop_with_trace("Linking bGrids");
    timeTracker.stop("initialization");

    timeTracker.infoAllStopped("Initialization Completed");
}

/**
 * @brief Calculate bitmask index for efficient voxel status tracking.
 *
 * Computes the flat array index and bit position within the 32-bit bitmask system
 * for efficient voxel status tracking across all resolution levels. This function
 * enables O(1) voxel activation/deactivation queries.
 *
 * ### Algorithm:
 * 1. Convert 3D block+local coordinates to 1D flattened index
 * 2. Divide by 32 to get array index (each element holds 32 bits)
 * 3. Use modulo 32 to get bit position within the element
 *
 * ### Memory Layout:
 * - Each bitmask element stores 32 voxel states as individual bits
 * - Achieves 32x compression compared to boolean arrays
 * - Cache-efficient sequential access patterns
 *
 * @param l Resolution level (0 = finest, depth-1 = coarsest)
 * @param blockID 3D block coordinates within the level
 * @param localChild Local position of voxel within the block [0, refFactor)³
 * @return Pair of (array index, bit position) for the bitmask
 *
 * @note Bit position is in range [0, 31] for standard 32-bit integers
 * @note Array index scales with total domain size and refinement factor
 */
template <typename SBlock>
auto mGrid<SBlock>::levelBitMaskIndex(int l, const Neon::index_3d& blockID, const Neon::index_3d& localChild) const -> std::pair<int64_t, int>
{
    constexpr uint32_t MaskSize = 32;
    const int64_t      index1D = mData->mDescriptor.flattened1DIndex(blockID, l, mData->mTotalNumBlocks[l], localChild);
    const int64_t      mask = index1D / MaskSize;
    const int          bitPosition = static_cast<int32_t>(index1D % MaskSize);
    return std::pair<int64_t, int>(mask, bitPosition);
};

/**
 * @brief Check if a voxel is active at a specific resolution level.
 *
 * Performs O(1) lookup in the compressed bitmask to determine voxel activation status.
 * This is the primary query operation for the multi-resolution grid system.
 *
 *
 * @param l Resolution level to query (0 = finest, depth-1 = coarsest)
 * @param blockID 3D block coordinates within the level
 * @param localChild Local position of voxel within the block [0, refFactor)³
 * @return true if voxel is active/refined, false if inactive
 *
 * @note Used extensively during grid traversal and computation phases
 * @see setLevelBitMask() for voxel activation
 * @see clearLevelBitMask() for voxel deactivation
 */
template <typename SBlock>
auto mGrid<SBlock>::levelBitMaskIsSet(int l, const Neon::index_3d& blockID, const Neon::index_3d& localChild) const -> bool
{
    auto bxyz =
        blockID * 2 +
        localChild;
    // auto id = levelBitMaskIndex(l, blockID, localChild);
    // return mData->denseLevelsBitmask.at(l).at(id.first) & (1 << id.second);
    return mData->sparseLevelsBitmask.at(l).isActivePoint(bxyz);
};


/**
 * @brief Activate a voxel at a specific resolution level.
 *
 * Sets the corresponding bit in the bitmask to mark the voxel as active.
 *
 * @param l Resolution level
 * @param blockID 3D block coordinates within the level
 * @param localChild Local position of voxel within the block
 */
template <typename SBlock>
auto mGrid<SBlock>::    setLevelBitMask(int l, const Neon::index_3d& blockID, const Neon::index_3d& localChild) -> void
{
    auto const bxyz =
        blockID * 2 +
        localChild;
    return mData->sparseLevelsBitmask.at(l).template activatePoint<false>(bxyz);
};

/**
 * @brief Deactivate a voxel at a specific resolution level.
 *
 * Clears the corresponding bit in the bitmask to mark the voxel as inactive.
 *
 * @param l Resolution level
 * @param blockID 3D block coordinates within the level
 * @param localChild Local position of voxel within the block
 */
template <typename SBlock>
auto mGrid<SBlock>::clearLevelBitMask(int l, const Neon::index_3d& blockID, const Neon::index_3d& localChild) -> void
{
    auto const bxyz =
        blockID * 2 +
        localChild;
    return mData->sparseLevelsBitmask.at(l).template removePoint<false>(bxyz);
};

/**
 * @brief Check if a given index is inside the domain at a specific resolution level.
 *
 * @param idx 3D index to check
 * @param level Resolution level to query
 * @return true if index is within domain bounds, false otherwise
 */
template <typename SBlock>
auto mGrid<SBlock>::isInsideDomain(const Neon::index_3d& idx, int level) const -> bool
{
    return mData->grids[level].isInsideDomain(idx);
}


/**
 * @brief Access the internal grid at a specific resolution level (non-const).
 *
 * @param level Resolution level to access
 * @return Reference to the internal block sparse grid at the specified level
 */
template <typename SBlock>
auto mGrid<SBlock>::operator()(int level) -> InternalGrid&
{
    return mData->grids[level];
}

/**
 * @brief Access the internal grid at a specific resolution level (const).
 *
 * @param level Resolution level to access
 * @return Const reference to the internal block sparse grid at the specified level
 */
template <typename SBlock>
auto mGrid<SBlock>::operator()(int level) const -> const InternalGrid&
{
    return mData->grids[level];
}


/**
 * @brief Get the origin block 3D index for a given voxel position at a specific resolution level.
 *
 * Computes the block origin by rounding down the voxel coordinates to the nearest multiple
 * of the level's spacing, effectively finding which block contains the given voxel. This is
 * a fundamental operation for mapping between global voxel coordinates and block-local coordinates.
 *
 * ### Algorithm:
 * For each dimension: `block_origin = (voxel_coord / spacing) * spacing`
 * This rounds down to the nearest multiple of spacing, giving the block's corner coordinate.
 *
 * ### Use Cases:
 * - **Voxel-to-Block Mapping**: Find which block contains a specific voxel
 * - **Interpolation Setup**: Identify blocks involved in multi-level operations
 * - **Neighbor Finding**: Locate adjacent blocks for stencil operations
 * - **Boundary Handling**: Determine block boundaries for domain decomposition
 *
 * ### Example:
 * ```
 * Level spacing = 4, voxel at (7, 9, 11)
 * Block origin = (4, 8, 8)  // Rounded down to nearest multiple of 4
 * Local coords = (3, 1, 3)  // Relative to block origin
 * ```
 *
 * @param idx Voxel position in 3D global coordinate space
 * @param level Resolution level for spacing calculation (0 = finest spacing)
 * @return 3D coordinates of the block origin that contains the voxel
 *
 * @note Block origins are always aligned to spacing boundaries
 * @note Returns coordinates in the same global coordinate system as input
 */
template <typename SBlock>
auto mGrid<SBlock>::getOriginBlock3DIndex(const Neon::int32_3d idx, int level) const -> Neon::int32_3d
{
    // Round down to nearest multiple of spacing
    auto roundDownToNearestMultiple = [](int32_t n, int32_t m) -> int32_t {
        return (n / m) * m;
    };

    Neon::int32_3d block_origin(roundDownToNearestMultiple(idx.x, mData->mDescriptor.getSpacing(level)),
                                roundDownToNearestMultiple(idx.y, mData->mDescriptor.getSpacing(level)),
                                roundDownToNearestMultiple(idx.z, mData->mDescriptor.getSpacing(level)));
    return block_origin;
}

/**
 * @brief Set the reduction engine for computational operations.
 *
 * Currently only supports CUB engine for reduction operations on multi-resolution grids.
 *
 * @param eng Reduction engine to use (must be CUB)
 * @throws NeonException if engine is not CUB
 */
template <typename SBlock>
auto mGrid<SBlock>::setReduceEngine(Neon::sys::patterns::Engine eng) -> void
{
    if (eng != Neon::sys::patterns::Engine::CUB) {
        NeonException exp("mGrid::setReduceEngine");
        exp << "mGrid only work on CUB engine for reduction";
        NEON_THROW(exp);
    }
}

/**
 * @brief Get parent block IDs for a specific resolution level.
 *
 * Returns memory set containing parent block identifiers for connecting to the next
 * coarser resolution level. The coarsest level has no parent.
 *
 * @param level Resolution level to query
 * @return Reference to memory set of parent block IDs
 * @throws NeonException if level has no parent (coarsest level)
 */
template <typename SBlock>
auto mGrid<SBlock>::getParentsBlockID(int level) const -> Neon::set::MemSet<uint32_t>&
{
    if (level >= mData->mDescriptor.getDepth() - 1) {
        NeonException exp("mGrid::getParentsBlockID");
        exp << "There is no parent for level " << level << " since the tree depth is " << mData->mDescriptor.getDepth();
        NEON_THROW(exp);
    }

    return mData->mParentBlockID[level];
}

/**
 * @brief Get child block IDs for a specific resolution level.
 *
 * Returns memory set containing child block identifiers for connecting to the next
 * finer resolution level. The finest level has no children.
 *
 * @param level Resolution level to query
 * @return Const reference to memory set of child block IDs
 */
template <typename SBlock>
auto mGrid<SBlock>::getChildBlockID(int level) const -> const Neon::set::MemSet<uint32_t>&
{
    return mData->mChildBlockID[level];
}

/**
 * @brief Get refinement factors for all resolution levels.
 *
 * @return Const reference to memory set containing refinement factors per level
 */
template <typename SBlock>
auto mGrid<SBlock>::getRefFactors() const -> const Neon::set::MemSet<int>&
{
    return mData->mRefFactors;
}

/**
 * @brief Get spacing values for all resolution levels.
 *
 * @return Const reference to memory set containing spacing values per level
 */
template <typename SBlock>
auto mGrid<SBlock>::getLevelSpacing() const -> const Neon::set::MemSet<int>&
{
    return mData->mSpacing;
}

/**
 * @brief Get the total number of resolution levels in the multi-resolution grid.
 *
 * @return Number of resolution levels
 */
template <typename SBlock>
auto mGrid<SBlock>::getLevelCount() const -> uint32_t
{
    return mData->grids.size();
}

/**
 * @brief Get the grid descriptor containing refinement structure information.
 *
 * @return Const reference to the descriptor object
 */
template <typename SBlock>
auto mGrid<SBlock>::getDescriptor() const -> const Descriptor&
{
    return mData->mDescriptor;
}

/**
 * @brief Get the 3D dimensions for a specific resolution level.
 *
 * @param level Resolution level to query
 * @return 3D dimensions of the level
 */
template <typename SBlock>
auto mGrid<SBlock>::getDimension(int level) const -> const Neon::index_3d
{
    return mData->mTotalNumBlocks[level] * mData->mDescriptor.getRefFactor(level);
}

/**
 * @brief Get the overall domain dimensions.
 *
 * @return 3D dimensions of the computational domain
 */
template <typename SBlock>
auto mGrid<SBlock>::getDimension() const -> const Neon::index_3d
{
    return mData->domainSize;
}

/**
 * @brief Get the number of blocks for a specific resolution level.
 *
 * @param level Resolution level to query
 * @return 3D block count for the level
 */
template <typename SBlock>
auto mGrid<SBlock>::getNumBlocks(int level) const -> const Neon::index_3d&
{
    return mData->mTotalNumBlocks[level];
}

/**
 * @brief Get the computational backend (const version).
 *
 * @return Const reference to the backend object
 */
template <typename SBlock>
auto mGrid<SBlock>::getBackend() const -> const Backend&
{
    return mData->backend;
}

/**
 * @brief Get the computational backend (non-const version).
 *
 * @return Reference to the backend object
 */
template <typename SBlock>
auto mGrid<SBlock>::getBackend() -> Backend&
{
    return mData->backend;
}
/**
 * @brief Generate a comprehensive string representation of the multi-resolution grid.
 *
 * Creates a detailed string representation including hierarchical information about all
 * resolution levels and their internal block sparse grids. This is primarily used for
 * debugging, logging, and grid structure analysis.
 *
 * ### Output Format:
 * ```
 * mGrid (level count: N)
 * ---
 * [Level 0 block sparse grid details]
 * ---
 * [Level 1 block sparse grid details]
 * ...
 * ```
 *
 * ### Information Included:
 * - Total number of resolution levels
 * - Per-level block sparse grid statistics
 * - Memory usage and active cell counts
 * - Spacing and refinement factor information
 * - Backend and device configuration
 *
 * ### Use Cases:
 * - **Debugging**: Verify grid construction correctness
 * - **Performance Analysis**: Analyze memory usage and load balancing
 * - **Logging**: Record grid configuration for reproducibility
 * - **Validation**: Compare different grid configurations
 *
 * @return Multi-line string representation of the complete grid hierarchy
 *
 * @note Output can be large for grids with many levels or large domains
 * @note Each level's block sparse grid provides its own detailed toString() output
 */
template <typename SBlock>
auto mGrid<SBlock>::toString() const -> std::string
{
    std::stringstream ss;
    ss << "mGrid (level count:" << getLevelCount() << ")";
    for (int l = 0; l < static_cast<int>(getLevelCount()); l++) {
        auto& bGrid = this->operator()(l);
        ss << "---\n";
        ss << bGrid.toString() << "\n";
    }
    return ss.str();
}


}  // namespace Neon::domain::details::mGrid

/**
 * ## Explicit Template Instantiations
 *
 * Pre-instantiate common block sizes to reduce compilation time and ensure
 * consistent behavior across different translation units.
 *
 * ### Supported Block Configurations:
 * - **8x8x8 blocks**: High memory efficiency for large-scale simulations
 * - **4x4x4 blocks**: Balanced performance for general-purpose AMR
 * - **2x2x2 blocks**: Minimal block size for fine-grained control
 *
 * All configurations use:
 * - **Memory block size**: Same as user block size for simplicity
 * - **Refinement factor**: 2x2x2 (octree structure)
 * - **Contiguous memory**: true for optimal cache performance
 *
 * @note Additional block sizes can be instantiated by including the header
 * @note All instantiations support the same multi-resolution grid interface
 */
template class Neon::domain::details::mGrid::mGrid<Neon::domain::details::StaticBlock<8, 8, 8, 2, 2, 2, true>>;
template class Neon::domain::details::mGrid::mGrid<Neon::domain::details::StaticBlock<4, 4, 4, 2, 2, 2, true>>;
template class Neon::domain::details::mGrid::mGrid<Neon::domain::details::StaticBlock<2, 2, 2, 2, 2, 2, true>>;
