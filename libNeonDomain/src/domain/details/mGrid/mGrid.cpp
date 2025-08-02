
#include "Neon/domain/details//mGrid/mGrid.h"
#include "Neon/domain/details/mGrid/mPartition.h"


namespace Neon::domain::details::mGrid {

/**
 * @brief Constructs a multi-resolution grid by stacking multiple block sparse grids.
 *
 * Creates a hierarchical multi-resolution grid system where each level represents a different
 * resolution/spacing. The levels are connected via parent-child relationships to form a unified
 * data structure for multi-scale computations.
 *
 * Construction phases:
 * 1. Parameter validation and setup
 * 2. Bitmask creation for active cells per level
 * 3. Optional overlap culling (removes redundant coarse cells)
 * 4. Optional strong balancing (ensures smooth resolution transitions)
 * 5. Internal block sparse grid creation per level
 * 6. Parent-child relationship linking between levels
 *
 * @param backend Computational backend (CPU/CUDA)
 * @param domainSize 3D dimensions of the computational domain
 * @param activeCellLambda Functions (one per level) determining which cells are active
 * @param stencil Computational stencil pattern (unused)
 * @param descriptor Refinement structure (depth, refinement factors, spacing)
 * @param isStrongBalanced Enable strong balancing for smooth resolution transitions
 * @param isCullOverlaps Enable overlap culling to remove redundant coarse cells
 * @param spacingData Physical spacing information (unused)
 * @param origin Physical origin of the domain (unused)
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
    Neon::TimerManagerSec mgridTimeTracker;
    mgridTimeTracker.start_with_info("initialization","mGrid");


    // Debug code for process identification - commented out

    // ==============================================
    // PHASE 1: Parameter Validation and Setup
    // ==============================================
    if (backend.devSet().setCardinality() > 1) {
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
    constexpr uint32_t MaskSize = 32;  // Size of each bitmask element (32-bit integers)
    for (int i = 0; i < mData->mDescriptor.getDepth(); ++i) {
        const int refFactor = mData->mDescriptor.getRefFactor(i);
        const int spacing = mData->mDescriptor.getSpacing(i);

        // Calculate total blocks needed per dimension
        mData->mTotalNumBlocks[i].set(NEON_DIVIDE_UP(domainSize.x, spacing),
                                      NEON_DIVIDE_UP(domainSize.y, spacing),
                                      NEON_DIVIDE_UP(domainSize.z, spacing));

        // Create bitmask for tracking active voxels (refFactor^3 voxels per block)
        std::vector<uint32_t> msk(NEON_DIVIDE_UP(static_cast<int64_t>(refFactor) *
                                                     static_cast<int64_t>(refFactor) *
                                                     static_cast<int64_t>(refFactor) *
                                                     mData->mTotalNumBlocks[i].template rMulTyped<int64_t>(),
                                                 static_cast<int64_t>(MaskSize)),
                                  0);
        mData->denseLevelsBitmask.push_back(msk);
    }

    // ==============================================
    // PHASE 2: Bitmask Creation for Each Resolution Level
    // ==============================================
    mgridTimeTracker.start_with_trace("Bitmask creation", "mGrid");

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
#pragma omp parallel for collapse(3)
        for (int bz = 0; bz < mData->mTotalNumBlocks[l].z; bz++) {
            for (int by = 0; by < mData->mTotalNumBlocks[l].y; by++) {
                for (int bx = 0; bx < mData->mTotalNumBlocks[l].x; bx++) {

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
                                    if (levelBitMaskIsSet(l, {bx, by, bz}, {x, y, z})) {
                                        containVoxels = true;
                                    } else {
                                        if (activeCellLambda[l](voxel)) {
                                            containVoxels = true;
                                            setLevelBitMask(l, {bx, by, bz}, {x, y, z});
                                        }
                                    }
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
                                        setLevelBitMask(l, {bx, by, bz}, {x, y, z});
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

                            // Activate the corresponding voxel in the parent block
                            setLevelBitMask(l + 1, parentBlock, indexInParentBlock);
                        }
                    }
                }
            }
        }
    }
    mgridTimeTracker.stop_with_trace("Bitmask creation", "mGrid");

    // ==============================================
    // PHASE 3: Overlap Culling (Optional)
    // ==============================================
    mgridTimeTracker.start_with_trace("Cull Overlaps", "mGrid");

    // Overlap culling removes coarse voxels that are fully covered by fine voxels
    // A coarse voxel is removed if:
    // 1. It is refined (has active children at finer level)
    // 2. ALL its neighbors at the same level are also refined
    // This ensures we don't have redundant representation at multiple levels
    if (mData->mCullOverlaps) {

        // Lambda function to check if a voxel at a given level is refined
        // (i.e., has active children at the next finer level)
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
#pragma omp parallel for collapse(3)
            for (int bz = 0; bz < mData->mTotalNumBlocks[l].z; bz++) {
                for (int by = 0; by < mData->mTotalNumBlocks[l].y; by++) {
                    for (int bx = 0; bx < mData->mTotalNumBlocks[l].x; bx++) {

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
    mgridTimeTracker.stop_with_trace("Cull Overlaps", "mGrid");

    // ==============================================
    // PHASE 4: Strong Balancing Between Resolution Levels (Optional)
    // ==============================================
    mgridTimeTracker.start_with_trace("Strong Balance", "mGrid");

    // Strong balancing ensures smooth transitions between the stacked grids by enforcing that
    // adjacent cells differ by at most one resolution level. This prevents sudden jumps in
    // resolution that could affect numerical accuracy when moving between the stacked grids.
    // The algorithm iteratively activates intermediate resolution levels until the constraint is satisfied
    if (mData->mStrongBalanced) {
        // Iteratively refine grid until strong balance condition is satisfied
        bool again = true;
        while (again) {
            again = false;

            // Check all levels for balance violations
            for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
                const int refFactor = mData->mDescriptor.getRefFactor(l);
                const int childSpacing = mData->mDescriptor.getSpacing(l - 1);

#pragma omp parallel for collapse(3)
                for (int bz = 0; bz < mData->mTotalNumBlocks[l].z; bz++) {
                    for (int by = 0; by < mData->mTotalNumBlocks[l].y; by++) {
                        for (int bx = 0; bx < mData->mTotalNumBlocks[l].x; bx++) {

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
                                                                        // Balance violation: activate intermediate level
                                                                        setLevelBitMask(l_n - 1, prv_nVoxelBlockOrigin, prv_nVoxelLocalID);
                                                                        again = true;  // Need another iteration
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
    mgridTimeTracker.stop_with_trace("Strong Balance", "mGrid");
    // ==============================================
    // PHASE 5: Internal Block Sparse Grid Creation
    // ==============================================
    mgridTimeTracker.start_with_trace("bGrid initialization", "mGrid");

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

    mgridTimeTracker.stop_with_trace("bGrid initialization", "mGrid");

    // ==============================================
    // PHASE 6: Linking Resolution Levels
    // ==============================================
    mgridTimeTracker.start_with_trace("Linking bGrids", "mGrid");

    // Establish parent-child relationships between different resolution levels
    // This creates the connections that allow traversal between the stacked grids

    // Initialize parent block ID storage for each level (except the coarsest)
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
    mgridTimeTracker.stop_with_trace("Linking bGrids", "mGrid");
    mgridTimeTracker.stop("initialization");

    mgridTimeTracker.infoAllStopped("Initialization Completed", "mGrid");
}

/**
 * @brief Calculate bitmask index for a voxel within a block at a specific resolution level.
 *
 * Computes the flat array index and bit position within the bitmask for efficient voxel
 * status tracking across all resolution levels.
 *
 * @param l Resolution level
 * @param blockID 3D block coordinates within the level
 * @param localChild Local position of voxel within the block
 * @return Pair of (array index, bit position) for the bitmask
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
 * @param l Resolution level to query
 * @param blockID 3D block coordinates within the level
 * @param localChild Local position of voxel within the block
 * @return true if voxel is active, false otherwise
 */
template <typename SBlock>
auto mGrid<SBlock>::levelBitMaskIsSet(int l, const Neon::index_3d& blockID, const Neon::index_3d& localChild) const -> bool
{
    auto id = levelBitMaskIndex(l, blockID, localChild);
    return mData->denseLevelsBitmask.at(l).at(id.first) & (1 << id.second);
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
auto mGrid<SBlock>::setLevelBitMask(int l, const Neon::index_3d& blockID, const Neon::index_3d& localChild) -> void
{
    auto id = levelBitMaskIndex(l, blockID, localChild);
    mData->denseLevelsBitmask.at(l).at(id.first) |= (1 << id.second);
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
    auto id = levelBitMaskIndex(l, blockID, localChild);
    mData->denseLevelsBitmask.at(l).at(id.first) &= ~(1 << id.second);
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
 * of the level's spacing, effectively finding which block contains the given voxel.
 *
 * @param idx Voxel position in 3D space
 * @param level Resolution level for spacing calculation
 * @return 3D coordinates of the block origin that contains the voxel
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
 * @brief Generate a string representation of the multi-resolution grid.
 *
 * Creates a detailed string representation including information about all
 * resolution levels and their internal block sparse grids.
 *
 * @return String representation of the grid structure
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

template class Neon::domain::details::mGrid::mGrid<Neon::domain::details::StaticBlock<8, 8, 8, 2, 2, 2, true>>;
template class Neon::domain::details::mGrid::mGrid<Neon::domain::details::StaticBlock<4, 4, 4, 2, 2, 2, true>>;
template class Neon::domain::details::mGrid::mGrid<Neon::domain::details::StaticBlock<2, 2, 2, 2, 2, 2, true>>;
