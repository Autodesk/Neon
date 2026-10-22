#pragma once

#include <cstdint>
#include <vector>

#include "Neon/core/core.h"

#include "Neon/set/Containter.h"
#include "Neon/set/memory/memSet.h"

#include "Neon/domain/interface/GridBaseTemplate.h"

#include "Neon/domain/tools/PointHashTable.h"
#include "Neon/domain/tools/SpanTable.h"

namespace Neon::domain::tool::partitioning {

/**
 * Defines the partition of the domain by slicing along the z axe.
 * Granularity of the slicing is a block.
 */
class SpanDecomposition
{
   public:
    SpanDecomposition() = default;

    template <typename ActiveCellLambda,
              typename Block3dIdxToBlockOrigin,
              typename GetVoxelAbsolute3DIdx>
    SpanDecomposition(const Neon::Backend&           backend,
                      const ActiveCellLambda&        activeCellLambda,
                      const Block3dIdxToBlockOrigin& block3dIdxToBlockOrigin,
                      const GetVoxelAbsolute3DIdx&   getVoxelAbsolute3DIdx,
                      const Neon::int32_3d&          block3DSpan,
                      const Neon::int32_3d&          blockSize,
                      const Neon::int32_3d&          domainSize,
                      const int&                     discreteVoxelSpacing);

    auto getNumBlockPerPartition() const
        -> const Neon::set::DataSet<int64_t>&;

    auto getFirstZSliceIdx() const
        -> const Neon::set::DataSet<int32_t>&;

    auto getLastZSliceIdx() const
        -> const Neon::set::DataSet<int32_t>&;

    /**
     * Per-block activity flags over the full block3DSpan, laid out with bx varying
     * fastest. A block is flagged when it holds at least one active voxel.
     *
     * Deciding this is the only thing the decomposition sweep does, and it is exactly
     * the test SpanClassifier would otherwise run a second time over the same block
     * space. Publishing it lets the classifier skip that repeat.
     *
     * An empty vector means the mask is not available (it has been released); callers
     * must fall back to evaluating the activation lambda themselves.
     */
    auto getBlockActiveMask() const
        -> const std::vector<uint8_t>&;

    /**
     * Frees the block activity mask. For a large finest level it is tens of MB that
     * would otherwise be retained for the lifetime of the grid.
     */
    auto releaseBlockActiveMask()
        -> void;

    auto toString(Neon::Backend const&) const
        -> std::string;

   private:
    Neon::set::DataSet<int32_t> mZFirstIdx;
    Neon::set::DataSet<int32_t> mZLastIdx;
    Neon::set::DataSet<int64_t> mNumBlocks;
    std::vector<uint8_t>        mBlockActive;

    size_t mDomainBlocksCount;
};

template <typename ActiveCellLambda,
          typename Block3dIdxToBlockOrigin,
          typename GetVoxelAbsolute3DIdx>
SpanDecomposition::SpanDecomposition(const Neon::Backend&           backend,
                                     const ActiveCellLambda&        activeCellLambda,
                                     const Block3dIdxToBlockOrigin& block3dIdxToBlockOrigin,
                                     const GetVoxelAbsolute3DIdx&   getVoxelAbsolute3DIdx,
                                     const Neon::int32_3d&          block3DSpan,
                                     const Neon::int32_3d&          blockSize,
                                     const Neon::int32_3d&          domainSize,
                                     const int&                     discreteVoxelSpacing)
{
    // Computing nBlockProjectedToZ and totalBlocks
    mDomainBlocksCount = 0;
    std::vector<size_t> nBlockProjectedToZ(block3DSpan.z);

    // Record which blocks turned out to be active so that SpanClassifier does not have
    // to rediscover it. One byte per block rather than one bit, so that neighbouring
    // blocks handled by different threads never touch the same location.
    size_t const sliceStride = static_cast<size_t>(block3DSpan.x) *
                               static_cast<size_t>(block3DSpan.y);
    mBlockActive.assign(sliceStride * static_cast<size_t>(block3DSpan.z), uint8_t(0));

    for (int bz = 0; bz < block3DSpan.z; bz++) {
        size_t count_on_bz = 0;
#pragma omp parallel for reduction(+ : count_on_bz) schedule(static) collapse(2)
        for (size_t by64 = 0; by64 < static_cast<size_t>(block3DSpan.y); by64++) {
            for (size_t bx64 = 0; bx64 < static_cast<size_t>(block3DSpan.x); bx64++) {
                int const      bx = static_cast<int>(bx64);
                int const      by = static_cast<int>(by64);
                Neon::int32_3d blockOrigin = block3dIdxToBlockOrigin({bx, by, bz});
                bool           doBreak = false;
                for (int z = 0; (z < blockSize.z && !doBreak); z++) {
                    for (int y = 0; (y < blockSize.y && !doBreak); y++) {
                        for (int x = 0; (x < blockSize.x && !doBreak); x++) {

                            Neon::int32_3d const id = getVoxelAbsolute3DIdx(blockOrigin, {x, y, z});
                            if (id < domainSize * discreteVoxelSpacing) {
                                if (activeCellLambda(id)) {
                                    doBreak = true;
                                    count_on_bz++;
                                    mBlockActive[bx64 +
                                                 by64 * static_cast<size_t>(block3DSpan.x) +
                                                 static_cast<size_t>(bz) * sliceStride] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        nBlockProjectedToZ[bz] += count_on_bz;
        mDomainBlocksCount += count_on_bz;
    }

    const int64_t avgBlocksPerPartition = NEON_DIVIDE_UP(mDomainBlocksCount,
                                                         backend.devSet().numDevs());

    mZFirstIdx = backend.devSet().newDataSet<int32_t>(0);
    mZLastIdx = backend.devSet().newDataSet<int32_t>(0);
    mNumBlocks = backend.devSet().newDataSet<int64_t>(0);


    // Slicing
    backend.devSet().forEachSetIdxSeq([&](Neon::SetIdx const& idx) {
        mZFirstIdx[idx] = [&] {
            if (idx.idx() == 0)
                return 0;
            return mZLastIdx[idx - 1] + 1;
        }();
        if (idx != backend.devSet().numDevs() - 1) {
            for (int i = mZFirstIdx[idx]; i < block3DSpan.z; i++) {
                mNumBlocks[idx] += nBlockProjectedToZ[i];
                mZLastIdx[idx] = i;

                if (mNumBlocks[idx] >= avgBlocksPerPartition) {
                    break;
                }
            }
        } else {
            mZLastIdx[idx] = block3DSpan.z - 1;
            for (int i = mZFirstIdx[idx]; i <= mZLastIdx[idx]; i++) {
                mNumBlocks[idx] += nBlockProjectedToZ[i];
            }
        }
    });

    if (backend.getDeviceCount() > 1) {
        const int ndevs = backend.getDeviceCount();
        const int minSlice = 3;
        for (int i = ndevs - 1; i > -1; i--) {
            int diff = minSlice - (mZLastIdx[i] - mZFirstIdx[i] + 1);
            if (diff > 0) {
                if (i == 0) {
                    NeonException exc("SpanDecomposition");
                    exc << "Distribution error\n";
                    exc << toString(backend);
                    NEON_THROW(exc);
                }
                mZFirstIdx[i] -= diff;
                mZLastIdx[i - 1] -= diff;

                mNumBlocks[i] = 0;
                mNumBlocks[i - 1] = 0;

                for (int j = mZFirstIdx[i]; j <= mZLastIdx[i]; j++) {
                    mNumBlocks[i] += nBlockProjectedToZ[j];
                }
                for (int j = mZFirstIdx[i - 1]; j <= mZLastIdx[i - 1]; j++) {
                    mNumBlocks[i - 1] += nBlockProjectedToZ[j];
                }
            }
        }
    }
}
}  // namespace Neon::domain::tool::partitioning
