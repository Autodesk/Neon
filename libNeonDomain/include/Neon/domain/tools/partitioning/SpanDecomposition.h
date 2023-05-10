#pragma once
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
                      const int&                     blockSize,
                      const Neon::int32_3d&          domainSize,
                      const int&                     discreteVoxelSpacing);

    auto getNumBlockPerPartition() const
        -> const Neon::set::DataSet<int64_t>&;

    auto getFirstZSliceIdx() const
        -> const Neon::set::DataSet<int32_t>&;

    auto getLastZSliceIdx() const
        -> const Neon::set::DataSet<int32_t>&;

    auto toString(Neon::Backend const&) const
        -> std::string;

   private:
    Neon::set::DataSet<int32_t> mZFirstIdx;
    Neon::set::DataSet<int32_t> mZLastIdx;
    Neon::set::DataSet<int64_t> mNumBlocks;

    int64_t mDomainBlocksCount;
};

template <typename ActiveCellLambda,
          typename Block3dIdxToBlockOrigin,
          typename GetVoxelAbsolute3DIdx>
SpanDecomposition::SpanDecomposition(const Neon::Backend&           backend,
                                     const ActiveCellLambda&        activeCellLambda,
                                     const Block3dIdxToBlockOrigin& block3dIdxToBlockOrigin,
                                     const GetVoxelAbsolute3DIdx&   getVoxelAbsolute3DIdx,
                                     const Neon::int32_3d&          block3DSpan,
                                     const int&                     blockSize,
                                     const Neon::int32_3d&          domainSize,
                                     const int&                     discreteVoxelSpacing)
{
    // Computing nBlockProjectedToZ and totalBlocks
    mDomainBlocksCount = 0;
    std::vector<int> nBlockProjectedToZ(block3DSpan.z);

    for (int bz = 0; bz < block3DSpan.z; bz++) {
        nBlockProjectedToZ[bz] = 0;

        for (int by = 0; by < block3DSpan.y; by++) {
            for (int bx = 0; bx < block3DSpan.x; bx++) {

                Neon::int32_3d blockOrigin = block3dIdxToBlockOrigin({bx, by, bz});
                bool           doBreak = false;
                for (int z = 0; (z < blockSize && !doBreak); z++) {
                    for (int y = 0; (y < blockSize && !doBreak); y++) {
                        for (int x = 0; (x < blockSize && !doBreak); x++) {

                            const Neon::int32_3d id = getVoxelAbsolute3DIdx(blockOrigin, {x, y, z});
                            if (id < domainSize * discreteVoxelSpacing) {
                                if (activeCellLambda(id)) {
                                    doBreak = true;
                                    nBlockProjectedToZ[bz]++;
                                    mDomainBlocksCount++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    const int64_t avgBlocksPerPartition = NEON_DIVIDE_UP(mDomainBlocksCount,
                                                         backend.devSet().setCardinality());

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
        if (idx != backend.devSet().setCardinality() - 1) {
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

   if(backend.getDeviceCount()>1) {
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
