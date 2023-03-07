#pragma once

#include "Neon/domain/aGrid.h"
#include "Neon/domain/tools/partitioning/Cassifications.h"
#include "Neon/domain/tools/partitioning/SpanClassifier.h"
#include "Neon/domain/tools/partitioning/SpanDecomposition.h"
#include "Neon/domain/tools/partitioning/SpanLayout.h"

namespace Neon::domain::tool {

class Partitioner1D
{
   public:
    Partitioner1D() = default;

    template <typename ActiveCellLambda,
              typename BcLambda>
    Partitioner1D(const Neon::Backend&    backend,
                  const ActiveCellLambda& activeCellLambda,
                  const BcLambda&         bcLambda,
                  const int&              blockSize,
                  const Neon::int32_3d&   domainSize,
                  const int&              discreteVoxelSpacing = 1)
    {
        mBlockSize = blockSize;
        mDiscreteVoxelSpacing = discreteVoxelSpacing;

        Neon::int32_3d block3DSpan(NEON_DIVIDE_UP(domainSize.x, blockSize),
                                   NEON_DIVIDE_UP(domainSize.y, blockSize),
                                   NEON_DIVIDE_UP(domainSize.z, blockSize));

        std::vector<int> nBlockProjectedToZ(block3DSpan.z);

        auto constexpr block3dIdxToBlockOrigin = [&](Neon::int32_3d const& block3dIdx) {
            Neon::int32_3d blockOrigin(block3dIdx.x * blockSize * discreteVoxelSpacing,
                                       block3dIdx.y * blockSize * discreteVoxelSpacing,
                                       block3dIdx.z * blockSize * discreteVoxelSpacing);
            return blockOrigin;
        };

        auto constexpr getVoxelAbsolute3DIdx = [&](Neon::int32_3d const& blockOrigin,
                                                   Neon::int32_3d const& voxelRelative3DIdx) {
            const Neon::int32_3d id(blockOrigin.x + voxelRelative3DIdx.x * discreteVoxelSpacing,
                                    blockOrigin.y + voxelRelative3DIdx.y * discreteVoxelSpacing,
                                    blockOrigin.z + voxelRelative3DIdx.z * discreteVoxelSpacing);
            return id;
        };

        mSpanPartitioner = partitioning::SpanDecomposition(
            backend,
            activeCellLambda,
            block3dIdxToBlockOrigin,
            getVoxelAbsolute3DIdx,
            block3DSpan,
            blockSize,
            domainSize,
            discreteVoxelSpacing);

        mSpanClassifier = partitioning::SpanClassifier(
            backend,
            activeCellLambda,
            bcLambda,
            block3dIdxToBlockOrigin,
            getVoxelAbsolute3DIdx,
            block3DSpan,
            blockSize,
            domainSize,
            discreteVoxelSpacing,
            mSpanPartitioner);

        mPartitionSpan = partitioning::SpanLayout(
            backend,
            mSpanPartitioner,
            mSpanClassifier);
    }

    auto getSpanClassifier();

    auto getSpanLayout();

   private:
    int mBlockSize = 0;
    int mDiscreteVoxelSpacing = 0;

    partitioning::SpanDecomposition mSpanPartitioner;
    partitioning::SpanClassifier    mSpanClassifier;
    partitioning::SpanLayout        mPartitionSpan;

    Neon::aGrid mTopology;
};

}  // namespace Neon::domain::tools