#pragma once
#include <assert.h>
#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/interface/NghInfo.h"
#include "Neon/domain/internal/experimantal/FeaGrid/FeaElement.h"
#include "Neon/domain/internal/experimantal/FeaGrid/FeaNode.h"
#include "Neon/set/DevSet.h"
#include "Neon/sys/memory/CudaIntrinsics.h"
#include "Neon/sys/memory/mem3d.h"

namespace Neon::domain::internal::experimental::FeaVoxelGrid {

template <typename BuildingBlockGridT,
          typename T_ta,
          int cardinality_ta = 0>
struct FeaNodePartition : public BuildingBlockGridT::template Partition<T_ta, cardinality_ta>
{
   public:
    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        using Partition = typename BuildingBlockGridT::template Partition<T_ta, cardinality_ta>;
    };

    using Self = FeaNodePartition<BuildingBlockGridT, T_ta, cardinality_ta>;
    using PartitionIndexSpace = typename BuildingBlockGridT::PartitionIndexSpace;
    using Node = FeaNode<BuildingBlockGridT>;
    using Element = FeaElement<BuildingBlockGridT>;
    using Type = T_ta;

   public:
    FeaNodePartition() = default;

    ~FeaNodePartition() = default;

    explicit FeaNodePartition(typename BuildingBlocks::Partition& partition)
        : BuildingBlocks::Partition(partition)
    {
    }


    NEON_CUDA_HOST_DEVICE inline auto operator()(const Self::Node& node,
                                                 int               cardinalityIdx) -> T_ta&
    {
        return BuildingBlocks::Partition::operator()(node, cardinalityIdx);
    }

    NEON_CUDA_HOST_DEVICE inline auto operator()(const Self::Node& node,
                                                 int               cardinalityIdx) const -> const T_ta&
    {
        return BuildingBlocks::Partition::operator()(node, cardinalityIdx);
    }

    template <int8_t sx,
              int8_t sy,
              int8_t sz>
    NEON_CUDA_HOST_DEVICE inline auto operator()(const Self::Element& element,
                                                 int                  cardinalityIdx) -> T_ta
    {
        return BuildingBlocks::Partition::nghVal() < sx == -1 ? 0 : sx,
               sy == -1 ? 0 : sy,
               sz == -1 ? 0 : sz > (element, cardinalityIdx);
    }

   private:
    using BuildingBlocks::Partition::operator();
    using BuildingBlocks::Partition::nghIdx;
    using BuildingBlocks::Partition::nghVal;
};
}  // namespace Neon::domain::internal::experimental::FeaVoxelGrid
