#include "Neon/domain/dGrid.h"
#include "Neon/domain/internal/experimantal/staggeredGrid/StaggeredGrid.h"

template <typename StaggeredGrid, typename T>
struct Containers
{
    static constexpr int errorCode = -11;
    static constexpr int noErrorCode = 11;

    using Self = Containers<StaggeredGrid, T>;

    using Type = T;
    using NodeField = typename StaggeredGrid::template NodeField<T, 1>;
    using VoxelField = typename StaggeredGrid::template VoxelField<T, 1>;

    static auto resetValue(NodeField  field,
                           const Type alpha)
        -> Neon::set::Container;

    static auto resetValue(VoxelField field,
                           const Type alpha)
        -> Neon::set::Container;

    static auto sumNodesOnVoxels(Self::VoxelField&      fieldVox,
                                 const Self::NodeField& fieldNode)
        -> Neon::set::Container;

    static auto sumVoxelsOnNodes(Self::NodeField&        fieldNode,
                                 const Self::VoxelField& fieldVox)
        -> Neon::set::Container;
};


extern template struct Containers<Neon::domain::internal::experimental::staggeredGrid::StaggeredGrid<Neon::domain::dGrid>, double>;
extern template struct Containers<Neon::domain::internal::experimental::staggeredGrid::StaggeredGrid<Neon::domain::dGrid>, float>;
