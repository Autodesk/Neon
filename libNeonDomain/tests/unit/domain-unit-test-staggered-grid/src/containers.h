#include "Neon/domain/dGrid.h"
#include "Neon/domain/internal/experimantal/staggeredGrid/StaggeredGrid.h"

#define TEST_TYPE float


template <typename StaggeredGrid, typename T>
struct Containers
{
    using Self = Containers<StaggeredGrid, T>;

    using Type = T;
    using NodeField = typename StaggeredGrid::template NodeField<T, 1>;
    using VoxelField = typename StaggeredGrid::template VoxelField<T, 1>;
    using NodeField_3 = typename StaggeredGrid::template NodeField<T, 3>;
    using VoxelField_3 = typename StaggeredGrid::template VoxelField<T, 3>;
    static auto addConstOnNodes(NodeField  field,
                                const Type alpha)
        -> Neon::set::Container;

    static auto addConstOnVoxels(VoxelField field,
                                 const Type alpha)
        -> Neon::set::Container;

    static auto sumNodesOnVoxels(Self::VoxelField fieldVox,
                                 Self::NodeField  fieldNode) -> Neon::set::Container;

    static auto sumNodesOnVoxels(Self::VoxelField_3 fieldVox,
                                 Self::NodeField_3  fieldNode) -> Neon::set::Container;
};


extern template struct Containers<Neon::domain::internal::experimental::staggeredGrid::StaggeredGrid<Neon::domain::dGrid>, TEST_TYPE>;