#include "containers.h"


template <typename StaggeredGrid, typename T>
auto Containers<StaggeredGrid, T>::addConstOnNodes(Self::NodeField  field,
                                                   const Self::Type alpha) -> Neon::set::Container
{
    return field.getGrid().getContainerOnNodes(
        "MAP-on-nodes",
        [&](Neon::set::Loader& loader) {
            auto& out = loader.load(field);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Self::NodeField::Node& e) mutable {
                for (int i = 0; i < out.cardinality(); i++) {
                    out(e, i) += alpha;
                }
            };
        });
}


template <typename StaggeredGrid, typename T>
auto Containers<StaggeredGrid, T>::addConstOnVoxels(Self::VoxelField field,
                                                    const Self::Type alpha) -> Neon::set::Container
{
    return field.getGrid().getContainerOnVoxels(
        "MAP-on-nodes",
        [&](Neon::set::Loader& loader) {
            auto& out = loader.load(field);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Self::VoxelField::Voxel& e) mutable {
                for (int i = 0; i < out.cardinality(); i++) {
                    out(e, i) += alpha;
                }
            };
        });
}

template <typename StaggeredGrid, typename T>
auto Containers<StaggeredGrid, T>::sumNodesOnVoxels(Self::VoxelField_3 fieldVox,
                                                    Self::NodeField_3  fieldNode) -> Neon::set::Container
{
    return fieldVox.getGrid().getContainerOnVoxels(
        "MAP-on-nodes",
        [&](Neon::set::Loader& loader) {
            auto&       vox = loader.load(fieldVox);
            const auto& nodes = loader.load(fieldNode);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Self::VoxelField::Voxel& e) mutable {
                TEST_TYPE voxId[3] = {vox(e, 0),
                                      vox(e, 1),
                                      vox(e, 2)};

#define CHECK_DIRECTION(X, Y, Z)                                               \
    {                                                                          \
        TEST_TYPE nIx[3] = {nodes.template getNodeValue<X, Y, Z>(e, 0, 1000),  \
                            nodes.template getNodeValue<X, Y, Z>(e, 1, 1000),  \
                            nodes.template getNodeValue<X, Y, Z>(e, 2, 1000)}; \
                                                                               \
        TEST_TYPE reference[3] = {X == -1 ? voxId[0] : voxId[0] + X,           \
                                  Y == -1 ? voxId[1] : voxId[1] + Y,           \
                                  Z == -1 ? voxId[2] : voxId[2] + Z};          \
        printf("%f %f %f vs %f %f %f ( center %f %f %f)\n",                          \
               nIx[0], nIx[1], nIx[2],                                         \
               reference[0], reference[1], reference[2],                       \
               voxId[0], voxId[1], voxId[2]);                                  \
    }

                CHECK_DIRECTION(1, 1, 1);
            };
        });
}

template struct Containers<Neon::domain::internal::experimental::staggeredGrid::StaggeredGrid<Neon::domain::dGrid>, TEST_TYPE>;
