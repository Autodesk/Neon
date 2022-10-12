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
auto Containers<StaggeredGrid, T>::sumNodesOnVoxels(Self::VoxelField fieldVox,
                                                    Self::NodeField  fieldNode) -> Neon::set::Container
{
    return fieldVox.getGrid().getContainerOnVoxels(
        "MAP-on-nodes",
        [&](Neon::set::Loader& loader) {
        auto&       vox = loader.load(fieldVox);
        const auto& nodes = loader.load(fieldNode);

        return [=] NEON_CUDA_HOST_DEVICE(const typename Self::VoxelField::Voxel& e) mutable {
            TEST_TYPE voxId[3] = {fieldVox(e, 0), fieldVox(e, 1), fieldVox(e, 2)};

#define CHECK_DIRECTION(X, Y, Z)                                               \
    {                                                                          \
        TEST_TYPE nIx[3] = {nodes.template getNodeValue<X, Y, Z>(e, 0, 1000),  \
                            nodes.template getNodeValue<X, Y, Z>(e, 1, 1000),  \
                            nodes.template getNodeValue<X, Y, Z>(e, 2, 1000)}; \
                                                                               \
           TEST_TYPE reference[3]                ={X == -1 ? voxId[0] }                                                     \
    }                                                                          \

            partial += nodes.template getNodeValue<-1, -1, 1>(e, 0, 1000);
            partial += nodes.template getNodeValue<-1, 1, -1>(e, 0, 1000);
            partial += nodes.template getNodeValue<-1, 1, 1>(e, 0, 1000);
            //
            // partial += nodes.template getNodeValue<1, -1, -1>(e, 0, 1000);
            //                    partial += nodes.template getNodeValue<1, -1, 1>(e, 0, 1000);
            //                    partial += nodes.template getNodeValue<1, 1, -1>(e, 0, 1000);
            partial += nodes.template getNodeValue<1, 1, 1>(e, 0, 1000);

            vox(e, 0) = partial;

            //                    for (constexpr auto d : nodeDirections) {
            //                        partial += nodes<d[0], d[1], d[2]>(e, 0);
            //                    }
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
                TEST_TYPE partial = 0;
                for (int i = 0; i < vox.cardinality(); i++) {
                    //                    {
                    //                        constexpr std::array<int8_t, 3> off {-1,-1,-1};
                    //                        auto idxNode = nodes.template getNodeValue<off[0],off[1], off[2]>(e, i, 1000);
                    //                        if(idxNode !=  vox(e, i) -1){
                    //                            printf("Error %f %f ",idxNode,  vox(e, i) + off)
                    //                        }
                    //                    }
                    partial += nodes.template getNodeValue<-1, -1, 1>(e, 0, 1000);
                    partial += nodes.template getNodeValue<-1, 1, -1>(e, 0, 1000);
                    partial += nodes.template getNodeValue<-1, 1, 1>(e, 0, 1000);
                    //
                    // partial += nodes.template getNodeValue<1, -1, -1>(e, 0, 1000);
                    //                    partial += nodes.template getNodeValue<1, -1, 1>(e, 0, 1000);
                    //                    partial += nodes.template getNodeValue<1, 1, -1>(e, 0, 1000);
                    partial += nodes.template getNodeValue<1, 1, 1>(e, 0, 1000);

                    vox(e, 0) = partial;

                    //                    for (constexpr auto d : nodeDirections) {
                    //                        partial += nodes<d[0], d[1], d[2]>(e, 0);
                    //                    }
                }
            };
        });
}

template struct Containers<Neon::domain::internal::experimental::staggeredGrid::StaggeredGrid<Neon::domain::dGrid>, TEST_TYPE>;
