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
                                                    Self::NodeField_3  fieldNode,
                                                    Self::VoxelField   errorFlagField) -> Neon::set::Container
{
    return fieldVox.getGrid().getContainerOnVoxels(
        "MAP-on-nodes",
        [&](Neon::set::Loader& loader) {
            auto& vox = loader.load(fieldVox);
            auto& errorFlag = loader.load(errorFlagField);

            const auto& nodes = loader.load(fieldNode);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Self::VoxelField::Voxel& voxHandle) mutable {
                TEST_TYPE voxId[3] = {vox(voxHandle, 0),
                                      vox(voxHandle, 1),
                                      vox(voxHandle, 2)};
                errorFlag(voxHandle, 0) = 1;

#define CHECK_DIRECTION(X, Y, Z)                                                          \
    {                                                                                     \
        TEST_TYPE nIx[3] = {nodes.template getNghNodeValue<X, Y, Z>(voxHandle, 0, 1000),     \
                            nodes.template getNghNodeValue<X, Y, Z>(voxHandle, 1, 1000),     \
                            nodes.template getNghNodeValue<X, Y, Z>(voxHandle, 2, 1000)};    \
                                                                                          \
        TEST_TYPE reference[3] = {X == -1 ? voxId[0] : voxId[0] + X,                      \
                                  Y == -1 ? voxId[1] : voxId[1] + Y,                      \
                                  Z == -1 ? voxId[2] : voxId[2] + Z};                     \
                                                                                          \
        if (nIx[0] != reference[0] || nIx[1] != reference[1] || nIx[2] != reference[2]) { \
            errorFlag(voxHandle, 0) = errorCode;                                          \
        } else {                                                                          \
            errorFlag(voxHandle, 0) = noErrorCode;                                        \
        }                                                                                 \
    }

                CHECK_DIRECTION(1, 1, 1);
                CHECK_DIRECTION(1, 1, -1);
                CHECK_DIRECTION(1, -1, 1);
                CHECK_DIRECTION(1, -1, -1);
                CHECK_DIRECTION(-1, 1, 1);
                CHECK_DIRECTION(-1, 1, -1);
                CHECK_DIRECTION(-1, -1, 1);
                CHECK_DIRECTION(-1, -1, -1);

#undef CHECK_DIRECTION
            };
        });
}


//template <typename StaggeredGrid, typename T>
//auto Containers<StaggeredGrid, T>::sumVoxelsOnNodes(Self::NodeField_3  fieldNode,
//                                                    Self::VoxelField_3 fieldVox,
//                                                    Self::NodeField    errorFlagField) -> Neon::set::Container
//{
//    return fieldNode.getGrid().getContainerOnNodes(
//        "MAP-on-nodes",
//        [&](Neon::set::Loader& loader) {
//            auto& vox = loader.load(fieldVox);
//            auto& errorFlag = loader.load(errorFlagField);
//
//            const auto& nodes = loader.load(fieldNode);
//
//            return [=] NEON_CUDA_HOST_DEVICE(const typename Self::NodeField::Node& nodeHandle) mutable {
//                TEST_TYPE nodeId[3] = {nodes(nodeHandle, 0),
//                                       nodes(nodeHandle, 1),
//                                       nodes(nodeHandle, 2)};
//                errorFlag(nodeHandle, 0) = 1;
//
//#define CHECK_DIRECTION(X, Y, Z)                                                          \
//    {                                                                                     \
//        TEST_TYPE nIx[3] = {vox.template getNghNodeValue<X, Y, Z>(nodeHandle, 0, 1000),      \
//                            vox.template getNghNodeValue<X, Y, Z>(nodeHandle, 1, 1000),      \
//                            vox.template getNghNodeValue<X, Y, Z>(nodeHandle, 2, 1000)};     \
//                                                                                          \
//        TEST_TYPE reference[3] = {X == +1 ? nodeId[0] : nodeId[0] - X,                    \
//                                  Y == +1 ? nodeId[1] : nodeId[1] - Y,                    \
//                                  Z == +1 ? nodeId[2] : nodeId[2] - Z};                   \
//                                                                                          \
//        if (nIx[0] != reference[0] || nIx[1] != reference[1] || nIx[2] != reference[2]) { \
//            errorFlag(nodeHandle, 0) = errorCode;                                         \
//        } else {                                                                          \
//            errorFlag(nodeHandle, 0) = noErrorCode;                                       \
//        }                                                                                 \
//    }
//
//                CHECK_DIRECTION(1, 1, 1);
//                CHECK_DIRECTION(1, 1, -1);
//                CHECK_DIRECTION(1, -1, 1);
//                CHECK_DIRECTION(1, -1, -1);
//                CHECK_DIRECTION(-1, 1, 1);
//                CHECK_DIRECTION(-1, 1, -1);
//                CHECK_DIRECTION(-1, -1, 1);
//                CHECK_DIRECTION(-1, -1, -1);
//            };
//        });
//}
template struct Containers<Neon::domain::internal::experimental::staggeredGrid::StaggeredGrid<Neon::domain::dGrid>, TEST_TYPE>;
