#include "Neon/domain/tools/Geometries.h"
#include "containers.h"

namespace tools {
template <typename StaggeredGrid, typename T>
auto Containers<StaggeredGrid, T>::resetValue(Self::NodeField  field,
                                              const Self::Type alpha) -> Neon::set::Container
{
    return field.getGrid().getContainerOnNodes(
        "addConstOnNodes",
        [&](Neon::set::Loader& loader) {
            auto& out = loader.load(field);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Self::NodeField::Node& e) mutable {
                for (int i = 0; i < out.cardinality(); i++) {
                    out(e, i) = alpha;
                }
            };
        });
}


template <typename StaggeredGrid, typename T>
auto Containers<StaggeredGrid, T>::resetValue(Self::VoxelField field,
                                              const Self::Type alpha) -> Neon::set::Container
{
    return field.getGrid().getContainerOnVoxels(
        "addConstOnVoxels",
        [&](Neon::set::Loader& loader) {
            auto& out = loader.load(field);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Self::VoxelField::Voxel& e) mutable {
                for (int i = 0; i < out.cardinality(); i++) {
                    out(e, i) = alpha;
                }
            };
        });
}

template <typename StaggeredGrid, typename T>
auto Containers<StaggeredGrid, T>::sumNodesOnVoxels(Self::VoxelField&      densityField,
                                                    const Self::NodeField& temperatureField)
    -> Neon::set::Container
{
    return densityField.getGrid().getContainerOnVoxels(
        "sumNodesOnVoxels",
        // Neon Loading Lambda
        [&](Neon::set::Loader& loader) {
            auto&       density = loader.load(densityField);
            const auto& temperature = loader.load(temperatureField, Neon::Compute::STENCIL);

            // Neon Compute Lambda
            return [=] NEON_CUDA_HOST_DEVICE(const typename Self::VoxelField::Voxel& voxHandle) mutable {
                Type          sum = 0;
                constexpr int componentId = 0;

                // We visit all the neighbour nodes around voxHandle.
                // Relative discrete offers are used to identify the neighbour node.
                // As by construction all nodes of a voxel are active, we don't have to do any extra check.
                sum += temperature.template getNghNodeValue<1, 1, 1>(voxHandle, componentId);
                sum += temperature.template getNghNodeValue<1, 1, -1>(voxHandle, componentId);
                sum += temperature.template getNghNodeValue<1, -1, 1>(voxHandle, componentId);
                sum += temperature.template getNghNodeValue<1, -1, -1>(voxHandle, componentId);
                sum += temperature.template getNghNodeValue<-1, 1, 1>(voxHandle, componentId);
                sum += temperature.template getNghNodeValue<-1, 1, -1>(voxHandle, componentId);
                sum += temperature.template getNghNodeValue<-1, -1, 1>(voxHandle, componentId);
                sum += temperature.template getNghNodeValue<-1, -1, -1>(voxHandle, componentId);

                // Storing the final result in the target voxel.
                density(voxHandle, 0) = sum;
            };
        });
}


template <typename StaggeredGrid, typename T>
auto Containers<StaggeredGrid, T>::sumVoxelsOnNodesAndDivideBy8(Self::NodeField&        temperatureField,
                                                                const Self::VoxelField& densityField) -> Neon::set::Container
{
    return temperatureField.getGrid().getContainerOnNodes(
        "sumVoxelsOnNodesAndDivideBy8",
        // Neon Loading Lambda
        [&](Neon::set::Loader& loader) {
            const auto& density = loader.load(densityField, Neon::Compute::STENCIL);
            auto&       temperature = loader.load(temperatureField);

            auto nodeSpaceDim = temperatureField.getGrid().getDimension();

            // Neon Compute Lambda
            return [=] NEON_CUDA_HOST_DEVICE(const typename Self::NodeField::Node& nodeHandle) mutable {
                Type sum = 0;

                constexpr int componentId = 0;
                constexpr int returnValueIfVoxelIsNotActive = 0;

                // We visit all the neighbouring voxels around nodeHandle.
                // Relative discrete offers are used to identify the neighbour node.
                // Note that some neighbouring nodes may be not active.
                // Rather than explicitly checking we ask Neon to return 0 if the node is not active.
                sum += density.template getNghVoxelValue<1, 1, 1>(nodeHandle, componentId, returnValueIfVoxelIsNotActive).value;
                sum += density.template getNghVoxelValue<1, 1, -1>(nodeHandle, componentId, returnValueIfVoxelIsNotActive).value;
                sum += density.template getNghVoxelValue<1, -1, 1>(nodeHandle, componentId, returnValueIfVoxelIsNotActive).value;
                sum += density.template getNghVoxelValue<1, -1, -1>(nodeHandle, componentId, returnValueIfVoxelIsNotActive).value;
                sum += density.template getNghVoxelValue<-1, 1, 1>(nodeHandle, componentId, returnValueIfVoxelIsNotActive).value;
                sum += density.template getNghVoxelValue<-1, 1, -1>(nodeHandle, componentId, returnValueIfVoxelIsNotActive).value;
                sum += density.template getNghVoxelValue<-1, -1, 1>(nodeHandle, componentId, returnValueIfVoxelIsNotActive).value;
                sum += density.template getNghVoxelValue<-1, -1, -1>(nodeHandle, componentId, returnValueIfVoxelIsNotActive).value;

                // Storing the final result in the target node.
                temperature(nodeHandle, 0) = sum / 8;
            };
        });
}

template struct Containers<Neon::domain::details::experimental::staggeredGrid::StaggeredGrid<Neon::dGrid>, double>;
template struct Containers<Neon::domain::details::experimental::staggeredGrid::StaggeredGrid<Neon::dGrid>, float>;
}  // namespace tools