#include "Neon/domain/tools/Geometries.h"
#include "containers.h"


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
    using Type = typename Self::NodeField::Type;

    return densityField.getGrid().getContainerOnVoxels(
        "sumNodesOnVoxels",
        [&](Neon::set::Loader& loader) {
            auto&       density = loader.load(densityField);
            const auto& temperature = loader.load(temperatureField, Neon::Compute::STENCIL);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Self::VoxelField::Voxel& voxHandle) mutable {
                Type sum = 0;

#define CHECK_DIRECTION(X, Y, Z)                                                         \
    {                                                                                    \
        Type nghNodeValue = temperature.template getNghNodeValue<X, Y, Z>(voxHandle, 0); \
                                                                                         \
        sum += nghNodeValue;                                                             \
    }

                CHECK_DIRECTION(1, 1, 1);
                CHECK_DIRECTION(1, 1, -1);
                CHECK_DIRECTION(1, -1, 1);
                CHECK_DIRECTION(1, -1, -1);
                CHECK_DIRECTION(-1, 1, 1);
                CHECK_DIRECTION(-1, 1, -1);
                CHECK_DIRECTION(-1, -1, 1);
                CHECK_DIRECTION(-1, -1, -1);

                density(voxHandle, 0) = sum;
#undef CHECK_DIRECTION
            };
        });
}


template <typename StaggeredGrid, typename T>
auto Containers<StaggeredGrid, T>::sumVoxelsOnNodes(Self::NodeField&        temperatureField,
                                                    const Self::VoxelField& densityField) -> Neon::set::Container
{

    using Type = typename Self::NodeField::Type;

    return temperatureField.getGrid().getContainerOnNodes(
        "sumVoxelsOnNodes",
        [&](Neon::set::Loader& loader) {
            const auto& density = loader.load(densityField, Neon::Compute::STENCIL);
            auto&       temperature = loader.load(temperatureField);

            auto nodeSpaceDim = temperatureField.getGrid().getDimension();

            return [=] NEON_CUDA_HOST_DEVICE(const typename Self::NodeField::Node& nodeHandle) mutable {
                Type sum = 0;

#define CHECK_DIRECTION(X, Y, Z)                                                              \
    {                                                                                         \
        Type nghDensity = density.template getNghVoxelValue<X, Y, Z>(nodeHandle, 0, 0).value; \
        sum += nghDensity;                                                                    \
    }

                CHECK_DIRECTION(1, 1, 1);
                CHECK_DIRECTION(1, 1, -1);
                CHECK_DIRECTION(1, -1, 1);
                CHECK_DIRECTION(1, -1, -1);
                CHECK_DIRECTION(-1, 1, 1);
                CHECK_DIRECTION(-1, 1, -1);
                CHECK_DIRECTION(-1, -1, 1);
                CHECK_DIRECTION(-1, -1, -1);

                temperature(nodeHandle, 0) = sum;
                ;
#undef CHECK_DIRECTION
            };
        });
}
template struct Containers<Neon::domain::internal::experimental::staggeredGrid::StaggeredGrid<Neon::domain::dGrid>, double>;
template struct Containers<Neon::domain::internal::experimental::staggeredGrid::StaggeredGrid<Neon::domain::dGrid>, float>;
