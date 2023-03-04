#include "gtest/gtest.h"

#include "Neon/Neon.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/tools/TestData.h"
#include "RunHelper.h"
#include "containers.h"

#include "Neon/domain/StaggeredGrid.h"
#include "TestInformation.h"

#define EXECUTE_IO_TO_VTK 0

using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C>
void StaggeredGrid_Map(TestData<G, T, C>& data)
{
    //
    Neon::int32_3d voxDim = [&] {
        auto dim = data.getGrid().getDimension();
        return dim;
    }();

    std::vector<Neon::domain::Stencil> empty;
    using FeaGrid = Neon::domain::experimental::StaggeredGrid<G>;
    FeaGrid FEA(
        data.getBackend(),
        voxDim,
        [](const Neon::index_3d&) -> bool {
            return true;
        });

    auto temperature = FEA.template newNodeField<TEST_TYPE, 1>("Temperature", 1, 0);
    temperature.forEachActiveCell([](const Neon::index_3d& idx,
                                     const int& /*cardinality*/,
                                     TEST_TYPE& value) {
        value = TEST_TYPE(idx.rMul());
    });


    auto density = FEA.template newVoxelField<TEST_TYPE, 1>("Density", 1, 0);
    density.forEachActiveCell([](const Neon::index_3d& idx,
                                 const int& /*cardinality*/,
                                 TEST_TYPE& value) {
        value = TEST_TYPE(idx.rSum());
    });


    temperature.updateCompute(Neon::Backend::mainStreamIdx);
    density.updateCompute(Neon::Backend::mainStreamIdx);

    const std::string appName = TestInformation::fullName("_map_", data.getGrid().getImplementationName());

    if constexpr (EXECUTE_IO_TO_VTK == 1) {
        temperature.ioToVtk(appName + "-temperature_asVoxels", "temperature", false, Neon::IoFileType::ASCII, false);
        density.ioToVtk(appName + "-density_0000", "density");
    }

    const TEST_TYPE valueToAddOnNodes = 50;
    const TEST_TYPE valueToAddOnVoxels = 33;

    Containers<FeaGrid, TEST_TYPE>::addConstOnNodes(temperature, valueToAddOnNodes).run(Neon::Backend::mainStreamIdx);
    Containers<FeaGrid, TEST_TYPE>::addConstOnVoxels(density, valueToAddOnVoxels).run(Neon::Backend::mainStreamIdx);

    temperature.updateIO(Neon::Backend::mainStreamIdx);
    density.updateIO(Neon::Backend::mainStreamIdx);
    data.getBackend().sync(Neon::Backend::mainStreamIdx);

    if constexpr (EXECUTE_IO_TO_VTK == 1) {
        temperature.ioToVtk(appName + "-temperature_0001", "temperature", false, Neon::IoFileType::ASCII);
        density.ioToVtk(appName + "-density_0001", "density");
    }

    bool errorDetected = false;

    temperature.forEachActiveCell([valueToAddOnNodes, &errorDetected](const Neon::index_3d& idx,
                                                               const int& /*cardinality*/,
                                                               TEST_TYPE& value) {
        auto target = TEST_TYPE(idx.rMul()) + valueToAddOnNodes;
        if (target != value) {
            errorDetected = true;
        }
    });

    density.forEachActiveCell([valueToAddOnVoxels, &errorDetected](const Neon::index_3d& idx,
                                                           const int& /*cardinality*/,
                                                           TEST_TYPE& value) {
        auto target = TEST_TYPE(idx.rSum()) + valueToAddOnVoxels;
        if (target != value) {
            errorDetected = true;
        }
    });
    ASSERT_TRUE(!errorDetected);
}

template <typename G, typename T, int C>
void StaggeredGrid_VoxToNodes(TestData<G, T, C>& data)
{
    //
    Neon::int32_3d voxDim = [&] {
        auto dim = data.getGrid().getDimension();
        return dim;
    }();

    std::vector<Neon::domain::Stencil> empty;
    using FeaGrid = Neon::domain::details::experimental::staggeredGrid::StaggeredGrid<G>;
    FeaGrid FEA(
        data.getBackend(),
        voxDim,
        [](const Neon::index_3d&) -> bool {
            return true;
        });

    auto nodeIDX = FEA.template newNodeField<TEST_TYPE, 3>("nodeIdx", 3, 0);
    nodeIDX.forEachActiveCell([](const Neon::index_3d& idx,
                                 const int&            cardinality,
                                 TEST_TYPE&            value) {
        value = static_cast<TEST_TYPE>(idx.v[cardinality]);
    });

    auto voxelIDX = FEA.template newVoxelField<TEST_TYPE, 3>("voxelIdx", 3, 0);
    voxelIDX.forEachActiveCell([](const Neon::index_3d& idx,
                                  const int&            cardinality,
                                  TEST_TYPE&            value) {
        value = static_cast<TEST_TYPE>(idx.v[cardinality]);
    });

    auto errorFlagField = FEA.template newVoxelField<TEST_TYPE, 1>("ErrorFlag", 1, 0);


    nodeIDX.updateCompute(Neon::Backend::mainStreamIdx);
    voxelIDX.updateCompute(Neon::Backend::mainStreamIdx);


    Containers<FeaGrid, TEST_TYPE>::sumNodesOnVoxels(voxelIDX,
                                                     nodeIDX,
                                                     errorFlagField)
        .run(Neon::Backend::mainStreamIdx);
    errorFlagField.updateIO(Neon::Backend::mainStreamIdx);
    data.getBackend().sync(Neon::Backend::mainStreamIdx);

    bool errorDetected = false;

    nodeIDX.forEachActiveCell([&](const Neon::index_3d& /*idx*/,
                                  const int& /*cardinality*/,
                                  TEST_TYPE& value) {
        if (value != 0) {
            if (value == Containers<FeaGrid, TEST_TYPE>::errorCode) {
                errorDetected = true;
                /// std::cout << "Error detected at " << idx << std::endl;
            }
        }
    });

    ASSERT_TRUE(!errorDetected);
}


template <typename G, typename T, int C>
void StaggeredGrid_NodeToVoxels(TestData<G, T, C>& data)
{
    //
    Neon::int32_3d                     dims{7, 10, 15};
    std::vector<Neon::domain::Stencil> empty;
    using FeaGrid = Neon::domain::details::experimental::staggeredGrid::StaggeredGrid<G>;
    FeaGrid FEA(
        data.getBackend(),
        dims,
        [](const Neon::index_3d&) -> bool {
            return true;
        });

    auto nodeIDX = FEA.template newNodeField<TEST_TYPE, 3>("nodeIdx", 3, 0);
    nodeIDX.forEachActiveCell([](const Neon::index_3d& idx,
                                 const int&            cardinality,
                                 TEST_TYPE&            value) {
        value = static_cast<TEST_TYPE>(idx.v[cardinality]);
    });

    auto voxelIDX = FEA.template newVoxelField<TEST_TYPE, 3>("voxelIdx", 3, 0);
    voxelIDX.forEachActiveCell([](const Neon::index_3d& idx,
                                  const int&            cardinality,
                                  TEST_TYPE&            value) {
        value = static_cast<TEST_TYPE>(idx.v[cardinality]);
    });

    auto errorFlagField = FEA.template newNodeField<TEST_TYPE, 1>("ErrorFlag", 1, 0);


    nodeIDX.updateCompute(Neon::Backend::mainStreamIdx);
    voxelIDX.updateCompute(Neon::Backend::mainStreamIdx);

    //    const std::string appName(testFilePrefix + "_VoxToNodes_" + grid.getImplementationName());
    //
    //    nodeIDX.ioToVtk(appName + "-nodeIDX_0000", "density");
    //    voxelIDX.ioToVtk(appName + "-voxelIDX_0000", "density");


    Containers<FeaGrid, TEST_TYPE>::sumVoxelsOnNodes(nodeIDX,
                                                     voxelIDX,
                                                     errorFlagField,
                                                     Neon::domain::tool::Geometry::FullDomain)
        .run(Neon::Backend::mainStreamIdx);
    errorFlagField.updateIO(Neon::Backend::mainStreamIdx);
    data.getBackend().sync(Neon::Backend::mainStreamIdx);

    bool errorDetected = false;

    nodeIDX.forEachActiveCell([&](const Neon::index_3d& /*idx*/,
                                  const int& /*cardinality*/,
                                  TEST_TYPE& value) {
        if (value != 0) {
            if (value == Containers<FeaGrid, TEST_TYPE>::errorCode) {
                errorDetected = true;
            }
        }
    });

    ASSERT_TRUE(!errorDetected);
}
namespace {
int getNGpus()
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        int maxGPUs = Neon::set::DevSet::maxSet().setCardinality();
        if (maxGPUs > 1) {
            return maxGPUs;
        } else {
            return 3;
        }
    } else {
        return 0;
    }
}
}  // namespace

TEST(DISABLED_Map, dGrid)
{
    Neon::init();
    int nGpus = getNGpus();
    using Grid = Neon::dGrid;
    using Type = TEST_TYPE;
    runAllTestConfiguration<Grid, Type, 0>("staggeredGrid", StaggeredGrid_Map<Grid, Type, 0>, nGpus, 1);
}

TEST(DISABLED_CodeVoxToNodes, dGrid)
{
    Neon::init();
    int nGpus = getNGpus();
    using Grid = Neon::dGrid;
    using Type = TEST_TYPE;
    runAllTestConfiguration<Grid, Type, 0>("staggeredGrid", StaggeredGrid_VoxToNodes<Grid, Type, 0>, nGpus, 1);
}

TEST(DISABLED_NodeToVoxels, dGrid)
{
    Neon::init();
    int nGpus = getNGpus();
    using Grid = Neon::dGrid;
    using Type = TEST_TYPE;
    runAllTestConfiguration<Grid, Type, 0>("staggeredGrid", StaggeredGrid_NodeToVoxels<Grid, Type, 0>, nGpus, 1);
}

//TEST(Map, eGrid)
//{
//    Neon::init();
//    int nGpus = getNGpus();
//    using Grid = Neon::domain::eGrid;
//    using Type = TEST_TYPE;
//    runAllTestConfiguration<Grid, Type, 0>("staggereeGrid", StaggereeGrid_Map<Grid, Type, 0>, nGpus, 1);
//}
//
//TEST(VoxToNodes, eGrid)
//{
//    Neon::init();
//    int nGpus = getNGpus();
//    using Grid = Neon::domain::eGrid;
//    using Type = TEST_TYPE;
//    runAllTestConfiguration<Grid, Type, 0>("staggereeGrid", StaggereeGrid_VoxToNodes<Grid, Type, 0>, nGpus, 1);
//}
//
//TEST(NodeToVoxels, eGrid)
//{
//    Neon::init();
//    int nGpus = getNGpus();
//    using Grid = Neon::domain::eGrid;
//    using Type = TEST_TYPE;
//    runAllTestConfiguration<Grid, Type, 0>("staggereeGrid", StaggereeGrid_NodeToVoxels<Grid, Type, 0>, nGpus, 1);
//}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
