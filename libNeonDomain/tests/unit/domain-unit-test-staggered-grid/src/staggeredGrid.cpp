#include "gtest/gtest.h"

#include "Neon/Neon.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/tools/TestData.h"
#include "RunHelper.h"
#include "containers.h"

#include "Neon/domain/internal/experimantal/staggeredGrid/StaggeredGrid.h"

using namespace Neon::domain::tool::testing;
static const std::string testFilePrefix("domain-unit-test-staggered-grid");


// template <typename Field>
// auto map(Field&                      input_field,
//          Field&                      output_field,
//          const typename Field::Type& alpha) -> Neon::set::Container
//{
//     return input_field.getGrid().getContainer(
//         "MAP",
//         [&](Neon::set::Loader& loader) {
//             const auto& inp = loader.load(input_field);
//             auto&       out = loader.load(output_field);
//
//             return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Cell& e) mutable {
//                 for (int i = 0; i < inp.cardinality(); i++) {
//                     out(e, i) = inp(e, i) + alpha;
//                 }
//             };
//         });
// }

template <typename G, typename T, int C>
void StaggeredGrid_Map(TestData<G, T, C>& data)
{
    auto& grid = data.getGrid();
    //
    Neon::int32_3d                     dims{2, 1, 1};
    std::vector<Neon::domain::Stencil> empty;
    using FeaGrid = Neon::domain::internal::experimental::staggeredGrid::StaggeredGrid<G>;
    FeaGrid FEA(
        data.getBackend(),
        dims,
        [](const Neon::index_3d&) -> bool {
            return true;
        });

    auto temperature = FEA.template newNodeField<TEST_TYPE, 1>("Temperature", 1, 0);
    temperature.forEachActiveCell([](const Neon::index_3d& idx,
                                     const int& /*cardinality*/,
                                     TEST_TYPE& value) {
        //        if (((idx.x + idx.y + idx.z) % 2) == 0) {
        //            value = 10;
        //        } else {
        //            value = 0;
        //        }
        value = TEST_TYPE(idx.x);
    });


    auto density = FEA.template newVoxelField<TEST_TYPE, 1>("Density", 1, 0);
    density.forEachActiveCell([](const Neon::index_3d& idx,
                                 const int& /*cardinality*/,
                                 TEST_TYPE& value) {
        value = TEST_TYPE(idx.rSum());
    });


    temperature.updateCompute(Neon::Backend::mainStreamIdx);
    density.updateCompute(Neon::Backend::mainStreamIdx);
    const std::string appName(testFilePrefix + "_Map_" + grid.getImplementationName());

    temperature.ioToVtk(appName + "-temperature_0000", "temperature");
    //    temperature.ioToVtk(appName + "-temperature_asVoxels", "temperature", false, Neon::IoFileType::ASCII, false);

    density.ioToVtk(appName + "-density_0000", "density");

    //    density.ioToVtk(appName + "-density_asNodes", "density", false, Neon::IoFileType::ASCII, true);
    // Containers<FeaGrid, TEST_TYPE>::sumNodesOnVoxels(density, temperature, 30).run(Neon::Backend::mainStreamIdx);

    Containers<FeaGrid, TEST_TYPE>::addConstOnNodes(temperature, 50).run(Neon::Backend::mainStreamIdx);
    temperature.updateIO(Neon::Backend::mainStreamIdx);
    density.updateIO(Neon::Backend::mainStreamIdx);
    data.getBackend().sync(Neon::Backend::mainStreamIdx);

    temperature.ioToVtk(appName + "-temperature_0001", "temperature");
    //    temperature.ioToVtk(appName + "-temperature_asVoxels", "temperature", false, Neon::IoFileType::ASCII, false);

    density.ioToVtk(appName + "-density_0001", "density");
    // const Type alpha = 11;
    // NEON_INFO(temperature.toString());
    // data.resetValuesToLinear(1, 100);
}

template <typename G, typename T, int C>
void StaggeredGrid_VoxToNodes(TestData<G, T, C>& data)
{
    auto& grid = data.getGrid();
    //
    Neon::int32_3d                     dims{1, 2, 1};
    std::vector<Neon::domain::Stencil> empty;
    using FeaGrid = Neon::domain::internal::experimental::staggeredGrid::StaggeredGrid<G>;
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
        value = idx.v[cardinality];
    });

    auto voxelIDX = FEA.template newVoxelField<TEST_TYPE, 3>("voxelIdx", 3, 0);
    voxelIDX.forEachActiveCell([](const Neon::index_3d& idx,
                                  const int&            cardinality,
                                  TEST_TYPE&            value) {
        value = idx.v[cardinality];
    });

    nodeIDX.updateCompute(Neon::Backend::mainStreamIdx);
    voxelIDX.updateCompute(Neon::Backend::mainStreamIdx);

    const std::string appName(testFilePrefix + "_VoxToNodes_" + grid.getImplementationName());
    

    nodeIDX.ioToVtk(appName + "-nodeIDX_0000", "density");
    voxelIDX.ioToVtk(appName + "-voxelIDX_0000", "density");

    Containers<FeaGrid, TEST_TYPE>::sumNodesOnVoxels(voxelIDX, nodeIDX).run(Neon::Backend::mainStreamIdx);
    nodeIDX.updateIO(Neon::Backend::mainStreamIdx);
    voxelIDX.updateIO(Neon::Backend::mainStreamIdx);

    data.getBackend().sync(Neon::Backend::mainStreamIdx);

    nodeIDX.ioToVtk(appName + "-temperature_0001", "temperature");
    voxelIDX.ioToVtk(appName + "-density_0001", "density");
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

TEST(Map, dGrid)
{
    Neon::init();
    int nGpus = getNGpus();
    using Grid = Neon::domain::dGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("staggeredGrid", StaggeredGrid_Map<Grid, Type, 0>, nGpus, 1);
}

TEST(VoxToNodes, dGrid)
{
    Neon::init();
    int nGpus = getNGpus();
    using Grid = Neon::domain::dGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("staggeredGrid", StaggeredGrid_VoxToNodes<Grid, Type, 0>, nGpus, 1);
}
//
// TEST(Swap, eGrid)
//{
//    Neon::init();
//    int nGpus = getNGpus();
//    using Grid = Neon::domain::eGrid;
//    using Type = int32_t;
//    runAllTestConfiguration<Grid, Type, 0>("sGrid", staggeredGrid<Grid, Type, 0>, nGpus, 1);
//}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
