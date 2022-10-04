#include "gtest/gtest.h"

#include "Neon/Neon.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/tools/TestData.h"
#include "RunHelper.h"

#include "Neon/domain/internal/experimantal/staggeredGrid/StaggeredGrid.h"

using namespace Neon::domain::tool::testing;
static const std::string testFilePrefix("domainUt_Swap");


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
void StaggeredGrid(TestData<G, T, C>& data)
{
    auto& grid = data.getGrid();
    //
    Neon::int32_3d                                                      dims{10, 10, 10};
    std::vector<Neon::domain::Stencil>                                  empty;
    Neon::domain::internal::experimental::staggeredGrid::StaggeredGrid<G> FEA(
        data.getBackend(),
        dims,
        [](const Neon::index_3d&) -> bool {
            return true;
        });

    //
    //    //auto density = FEA.template newElementField<double, 1>("Density", 1, 0);
    auto temperature = FEA.template newNodeField<double, 1>("Temperature", 1, 0);
    temperature.forEachActiveCell([](const Neon::index_3d& idx,
                                     const int& /*cardinality*/,
                                     double& value) {
        if (((idx.x + idx.y + idx.z) % 2) == 0) {
            value = 10;
        } else {
            value = 0;
        }
    });


    const std::string appName(testFilePrefix + "_" + grid.getImplementationName());

    temperature.ioToVtk(appName + "-temperature", "temperature");

    // const Type alpha = 11;
    // NEON_INFO(temperature.toString());
    data.resetValuesToLinear(1, 100);
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

TEST(Swap, dGrid)
{
    Neon::init();
    int nGpus = getNGpus();
    using Grid = Neon::domain::dGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("sGrid", StaggeredGrid<Grid, Type, 0>, nGpus, 1);
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
