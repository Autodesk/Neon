#include "gtest/gtest.h"

#include "Neon/Neon.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/tools/TestData.h"
#include "RunHelper.h"

#include "Neon/domain/internal/experimantal/FeaGrid/FeaVoxelGrid.h"

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
void FeaVoxelGrid(TestData<G, T, C>& data)
{
    auto& grid = data.getGrid();
    //
    Neon::domain::internal::experimental::FeaVoxelGrid::FeaVoxelGrid<G> FEA;
    //
    //    //auto density = FEA.template newElementField<double, 1>("Density", 1, 0);
    //    auto displacement = FEA.template newNodeField<double, 1>("Displacement", 1, 0);

    // const Type alpha = 11;
    NEON_INFO(grid.toString());

    const std::string appName(testFilePrefix + "_" + grid.getImplementationName());
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
    runAllTestConfiguration<Grid, Type, 0>("sGrid", FeaVoxelGrid<Grid, Type, 0>, nGpus, 1);
}

TEST(Swap, eGrid)
{
    Neon::init();
    int nGpus = getNGpus();
    using Grid = Neon::domain::eGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("sGrid", FeaVoxelGrid<Grid, Type, 0>, nGpus, 1);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}