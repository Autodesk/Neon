#include "gtest/gtest.h"

#include "Neon/Neon.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/tools/TestData.h"
#include "RunHelper.h"
using namespace Neon::domain::tool::testing;
static const std::string testFilePrefix("domainUt_Swap");


template <typename Field>
auto map(Field&                      input_field,
         Field&                      output_field,
         const typename Field::Type& alpha) -> Neon::set::Container
{
    return input_field.getGrid().getContainer(
        "MAP",
        [&](Neon::set::Loader& loader) {
            const auto& inp = loader.load(input_field);
            auto&       out = loader.load(output_field);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Cell& e) mutable {
                for (int i = 0; i < inp.cardinality(); i++) {
                    out(e, i) = inp(e, i) + alpha;
                }
            };
        });
}

template <typename G, typename T, int C>
void SwapContainerRun(TestData<G, T, C>& data)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        /*
     * Grid structure:
     * - (x,y,z) in sGrid if X(x,y,z) %2 ==0
     * Computations on Neon
     * a. sX = X
     * b. sY = 2* sX
     * c. Y = sY
     *
     * Computation on Golden reference
     * - if X(x,y,z) %2, Y(x,y,z) = 2*X(x,y,z)
     * - else Y(x,y,z) =  Y_{t0}(x,y,z)
     *
     * Check
     * Y
     */
        using Type = typename TestData<G, T, C>::Type;
        auto& grid = data.getGrid();

        const Type alpha = 11;
        NEON_INFO(grid.toString());

        const std::string appName(testFilePrefix + "_" + grid.getImplementationName());
        data.resetValuesToLinear(1, 100);

        {  // NEON
            auto& X = data.getField(FieldNames::X);
            auto& Y = data.getField(FieldNames::Y);

            map(X, Y, alpha).run(0);
            X.swap(X, Y);
            map(X, Y, alpha).run(0);
            X.swap(X, Y);
            map(X, Y, alpha).run(0);

            data.getBackend().sync(0);
            Y.updateIO(0);
        }

        {  // Golden data

            auto& X = data.getIODomain(FieldNames::X);
            auto& Y = data.getIODomain(FieldNames::Y);

            auto run = [&](auto A, auto B) {
                data.forEachActiveIODomain([&](const Neon::index_3d& idx,
                                               int                   cardinality,
                                               Type&                 a,
                                               Type&                 b) {
                    b = alpha + a;
                },
                                           A, B);
            };

            run(X, Y);
            run(Y, X);
            run(X, Y);
        }

        // storage.ioToVti("After");
        {  // DEBUG
            data.getIODomain(FieldNames::Y).ioToVti("getIODomain_Y", "Y");
            data.getField(FieldNames::Y).ioToVtk("getField_Y", "Y");
        }


        bool isOk = data.compare(FieldNames::Y);
        isOk = isOk && data.compare(FieldNames::X);

        ASSERT_TRUE(isOk);
    }
}

namespace {
int getNGpus()
{
    int maxGPUs = Neon::set::DevSet::maxSet().setCardinality();
    if (maxGPUs > 1) {
        return maxGPUs;
    } else {
        return 3;
    }
}
}  // namespace

TEST(Swap, dGrid)
{
    Neon::init();
    int nGpus = getNGpus();
    using Grid = Neon::domain::dGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("sGrid", SwapContainerRun<Grid, Type, 0>, nGpus, 1);
}

TEST(Swap, eGrid)
{
    Neon::init();
    int nGpus = getNGpus();
    using Grid = Neon::domain::eGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("sGrid", SwapContainerRun<Grid, Type, 0>, nGpus, 1);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
