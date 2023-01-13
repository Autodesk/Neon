#include <functional>
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"

#include "Neon/domain/tools/TestData.h"
#include "TestInformation.h"
#include "cuda_fp16.h"
#include "gtest/gtest.h"

namespace map {

template <typename ComputeType, typename Field>
auto mapContainer_axpy(int                   streamIdx,
                       typename Field::Type& val,
                       const Field&          filedA,
                       Field&                fieldB)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();

    return grid.getContainer("mapContainer_axpy",
                             [&, val](Neon::set::Loader& loader) {
                                 const auto a = loader.load(filedA);
                                 auto       b = loader.load(fieldB);

                                 return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Cell& e) mutable {
                                     for (int i = 0; i < a.cardinality(); i++) {
                                         // printf("GPU %ld <- %ld + %ld\n", lc(e, i) , la(e, i) , val);
                                         ComputeType x = a.template castRead<ComputeType>(e, i);
                                         ComputeType y = b.template castRead<ComputeType>(e, i);
                                         ComputeType val_c = static_cast<ComputeType>(val);
                                         ComputeType output = val_c * x + y;
                                         b.castWrite(e, i, output);
                                     }
                                 };
                             });
}

using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C, typename ComputeType>
auto run(TestData<G, T, C>& data) -> void
{

    using Type = typename TestData<G, T, C>::Type;
    auto&             grid = data.getGrid();
    const std::string appName = TestInformation::fullName(grid.getImplementationName());

    NEON_INFO(grid.toString());

    data.resetValuesToLinear(1, 100);
    T val = T(33);

    {  // NEON
        const Neon::index_3d        dim = grid.getDimension();
        std::vector<Neon::index_3d> elements;

        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);


        mapContainer_axpy<ComputeType>(Neon::Backend::mainStreamIdx,
                                                  val, X, Y)
            .run(0);

        data.getBackend().sync(0);
    }

    {  // Golden data
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);
        data.forEachActiveIODomain([&](const Neon::index_3d& /*idx*/,
                                       int /*cardinality*/,
                                       Type& a,
                                       Type& b) {
            ComputeType a_c = static_cast<ComputeType>(a);
            ComputeType b_c = static_cast<ComputeType>(b);
            ComputeType val_c = static_cast<ComputeType>(val);


            b_c += val_c * a_c;
            b = static_cast<Type>(b_c);
        },
                                   X, Y);
    }

    bool isOk = data.compare(FieldNames::Y);
    ASSERT_TRUE(isOk);
}

template auto run<Neon::domain::eGrid, int64_t, 0, double>(TestData<Neon::domain::eGrid, int64_t, 0>&) -> void;
template auto run<Neon::domain::dGrid, int64_t, 0, double>(TestData<Neon::domain::dGrid, int64_t, 0>&) -> void;

}  // namespace map