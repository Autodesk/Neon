#include <functional>
#include "Neon/domain/internal/experimental/dGrid/dGrid.h"
#include "Neon/domain/eGrid.h"

#include "Neon/domain/tools/TestData.h"
#include "TestInformation.h"
#include "gtest/gtest.h"


namespace device {
template <typename Field>
auto setToPitch(Field& fieldB)
    -> Neon::set::Container
{
    const auto& grid = fieldB.getGrid();
    return grid.newContainer(
        "DeviceSetToPitch",
        [&](Neon::set::Loader& loader) {
            auto b = loader.load(fieldB);
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                for (int i = 0; i < b.cardinality(); i++) {
                    Neon::index_3d const global = b.mapToGlobal(e);
                    auto const           domainSize = b.getDomainSize();
                    typename Field::Type result = (global + domainSize * i).mPitch(domainSize);
                    b(e, i) = result;
                }
            };
        },
        Neon::Execution::device);
}

using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C>
auto runDevice(TestData<G, T, C>& data) -> void
{
    using Type = typename TestData<G, T, C>::Type;
    auto&             grid = data.getGrid();
    const std::string appName = TestInformation::fullName(grid.getImplementationName());

    data.resetValuesToLinear(1, 100);

    {  // NEON
        const Neon::index_3d        dim = grid.getDimension();
        std::vector<Neon::index_3d> elements;

        auto& Y = data.getField(FieldNames::Y);


        setToPitch(Y)
            .run(0);

        data.getBackend().sync(0);
    }

    {  // Golden data
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);
        data.forEachActiveIODomain([&](const Neon::index_3d& global,
                                       int                   i,
                                       Type&                 b) {
            Neon::index_3d const domainSize = Y.getDimension();
            b = (global + domainSize * i).mPitch(domainSize);
        },
                                   Y);
    }

    bool isOk = data.compare(FieldNames::Y);
    ASSERT_TRUE(isOk);
}

// template auto run<Neon::domain::eGrid, int64_t, 0>(TestData<Neon::domain::eGrid, int64_t, 0>&) -> void;
template auto runDevice<Neon::domain::internal::exp::dGrid::dGrid, int64_t, 0>(TestData<Neon::domain::internal::exp::dGrid::dGrid, int64_t, 0>&) -> void;

}  // namespace device