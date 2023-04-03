//#include <functional>
//#include "Neon/domain/dGrid.h"
//#include "Neon/domain/eGrid.h"
//
//#include "Neon/domain/tools/TestData.h"
//#include "TestInformation.h"
//#include "gtest/gtest.h"
//
//
//namespace host {
//
//template <typename Field>
//auto setToPitch(Field& fieldB)
//    -> Neon::set::Container
//{
//    const auto& grid = fieldB.getGrid();
//    return grid.getHostContainer(
//        "HostSetToPitch",
//        [&](Neon::set::Loader& loader) {
//            auto b = loader.load(fieldB);
//
//            return [=](const typename Field::Index& e) mutable {
//                Neon::index_3d const global = b.mapToGlobal(e);
//                for (int i = 0; i < b.cardinality(); i++) {
//                    auto const           domainSize = b.getDomainSize();
//                    typename Field::Type result = (global + domainSize * i).mPitch(domainSize);
//                    b(e, i) = result;
//                }
//            };
//        });
//}
//
//using namespace Neon::domain::tool::testing;
//
//template <typename G, typename T, int C>
//auto runHost(TestData<G, T, C>& data) -> void
//{
//    using Type = typename TestData<G, T, C>::Type;
//    auto&             grid = data.getGrid();
//    const std::string appName = TestInformation::fullName(grid.getImplementationName());
//
//    // NEON_INFO(grid.toString());
//
//    data.resetValuesToConst(1, 1);
//
//    {  // NEON
//        const Neon::index_3d        dim = grid.getDimension();
//        std::vector<Neon::index_3d> elements;
//
//        auto& Y = data.getField(FieldNames::Y);
//
//
//        setToPitch(Y)
//            .run(Neon::Backend::mainStreamIdx);
//        Y.updateCompute(Neon::Backend::mainStreamIdx);
//
//        // The TestData compare capabilities assumes that all data is on the device
//        // We need therefore to update the compute part with the data from the host.
//        data.getBackend().sync(Neon::Backend::mainStreamIdx);
//    }
//
//    {  // Golden data
//        auto& Y = data.getIODomain(FieldNames::Y);
//        data.forEachActiveIODomain([&](const Neon::index_3d& global,
//                                       int                   i,
//                                       Type&                 b) {
//            Neon::index_3d const domainSize = Y.getDimension();
//            Type                 result = (global + domainSize * i).mPitch(domainSize);
//            b = result;
//        },
//                                   Y);
//    }
//
//    bool isOk = data.compare(FieldNames::Y);
//
//    ASSERT_TRUE(isOk);
//}
//
//// template auto runHost<Neon::domain::eGrid, int64_t, 0>(TestData<Neon::domain::eGrid, int64_t, 0>&) -> void;
//template auto runHost<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;
//
//}  // namespace host