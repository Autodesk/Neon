#include <functional>
#include "Neon/domain/eGrid.h"
#include "Neon/domain/details/dGrid/dGrid.h"

#include "Neon/domain/tools/TestData.h"
#include "TestInformation.h"
#include "gtest/gtest.h"


namespace device {
template <typename Field>
auto setToPitch(Field& fieldA,
                Field& fieldB,
                Field& fieldC)
    -> Neon::set::Container
{
    const auto& grid = fieldB.getGrid();
    return grid.newContainer(
        "DeviceSetToPitch",
        [&](Neon::set::Loader& loader) {
            auto a = loader.load(fieldA);
            auto b = loader.load(fieldB);
            auto c = loader.load(fieldC);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                for (int i = 0; i < b.cardinality(); i++) {
                    Neon::index_3d const global = b.mapToGlobal(e);

                    a(e, i) = global.x;
                    //printf("E (%d, %d, %d) Val (%d, %d %d)\n", e.mLocation.x, e.mLocation.y, e.mLocation.z, global.x, global.y, global.z);
                    b(e, i) = global.y;
                    c(e, i) = global.z;
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

        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);
        auto& Z = data.getField(FieldNames::Z);


        setToPitch(X, Y, Z).run(0);

        X.updateHostData(0);
        Y.updateHostData(0);
        Z.updateHostData(0);

        data.getBackend().sync(0);
    }

    {  // Golden data
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);
        auto& Z = data.getIODomain(FieldNames::Z);

        data.forEachActiveIODomain([&](const Neon::index_3d& global,
                                       int                   i,
                                       Type&                 a,
                                       Type&                 b,
                                       Type&                 c) {
            a = global.x;
            b = global.y;
            c = global.z;
        },
                                   X, Y, Z);
    }

    data.updateHostData();

    //    auto printSome = [&](FieldNames fieldNames){
    //        for(int i=0; i< 6 ; i++){
    //            std::cout << data.getField(fieldNames).getReference({0, 0, i}, 0) <<" ";
    //        }
    //        std::cout <<std::endl;
    //    };
    //    printSome(FieldNames::X);
    //    printSome(FieldNames::Y);
    //    printSome(FieldNames::Z);


    //    data.getField(FieldNames::X).ioToVtk("X", "X", false);
    //    data.getField(FieldNames::Y).ioToVtk("Y", "Y", false);
    //    data.getField(FieldNames::Z).ioToVtk("Z", "Z", false);
    //
    //    data.getField(FieldNames::X).ioVtiAllocator("X_");
    //    data.getField(FieldNames::Y).ioVtiAllocator("Y_");
    //    data.getField(FieldNames::Z).ioVtiAllocator("Z_");

    data.updateHostData();
    bool isOk = data.compare(FieldNames::Z);
    //    isOk = isOk && data.compare(FieldNames::Y);
    //    isOk = isOk && data.compare(FieldNames::Z);

    ASSERT_TRUE(isOk);
}

// template auto run<Neon::domain::eGrid, int64_t, 0>(TestData<Neon::domain::eGrid, int64_t, 0>&) -> void;
template auto runDevice<Neon::domain::details::dGrid::dGrid, int64_t, 0>(TestData<Neon::domain::details::dGrid::dGrid, int64_t, 0>&) -> void;

}  // namespace device