#include <functional>
#include "Neon/domain/Grids.h"
#include "Neon/domain/interface/GridConcept.h"
#include "Neon/domain/tools/TestData.h"
#include "TestInformation.h"
#include "gtest/gtest.h"


namespace map {

template <typename Field>
auto haloSetGlobalPosition(Field& fieldB)
    -> Neon::set::Container
{
    const auto& grid = fieldB.getGrid();
    return grid.newContainer(
        "haloSetGlobalPosition",
        [&](Neon::set::Loader& loader) {
            auto b = loader.load(fieldB);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& idx) mutable {
                auto globalIdx = b.getGlobalIndex(idx);
                b(idx, 0) = globalIdx.x;
                b(idx, 1) = globalIdx.y;
                b(idx, 2) = globalIdx.z;
            };
        });
}

template <typename Field>
auto haloCheckContainer(const Field&        filedA,
                        const Neon::int8_3d offset,
                        Field&              fieldB)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();
    return grid.newContainer(
        "haloCheckContainer",
        [&, offset](Neon::set::Loader& loader) {
            const auto     a = loader.load(filedA, Neon::Compute::STENCIL);
            auto           b = loader.load(fieldB);
            Neon::index_3d domainSize = filedA.getGrid().getDimension();
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& idx) mutable {
                auto globalIdx = b.getGlobalIndex(idx);
                auto nghGlobal = globalIdx + offset.newType<int32_t>();

                bool goldenActive = false;
                if (nghGlobal >= 0 && nghGlobal < domainSize) {
                    goldenActive = true;
                }

                bool checkOnActivity[3] = {false, false, false};
                bool checkOnValue[3] = {false, false, false};

                for (int i = 0; i < a.cardinality(); i++) {
                    typename Field::NghData const nghData = a.getNghData(idx, offset, i);

                    // Checking data
                    if (nghData.isValid()) {
                        if (!goldenActive) {
                            checkOnActivity[i] = false;
                            continue;
                        }
                        checkOnActivity[i] = true;
                        const auto data = nghData.getData();
                        const auto golden = nghGlobal.v[i];
                        if (data != golden) {
                            checkOnValue[i] = false;
                        } else {
                            checkOnValue[i] = true;
                        }
                    }

                    typename Field::Type resBycard = 0;
                    if (goldenActive == false) {
                        resBycard = checkOnActivity[i] == false ? 1 : 0;
                    } else {
                        resBycard = checkOnActivity[i] == true && checkOnValue[i] == true ? 1 : 0;
                    }
                    if (resBycard == 0) {
                        printf("here cell %d %d %d, ngh %d %d %d, card %d, val %d vs %d\n",
                               globalIdx.x, globalIdx.y, globalIdx.z,
                               nghGlobal.x, nghGlobal.y, nghGlobal.z,
                               i,
                               int(nghData.getData()),
                               nghGlobal.v[i]);
                    }
                    b(idx, i) = resBycard;
                }
            };
        });
}

using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C>
auto run(TestData<G, T, C>& data) -> void
{

    using Type = typename TestData<G, T, C>::Type;
    auto&             grid = data.getGrid();
    const std::string appName = TestInformation::fullName(grid.getImplementationName());

    NEON_INFO(grid.toString());

    data.resetValuesToMasked(1, 1, 2);
    int iterations = 1;
    {  // NEON
        const Neon::index_3d        dim = grid.getDimension();
        std::vector<Neon::index_3d> elements;

        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);
        auto& Z = data.getField(FieldNames::Z);
        auto& W = data.getField(FieldNames::W);

        haloSetGlobalPosition(X).run(Neon::Backend::mainStreamIdx);
        haloSetGlobalPosition(Y).run(Neon::Backend::mainStreamIdx);

        //        data.updateHostData();
        //        data.getField(FieldNames::X).ioToVtk("X_before", "X", true);

        X.newHaloUpdate(Neon::set::StencilSemantic::standard,
                        data.getTransferMode(),
                        Neon::Execution::device)
            .run(Neon::Backend::mainStreamIdx);

//        data.updateHostData();
//        data.getField(FieldNames::X).ioToVtk("X_after", "X", true);

        Y.newHaloUpdate(Neon::set::StencilSemantic::standard,
                        data.getTransferMode(),
                        Neon::Execution::device)
            .run(Neon::Backend::mainStreamIdx);

        haloCheckContainer(X, Neon::int8_3d(0, 0, 1), Z).run(Neon::Backend::mainStreamIdx);
        haloCheckContainer(Y, Neon::int8_3d(0, 0, -1), W).run(Neon::Backend::mainStreamIdx);

        data.getBackend().sync(0);
    }

    {  // Golden data
        auto haloSetGlobalPositionFun = [&data](auto&& domainA) {
            data.forEachActiveIODomain([&](const Neon::index_3d& idx,
                                           int                   cardinality,
                                           Type&                 a) {
                // Laplacian stencil operates on 6 neighbors (assuming 3D)
                a = idx.v[cardinality];
            },
                                       domainA);
        };

        auto haloCheckContainerFun = [&](auto&& XX, auto&& YY, Neon::int8_3d offset) {
            data.forEachActiveIODomain([&, offset](const Neon::index_3d& globalIdx,
                                                   int                   cardinality,
                                                   Type&                 a,
                                                   Type&                 b) {
                auto nghGlobal = globalIdx + offset.newType<int32_t>();
                auto domainSize = data.getDimention();

                bool goldenActive = false;
                if (nghGlobal >= 0 && nghGlobal < domainSize) {
                    goldenActive = true;
                }

                bool checkOnActivity[3] = {false, false, false};
                bool checkOnValue[3] = {false, false, false};

                int  i = cardinality;
                bool isValid = false;
                auto neighborVal = XX.nghVal(globalIdx, offset, cardinality, &isValid);
                if (isValid) {
                    if (!goldenActive) {
                        checkOnActivity[i] = false;
                    } else {
                        checkOnActivity[i] = true;
                        if (neighborVal != nghGlobal.v[i]) {
                            checkOnValue[i] = false;
                        } else {
                            checkOnValue[i] = true;
                        }
                    }
                }
                Type resBycard = 0;
                if (goldenActive == false) {
                    resBycard = checkOnActivity[i] == false ? 1 : 0;
                } else {
                    resBycard = checkOnActivity[i] == true && checkOnValue[i] == true ? 1 : 0;
                }
                b = resBycard;
            },
                                       XX, YY);
        };
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);
        auto& Z = data.getIODomain(FieldNames::Z);
        auto& W = data.getIODomain(FieldNames::W);

        haloSetGlobalPositionFun(X);
        haloSetGlobalPositionFun(Y);

        haloCheckContainerFun(X, Z, Neon::int8_3d(0, 0, 1));
        haloCheckContainerFun(Y, W, Neon::int8_3d(0, 0, -1));
    }

    data.updateHostData();

    bool doVti = false;
    bool isOk = data.compare(FieldNames::X);
    doVti = doVti || !isOk;
    ASSERT_TRUE(isOk);

    isOk = data.compare(FieldNames::Y);
    doVti = doVti || !isOk;
    ASSERT_TRUE(isOk);

    isOk = data.compare(FieldNames::Z);
    doVti = doVti || !isOk;
    ASSERT_TRUE(isOk);

    isOk = data.compare(FieldNames::W);
    doVti = doVti || !isOk;
    ASSERT_TRUE(isOk);

    if (doVti) {
        data.getField(FieldNames::X).ioToVtk("X", "X", true);
        data.getField(FieldNames::Y).ioToVtk("Y", "Y", false);
        data.getField(FieldNames::Z).ioToVtk("Z", "Z", false);
        data.getField(FieldNames::W).ioToVtk("W", "W", false);


        data.getIODomain(FieldNames::X).ioToVti("X_", "X_");
        data.getIODomain(FieldNames::Y).ioToVti("Y_", "Y_");
        data.getIODomain(FieldNames::Z).ioToVti("Z_", "Z_");
        data.getIODomain(FieldNames::W).ioToVti("W_", "W_");
    }
    //    std::cout << "COMPLETED" << std::endl;
}

template auto run<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;
template auto run<Neon::eGrid, int64_t, 0>(TestData<Neon::eGrid, int64_t, 0>&) -> void;

}  // namespace map