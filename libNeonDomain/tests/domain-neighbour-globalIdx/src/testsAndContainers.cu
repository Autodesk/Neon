#include <functional>
#include "Neon/domain/Grids.h"

#include "Neon/domain/tools/TestData.h"
#include "TestInformation.h"
#include "gtest/gtest.h"


namespace globalIdx {

template <typename Field>
auto initData(Field& filedA,
              Field& filedB,
              Field& filedC)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();
    return grid.newContainer(
        "defContainer",
        [&](Neon::set::Loader& loader) {
            auto a = loader.load(filedA);
            auto b = loader.load(filedB);
            auto c = loader.load(filedC);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                // printf("GPU %ld <- %ld + %ld\n", lc(e, i) , la(e, i) , val);
                Neon::index_3d globalPoint = a.getGlobalIndex(e);
                a(e, 0) = globalPoint.x;
                b(e, 0) = globalPoint.y;
                c(e, 0) = globalPoint.z;
            };
        });
}

template <typename Field>
auto reset(Field& filedA,
           Field& filedB,
           Field& filedC)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();
    return grid.newContainer(
        "defContainer",
        [&](Neon::set::Loader& loader) {
            auto a = loader.load(filedA);
            auto b = loader.load(filedB);
            auto c = loader.load(filedC);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                // printf("GPU %ld <- %ld + %ld\n", lc(e, i) , la(e, i) , val);
                Neon::index_3d globalPoint = a.getGlobalIndex(e);
                a(e, 0) = -33;
                b(e, 0) = -33;
                c(e, 0) = -33;
            };
        });
}

template <typename Field>
auto checkNeighbourData(Field const&   filedA,
                        Field const&   filedB,
                        Field const&   filedC,
                        Neon::index_3d testDirection,
                        Field const&   checkFlatA,
                        Field const&   checkFlatB,
                        Field const&   checkFlatC)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();
    return grid.newContainer(
        "defContainer",
        [&](Neon::set::Loader& loader) {
            auto a = loader.load(filedA, Neon::Pattern::STENCIL);
            auto b = loader.load(filedB, Neon::Pattern::STENCIL);
            auto c = loader.load(filedC, Neon::Pattern::STENCIL);

            auto resA = loader.load(checkFlatA, Neon::Pattern::MAP);
            auto resB = loader.load(checkFlatB, Neon::Pattern::MAP);
            auto resC = loader.load(checkFlatC, Neon::Pattern::MAP);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                // printf("GPU %ld <- %ld + %ld\n", lc(e, i) , la(e, i) , val);
                Neon::index_3d globalPoint = a.getGlobalIndex(e);
                auto           ngh = globalPoint + testDirection;

                decltype(a)* nghInfo[3] = {&a, &b, &c};
                decltype(a)* results[3] = {&resA, &resB, &resC};

                for (int i = 0; i < 3; i++) {
                    auto d = nghInfo[i]->getNghData(e, testDirection.newType<int8_t>(), 0);
                    if (d.isValid()) {
                        results[i]->operator()(e, 0) = d.getData() == ngh.v[i] ? +1 : -1;
                    } else {
                        results[i]->operator()(e, 0) = 0;
                    }
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

    data.resetValuesToLinear(1, 100);

    auto aField = grid.template newField<int64_t>("a", 1, 0);
    auto bField = grid.template newField<int64_t>("a", 1, 0);
    auto cField = grid.template newField<int64_t>("a", 1, 0);

    auto& X = data.getField(FieldNames::X);
    auto& Y = data.getField(FieldNames::Y);
    auto& Z = data.getField(FieldNames::Z);

    const Neon::index_3d dim = grid.getDimension();
    auto                 bk = grid.getBackend();

    {  // NEON
        {
            initData(aField, bField, cField).run(Neon::Backend::mainStreamIdx);
            bk.sync(Neon::Backend::mainStreamIdx);
            aField.newHaloUpdate(Neon::set::StencilSemantic::standard, Neon::set::TransferMode::put, Neon::Execution::device).run(Neon::Backend::mainStreamIdx);
            bField.newHaloUpdate(Neon::set::StencilSemantic::standard, Neon::set::TransferMode::put, Neon::Execution::device).run(Neon::Backend::mainStreamIdx);
            cField.newHaloUpdate(Neon::set::StencilSemantic::standard, Neon::set::TransferMode::put, Neon::Execution::device).run(Neon::Backend::mainStreamIdx);
            bk.sync(Neon::Backend::mainStreamIdx);
        }
    }
    using Ngh3DIdx = Neon::int32_3d;

    auto setGolden = [&](Ngh3DIdx const& direction) {  // Golden data
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);
        auto& Z = data.getIODomain(FieldNames::Z);

        data.forEachActiveIODomain([&](const Neon::index_3d& idx,
                                       int                   cardinality,
                                       Type&                 a,
                                       Type&                 b,
                                       Type&                 c) {
            a = 1;
            b = 1;
            c = 1;
            auto ngh = direction + idx;
            if (!(ngh >= 0)) {
                a = 0;
                b = 0;
                c = 0;
            }
            if (!(dim > ngh)) {
                a = 0;
                b = 0;
                c = 0;
            }
        },
                                   X, Y, Z);
    };

    constexpr std::array<const Ngh3DIdx, 6>
        stencil{Ngh3DIdx(1, 0, 0),
                Ngh3DIdx(-1, 0, 0),
                Ngh3DIdx(0, 1, 0),
                Ngh3DIdx(0, -1, 0),
                Ngh3DIdx(0, 0, 1),
                Ngh3DIdx(0, 0, -1)};


    for (auto const& direction : stencil) {
        reset(aField, bField, cField).run(Neon::Backend::mainStreamIdx);
        reset(X, Y, Z).run(Neon::Backend::mainStreamIdx);
        {  // Updating halo with wrong data
            bk.sync(Neon::Backend::mainStreamIdx);
            aField.newHaloUpdate(Neon::set::StencilSemantic::standard, Neon::set::TransferMode::put, Neon::Execution::device).run(Neon::Backend::mainStreamIdx);
            bField.newHaloUpdate(Neon::set::StencilSemantic::standard, Neon::set::TransferMode::put, Neon::Execution::device).run(Neon::Backend::mainStreamIdx);
            cField.newHaloUpdate(Neon::set::StencilSemantic::standard, Neon::set::TransferMode::put, Neon::Execution::device).run(Neon::Backend::mainStreamIdx);
            bk.sync(Neon::Backend::mainStreamIdx);
        }
        {
            initData(aField, bField, cField).run(Neon::Backend::mainStreamIdx);
            bk.sync(Neon::Backend::mainStreamIdx);
            aField.newHaloUpdate(Neon::set::StencilSemantic::standard, Neon::set::TransferMode::put, Neon::Execution::device).run(Neon::Backend::mainStreamIdx);
            bField.newHaloUpdate(Neon::set::StencilSemantic::standard, Neon::set::TransferMode::put, Neon::Execution::device).run(Neon::Backend::mainStreamIdx);
            cField.newHaloUpdate(Neon::set::StencilSemantic::standard, Neon::set::TransferMode::put, Neon::Execution::device).run(Neon::Backend::mainStreamIdx);
            bk.sync(Neon::Backend::mainStreamIdx);
        }


        checkNeighbourData(aField, bField, cField, direction, X, Y, Z).run(Neon::Backend::mainStreamIdx);
        setGolden(direction);

        bk.sync(Neon::Backend::mainStreamIdx);
        bool isOk = data.compare(FieldNames::X);
        isOk = isOk && data.compare(FieldNames::Y);
        isOk = isOk && data.compare(FieldNames::Z);

        if (!isOk) {
            std::cout << "Direction with errors " << direction << std::endl;
            data.getField(FieldNames::X).ioToVtk(grid.getImplementationName() + "X", "X", true);
            data.getField(FieldNames::Y).ioToVtk(grid.getImplementationName() + "Y", "Y", true);
            data.getField(FieldNames::Z).ioToVtk(grid.getImplementationName() + "Z", "Z", true);
            exit(99);
            ASSERT_TRUE(isOk);
        }
    }
}

template auto run<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;
template auto run<Neon::eGrid, int64_t, 0>(TestData<Neon::eGrid, int64_t, 0>&) -> void;
template auto run<Neon::bGrid, int64_t, 0>(TestData<Neon::bGrid, int64_t, 0>&) -> void;

}  // namespace globalIdx