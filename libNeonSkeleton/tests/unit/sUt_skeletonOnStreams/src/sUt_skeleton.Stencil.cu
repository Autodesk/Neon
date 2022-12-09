#include <cuda_profiler_api.h>

#include "Neon/core/types/chrono.h"

#include "Neon/set/Containter.h"

#include "Neon/domain/bGrid.h"
#include "Neon/domain/tools/Geometries.h"
#include "Neon/domain/tools/TestData.h"

#include "Neon/skeleton/Options.h"
#include "Neon/skeleton/Skeleton.h"

#include <cctype>
#include <string>

#include "gtest/gtest.h"
#include "sUt.runHelper.h"

using namespace Neon::domain::tool::testing;
static const std::string testFilePrefix("sUt_skeleton_MapStencilMap");

template <typename Field, typename T>
auto axpy(const T&     val,
          const Field& y,
          Field&       x,
          size_t       sharedMem = 0) -> Neon::set::Container
{
    return y.getGrid().getContainer(
        "AXPY",
        y.getGrid().getDefaultBlock(),
        sharedMem,
        [&](Neon::set::Loader & L) -> auto {
            auto& yLocal = L.load(y);
            auto& xLocal = L.load(x);
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Cell& e) mutable {
                //Neon::sys::ShmemAllocator shmemAlloc;
                //yLocal.loadInSharedMemory(e, 1, shmemAlloc);

                //Neon::index_3d global = xLocal.mapToGlobal(e);

                for (int i = 0; i < yLocal.cardinality(); i++) {
                    xLocal(e, i) += val * yLocal(e, i);
                }
            };
        });
}

template <typename FieldT>
auto laplace(const FieldT& x, FieldT& y, size_t sharedMem = 0) -> Neon::set::Container
{
    return x.getGrid().getContainer(
        "Laplace",
        x.getGrid().getDefaultBlock(),
        sharedMem,
        [&](Neon::set::Loader & L) -> auto {
            auto& xLocal = L.load(x, Neon::Compute::STENCIL);
            auto& yLocal = L.load(y);

            return [=] NEON_CUDA_HOST_DEVICE(const typename FieldT::Cell& cell) mutable {
                //Neon::sys::ShmemAllocator shmemAlloc;
                //xLocal.loadInSharedMemory(cell, 1, shmemAlloc);


                using Type = typename FieldT::Type;
                for (int card = 0; card < xLocal.cardinality(); card++) {
                    typename FieldT::Type res = 0;


                    auto checkNeighbor = [&res](Neon::domain::NghInfo<Type>& neighbor) {
                        if (neighbor.isValid) {
                            res += neighbor.value;
                        }
                    };

                    for (int8_t nghIdx = 0; nghIdx < 6; ++nghIdx) {
                        auto neighbor = xLocal.nghVal(cell, nghIdx, card, Type(0));
                        checkNeighbor(neighbor);
                    }

                    yLocal(cell, card) = -6 * res;
                }
            };
        });
}


template <typename G, typename T, int C>
void SingleStencil(TestData<G, T, C>&      data,
                   Neon::skeleton::Occ     occ,
                   Neon::set::TransferMode transfer)
{
    using Type = typename TestData<G, T, C>::Type;

    auto occName = Neon::skeleton::OccUtils::toString(occ);
    occName[0] = toupper(occName[0]);
    const std::string appName(testFilePrefix + "_" + occName);

    Neon::skeleton::Skeleton skl(data.getBackend());
    Neon::skeleton::Options  opt(occ, transfer);

    const int nIterations = 5;

    const T val = 89;

    data.getBackend().syncAll();

    data.resetValuesToRandom(1, 50);


    {  // SKELETON
        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);

        std::vector<Neon::set::Container> ops;


        /*X.forEachActiveCell([&](const Neon::index_3d& idx,
                                const int&            cardinality,
                                T&) {


        });
        X.updateCompute(0);
        X.ioToVtk("X", "X");*/

        ops.push_back(laplace(X, Y, Y.getSharedMemoryBytes(1)));
        ops.push_back(axpy(val, Y, X, Y.getSharedMemoryBytes(1)));

        skl.sequence(ops, appName, opt);

        NEON_CUDA_CHECK_LAST_ERROR
        cudaProfilerStart();
        for (int i = 0; i < nIterations; i++) {
            skl.run();
        }
        data.getBackend().syncAll();
        cudaProfilerStop();
        NEON_CUDA_CHECK_LAST_ERROR

        //X.ioToVtk("X", "X");
        //Y.ioToVtk("Y", "Y");
    }

    {  // Golden data
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);

        for (int i = 0; i < nIterations; i++) {
            data.laplace(X, Y);
            data.axpy(&val, Y, X);
        }
    }

    bool isOk = data.compare(FieldNames::X);
    isOk = isOk && data.compare(FieldNames::Y);

    ASSERT_TRUE(isOk);
}

template <typename G, typename T, int C>
void SingleStencilOCC(TestData<G, T, C>& data)
{
    SingleStencil<G, T, C>(data, Neon::skeleton::Occ::standard, Neon::set::TransferMode::get);
}

template <typename G, typename T, int C>
void SingleStencilExtendedOCC(TestData<G, T, C>& data)
{
    SingleStencil<G, T, C>(data, Neon::skeleton::Occ::extended, Neon::set::TransferMode::get);
}

template <typename G, typename T, int C>
void SingleStencilTwoWayExtendedOCC(TestData<G, T, C>& data)
{
    SingleStencil<G, T, C>(data, Neon::skeleton::Occ::twoWayExtended, Neon::set::TransferMode::get);
}

template <typename G, typename T, int C>
void SingleStencilNoOCC(TestData<G, T, C>& data)
{
    SingleStencil<G, T, C>(data, Neon::skeleton::Occ::none, Neon::set::TransferMode::get);
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


TEST(SingleStencil_NoOCC, bGrid)
{
    int nGpus = 1;
    using Grid = Neon::domain::bGrid;
    //using Grid = Neon::domain::eGrid;
    //using Grid = Neon::domain::dGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("bGrid_t", SingleStencilNoOCC<Grid, Type, 0>, nGpus, 1);
}