#include <functional>
#include "Neon/domain/Grids.h"

#include "Neon/domain/tools/TestData.h"
#include "TestInformation.h"
#include "gtest/gtest.h"


namespace eGridTesting {


using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C>
auto run(TestData<G, T, C>& data) -> void
{

    using Type = typename TestData<G, T, C>::Type;
    Neon::eGrid&      grid = data.getGrid();
    Neon::Backend     backend = grid.getBackend();
    const std::string appName = TestInformation::fullName(grid.getImplementationName());

    NEON_INFO(grid.toString());
    const Neon::index_3d dim = grid.getDimension();
    bool                 testDetected = false;

    if (dim == Neon::index_3d(1, 1, 3 * backend.getDeviceCount()) && backend.getDeviceCount() >= 2) {
        backend.forEachDeviceSeq([&](Neon::SetIdx setIdx) {
            int boundaryOneDirectionCount = 1;
            int internalCount = 1;

            auto& data = grid.helpGetData();
            ASSERT_EQ(data.partitioner1D.getStandardCount()[setIdx], 3) << "Incoherent value from getStandardCount @ device" << setIdx.idx();
            ASSERT_EQ(data.partitioner1D.getStandardAndGhostCount()[setIdx], 5) << "Incoherent value from getStandardAndGhostCount @ device 0";

            ASSERT_EQ(data.partitioner1D.getSpanLayout().getBoundsInternal(setIdx).first, 0) << "Incoherent value from getBoundsInternal.first @ device " << setIdx.idx();
            ASSERT_EQ(data.partitioner1D.getSpanLayout().getBoundsInternal(setIdx).count, 1) << "Incoherent value from getBoundsInternal.count @ device " << setIdx.idx();

            auto bounds = data.partitioner1D.getSpanLayout().getBoundsBoundary(setIdx, Neon::domain::tool::partitioning::ByDirection::up);
            ASSERT_EQ(bounds.first, 1) << "Incoherent value from getBoundsBoundaryUp.first @ device " << setIdx.idx();
            ASSERT_EQ(bounds.count, 1) << "Incoherent value from getBoundsBoundaryUp.count @ device " << setIdx.idx();

            bounds = data.partitioner1D.getSpanLayout().getBoundsBoundary(setIdx, Neon::domain::tool::partitioning::ByDirection::down);
            ASSERT_EQ(bounds.first, 2) << "Incoherent value from getBoundsBoundaryDown.first @ device " << setIdx.idx();
            ASSERT_EQ(bounds.count, 1) << "Incoherent value from getBoundsBoundaryDown.count @ device " << setIdx.idx();

            bounds = data.partitioner1D.getSpanLayout().getGhostBoundary(setIdx, Neon::domain::tool::partitioning::ByDirection::up);
            ASSERT_EQ(bounds.first, internalCount + 2 * boundaryOneDirectionCount) << "Incoherent value from getBoundsBoundaryDown.first @ device " << setIdx.idx();
            ASSERT_EQ(bounds.count, boundaryOneDirectionCount) << "Incoherent value from getBoundsBoundaryDown.count @ device " << setIdx.idx();

            bounds = data.partitioner1D.getSpanLayout().getGhostBoundary(setIdx, Neon::domain::tool::partitioning::ByDirection::down);
            ASSERT_EQ(bounds.first, internalCount + 3 * boundaryOneDirectionCount) << "Incoherent value from getBoundsBoundaryDown.first @ device " << setIdx.idx();
            ASSERT_EQ(bounds.count, boundaryOneDirectionCount) << "Incoherent value from getBoundsBoundaryDown.count @ device " << setIdx.idx();
        });
        testDetected = true;
    }

    if (dim == Neon::index_3d(1, 1, 6 * backend.getDeviceCount()) && backend.getDeviceCount() >= 2) {
        backend.forEachDeviceSeq([&](Neon::SetIdx setIdx) {
            auto& data = grid.helpGetData();
            ASSERT_EQ(data.partitioner1D.getStandardCount()[setIdx], 6) << "Incoherent value from getStandardCount @ device" << setIdx.idx();
            ASSERT_EQ(data.partitioner1D.getStandardAndGhostCount()[setIdx], 8) << "Incoherent value from getStandardAndGhostCount @ device 0";

            ASSERT_EQ(data.partitioner1D.getSpanLayout().getBoundsInternal(setIdx).first, 0) << "Incoherent value from getBoundsInternal.first @ device " << setIdx.idx();
            ASSERT_EQ(data.partitioner1D.getSpanLayout().getBoundsInternal(setIdx).count, 4) << "Incoherent value from getBoundsInternal.count @ device " << setIdx.idx();

            auto bounds = data.partitioner1D.getSpanLayout().getBoundsBoundary(setIdx, Neon::domain::tool::partitioning::ByDirection::up);
            ASSERT_EQ(bounds.first, 4) << "Incoherent value from getBoundsBoundaryUp.first @ device " << setIdx.idx();
            ASSERT_EQ(bounds.count, 1) << "Incoherent value from getBoundsBoundaryUp.count @ device " << setIdx.idx();

            bounds = data.partitioner1D.getSpanLayout().getBoundsBoundary(setIdx, Neon::domain::tool::partitioning::ByDirection::down);
            ASSERT_EQ(bounds.first, 5) << "Incoherent value from getBoundsBoundaryDown.first @ device " << setIdx.idx();
            ASSERT_EQ(bounds.count, 1) << "Incoherent value from getBoundsBoundaryDown.count @ device " << setIdx.idx();
        });
        testDetected = true;
    }

    if (dim == Neon::index_3d(3, 5, 6 * backend.getDeviceCount()) && backend.getDeviceCount() >= 2) {
        backend.forEachDeviceSeq([&](Neon::SetIdx setIdx) {
            auto&          data = grid.helpGetData();
            Neon::index_3d partDim = Neon::index_3d(3, 5, 6);
            ASSERT_EQ(data.partitioner1D.getStandardCount()[setIdx], partDim.rMul()) << "Incoherent value from getStandardCount @ device" << setIdx.idx();
            ASSERT_EQ(data.partitioner1D.getStandardAndGhostCount()[setIdx], partDim.rMul() + 2 * partDim.x * partDim.y) << "Incoherent value from getStandardAndGhostCount @ device 0";

            int internalCount = partDim.rMul() - 2 * partDim.x * partDim.y;
            ASSERT_EQ(data.partitioner1D.getSpanLayout().getBoundsInternal(setIdx).first, 0) << "Incoherent value from getBoundsInternal.first @ device " << setIdx.idx();
            ASSERT_EQ(data.partitioner1D.getSpanLayout().getBoundsInternal(setIdx).count, internalCount) << "Incoherent value from getBoundsInternal.count @ device " << setIdx.idx();

            auto bounds = data.partitioner1D.getSpanLayout().getBoundsBoundary(setIdx, Neon::domain::tool::partitioning::ByDirection::up);
            int  boundaryOneDirectionCount = partDim.x * partDim.y;
            ASSERT_EQ(bounds.first, internalCount) << "Incoherent value from getBoundsBoundaryUp.first @ device " << setIdx.idx();
            ASSERT_EQ(bounds.count, boundaryOneDirectionCount) << "Incoherent value from getBoundsBoundaryUp.count @ device " << setIdx.idx();

            bounds = data.partitioner1D.getSpanLayout().getBoundsBoundary(setIdx, Neon::domain::tool::partitioning::ByDirection::down);
            ASSERT_EQ(bounds.first, internalCount + boundaryOneDirectionCount) << "Incoherent value from getBoundsBoundaryDown.first @ device " << setIdx.idx();
            ASSERT_EQ(bounds.count, boundaryOneDirectionCount) << "Incoherent value from getBoundsBoundaryDown.count @ device " << setIdx.idx();

            bounds = data.partitioner1D.getSpanLayout().getGhostBoundary(setIdx, Neon::domain::tool::partitioning::ByDirection::up);
            ASSERT_EQ(bounds.first, internalCount + 2 * boundaryOneDirectionCount) << "Incoherent value from getBoundsBoundaryDown.first @ device " << setIdx.idx();
            ASSERT_EQ(bounds.count, boundaryOneDirectionCount) << "Incoherent value from getBoundsBoundaryDown.count @ device " << setIdx.idx();

            bounds = data.partitioner1D.getSpanLayout().getGhostBoundary(setIdx, Neon::domain::tool::partitioning::ByDirection::down);
            ASSERT_EQ(bounds.first, internalCount + 3 * boundaryOneDirectionCount) << "Incoherent value from getBoundsBoundaryDown.first @ device " << setIdx.idx();
            ASSERT_EQ(bounds.count, boundaryOneDirectionCount) << "Incoherent value from getBoundsBoundaryDown.count @ device " << setIdx.idx();
        });
        testDetected = true;
    }

    ASSERT_TRUE(testDetected);
}

template auto run<Neon::eGrid, int64_t, 0>(TestData<Neon::eGrid, int64_t, 0>&) -> void;


}  // namespace eGridTesting