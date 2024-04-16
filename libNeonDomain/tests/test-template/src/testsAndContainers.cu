#include <functional>
#include "Neon/domain/Grids.h"
#include "Neon/domain/details/dGridDisg/dGrid.h"
#include "Neon/domain/details/dGridSoA/dGridSoA.h"

#include "Neon/domain/tools/TestData.h"

auto NEON_CUDA_HOST_DEVICE idxToInt(Neon::index_3d idx)
{
    return 33 * 1000000 + 10000 * idx.x + 100 * idx.y + idx.z;
};

int main()
{
    using Grid = Neon::bGridMgpu;
    Neon::init();
    Neon::Backend bk({0, 0}, Neon::Runtime::openmp);
    int           blockSize = Neon::domain::details::bGridMgpu::defaultBlockSize;
    Grid          bGridMgpu(
        bk,
        {blockSize, blockSize, 6 * blockSize},
        [](Neon::index_3d idx) { return true; },
        Neon::domain::Stencil::s6_Jacobi_t(),
        {1, 1, 1},
        {0, 0, 0},
        Neon::domain::tool::spaceCurves::EncoderType::sweep);
    using Field = typename Grid::Field<int, 1>;
    Field A = bGridMgpu.newField<int, 1>("A", 1, 0);

    auto setupOp = bGridMgpu.newContainer(
        "setup",
        [&A](Neon::set::Loader& loader) {
            auto a = loader.load(A);
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& gIdx) mutable {
                auto const global_idx = a.getGlobalIndex(gIdx);
                a(gIdx, 0) = idxToInt(global_idx);
            };
        });

    setupOp.run(0);
    bk.sync(0);

    A.newHaloUpdate(
         Neon::set::StencilSemantic::standard,
         Neon::set::TransferMode::get,
         Neon::Execution::device)
        .run(0);

    bk.sync(0);

    auto stencilOp = bGridMgpu.newContainer(
        "setup",
        [&A](Neon::set::Loader& loader) {
            auto a = loader.load(A);
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& gIdx) mutable {
                auto const global_idx = a.getGlobalIndex(gIdx);
                auto       expectedNgh = global_idx + Neon::index_3d(0, 0, 1);
                auto       val = a.getNghData<0, 0, 1>(gIdx, 0, -1).getData();
                if (val != -1 &&
                    val != idxToInt(expectedNgh)) {
                    printf("ERROR: %d (%d %d %d) %d\n", val,
                           global_idx.x, global_idx.y, global_idx.z,
                           a(gIdx, 0));
                    if (global_idx.x == 0 && global_idx.y == 0 && global_idx.z == 11) {
                        printf("ERROR: %d (%d %d %d) %d\n", val,
                               global_idx.x, global_idx.y, global_idx.z,
                               a(gIdx, 0));
                        val = a.getNghData<0, 0, 1>(gIdx, 0, -1).getData();
                    }
                }
            };
        });
    stencilOp.run(0);
    bk.sync(0);
};
