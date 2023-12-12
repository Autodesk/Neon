
#include "Neon/Neon.h"
#include "Neon/domain/tools/SpaceCurves.h"
#include "domain-space-filling-curves.h"
#include "goldenEncoding.h"
#include "gtest/gtest.h"
#include "runHelper.h"

TEST(domain_space_filling_curves, morton)
{
    Neon::int32_3d dim = {16, 16, 16};
    for (int x = 0; x < dim.x; x++) {
        for (int y = 0; y < dim.y; y++) {
            for (int z = 0; z < dim.z; z++) {
                using namespace Neon::domain::tool::spaceCurves;
                Neon::int32_3d idx = {x, y, z};
                auto           morton = Encoder::encode(EncoderType::morton, dim, idx);
                auto           sweep = Encoder::encode(EncoderType::sweep, dim, {z, y, x});

                ASSERT_EQ(morton_grid_16_16_16[sweep], morton) << dim << " " << idx << " " << morton;
            }
        }
    }
}

TEST(domain_space_filling_curves, hilbert)
{
    Neon::int32_3d dim = {16, 16, 16};
    for (int x = 0; x < dim.x; x++) {
        for (int y = 0; y < dim.y; y++) {
            for (int z = 0; z < dim.z; z++) {

                using namespace Neon::domain::tool::spaceCurves;
                Neon::int32_3d idx = {x, y, z};
                auto           hilbert = Encoder::encode(EncoderType::hilbert, dim, idx);
                auto           sweep = Encoder::encode(EncoderType::sweep, dim, {z, y, x});

                ASSERT_EQ(hilbert_grid_16_16_16[sweep], hilbert) << dim << " " << idx << " " << hilbert;
            }
        }
    }
}

TEST(domain_space_filling_curves, hilbert_hilbert)
{
    auto run = [](Neon::domain::tool::spaceCurves::EncoderType encodingType, int dimEdge) {
        // Step 1 -> Neon backend: choosing the hardware for the computation
        Neon::init();
        // auto runtime = Neon::Runtime::openmp;
        auto runtime = Neon::Runtime::openmp;
        // We are overbooking GPU 0 three times
        std::vector<int> devIds{0};
        Neon::Backend    backend(devIds, runtime);

        // Step 2 -> Neon grid: setting up a dense cartesian domain
        Neon::index_3d dim(dimEdge, dimEdge, dimEdge);  // Size of the domain

        using Grid = Neon::eGrid;  // Selecting one of the grid provided by Neon
        Neon::domain::Stencil gradStencil([] {
            // We use a center difference scheme to compute the grad
            // The order of the points is important,
            // as we'll leverage the specific order when computing the grad.
            // First positive direction on x, y and z,
            // then negative direction on x, y, z respectively.
            return std::vector<Neon::index_3d>{
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1},
                {-1, 0, 0},
                {0, -1, 0},
                {0, 0, -1}};
        }());
        // Actual Neon grid allocation
        Grid grid(
            backend,
            dim,
            [&](const Neon::index_3d&) -> bool {
                return true;
            },  // <-  defining the active cells.
            gradStencil,
            1.0,
            0.0, encodingType);

        auto field = grid.newField<int>("spaceCode", 1, 0);

        grid.newContainer<Neon::Execution::host>("DecoceFromId",
                                                 [&](Neon::set::Loader& l) {
                                                     auto f = l.load(field);
                                                     return [=] NEON_CUDA_HOST_DEVICE(const Grid::Idx& gidx) mutable {
                                                         auto internalId = gidx.helpGet();
                                                         auto global = f.getGlobalIndex(gidx);
#pragma omp critical
                                                         {
                                                             using namespace Neon::domain::tool::spaceCurves;
                                                             auto encoded = Encoder::encode(encodingType, dim, global);
                                                             // std::cout << global << " -> internal " << internalId << " code " << encoded << std::endl;
                                                             EXPECT_EQ(internalId, encoded);
                                                         }
                                                         f(gidx, 0) = internalId;
                                                     };
                                                 })
            .run(Neon::Backend::mainStreamIdx);
        field.ioToVtk("DecoceFromId", "grad");
        printf("DONE\n");
    };
    run(Neon::domain::tool::spaceCurves::EncoderType::sweep, 32);
    run(Neon::domain::tool::spaceCurves::EncoderType::morton,32);
    run(Neon::domain::tool::spaceCurves::EncoderType::hilbert,32);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
