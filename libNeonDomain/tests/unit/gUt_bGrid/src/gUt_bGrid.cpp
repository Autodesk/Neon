#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/domain/bGrid.h"

TEST(bGrid, activeCell)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        int              nGPUs = 1;
        Neon::int32_3d   dim(10, 10, 1);
        std::vector<int> gpusIds(nGPUs, 0);
        auto             bk = Neon::Backend(gpusIds, Neon::Runtime::stream);

        Neon::domain::bGrid b_grid(
            bk,
            dim,
            [&](const Neon::index_3d& id) -> bool {
                if ((id.x == 0 && id.y == 0) ||
                    (id.x == 4 && id.y == 4) ||
                    (id.x == 9 && id.y == 0)) {
                    return true;
                } else {
                    return false;
                }
            },
            Neon::domain::Stencil::s7_Laplace_t(), 2, 1);

        auto field = b_grid.newField<float>("myField", 1, -5);

        //field.ioToVtk("f", "f");

        field.forEachActiveCell(
            [](const Neon::int32_3d id, const int card, float) {
                EXPECT_TRUE(((id.x == 0 && id.y == 0) ||
                             (id.x == 4 && id.y == 4) ||
                             (id.x == 9 && id.y == 0)) &&
                            card == 0);
            },
            Neon::computeMode_t::computeMode_e::seq);
    }
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
