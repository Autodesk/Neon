#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/domain/bGrid.h"

TEST(bGrid, activeCell)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        int              nGPUs = 1;
        Neon::int32_3d   dim(16, 16, 16);
        std::vector<int> gpusIds(nGPUs, 0);
        auto             bk = Neon::Backend(gpusIds, Neon::Runtime::stream);

        Neon::domain::bGrid b_grid(
            bk,
            dim,
            [&](const Neon::index_3d& id) -> bool {
                if (id.x % 8 == 0 && id.y % 8 == 0 && id.z % 8 == 0 && id.x == id.y && id.y == id.z) {
                    return true;
                } else {
                    return false;
                }
            },
            Neon::domain::Stencil::s7_Laplace_t());

        auto field = b_grid.newField<float>("myField", 1, 0);

        //field.ioToVtk("f", "f");

        field.forEachActiveCell(
            [](const Neon::int32_3d id, const int card, float) {
                EXPECT_TRUE(((id.x == 0 && id.y == 0 && id.z == 0) ||
                             (id.x == 8 && id.y == 8 && id.z == 8)) &&
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
