#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/domain/mGrid.h"

TEST(mGrid, multiRes)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        int              nGPUs = 1;
        Neon::int32_3d   dim(16, 16, 16);
        std::vector<int> gpusIds(nGPUs, 0);
        auto             bk = Neon::Backend(gpusIds, Neon::Runtime::stream);

        Neon::mGridDescriptor descr = Neon::mGridOctreeDescr;


        Neon::domain::mGrid grid(
            bk,
            dim,
            {[&](Neon::index_3d id) -> bool {
                 return id.x == 8 && id.y == 8 && id.z == 0;
             },
             [&](const Neon::index_3d&) -> bool {
                 return false;
             },
             [&](const Neon::index_3d& id) -> bool {
                 return id.x == 4 && id.y == 4 && id.z == 0;
             }},
            Neon::domain::Stencil::s7_Laplace_t(),
            descr);

        auto field = grid.newField<float>("myField", 1, 0);

        for (int l = 0; l < descr.getDepth(); ++l) {
            field.forEachActiveCell(
                l,
                [&]([[maybe_unused]] const Neon::int32_3d idx, const int /*card*/, float& val) {
                    val = 20 + float(l);
                },
                Neon::computeMode_t::computeMode_e::seq);
        }

        field.ioToVtk("f");
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
