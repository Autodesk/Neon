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

        field.forEachActiveCell<Neon::computeMode_t::computeMode_e::seq>(
            [](const Neon::int32_3d id, const int card, float) {
                EXPECT_TRUE(((id.x == 0 && id.y == 0 && id.z == 0) ||
                             (id.x == 8 && id.y == 8 && id.z == 8)) &&
                            card == 0);
            });
    }
}

TEST(bGrid, multiRes)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        int              nGPUs = 1;
        Neon::int32_3d   dim(16, 16, 16);
        std::vector<int> gpusIds(nGPUs, 0);
        auto             bk = Neon::Backend(gpusIds, Neon::Runtime::stream);

        Neon::domain::internal::bGrid::bGridDescriptor descriptor({1, 1, 2});

        Neon::domain::bGrid b_grid(
            bk,
            dim,
            {[&](Neon::index_3d id) -> bool {
                 return id.x == 8 && id.y == 8 && id.z == 0;
                 //return id.norm() < 18;

                 //Link SDF https://www.shadertoy.com/view/wlXSD7
                 /*Neon::index_3d tid(id.x - dim.x / 2,
                                    id.y - dim.y / 2,
                                    id.z - dim.z / 2);

                 int le = 6;  //length
                 int r1 = 8;  //inner void
                 int r2 = 4;  //diameter

                 Neon::index_3d q(tid.x, std::max(std::abs(tid.y) - le, 0), tid.z);
                 Neon::index_2d d(q.x, q.y);
                 Neon::index_2d j(d.norm() - r1, q.z);
                 bool           is_link = (j.norm() - r2) < 0;


                 //Cut Hollow Sphere https://www.shadertoy.com/view/7tVXRt
                 tid.x = 3 * id.x - dim.x / 2;
                 tid.y = id.y - dim.y / 2;
                 tid.z = 3 * id.z - dim.z / 2;
                 int            r = 20;  //radius
                 int            h = 5;   //height
                 int            t = 2;   //thickness
                 int            w = static_cast<int>(std::sqrt(r * r - h * h));
                 Neon::index_2d xz(tid.x, tid.z);
                 Neon::index_2d v(xz.norm(), tid.y);
                 Neon::index_2d s(w, h);
                 bool           is_cut_sphere = (((h * v.x < w * v.y) ? (v - s).norm() : abs(v.norm() - r)) - t) < 0;

                 return is_link || is_cut_sphere;*/
             },
             [&](const Neon::index_3d&) -> bool {
                 return false;
             },
             [&](const Neon::index_3d& id) -> bool {
                 return id.x == 4 && id.y == 4 && id.z == 0;
                 //return false;
             }},
            Neon::domain::Stencil::s7_Laplace_t(),
            descriptor);

        b_grid.topologyToVTK("bGrid112.vtk");

        //auto field = b_grid.newField<float>("myField", 1, 0);

        //field.ioToVtk("f", "f");

        //field.forEachActiveCell<Neon::computeMode_t::computeMode_e::seq>(
        //    [](const Neon::int32_3d id, const int card, float) {
        //        EXPECT_TRUE(((id.x == 0 && id.y == 0 && id.z == 0) ||
        //                     (id.x == 8 && id.y == 8 && id.z == 8)) &&
        //                    card == 0);
        //    });
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
