#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/domain/mGrid.h"

TEST(mGrid, multiRes)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        int              nGPUs = 1;
        Neon::int32_3d   dim(32, 32, 32);
        std::vector<int> gpusIds(nGPUs, 0);
        auto             bk = Neon::Backend(gpusIds, Neon::Runtime::stream);

        Neon::mGridDescriptor<1> descr(3);


        Neon::domain::mGrid grid(
            bk,
            dim,
            {[&](Neon::index_3d id) -> bool {
                 return (id.x == 8 && id.y == 8 && id.z == 0) || (id.x == 24 && id.y == 24 && id.z == 0);
             },
             [&](const Neon::index_3d&) -> bool {
                 return false;
             },
             [&](const Neon::index_3d& id) -> bool {
                 return (id.x == 4 && id.y == 4 && id.z == 0) || (id.x == 20 && id.y == 20 && id.z == 0);
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

        field.ioToVtk("f", true, true, true, false);
    }
}

TEST(mGrid, checkpoint_roundtrip)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        int              nGPUs = 1;
        Neon::int32_3d   dim(32, 32, 32);
        std::vector<int> gpusIds(nGPUs, 0);
        auto             bk = Neon::Backend(gpusIds, Neon::Runtime::stream);

        Neon::mGridDescriptor<1> descr(2);

        Neon::domain::mGrid grid(
            bk,
            dim,
            {[&](Neon::index_3d id) -> bool {
                 return id.x < 16 && id.y < 16 && id.z < 8;
             },
             [&](Neon::index_3d) -> bool {
                 return true;
             }},
            Neon::domain::Stencil::s7_Laplace_t(),
            descr);

        auto field = grid.newField<float>("rho", 1, 0.f);
        for (int l = 0; l < descr.getDepth(); ++l) {
            field.forEachActiveCell(
                l,
                [&](const Neon::int32_3d idx, const int /*card*/, float& val) {
                    val = static_cast<float>(idx.x + idx.y + idx.z + l * 100);
                },
                true,
                Neon::computeMode_t::computeMode_e::seq);
        }

        const std::string topoPath = "gUt_mGrid_topology.ngrid";
        grid.ioToTopologyFile(topoPath);
        auto restored = Neon::domain::mGrid::fromTopologyFile(bk, topoPath);

        using Neon::domain::details::mGrid::CheckpointFormat;
        for (auto [format, suffix] : {std::pair{CheckpointFormat::bin, "bin"},
                                      std::pair{CheckpointFormat::npy, "npy"},
                                      std::pair{CheckpointFormat::npz, "npz"}}) {
            SCOPED_TRACE(suffix);
            const std::string fieldBase = std::string("gUt_mGrid_rho_") + suffix;
            field.ioToCheckpoint(fieldBase, "rho", format);

            auto restoredField = restored.newField<float>("rho", 1, 0.f);
            restoredField.ioFromCheckpoint(fieldBase);

            for (int l = 0; l < descr.getDepth(); ++l) {
                field.forEachActiveCell(
                    l,
                    [&](const Neon::int32_3d idx, const int card, float& val) {
                        (void)card;
                        const float expected = static_cast<float>(idx.x + idx.y + idx.z + l * 100);
                        EXPECT_FLOAT_EQ(val, expected);
                        const float restoredVal = restoredField(idx, 0, l);
                        EXPECT_FLOAT_EQ(restoredVal, expected);
                    },
                    true,
                    Neon::computeMode_t::computeMode_e::seq);
            }
        }
    }
}

TEST(mGrid, host_access_by_coordinate)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        int              nGPUs = 1;
        Neon::int32_3d   dim(32, 32, 32);
        std::vector<int> gpusIds(nGPUs, 0);
        auto             bk = Neon::Backend(gpusIds, Neon::Runtime::stream);

        Neon::mGridDescriptor<1> descr(3);

        Neon::domain::mGrid grid(
            bk,
            dim,
            {[&](Neon::index_3d id) -> bool {
                 return id.x < 8 && id.y < 8 && id.z < 8;
             },
             [&](Neon::index_3d id) -> bool {
                 return id.x < 16 && id.y < 16 && id.z < 16;
             },
             [&](Neon::index_3d) -> bool {
                 return true;
             }},
            Neon::domain::Stencil::s7_Laplace_t(),
            descr);

        auto field = grid.newField<float>("f", 2, 0.f);
        auto value = [](const Neon::index_3d& idx, int c, int l, float offset) {
            return static_cast<float>(idx.x + 32 * idx.y + 1024 * idx.z + 100000 * l + 0.5 * c) + offset;
        };

        for (int l = 0; l < descr.getDepth(); ++l) {
            SCOPED_TRACE(l);
            int numCells = 0;
            field.forEachActiveCell(
                l,
                [&](const Neon::int32_3d idx, const int c, float& val) {
                    val = value(idx, c, l, 0.f);
                    ++numCells;
                },
                true,
                Neon::computeMode_t::computeMode_e::seq);
            ASSERT_GT(numCells, 0);

            const auto& constField = field;
            field.forEachActiveCell(
                l,
                [&](const Neon::int32_3d idx, const int c, float&) {
                    EXPECT_FLOAT_EQ(field(idx, c, l), value(idx, c, l, 0.f)) << idx;
                    EXPECT_FLOAT_EQ(constField(idx, c, l), value(idx, c, l, 0.f)) << idx;
                    field(idx, c, l) = value(idx, c, l, 0.25f);
                },
                true,
                Neon::computeMode_t::computeMode_e::seq);

            field.forEachActiveCell(
                l,
                [&](const Neon::int32_3d idx, const int c, float& val) {
                    EXPECT_FLOAT_EQ(val, value(idx, c, l, 0.25f)) << idx;
                },
                true,
                Neon::computeMode_t::computeMode_e::seq);
        }
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
