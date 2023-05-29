
#include "Neon/Neon.h"
#include "Neon/domain/Grids.h"
#include "gtest/gtest.h"



template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
void test_backToBackConversion()
{
    using bGrid = Neon::domain::details::bGrid::bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>;
    using MicroIndex = Neon::domain::details::bGrid::MicroIndex;
    typename bGrid::Idx bIdx;
    MicroIndex          microIdx;
    uint32_t            ratioOnX = (memBlockSizeX) / (userBlockSizeX);
    uint32_t            ratioOnY = (memBlockSizeY) / (userBlockSizeY);
    uint32_t            ratioOnZ = (memBlockSizeZ) / (userBlockSizeZ);

    for (uint32_t memBlockIdx = 0; memBlockIdx < 10; memBlockIdx++) {
      const uint32_t  memBlockJump = (ratioOnX*ratioOnY*ratioOnZ)*memBlockIdx;
        for (uint32_t rZ = 0; rZ < ratioOnZ; rZ++) {
            for (uint32_t rY = 0; rY < ratioOnY; rY++) {
                for (uint32_t rX = 0; rX < ratioOnX; rX++) {
                    for (uint8_t k = 0; k < userBlockSizeX; k++) {
                        for (uint8_t j = 0; j < userBlockSizeY; j++) {
                            for (uint8_t i = 0; i < userBlockSizeZ; i++) {  // Set the micro idx to the first voxel
                                // Check that bIdx point to the first voxels too
                                microIdx.setTrayBlockIdx(memBlockJump + rX + rY * ratioOnX + rZ * ratioOnY * ratioOnX);
                                microIdx.setInTrayBlockIdx({i, j, k});
                                bIdx.init(microIdx);

                                auto res = bIdx.getMicroIndex();

                                ASSERT_EQ(bIdx.getDataBlockIdx(), memBlockIdx);
                                ASSERT_EQ(bIdx.getInDataBlockIdx(), Neon::uint8_3d(i + rX * userBlockSizeX,
                                                                                   j + rY * userBlockSizeY,
                                                                                   k + rZ * userBlockSizeZ))
                                    << bIdx.getInDataBlockIdx() << " instead of " << Neon::uint8_3d(i + rX * userBlockSizeX, j + rY * userBlockSizeY, k + rZ * userBlockSizeZ) << " with rX,Ry,rZ " << rX << "," << rY << "," << rZ << " and i,j,k = " << i << "," << j << "," << k;


                                ASSERT_EQ(res.getTrayBlockIdx(), microIdx.getTrayBlockIdx());
                                ASSERT_EQ(res.getInTrayBlockIdx(), microIdx.getInTrayBlockIdx());
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(bGrid_tray, init_4_4_4_2_2_2)
{
    test_backToBackConversion<4, 4, 4, 2, 2, 2>();
}

TEST(bGrid_tray, init_8_8_8_2_2_2)
{
    test_backToBackConversion<8, 8, 8, 2, 2, 2>();
}

TEST(bGrid_tray, init_8_8_8_1_1_1)
{
    test_backToBackConversion<8, 8, 8, 1, 1, 1>();
}

TEST(bGrid_tray, init_8_8_8_4_4_4)
{
    test_backToBackConversion<8, 8, 8, 4, 4, 4>();
}

TEST(bGrid_tray, init_4_4_4_2_1_2)
{
    test_backToBackConversion<4,4,4, 2, 1, 2>();
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
