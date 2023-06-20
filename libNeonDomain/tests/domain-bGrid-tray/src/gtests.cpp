
#include "Neon/Neon.h"
#include "Neon/domain/Grids.h"
#include "gtest/gtest.h"


template <typename SBlock>
void test_backToBackConversion()
{
    using bGrid = Neon::domain::details::bGrid::bGrid<SBlock>;
    using MicroIndex = Neon::domain::details::bGrid::MicroIndex;
    typename bGrid::Idx bIdx;
    MicroIndex          microIdx;

    for (uint32_t memBlockIdx = 0; memBlockIdx < 10; memBlockIdx++) {
        const uint32_t memBlockJump = (SBlock::blockRatioX * SBlock::blockRatioY * SBlock::blockRatioZ) * memBlockIdx;
        for (uint32_t rZ = 0; rZ < SBlock::blockRatioZ; rZ++) {
            for (uint32_t rY = 0; rY < SBlock::blockRatioY; rY++) {
                for (uint32_t rX = 0; rX < SBlock::blockRatioX; rX++) {
                    for (int8_t k = 0; k < int8_t(SBlock::userBlockSizeX); k++) {
                        for (int8_t j = 0; j < int8_t(SBlock::userBlockSizeY); j++) {
                            for (int8_t i = 0; i < int8_t(SBlock::userBlockSizeZ); i++) {  // Set the micro idx to the first voxel
                                // Check that bIdx point to the first voxels too
                                microIdx.setTrayBlockIdx(memBlockJump + rX + rY * SBlock::blockRatioX + rZ * SBlock::blockRatioY * SBlock::blockRatioX);
                                microIdx.setInTrayBlockIdx({i, j, k});
                                bIdx.init(microIdx);

                                auto res = bIdx.getMicroIndex();

                                ASSERT_EQ(bIdx.getDataBlockIdx(), memBlockIdx);
                                ASSERT_EQ(bIdx.getInDataBlockIdx(), Neon::int8_3d(static_cast<int8_t>(i + rX * SBlock::userBlockSizeX),
                                                                                  static_cast<int8_t>(j + rY * SBlock::userBlockSizeY),
                                                                                  static_cast<int8_t>(k + rZ * SBlock::userBlockSizeZ)))
                                    << bIdx.getInDataBlockIdx() << " instead of " << Neon::int8_3d(static_cast<int8_t>(i + rX * SBlock::userBlockSizeX), static_cast<int8_t>(j + rY * SBlock::userBlockSizeY), static_cast<int8_t>(k + rZ * SBlock::userBlockSizeZ)) << " with rX,Ry,rZ " << rX << "," << rY << "," << rZ << " and i,j,k = " << i << "," << j << "," << k;


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
    test_backToBackConversion<Neon::domain::details::bGrid::StaticBlock<4, 4, 4, 2, 2, 2>>();
}

TEST(bGrid_tray, init_8_8_8_2_2_2)
{
    test_backToBackConversion<Neon::domain::details::bGrid::StaticBlock<8, 8, 8, 2, 2, 2>>();
}

TEST(bGrid_tray, init_8_8_8_1_1_1)
{
    test_backToBackConversion<Neon::domain::details::bGrid::StaticBlock<8, 8, 8, 1, 1, 1>>();
}

TEST(bGrid_tray, init_8_8_8_4_4_4)
{
    test_backToBackConversion<Neon::domain::details::bGrid::StaticBlock<8, 8, 8, 4, 4, 4>>();
}

TEST(bGrid_tray, init_4_4_4_2_1_2)
{
    test_backToBackConversion<Neon::domain::details::bGrid::StaticBlock<4, 4, 4, 2, 1, 2>>();
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
