
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
                auto           sweep = Encoder::encode(EncoderType::sweep, dim, {z,y,x});

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
                auto           sweep = Encoder::encode(EncoderType::sweep, dim, {z,y,x});

                ASSERT_EQ(hilbert_grid_16_16_16[sweep], hilbert) << dim << " " << idx << " " << hilbert;
            }
        }
    }
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
