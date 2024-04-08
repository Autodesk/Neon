
#include <Neon/core/types/vec.h>

#include "Neon/core/core.h"
#include "gtest/gtest.h"

#include <cstring>
#include <iostream>


namespace tuple3dTests {
template <typename T_ta>
void constructor()
{
    for (T_ta z = 0; z < (T_ta)10; z++) {
        Neon::Vec_3d<T_ta> a(z);
        ASSERT_EQ(a.x, z);
        ASSERT_EQ(a.y, z);
        ASSERT_EQ(a.z, z);

        for (T_ta y = 0; y < (T_ta)10; y++) {
            for (T_ta x = 0; x < (T_ta)10; x++) {
                Neon::Vec_3d<T_ta> p(x, y, z);
                ASSERT_EQ(p.x, p.getVectorView()[0]);
                ASSERT_EQ(p.y, p.getVectorView()[1]);
                ASSERT_EQ(p.z, p.getVectorView()[2]);

                ASSERT_EQ(p.x, x);
                ASSERT_EQ(p.y, y);
                ASSERT_EQ(p.z, z);


                Neon::Vec_3d<T_ta> k(p);
                ASSERT_EQ(p.x, p.x);
                ASSERT_EQ(p.y, p.y);
                ASSERT_EQ(p.z, p.z);

                Neon::Vec_3d<T_ta> l(z);
                l = p;
                ASSERT_EQ(p.x, l.x);
                ASSERT_EQ(p.y, l.y);
                ASSERT_EQ(p.z, l.z);
            }
        }
    }
}
}  // END of namespace tuple3dTests

TEST(Vec_3d, constructor)
{
    tuple3dTests::constructor<int32_t>();
    tuple3dTests::constructor<int64_t>();
    tuple3dTests::constructor<float>();
    tuple3dTests::constructor<double>();
    tuple3dTests::constructor<char>();
}

namespace tuple3dTests {
template <typename T_ta>
void sum()
{
    Neon::Vec_2d<int> a;

    for (T_ta z = 0; z < (T_ta)10; z++) {
        for (T_ta y = 0; y < (T_ta)10; y++) {
            for (T_ta x = 0; x < (T_ta)10; x++) {
                {  // SUM  and SUB
                    const Neon::Vec_3d<T_ta> a(x, y, z);
                    const Neon::Vec_3d<T_ta> b(y, z, x);

                    Neon::Vec_3d<T_ta> p = a + a;

                    ASSERT_EQ(p.x, a.x + a.x);
                    ASSERT_EQ(p.y, a.y + a.y);
                    ASSERT_EQ(p.z, a.z + a.z);

                    p = a + b;

                    ASSERT_EQ(p.x, b.x + a.x);
                    ASSERT_EQ(p.y, b.y + a.y);
                    ASSERT_EQ(p.z, b.z + a.z);

                    p = a - a;
                    ASSERT_EQ(p.x, 0);
                    ASSERT_EQ(p.y, 0);
                    ASSERT_EQ(p.z, 0);

                    p = a - b;
                    ASSERT_EQ(p.x, a.x - b.x);
                    ASSERT_EQ(p.y, a.y - b.y);
                    ASSERT_EQ(p.z, a.z - b.z);


                    p = a + b.z;
                    ASSERT_EQ(p.x, b.z + a.x);
                    ASSERT_EQ(p.y, b.z + a.y);
                    ASSERT_EQ(p.z, b.z + a.z);

                    p = a - b.z;
                    ASSERT_EQ(p.x, a.x - b.z);
                    ASSERT_EQ(p.y, a.y - b.z);
                    ASSERT_EQ(p.z, a.z - b.z);


                }
                {  // MUL DIV
                    const Neon::Vec_3d<T_ta> a(x, y, z);
                    const Neon::Vec_3d<T_ta> b(y, z, x);

                    Neon::Vec_3d<T_ta> p = a * a;

                    ASSERT_EQ(p.x, a.x * a.x);
                    ASSERT_EQ(p.y, a.y * a.y);
                    ASSERT_EQ(p.z, a.z * a.z);

                    p = a * b;

                    ASSERT_EQ(p.x, b.x * a.x);
                    ASSERT_EQ(p.y, b.y * a.y);
                    ASSERT_EQ(p.z, b.z * a.z);

                    p = a * b.z;
                    ASSERT_EQ(p.x, b.z * a.x);
                    ASSERT_EQ(p.y, b.z * a.y);
                    ASSERT_EQ(p.z, b.z * a.z);

                    if (b.rMul() != 0) {
                        p = b / b;
                        ASSERT_EQ(p.x, 1);
                        ASSERT_EQ(p.y, 1);
                        ASSERT_EQ(p.z, 1);

                        p = a / b;
                        ASSERT_EQ(p.x, a.x / b.x);
                        ASSERT_EQ(p.y, a.y / b.y);
                        ASSERT_EQ(p.z, a.z / b.z);

                        p = a / b.z;
                        ASSERT_EQ(p.x, a.x / b.z);
                        ASSERT_EQ(p.y, a.y / b.z);
                        ASSERT_EQ(p.z, a.z / b.z);
                    }
                }
                {
                    const Neon::Vec_3d<T_ta> a(x, y, z);
                    const Neon::Vec_3d<T_ta> b(y, z, x);

                    Neon::Vec_3d<T_ta> c = T_ta(0);
                    c += a;
                    ASSERT_TRUE(c == a) << c << " " << a;

                    c = 1;
                    c *= a;
                    ASSERT_TRUE(c == a) << c << " " << a;
                    ;
                }
            }
        }
    }
}
}  // namespace tuple3dTests

TEST(Vec_3d, algebra)
{
    tuple3dTests::sum<int32_t>();
    tuple3dTests::sum<int64_t>();
    tuple3dTests::sum<float>();
    tuple3dTests::sum<double>();
    tuple3dTests::sum<char>();
}

TEST(Vec_3d, getFunctions)
{
    using namespace Neon;
    const Neon::Vec_3d<index_t> blockDim(8);

    for (int z = 8; z < 3 * 20 + 8; z += 3) {
        for (int y = 8; y < 3 * 20 + 8; y += 3) {
            for (int x = 8; x < 3 * 20 + 8; x += 3) {
                {
                    Neon::Vec_3d<Neon::index_t> gridDim(x, y, z);
                    auto                        blockGridDim = gridDim.cudaGridDim(blockDim);
                    ASSERT_TRUE(blockGridDim * blockDim >= gridDim) << blockGridDim * blockDim << " >= " << gridDim;
                    ASSERT_TRUE((blockGridDim - 1) * blockDim < gridDim) << blockGridDim * blockDim << " < " << gridDim;
                }

                for (int a = z * 3; a < 3 * 20 + 8; a += 3) {
                    for (int b = y * 3; b < 3 * 20 + 8; b += 3) {
                        for (int c = x * 3; c < 3 * 20 + 8; c += 3) {
                            Vec_3d<index_t> gridDim(c, b, a);
                            Vec_3d<index_t> index(x, y, z);

                            size_t pitch = index.mPitch(gridDim);
                            size_t computedPitch = index.x + index.y * (size_t)gridDim.x + index.z * (size_t)gridDim.x * (size_t)gridDim.y;
                            ASSERT_TRUE(pitch == computedPitch);
                        }
                    }
                }
            }
        }
    }
}

namespace tuple3dTests {
template <typename T_ta>
void reduce()
{
    for (T_ta z = 0; z < (T_ta)10; z++) {
        for (T_ta y = 0; y < (T_ta)10; y++) {
            for (T_ta x = 0; x < (T_ta)10; x++) {
                {  // Reduce
                    const Neon::Vec_3d<T_ta> a(x, y, z);

                    T_ta p = a.rMul();
                    ASSERT_EQ(p, a.x * a.y * a.z);
                    p = a.rSum();
                    ASSERT_EQ(p, a.x + a.y + a.z);

                    p = a.rMin();
                    ASSERT_EQ(p, std::min(a.x, std::min(a.y, a.z)));

                    p = a.rMax();
                    ASSERT_EQ(p, std::max(a.x, std::max(a.y, a.z)));

                }
                {  // mul Reduce
                    const Neon::Vec_3d<T_ta> a(x, y, z);
                    const Neon::Vec_3d<T_ta> b(y, z, x);

                    //                                T_ta p = Neon::Vec_3d<T_ta>::mulReduceSum(a, b);
                    //                                auto c = a * b;
                    //                                ASSERT_EQ(c.reduceSum(), p);
                }
            }
        }
    }
}
}  // namespace tuple3dTests

TEST(Vec_3d, reduce)
{
    tuple3dTests::reduce<int32_t>();
    tuple3dTests::reduce<int64_t>();
    tuple3dTests::reduce<float>();
    tuple3dTests::reduce<double>();
}

namespace tuple3dTests {
struct userType_t
{
    int    a;
    int    b;
    double k;
};

void userTypes()
{
    Neon::Vec_3d<userType_t> userP3d;
}
}  // namespace tuple3dTests

TEST(Vec_3d, userTypes)
{
    tuple3dTests::userTypes();
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
