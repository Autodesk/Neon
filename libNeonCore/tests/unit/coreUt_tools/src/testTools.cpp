
#include "gtest/gtest.h"

#include "Neon/core/core.h"
#include "Neon/core/tools/metaprogramming.h"

#include <cstring>
#include <iostream>
#include <vector>

namespace applyTupleTest {
int fun(int a, std::vector<double> b, float c)
{
    for (auto&& el : b) {
        el = a - int(c);
    }
    return 33;
}
}  // namespace applyTupleTest

TEST(tools, applyTuple)
{
    std::vector<double> vals = {1, 2, 3, 4};
    auto                input = make_tuple(55, vals, float(3.0));
    auto                retval = Neon::meta::computeFunOnTuple(applyTupleTest::fun, input);

    ASSERT_TRUE(retval == 33);
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
