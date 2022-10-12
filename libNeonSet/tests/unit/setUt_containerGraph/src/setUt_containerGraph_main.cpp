#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include <map>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
