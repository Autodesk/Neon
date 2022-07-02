#include "gtest/gtest.h"

#include "Neon/core/types/Exceptions.h"

#include <cstring>
#include <iostream>

TEST(exceptions, noMessageNoComponet)
{

    auto runThrow = []() {
        Neon::NeonException exp;
        NEON_THROW(exp);
    };

    ASSERT_ANY_THROW(runThrow());
}

TEST(exceptions, noMessage)
{

    auto runThrow = []() {
        Neon::NeonException exp("component Name");
        NEON_THROW(exp);
    };

    ASSERT_ANY_THROW(runThrow());
}

TEST(exceptions, all)
{

    auto runThrow = []() {
        Neon::NeonException exp("component Name");
        exp << "This is the user message...";
        NEON_THROW(exp);
    };

    ASSERT_ANY_THROW(runThrow());
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
