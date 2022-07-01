#include "gtest/gtest.h"

#include "Neon/core/tools/Logger.h"


TEST(logging, trace)
{
    NEON_TRACE("test trace message ...");
}

TEST(logging, info)
{
    NEON_INFO("test info message ...");
}

TEST(logging, warning)
{
    NEON_WARNING("test warning message ...");
}

TEST(logging, error)
{
    NEON_ERROR("test error message ...");
}

TEST(logging, critical)
{
    NEON_CRITICAL("test critical message ...");
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}