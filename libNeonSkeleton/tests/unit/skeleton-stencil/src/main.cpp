#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/core/types/chrono.h"

#include "Neon/domain/dGrid.h"
#include "Neon/domain/tools/Geometries.h"
#include "Neon/domain/tools/TestData.h"

#include "Neon/skeleton/Options.h"
#include "Neon/skeleton/Skeleton.h"

using namespace Neon::domain::tool::testing;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}