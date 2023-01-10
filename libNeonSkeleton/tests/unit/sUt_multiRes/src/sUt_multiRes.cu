#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/domain/mGrid.h"

#include "Neon/skeleton/Options.h"
#include "Neon/skeleton/Skeleton.h"

#include "MultiResChild.h"
#include "MultiResDemo.h"
#include "MultiResMap.h"
#include "MultiResParent.h"
#include "MultiResSkeleton.h"
#include "MultiResStencil.h"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}