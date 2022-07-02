
#include <map>

#include "Neon/Neon.h"

#include "Neon/core/core.h"
#include "gUt_map.storage.h"
#include "gtest/gtest.h"
using namespace Neon;
using namespace Neon::domain;


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
