
#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/core/types/vec.h"

#include "Neon/sys/global/CpuSysGlobal.h"
#include "Neon/sys/global/GpuSysGlobal.h"

#include "Neon/sys/memory/mem3d.h"

#include <cstring>
#include <iostream>

#include "sysUt_mem3d.h"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
