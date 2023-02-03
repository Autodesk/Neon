#pragma once

#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>

#include "Neon/Report.h"
#include "Neon/core/core.h"
#include "Neon/set/MemoryOptions.h"
//#include "Neon/core/types/mode.h"
//#include "Neon/core/types/devType.h"

namespace Neon {

enum struct Execution
{
    seq = 0,
    par = 0,
};

}  // namespace Neon::set
