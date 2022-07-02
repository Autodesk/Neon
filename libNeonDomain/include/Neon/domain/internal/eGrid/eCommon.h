#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::internal::eGrid {


using count_t = int32_t;
using eIdx_t = int32_t;
using ePitch_t = Neon::index64_2d;

struct ComDirection_e
{
    enum e
    {
        COM_DW = 0,
        COM_UP = 1,
        COM_NUM = 2,
    };
};

struct BdrDepClass_e
{
    enum e
    {
        DW = ComDirection_e::COM_DW,
        UP = ComDirection_e::COM_UP,
        BOTH = ComDirection_e::COM_NUM,
        num = ComDirection_e::COM_NUM + 1
    };
};

struct indexing_e
{
    enum e
    {
        standard = 0,
        internal = 1,
        boundary = 2,        
        num = 3
    };
};

}  // namespace eGrid

