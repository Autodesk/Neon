#pragma once

#include "Neon/core/core.h"

namespace Neon {
namespace domain {

template <typename T_ta>
struct NghInfo
{
    T_ta value;
    bool isValid;
    NghInfo() = default;
    NEON_CUDA_HOST_DEVICE NghInfo(const T_ta& val, bool status)
    {
        this->value = val;
        this->isValid = status;
    }
    NEON_CUDA_HOST_DEVICE void set(const T_ta& val_, bool status_)
    {
        this->value = val_;
        this->isValid = status_;
    }
};
}  // namespace grids
}  // namespace Neon