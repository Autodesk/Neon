#pragma once

#include "Neon/core/core.h"

namespace Neon {
namespace domain {

template <typename Type>
struct NghData
{
    Type mData;
    bool mIsValid;
    NghData() = default;
    NEON_CUDA_HOST_DEVICE NghData(const Type& val, bool status)
    {
        this->mData = val;
        this->mIsValid = status;
    }
    NEON_CUDA_HOST_DEVICE void set(const Type& val_, bool status_)
    {
        this->mData = val_;
        this->mIsValid = status_;
    }
    auto isValid() const -> bool
    {
        return mIsValid;
    }

    auto getData() -> Type&
    {
        return mData;
    }

    auto getData() const -> const Type&
    {
        return mData;
    }
};
}  // namespace domain
}  // namespace Neon