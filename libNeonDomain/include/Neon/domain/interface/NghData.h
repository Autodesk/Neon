#pragma once

#include "Neon/core/core.h"

namespace Neon {
namespace domain {

template <typename Type>
struct NghData
{
    Type                  mData;
    bool                  mIsValid;
    NEON_CUDA_HOST_DEVICE NghData(bool status = false)
    {
        this->mIsValid = false;
    }

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

    NEON_CUDA_HOST_DEVICE void invalidate()
    {
        this->mIsValid = false;
    }

    NEON_CUDA_HOST_DEVICE auto isValid() const -> bool
    {
        return mIsValid;
    }

    NEON_CUDA_HOST_DEVICE auto getData() const -> const Type&
    {
        return mData;
    }

    NEON_CUDA_HOST_DEVICE auto operator()() const -> const Type&
    {
        return mData;
    }
};
}  // namespace domain
}  // namespace Neon