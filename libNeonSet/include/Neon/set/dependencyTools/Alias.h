#pragma once
#include "Neon/set/Backend.h"

namespace Neon {
namespace set {
namespace internal {
namespace dependencyTools {

/**
 * Unique identifier for a kernel parameter
 */
using DataUId_t = int64_t;

/**
 * Unique identifier for a kernel parameter
 */
using DataIdx_t = int64_t;

/**
 * Index of a KernelContainer
 */
using KernelContainerIdx_t = int64_t;

}  // namespace dependencyTools
}  // namespace internal
}  // namespace set
}  // namespace Neon