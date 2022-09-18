#pragma once
#include "Neon/set/Backend.h"
#include "Neon/set/MultiDeviceObjectUid.h"

namespace Neon::internal::dataDependency {

/**
 * Unique identifier for a kernel parameter
 */
using DataUId = Neon::set::MultiDeviceObjectUid;

/**
 * Unique identifier for a kernel parameter
 */
using DataIdx = Neon::set::MultiDeviceObjectUid;

}  // namespace Neon::internal::dataDependency
