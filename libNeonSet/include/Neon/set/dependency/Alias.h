#pragma once
#include "Neon/set/Backend.h"
#include "Neon/set/MultiDeviceObjectUid.h"

namespace Neon::internal::dataDependency {

/**
 * Unique identifier for a kernel parameter
 */
using MdObjUid = Neon::set::MultiDeviceObjectUid;

/**
 * Unique identifier for a kernel parameter
 */
using MdObjIdx = Neon::set::MultiDeviceObjectUid;

}  // namespace Neon::internal::dataDependency
