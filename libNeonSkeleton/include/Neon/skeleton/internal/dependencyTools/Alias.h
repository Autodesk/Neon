#pragma once
#include "Neon/set/Backend.h"
#include "Neon/set/dependencyTools/Alias.h"
#include "Neon/set/dependencyTools/DataParsing.h"
#include "Neon/set/dependencyTools/enum.h"

namespace Neon::skeleton::internal {

/**
 * Index of a KernelContainer
 */
using ContainerIdx = int64_t;
using DataUId_t = Neon::set::internal::dependencyTools::DataUId_t;
using DataIdx_t = Neon::set::internal::dependencyTools::DataIdx_t;

using Dependencies_e = Neon::set::internal::dependencyTools::Dependencies_e;
using Dependencies_et = Neon::set::internal::dependencyTools::Dependencies_et;

using Access_e = Neon::set::internal::dependencyTools::Access_e;
using Access_et = Neon::set::internal::dependencyTools::Access_et;

using DataToken_t = Neon::set::internal::dependencyTools::DataToken;


using NodeId = size_t;              /** id for DiGraph */
using MetaNodeExtendedIdx = size_t; /** index for the extended node information */
using LevelIdx = int;               /** level index */

}  // namespace Neon