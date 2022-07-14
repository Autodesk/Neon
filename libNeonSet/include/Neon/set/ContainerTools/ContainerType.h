#pragma once

#include "Neon/set/DevSet.h"
#include "Neon/set/dependencyTools/DataParsing.h"
#include "functional"
#include "type_traits"

/**
 * Abstract interface to hide
 */

namespace Neon::set::internal {

enum struct ContainerType
{
    device = 0 /** the operation of the containers are only for the device (note: device can be CPU too) */,
    deviceManaged = 1 /** manage version of the device type of Container, i.e. the launch is managed by the container itself. Useful to wrap calls to cuBlas operation for example*/,
    deviceThenHostManaged = 2, /** a container that stores operation on both device and host. For this type of Container a getHostContainer method is enabled to retrieved a container with the host code */
    hostManaged = 3
};


struct ContainerTypeUtils
{
    static constexpr int nOptions = 3;

    static auto toString(ContainerType option) -> std::string;
    static auto fromString(const std::string& option) -> ContainerType;
    static auto getOptions() -> std::array<ContainerType, nOptions>;
    static auto isExpandable(ContainerType option)  -> bool;
};


}  // namespace Neon
