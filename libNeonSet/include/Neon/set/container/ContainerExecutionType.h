#pragma once

#include "Neon/set/DevSet.h"
#include "Neon/set/dependency/Token.h"
#include "functional"
#include "type_traits"

/**
 * Abstract interface to hide
 */

namespace Neon::set {

enum struct ContainerExecutionType
{
    device = 0 /** the operation of the containers are only for the device (note: device can be CPU too) */,
    deviceManaged = 1 /** manage version of the device type of Container, i.e. the launch is managed by the container itself. Useful to wrap calls to cuBlas operation for example*/,
    deviceThenHostManaged = 2, /** a container that stores operation on both device and host. For this type of Container a getHostContainer method is enabled to retrieved a container with the host code */
    hostManaged = 3 /** host managed container */,
    graph = 4 /** A complex container */,
    none = 5
};


struct ContainerExecutionTypeUtils
{
    static constexpr int nOptions = 5;

    static auto toString(ContainerExecutionType option) -> std::string;
    static auto fromString(const std::string& option) -> ContainerExecutionType;
    static auto getOptions() -> std::array<ContainerExecutionType, nOptions>;
    static auto isExpandable(ContainerExecutionType option) -> bool;
};

/**
 * operator<<
 */
std::ostream& operator<<(std::ostream& os, Neon::set::ContainerExecutionType const& m);
}  // namespace Neon::set
