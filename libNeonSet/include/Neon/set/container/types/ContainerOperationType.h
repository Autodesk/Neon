#pragma once

#include "Neon/set/DevSet.h"
#include "functional"
#include "type_traits"

/**
 * Abstract interface to hide
 */

namespace Neon::set {

enum struct ContainerOperationType
{
    compute = 0 /**< Compute container, can be on host or device */,
    graph = 1 /**<  A graph based container */,
    sequence = 2,
    communication = 3 /**< Halo update container **/,
    synchronization = 4 /**< Synchronization Container */,
    anchor = 5 /**< Synchronization Container: begin or end */
};


struct ContainerOperationTypeUtils
{
    static constexpr int nOptions = 6;

    /**
     * Convert type to string
     */
    static auto toString(ContainerOperationType option) -> std::string;

    /**
     * Returns the type associated to a string
     */
    static auto fromString(const std::string& option) -> ContainerOperationType;

    /**
     * Return available options
     */
    static auto getOptions() -> std::array<ContainerOperationType, nOptions>;
};

/**
 * operator<<
 */
std::ostream& operator<<(std::ostream& os, Neon::set::ContainerOperationType const& m);

}  // namespace Neon::set
