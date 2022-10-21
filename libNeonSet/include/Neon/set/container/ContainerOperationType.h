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
    halo = 2 /**< Halo update container **/,
    sync = 3 /**< Synchronization Container */,
    anchor = 4 /**< Synchronization Container: begin or end */
};


struct ContainerOperationTypeUtils
{
    static constexpr int nOptions = 5;

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

