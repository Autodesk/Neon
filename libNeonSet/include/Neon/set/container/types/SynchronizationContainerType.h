#pragma once

#include "Neon/set/DevSet.h"
#include "functional"
#include "type_traits"

/**
 * Abstract interface to hide
 */

namespace Neon::set {

enum struct SynchronizationContainerType
{
    hostOmpBarrier = 0,
    none = 6
};


struct SynchronizationContainerTypeUtils
{
    static constexpr int nOptions = 1;

    static auto toString(SynchronizationContainerType option)
        -> std::string;

    static auto fromString(const std::string& option)
        -> SynchronizationContainerType;

    static auto getOptions()
        -> std::array<SynchronizationContainerType, nOptions>;
};

/**
 * operator<<
 */
std::ostream& operator<<(std::ostream& os, Neon::set::SynchronizationContainerType const& m);
}  // namespace Neon::set
