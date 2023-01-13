#pragma once
#include "Neon/set/Backend.h"

namespace Neon::set::dataDependency {

/**
 * Classical definition of dependency types.
 */
enum struct DataDependencyType
{
    RAW /**< Read after write */,
    WAR /**< Write after read */,
    RAR /**< Read after read. This is not actually a true dependency */,
    WAW /**< This configuration may indicate something wrong in the user graph */,
    NONE /**< Not defined */
};

struct DataDependencyTypeUtils
{
    /**
     * Returns a string for the selected allocator
     *
     * @param allocator
     * @return
     */
    static auto toString(DataDependencyType type) -> std::string;
};

std::ostream& operator<<(std::ostream& os, Neon::set::dataDependency::DataDependencyType const& m);
}
