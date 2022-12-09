#pragma once
#include "Neon/set/Backend.h"

namespace Neon::set::dataDependency {


/**
 * Define type of operation for a kernel parameter: Read or write.
 * Write includes also complex operation where the parameter is both read and written.
 */
enum struct AccessType
{
    READ /**< The field or kernel parameter is in read only mode */,
    WRITE /**< The field or kernel parameter is in write mode (this include read and write) */,
    NONE /**< Not defined */

};

struct AccessTypeUtils
{
    static auto toString(AccessType val) -> std::string;
    static auto merge(AccessType valA, AccessType valB) -> AccessType;
};

}  // namespace Neon::dependency