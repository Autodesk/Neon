#pragma once
#include "Neon/set/Backend.h"

namespace Neon {
namespace set {
namespace internal {
namespace dependencyTools {

/**
 * Define type of operation for a kernel parameter: Read or write.
 * Write includes also complex operation where the parameter is both read and written.
 */
struct Access_et
{
    Access_et() = delete;
    Access_et(const Access_et&) = delete;

    enum e
    {
        READ /**< The field or kernel parameter is in read only mode */,
        WRITE /**< The field or kernel parameter is in write mode (this include read and write) */,
        NONE /**< Not defined */
    };
    static auto toString(e val) -> std::string;
    static auto merge(e valA, e valB) -> e;

};
using Access_e = Access_et::e;

/**
 * Classical definition of dependency types.
 */
struct Dependencies_et
{
    Dependencies_et() = delete;
    Dependencies_et(const Dependencies_et&) = delete;

    enum e
    {
        RAW /**< Read after write */,
        WAR /**< Write after read */,
        RAR /**< Read after read. This is not actually a true dependency */,
        WAW /**< This configuration may indicate something wrong in the user graph */,
        NONE /**< Not defined */
    };
    static auto toString(e val) -> std::string;
};
using Dependencies_e = Dependencies_et::e;
}  // namespace dependencyTools
}  // namespace internal
}  // namespace set

/**
 * Enumeration for the supported type of computation by the skeleton
 * */
enum struct Compute
{
    MAP /**< Map operation */,
    STENCIL /**< Stencil operation */,
    REDUCE /**< Reduction operation */
};

struct ComputeUtils
{
    /**
     * Returns a string for the selected allocator
     *
     * @param allocator
     * @return
     */
    static auto toString(Compute allocator) -> const char*;
};

}  // namespace Neon