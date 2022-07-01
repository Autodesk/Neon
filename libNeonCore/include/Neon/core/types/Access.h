#pragma once

#include <string>
#include <iostream>

namespace Neon {

/**
 * Define an enumeration of access to resources:
 * a. read
 * b. readWrite
 */
enum class Access
{
    read /**< read access **/,
    readWrite /**< read and write access **/
};

/**
 * A set of utilities to import export Access enums
 */
struct AccessUtils
{
    /**
     * Convert an access value to string
     *
     * @param mode
     * @return
     */
    static auto toString(Access mode) -> const char*;
};


namespace metaProgramming {
template <Access access>
struct isReadOnly_type_t
{
};

template <>
struct isReadOnly_type_t<Access::read> : std::integral_constant<bool, true>
{
};
template <>
struct isReadOnly_type_t<Access::readWrite> : std::integral_constant<bool, false>
{
};

template <Access access>
struct isreadAndWrite_type_t
{
};
template <>
struct isreadAndWrite_type_t<Access::read> : std::integral_constant<bool, false>
{
};
template <>
struct isreadAndWrite_type_t<Access::readWrite> : std::integral_constant<bool, true>
{
};

}  // namespace metaProgramming
}  // END of namespace Neon

/**
 * operator<<
 *
 * @param os
 * @param m
 * @return
 */
std::ostream& operator<<(std::ostream& os, Neon::Access const& m);