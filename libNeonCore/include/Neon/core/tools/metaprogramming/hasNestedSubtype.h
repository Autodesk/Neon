#pragma once

#include <type_traits>
#include "Neon/core/types/macros.h"

namespace Neon {
namespace meta {

namespace internal {
namespace hasNestedType {
template <class T>
struct Void
{
    typedef void type;
};
}  // namespace hasNestedType
}  // namespace internal
// namespace internal
template <class T, class U = void>
struct HasNestedType_t
{
    enum
    {
        value = 0
    };
};

template <class T>
struct HasNestedType_t<T, typename internal::hasNestedType::Void<typename T::bar>::type>
{
    enum
    {
        value = 1
    };
};

}  // namespace meta
}  // namespace Neon
