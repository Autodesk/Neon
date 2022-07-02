#pragma once

#include <type_traits>
#include "Neon/core/types/Macros.h"

namespace Neon {
namespace meta {


namespace debug {
/**
 * This function will create a compile time error.
 * Withing the error the type of the object will be printed.
 * @tparam T_ta
 */
template <typename T_ta>
void printType(T_ta& obj)
{
    bool NEON_ATTRIBUTE_UNUSED x;
    x = decltype(obj)::NeonNonExistingFunciton;
}
template <typename T_ta>
void printType()
{
    auto x = T_ta::NeonNonExistingFunciton;
    printf("%p", &x);
}

}  // namespace debug

namespace assert {


namespace privateSup {

template <typename A_ta, typename B_ta, bool>
struct CompareObjType
{
};


template <typename A_ta, typename B_ta>
struct CompareObjType<A_ta, B_ta, false>
{
    CompareObjType(A_ta&, B_ta&)
    {
        B_ta::CompileErrorMessage;
    }
};
template <typename A_ta, typename B_ta>
struct CompareObjType<A_ta, B_ta, true>
{
    CompareObjType(A_ta&, B_ta&)
    {
        /* nothing */
    }
};
}  // End of namespace privateSup

template <typename A_ta, typename B_ta>
void CompareObjType(A_ta&, B_ta&)
{
    using A_taReferenceRemoved = std::remove_reference<A_ta>;
    using B_taReferenceRemoved = std::remove_reference<B_ta>;
    privateSup::CompareObjType<A_ta, B_ta, std::is_same<A_taReferenceRemoved, B_taReferenceRemoved>::value> trash;
}


}  // namespace assert


}  // namespace meta
}  // namespace Neon
