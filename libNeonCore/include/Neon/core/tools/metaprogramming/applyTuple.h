#pragma once
#include <stdint.h>
#include <iostream>
#include <tuple>
/**
 * this function contains some template function to call a function
 * when the parameters are actually packed in a tuple.
 *
 * Part of the code has been inspired by the following post on stackoverflow.com:
 * https://stackoverflow.com/questions/687490/how-do-i-expand-a-tuple-into-variadic-template-functions-arguments
 *
 * The solution in stackoverflow.com has been extended to handle also the return type.
 * */
namespace Neon {
namespace meta {
/**
 * Object Function Tuple Argument Unpacking
 *
 * This recursive template unpacks the tuple parameters into
 * variadic template arguments until we reach the count of 0 where the function
 * is called with the correct parameters
 *
 * @tparam N Number of tuple arguments to unroll
 *
 * @ingroup g_util_tuple
 */
template <uint32_t N>
struct apply_obj_func
{
    template <typename T, typename... ArgsF, typename... ArgsT, typename... Args>
    static void applyTuple(T* pObj,
                           void (T::*f)(ArgsF...),
                           const std::tuple<ArgsT...>& t,
                           Args... args)
    {
        apply_obj_func<N - 1>::applyTuple(pObj, f, t, std::get<N - 1>(t), args...);
    }
};

//-----------------------------------------------------------------------------

/**
 * Object Function Tuple Argument Unpacking End Point
 *
 * This recursive template unpacks the tuple parameters into
 * variadic template arguments until we reach the count of 0 where the function
 * is called with the correct parameters
 *
 * @ingroup g_util_tuple
 */
template <>
struct apply_obj_func<0>
{
    template <typename T, typename... ArgsF, typename... ArgsT, typename... Args>
    static void applyTuple(T* pObj,
                           void (T::*f)(ArgsF...),
                           const std::tuple<ArgsT...>& /* t */,
                           Args... args)
    {
        (pObj->*f)(args...);
    }
};

//-----------------------------------------------------------------------------

/**
 * Object Function Call Forwarding Using Tuple Pack Parameters
 */
// Actual apply function
template <typename T, typename... ArgsF, typename... ArgsT>
void applyTuple(T* pObj,
                void (T::*f)(ArgsF...),
                std::tuple<ArgsT...> const& t)
{
    apply_obj_func<sizeof...(ArgsT)>::applyTuple(pObj, f, t);
}

//-----------------------------------------------------------------------------

/**
 * Static Function Tuple Argument Unpacking
 *
 * This recursive template unpacks the tuple parameters into
 * variadic template arguments until we reach the count of 0 where the function
 * is called with the correct parameters
 *
 * @tparam N Number of tuple arguments to unroll
 *
 * @ingroup g_util_tuple
 */
template <uint32_t N>
struct apply_func
{
    template <typename retVal, typename... ArgsF, typename... ArgsT, typename... Args>
    static retVal applyTuple(retVal (*f)(ArgsF...),
                             const std::tuple<ArgsT...>& t,
                             Args... args)
    {
        return apply_func<N - 1>::applyTuple(f, t, std::get<N - 1>(t), args...);
    }
};

//-----------------------------------------------------------------------------

/**
 * Static Function Tuple Argument Unpacking End Point
 *
 * This recursive template unpacks the tuple parameters into
 * variadic template arguments until we reach the count of 0 where the function
 * is called with the correct parameters
 *
 * @ingroup g_util_tuple
 */
template <>
struct apply_func<0>
{
    template <typename retVal, typename... ArgsF, typename... ArgsT, typename... Args>
    static retVal applyTuple(retVal (*f)(ArgsF...),
                             const std::tuple<ArgsT...>& /* t */,
                             Args... args)
    {
        return f(args...);
    }
};

#if 0
// nvcc does not support c++17
// We use the reference implementation of apply, which is a c++17 feature
// https://en.cppreference.com/w/cpp/utility/apply
template <class F, class Tuple>
constexpr decltype(auto) apply(F&& f, Tuple&& t)
{
 // https://www.reddit.com/r/cpp/comments/a48di2/stdinvokestdapply_analogs_for_c14/
}
#endif

namespace impl {
template <class Tuple, class F, size_t... Is>
constexpr auto apply_impl(Tuple t, F f, std::index_sequence<Is...>)
{
    return f(std::get<Is>(t)...);
}
}  // namespace impl

template <class Tuple, class F>
constexpr auto apply(F f, Tuple t)
{
    return impl::apply_impl(
        t, f, std::make_index_sequence<std::tuple_size<Tuple>{}>{});
}

/**
 * Static Function Call Forwarding Using Tuple Pack Parameters
 */
// Actual apply function
template <typename retVal, typename... ArgsF, typename... ArgsT>
retVal applyTuple(retVal (*f)(ArgsF...),
                  std::tuple<ArgsT...> const& t)
{
    return apply_func<sizeof...(ArgsT)>::applyTuple(f, t);
}

}  // namespace meta
}  // namespace Neon