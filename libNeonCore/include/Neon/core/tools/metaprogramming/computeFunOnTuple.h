#pragma once

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
namespace computeFunOnTuplePrivate {
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
template <unsigned int N>
struct computeFunOnTuple
{
    template <typename retVal, typename... ArgsF, typename... ArgsT, typename... Args>
    static retVal run(retVal (*f)(ArgsF...), const std::tuple<ArgsT...>& t, Args... args)
    {
        return computeFunOnTuple<N - 1>::run(
            f, t, std::get<N - 1>(t), args...);
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
struct computeFunOnTuple<0>
{
    template <typename retVal, typename... ArgsF, typename... ArgsT, typename... Args>
    static retVal run(retVal (*f)(ArgsF...), const std::tuple<ArgsT...>& /* t */
                      ,
                      Args... args)
    {
        return f(args...);
    }
};

}  // namespace computeFunOnTuplePrivate

//-----------------------------------------------------------------------------

/**
 * Static Function Call Forwarding Using Tuple Pack Parameters
 */
// Actual apply function
template <typename retVal, typename... ArgsF, typename... ArgsT>
retVal computeFunOnTuple(retVal (*f)(ArgsF...),
                         std::tuple<ArgsT...> const& t)
{
    return computeFunOnTuplePrivate::computeFunOnTuple<sizeof...(ArgsT)>::run(f, t);
}


}  // namespace meta
}  // namespace Neon
