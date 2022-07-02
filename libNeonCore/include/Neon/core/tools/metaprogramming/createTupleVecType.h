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


namespace privateImplementation {
template <typename vectorTuple_ta>
struct tupleCreator_ta
{

    template <std::size_t N, typename... Args>
    struct recursion_t
    {
        // Extracting N-1 type: j_t
        using elementType = typename std::tuple_element<N - 1, vectorTuple_ta>::type;
        // Creating vector N-1 type: vector<j_t>
        using vectorType = typename std::vector<velementType>;
        // adding recursion: 1. Reducing counter 2. adding the extracted element to the list
        using tupleType = typename recursion_t<N - 1, elementType, Args...>::tupleType;
    };

    template <typename... Args>
    struct recursion_t<0, Args...>
    {
        using tupleType = std::tuple<Args...>;
    };

    static constexpr std::size_t TupleOfVectorLEN_tv = std::tuple_size<vectorTuple_ta>::value;
    using vectorTypeOfLastVector = typename std::tuple_element<TupleOfVectorLEN_tv - 1, vectorTuple_ta>::type;
    using elementTypeOfLastVector = typename vectorTypeOfLastVector::value_type;
    using linearlizedType = typename recursion_t<TupleOfVectorLEN_tv - 1, elementTypeOfLastVector>::tupleType;
};

}  // namespace privateImplementation

// input  -> std::tuple<vector<a_t>, vector<b_t>, vector<c_t>>
// output -> std::tuple<a_t, b_t, c_t>
template <typename tupleOfVectors_ta>
using TupleVecTypeList_t = typename privateImplementation::tupleExtractor_ta<tupleOfVectors_ta>::linearlizedType;


}  // namespace meta
}  // namespace Neon
