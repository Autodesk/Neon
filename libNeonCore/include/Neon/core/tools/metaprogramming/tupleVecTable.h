#pragma once
#if !defined(NEON_WARP_COMPILATION)
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
namespace tupleVecTable {

/**
 * We represent a table as a tuple of std::vectors.
 * the ith row of the table is a tuple containing the ith element of the vector
 * a column of the table is a vector.
 */


namespace privateImplementation {
template <typename vectorTuple_ta>
struct rowAsTuple_ta
{

    template <std::size_t N, typename... Args>
    struct recursionforTypes_t
    {
        // Extracting N-1 type: std::vector<j_t>
        using vectorType = typename std::tuple_element<N - 1, vectorTuple_ta>::type;
        // Extracting N-1 type: j_t
        using elementType = typename vectorType::value_type;
        // adding recursion: 1. Reducing counter 2. adding the extracted element to the list
        using tupleDeVectorizedFunType = typename recursionforTypes_t<N - 1, elementType, Args...>::tupleDeVectorizedFunType;
        using tupleDeVectorizedType = typename recursionforTypes_t<N - 1, elementType, Args...>::tupleDeVectorizedType;
    };

    template <typename... Args>
    struct recursionforTypes_t<0, Args...>
    {
        using tupleDeVectorizedFunType = void (*)(Args...);
        using tupleDeVectorizedType = std::tuple<Args...>;
    };

    static constexpr std::size_t TupleOfVectorLEN_tv = std::tuple_size<vectorTuple_ta>::value;
    using vectorTypeOfLastVector = typename std::tuple_element<TupleOfVectorLEN_tv - 1, vectorTuple_ta>::type;
    using elementTypeOfLastVector = typename vectorTypeOfLastVector::value_type;
    using tupleDeVectorizedFunType = typename recursionforTypes_t<TupleOfVectorLEN_tv - 1, elementTypeOfLastVector>::tupleDeVectorizedFunType;
    using tupleDeVectorizedType = typename recursionforTypes_t<TupleOfVectorLEN_tv - 1, elementTypeOfLastVector>::tupleDeVectorizedType;
};

}  // namespace privateImplementation

// input type -> IN_T = std::tuple<vector<a_t>, vector<b_t>, vector<c_t>>
// output TupleVecTypeList_t<IN_T>::tupleType -> std::tuple<a_t, b_t, c_t>
// output TupleVecTypeList_t<IN_T>::typeList -> std::tuple<a_t, b_t, c_t>
template <typename tupleOfVectors_ta>
using rowAsTupleType_t = typename privateImplementation::rowAsTuple_ta<tupleOfVectors_ta>::tupleDeVectorizedType;
template <typename tupleOfVectors_ta>
using rowAsTupleFunctionType_t = typename privateImplementation::rowAsTuple_ta<tupleOfVectors_ta>::tupleDeVectorizedFunType;

namespace privateImplementation {
template <typename vectorTuple_ta>
struct rowAsReader_ta
{
};

}  // namespace privateImplementation

}  // End of namespace tupleVecTable
}  // namespace meta
}  // namespace Neon
#endif