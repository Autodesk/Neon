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

namespace PrivateTupleOfVecInnerTypeExtractor {
template <typename vectorTuple_ta>
struct TupleOfVecInnerTypeExtractor_t
{

    template <std::size_t N, typename... Args>
    struct recursion_t
    {
        using targetVectorType = typename std::tuple_element<N - 1, vectorTuple_ta>::type;
        using tupleType = typename recursion_t<N - 1, typename targetVectorType::value_type, Args...>::tupleType;
    };

    template <typename... Args>
    struct recursion_t<0, Args...>
    {
        using tupleType = std::tuple<Args...>;
    };


    static constexpr std::size_t VectorTupleSize_tv = std::tuple_size<vectorTuple_ta>::value;
    using targetVectorType = typename std::tuple_element<VectorTupleSize_tv - 1, vectorTuple_ta>::type;

    using linearType = typename recursion_t<VectorTupleSize_tv - 1, typename targetVectorType::value_type>::tupleType;
};

}  // namespace PrivateTupleOfVecInnerTypeExtractor
template <typename vectorTuple_ta>
using TupleOfVecInnertType_t = typename PrivateTupleOfVecInnerTypeExtractor::TupleOfVecInnerTypeExtractor_t<vectorTuple_ta>::linearType;


namespace privateImplementation {
template <typename vectorTuple_ta>
struct tupleExtractor_ta
{

    template <std::size_t N, typename... Args>
    struct recursion_t
    {
        // Extracting N-1 type: std::vector<j_t>
        using vectorType = typename std::tuple_element<N - 1, vectorTuple_ta>::type;
        // Extracting N-1 type: j_t
        using elementType = typename vectorType::value_type;
        // adding recursion: 1. Reducing counter 2. adding the extracted element to the list
        using tupleDeVectorizedFunType = typename recursion_t<N - 1, elementType, Args...>::tupleDeVectorizedFunType;
        using tupleDeVectorizedType = typename recursion_t<N - 1, elementType, Args...>::tupleDeVectorizedType;
    };

    template <typename... Args>
    struct recursion_t<0, Args...>
    {
        using tupleDeVectorizedFunType = void (*)(Args...);
        using tupleDeVectorizedType = std::tuple<Args...>;
    };

    static constexpr std::size_t TupleOfVectorLEN_tv = std::tuple_size<vectorTuple_ta>::value;
    using vectorTypeOfLastVector = typename std::tuple_element<TupleOfVectorLEN_tv - 1, vectorTuple_ta>::type;
    using elementTypeOfLastVector = typename vectorTypeOfLastVector::value_type;
    using tupleDeVectorizedFunType = typename recursion_t<TupleOfVectorLEN_tv - 1, elementTypeOfLastVector>::tupleDeVectorizedFunType;
    using tupleDeVectorizedType = typename recursion_t<TupleOfVectorLEN_tv - 1, elementTypeOfLastVector>::tupleDeVectorizedType;
};

}  // namespace privateImplementation

// input type -> IN_T = std::tuple<vector<a_t>, vector<b_t>, vector<c_t>>
// output TupleVecTypeList_t<IN_T>::tupleType -> std::tuple<a_t, b_t, c_t>
// output TupleVecTypeList_t<IN_T>::typeList -> std::tuple<a_t, b_t, c_t>
template <typename tupleOfVectors_ta>
using tupleDeVectorizedType_t = typename privateImplementation::tupleExtractor_ta<tupleOfVectors_ta>::tupleDeVectorizedType;

template <typename tupleOfVectors_ta>
using tupleDeVectorizedFunType_t = typename privateImplementation::tupleExtractor_ta<tupleOfVectors_ta>::tupleDeVectorizedFunType;


}  // namespace meta
}  // namespace Neon

#endif