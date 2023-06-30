#pragma once

namespace Neon {

/**
 * Implementation of a constexpr for loop.
 * Reference: https://artificial-mind.net/blog/2020/10/31/constexpr-for
 *
 * The loop is implemented as a recursive template function.
 * It is equicalent to the following code:
 *
 * for(int i = Start; i < End; i += Inc) {
 *    f(i);
 *    // do something
 *    // ...
 *    // ...
 * }
 */
template <auto Start /**< First index for the loop */,
          auto End /**< Last index for the loop is (End-1) */,
          auto Inc /**< Loop increment */,
          class F>
constexpr void ConstexprFor(F&& f)
{
    if constexpr (Start < End) {
        f(std::integral_constant<decltype(Start), Start>());
        ConstexprFor<Start + Inc, End, Inc>(f);
    }
}

}  // namespace Neon