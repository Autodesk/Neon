#pragma once

namespace Neon {

template <auto Start, auto End, auto Inc, class F>
constexpr void ConstexprFor(F&& f)
{
    if constexpr (Start < End) {
        f(std::integral_constant<decltype(Start), Start>());
        ConstexprFor<Start + Inc, End, Inc>(f);
    }
}

}  // namespace Neon