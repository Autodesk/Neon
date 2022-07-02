#pragma once
#include "Neon/core/core.h"
#include "memory"

namespace Neon::domain::internal::eGrid {
namespace help {

/**
 *
 * @return
 */
template <typename Integer /**< type of the indexing (int16, int32 int64)*/,
          typename Lambda /**< function defining the sparsity */>
auto sparsityMap(Lambda                     fun /** function defining the sparsity: f(Integer_3d<Index>) -> bool */,
                 const Integer_3d<Integer>& dim /** Dimension of the space */)
    -> std::shared_ptr<uint8_t[]>
{
    std::shared_ptr<uint8_t[]> map(new Integer[dim.template rMulTyped<size_t>()]);
    dim.forEach([&](const Integer& x,
                    const Integer& y,
                    const Integer& z) {
        Integer_3d<Integer> xyz(x, y, z);
        const bool          isActive = fun(x, y, z);
        const uint8_t       val = isActive ? 1 : 0;
        map[xyz.mPitch(dim)] = val;
    });
    return map;
}

}  // namespace help
}  // namespace Neon::domain::internal::eGrid
