#pragma once

#include <functional>
#include "Neon/domain/aGrid.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/set/Containter.h"
#include "Neon/set/container/ContainerAPI.h"
#include "Neon/skeleton/Skeleton.h"
#include "gtest/gtest.h"

namespace UserTools {
/**
 *
 * @tparam Field
 * @param x
 * @param y
 * @return
 */
template <typename Field>
auto xpy(const Field& x,
         Field&       y) -> Neon::set::Container
{
    auto c = x.getGrid().newContainer(
        "xpy", [&](Neon::set::Loader& L) -> auto {
            auto& xLocal = L.load(x);
            auto& yLocal = L.load(y);
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                for (int i = 0; i < yLocal.cardinality(); i++) {
                    yLocal(e, i) += xLocal(e, i);
                }
            };
        });
    return c;
}

/**
 *
 * @tparam Field
 * @param x
 * @param y
 * @param z
 * @return
 */
template <typename Field>
auto xpypz(const Field& x,
           const Field& y,
           Field&       z) -> Neon::set::Container
{
    auto container = x.grid().container([&](Neon::set::Loader& L) -> auto {
        auto& xLocal = L.load(x);
        auto& yLocal = L.load(y);
        auto& zLocal = L.load(z);
        return [=] NEON_CUDA_HOST_DEVICE(const typename Field::e_idx& e) mutable {
            for (int i = 0; i < yLocal.cardinality(); i++) {
                zLocal(e, i) += xLocal(e, i) + yLocal(e, i);
            }
        };
    });
    return container;
}

}  // namespace UserTools