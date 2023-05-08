#pragma once

#include <functional>
#include "Neon/domain/aGrid.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/set/Containter.h"
#include "Neon/set/container/ContainerAPI.h"
#include "Neon/skeleton/Skeleton.h"
#include "gtest/gtest.h"
#include "sUt.runHelper.h"
#include "sUt_common.h"
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
        },
        Neon::Execution::device);
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
    auto Kontainer = x.grid().container([&](Neon::set::Loader& L) -> auto {
        auto& xLocal = L.load(x);
        auto& yLocal = L.load(y);
        auto& zLocal = L.load(z);
        return [=] NEON_CUDA_HOST_DEVICE(const typename Field::e_idx& e) mutable {
            for (int i = 0; i < yLocal.cardinality(); i++) {
                zLocal(e, i) += xLocal(e, i) + yLocal(e, i);
            }
        };
    });
    return Kontainer;
}

/**
 *
 * @tparam Field
 */
template <typename Field>
struct blas_t
{
    Field x;
    Field y;
    Field z;
    blas_t(Field x_,
           Field y_,
           Field z_)
        : x(x_),
          y(y_),
          z(z_)
    {
    }

    auto xpy() -> Neon::set::Container
    {
        auto Kontainer = x.grid().container([&](Neon::set::Loader& L) -> auto {
            auto& xLocal = L.load(x.cself());
            auto& yLocal = L.load(y);
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::eIdx_t& e) mutable {
                for (int i = 0; i < yLocal.cardinality(); i++) {
                    yLocal(e, i) += xLocal(e, i);
                }
            };
        });
        return Kontainer;
    }

    auto xpypz() -> Neon::set::Container
    {
        auto Kontainer = x.grid().container([&](Neon::set::Loader& L) -> auto {
            auto& xLocal = L.load(std::add_const(x));
            auto& yLocal = L.load(std::add_const(y));
            auto& zLocal = L.load(std::add_const(z));
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::e_idx& e) mutable {
                for (int i = 0; i < yLocal.cardinality(); i++) {
                    zLocal(e, i) += (xLocal(e, i) + yLocal(e, i));
                }
            };
        });
        return Kontainer;
    }
};
}  // namespace UserTools