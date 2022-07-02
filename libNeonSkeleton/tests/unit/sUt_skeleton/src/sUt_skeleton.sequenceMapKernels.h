#pragma once

#include <functional>
#include "Neon/domain/aGrid.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/set/Containter.h"
#include "Neon/set/ContainerTools/ContainerAPI.h"
#include "Neon/skeleton/Skeleton.h"
#include "gtest/gtest.h"
#include "sUt.runHelper.h"
#include "sUt_common.h"
namespace UserTools {
/**
 *
 * @tparam Field_ta
 * @param x
 * @param y
 * @return
 */
template <typename Field_ta>
auto xpy(const Field_ta& x,
         Field_ta&       y) -> Neon::set::Container
{
    auto Kontainer = x.getGrid().getContainer(
        "xpy", [&](Neon::set::Loader & L) -> auto {
            auto& xLocal = L.load(x);
            auto& yLocal = L.load(y);
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field_ta::Cell& e) mutable {
                for (int i = 0; i < yLocal.cardinality(); i++) {
                    yLocal(e, i) += xLocal(e, i);
                }
            };
        });
    return Kontainer;
}

/**
 *
 * @tparam Field_ta
 * @param x
 * @param y
 * @param z
 * @return
 */
template <typename Field_ta>
auto xpypz(const Field_ta& x,
           const Field_ta& y,
           Field_ta&       z) -> Neon::set::Container
{
    auto Kontainer = x.grid().container([&](Neon::set::Loader & L) -> auto {
        auto& xLocal = L.load(x);
        auto& yLocal = L.load(y);
        auto& zLocal = L.load(z);
        return [=] NEON_CUDA_HOST_DEVICE(const typename Field_ta::e_idx& e) mutable {
            for (int i = 0; i < yLocal.cardinality(); i++) {
                zLocal(e, i) += xLocal(e, i) + yLocal(e, i);
            }
        };
    });
    return Kontainer;
}

/**
 *
 * @tparam Field_ta
 */
template <typename Field_ta>
struct blas_t
{
    Field_ta x;
    Field_ta y;
    Field_ta z;
    blas_t(Field_ta x_,
           Field_ta y_,
           Field_ta z_)
        : x(x_),
          y(y_),
          z(z_)
    {
    }

    auto xpy() -> Neon::set::Container
    {
        auto Kontainer = x.grid().container([&](Neon::set::Loader & L) -> auto {
            auto& xLocal = L.load(x.cself());
            auto& yLocal = L.load(y);
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field_ta::eIdx_t& e) mutable {
                for (int i = 0; i < yLocal.cardinality(); i++) {
                    yLocal(e, i) += xLocal(e, i);
                }
            };
        });
        return Kontainer;
    }

    auto xpypz() -> Neon::set::Container
    {
        auto Kontainer = x.grid().container([&](Neon::set::Loader & L) -> auto {
            auto& xLocal = L.load(std::add_const(x));
            auto& yLocal = L.load(std::add_const(y));
            auto& zLocal = L.load(std::add_const(z));
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field_ta::e_idx& e) mutable {
                for (int i = 0; i < yLocal.cardinality(); i++) {
                    zLocal(e, i) += (xLocal(e, i) + yLocal(e, i));
                }
            };
        });
        return Kontainer;
    }
};
}  // namespace UserTools