#include "Neon/core/types/DataView.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/domain/bGrid.h"
#include "Neon/set/container/Loader.h"
#include "Neon/solver/linear/krylov/CGContainers.h"

namespace Neon::solver {

template <typename Grid, typename Real>
auto set(typename Grid::template Field<Real>& input,
         const Real                                val) -> Neon::set::Container
{
    auto container = input.getGrid().getContainer("set", [&, val](Neon::set::Loader& loader) {
        auto& inp = loader.load(input);
        return [=] NEON_CUDA_HOST_DEVICE(const typename Grid::template Field<Real>::Cell& e) mutable {
            for (int i = 0; i < inp.cardinality(); ++i) {
                inp(e, i) = val;
            }
        };
    });

    return container;
}

template <typename Grid, typename Real>
auto copy(typename Grid::template Field<Real>&       target,
          const typename Grid::template Field<Real>& source) -> Neon::set::Container
{
    auto container = target.getGrid().getContainer("copy", [&](Neon::set::Loader& loader) {
        auto&       tar = loader.load(target);
        const auto& src = loader.load(source);
        return [=] NEON_CUDA_HOST_DEVICE(const typename Grid::template Field<Real>::Cell& e) mutable {
            for (int i = 0; i < tar.cardinality(); ++i) {
                tar(e, i) = src(e, i);
            }
        };
    });

    return container;
}

template <typename Grid, typename Real>
auto initR(typename Grid::template Field<Real>&       r,
           const typename Grid::template Field<Real>& x,
           const typename Grid::template Field<Real>& b,
           const typename Grid::template Field<int8_t>&  bd) -> Neon::set::Container
{
    // r := (bnd == 1) ? b : x

    auto container = r.getGrid().getContainer("initR", [&](Neon::set::Loader& loader) {
        auto&       in_r = loader.load(r);
        const auto& in_x = loader.load(x);
        const auto& in_b = loader.load(b);
        const auto& in_bd = loader.load(bd);

        return [=] NEON_CUDA_HOST_DEVICE(const typename Grid::template Field<Real>::Cell& e) mutable {
            for (int i = 0; i < in_r.cardinality(); ++i) {
                if (in_bd(e, i) == 1) {
                    in_r(e, i) = in_b(e, i);
                } else {
                    in_r(e, i) = in_x(e, i);
                }
            }
        };
    });

    return container;
}

template <typename Grid, typename Real>
auto AXPY(typename Grid::template Field<Real>&       r,
          const typename Grid::template Field<Real>& s) -> Neon::set::Container
{
    // r := r - Ax = r - s
    auto container = r.getGrid().getContainer("axpy", [&](Neon::set::Loader& loader) {
        auto&       in_r = loader.load(r);
        const auto& in_s = loader.load(s);

        return [=] NEON_CUDA_HOST_DEVICE(const typename Grid::template Field<Real>::Cell& e) mutable {
            for (int i = 0; i < in_r.cardinality(); ++i) {
                in_r(e, i) -= in_s(e, i);
            }
        };
    });

    return container;
}

template <typename Grid, typename Real>
auto updateXandR(typename Grid::template Field<Real, 0>&       x,
                 typename Grid::template Field<Real, 0>&       r,
                 const typename Grid::template Field<Real, 0>& p,
                 const typename Grid::template Field<Real, 0>& s,
                 const Real&                                        delta_new,
                 const Real&                                        pAp,
                 Real&                                              delta_old) -> Neon::set::Container
{
    auto container = x.getGrid().getContainer("Update X, \\n update R", [&x, &r, &p, &s, &delta_new, &pAp, &delta_old](Neon::set::Loader& loader) {
        auto&       p_x = loader.load(x);
        auto&       p_r = loader.load(r);
        const auto& p_p = loader.load(p);
        const auto& p_s = loader.load(s);

        // alpha := rr / p.Ap;
        const Real alpha = delta_new / pAp;
        delta_old = delta_new;

        return [=] NEON_CUDA_HOST_DEVICE(const typename Grid::template Field<Real>::Cell& e) mutable {
            for (int i = 0; i < p_x.cardinality(); ++i) {
                // x := x + alpha p
                p_x(e, i) += alpha * p_p(e, i);

                // r := r - alpha*s
                p_r(e, i) -= alpha * p_s(e, i);
            }
        };
    });
    return container;
}

template <typename Grid, typename Real>
auto updateP(typename Grid::template Field<Real, 0>&       p,
             const typename Grid::template Field<Real, 0>& r,
             const Real&                                        delta_new,
             const Real&                                        delta_old) -> Neon::set::Container
{
    auto container = p.getGrid().getContainer("Update P", [&p, &r, &delta_new, &delta_old](Neon::set::Loader& loader) {
        auto&       p_p = loader.load(p);
        const auto& p_r = loader.load(r);

        // beta := delta_new / delta_old;
        //unless if we are at first iteration, then delta_old =0 and beta = 0
        Real beta;
        if (std::abs(delta_old) > std::numeric_limits<Real>::epsilon()) {
            beta = delta_new / delta_old;
        } else {
            beta = 0;
        }

        return [=] NEON_CUDA_HOST_DEVICE(const typename Grid::template Field<Real>::Cell& e) mutable {
            // p := r + beta p
            for (int i = 0; i < p_p.cardinality(); ++i) {
                p_p(e, i) = p_r(e, i) + beta * p_p(e, i);
            }
        };
    });
    return container;
}


template <typename Grid, typename Real>
auto printField(typename Grid::template Field<Real, 0>& p) -> Neon::set::Container
{
    auto container = p.getGrid().getContainer("printField", [&p](Neon::set::Loader& loader) {
        auto& p_p = loader.load(p);
        return [=] NEON_CUDA_HOST_DEVICE(const typename Grid::template Field<Real>::Cell& e) mutable {
            for (int i = 0; i < p_p.cardinality(); ++i) {
                //printf("p_rr %ld %d %f \n", e, i, p_p.eRef(e, i));
            }
        };
    });
    return container;
}


#define CG_EXTERN_TEMPLATE(GRID, DATA)                                                                                                                                                                                                                \
    template auto updateXandR<GRID, DATA>(GRID::template Field<DATA, 0>&, GRID::template Field<DATA, 0>&, const GRID::template Field<DATA, 0>&, const GRID::template Field<DATA, 0>&, const DATA&, const DATA&, DATA&)->Neon::set::Container; \
    template auto set<GRID, DATA>(GRID::template Field<DATA, 0>&, const DATA)->Neon::set::Container;                                                                                                                                                \
    template auto copy<GRID, DATA>(GRID::template Field<DATA, 0>&, const GRID::template Field<DATA, 0>&)->Neon::set::Container;                                                                                                                   \
    template auto initR<GRID, DATA>(GRID::template Field<DATA, 0>&, const GRID::template Field<DATA, 0>&, const GRID::template Field<DATA, 0>&, const GRID::template Field<int8_t, 0>&)->Neon::set::Container;                                \
    template auto AXPY<GRID, DATA>(GRID::template Field<DATA, 0>&, const GRID::template Field<DATA, 0>&)->Neon::set::Container;                                                                                                                   \
    template auto updateP<GRID, DATA>(typename GRID::template Field<DATA, 0>&, const GRID::template Field<DATA, 0>&, const DATA&, const DATA&)->Neon::set::Container;                                                                             \
    template auto printField<GRID, DATA>(typename GRID::template Field<DATA, 0>&)->Neon::set::Container;

CG_EXTERN_TEMPLATE(Neon::domain::dGrid, double);
CG_EXTERN_TEMPLATE(Neon::domain::bGrid, double);
CG_EXTERN_TEMPLATE(Neon::domain::eGrid, double);
CG_EXTERN_TEMPLATE(Neon::domain::dGrid, float);
CG_EXTERN_TEMPLATE(Neon::domain::bGrid, float);
CG_EXTERN_TEMPLATE(Neon::domain::eGrid, float);
#undef CG_EXTERN_TEMPLATE


}  // namespace Neon
