#pragma once

#include "Neon/domain/eGrid.h"
#include "Neon/domain/bGrid.h"
#include "Neon/set/Containter.h"
#include "Neon/set/DevSet.h"

namespace Neon {
namespace solver {

template <typename Grid, typename Real>
auto set(typename Grid::template Field<Real>& input,
         const Real                                val) -> Neon::set::Container;

template <typename Grid, typename Real>
auto initR(typename Grid::template Field<Real>&       r,
           const typename Grid::template Field<Real>& x,
           const typename Grid::template Field<Real>& b,
           const typename Grid::template Field<int8_t>&  bd) -> Neon::set::Container;

template <typename Grid, typename Real>
auto AXPY(typename Grid::template Field<Real>&       r,
          const typename Grid::template Field<Real>& s) -> Neon::set::Container;

template <typename Grid, typename Real>
auto copy(typename Grid::template Field<Real>&       target,
          const typename Grid::template Field<Real>& source) -> Neon::set::Container;

template <typename Grid, typename Real>
auto updateXandR(typename Grid::template Field<Real, 0>&       x,
                 typename Grid::template Field<Real, 0>&       r,
                 const typename Grid::template Field<Real, 0>& p,
                 const typename Grid::template Field<Real, 0>& s,
                 const Real&                                        delta_new,
                 const Real&                                        pAp,
                 Real&                                              delta_old) -> Neon::set::Container;

template <typename Grid, typename Real>
auto updateP(typename Grid::template Field<Real, 0>&       p,
             const typename Grid::template Field<Real, 0>& r,
             const Real&                                        delta_new,
             const Real&                                        delta_old) -> Neon::set::Container;

template <typename Grid, typename Real>
auto printField(typename Grid::template Field<Real, 0>& p) -> Neon::set::Container;


#define CG_EXTERN_TEMPLATE(GRID, DATA)                                                                                                                                                                                                                \
    extern template auto updateXandR<GRID, DATA>(GRID::template Field<DATA, 0>&, GRID::template Field<DATA, 0>&, const GRID::template Field<DATA, 0>&, const GRID::template Field<DATA, 0>&, const DATA&, const DATA&, DATA&)->Neon::set::Container; \
    extern template auto set<GRID, DATA>(GRID::template Field<DATA, 0>&, const DATA)->Neon::set::Container;                                                                                                                                         \
    extern template auto copy<GRID, DATA>(GRID::template Field<DATA, 0>&, const GRID::template Field<DATA, 0>&)->Neon::set::Container;                                                                                                            \
    extern template auto initR<GRID, DATA>(GRID::template Field<DATA, 0>&, const GRID::template Field<DATA, 0>&, const GRID::template Field<DATA, 0>&, const GRID::template Field<int8_t, 0>&)->Neon::set::Container;                         \
    extern template auto AXPY<GRID, DATA>(GRID::template Field<DATA, 0>&, const GRID::template Field<DATA, 0>&)->Neon::set::Container;                                                                                                            \
    extern template auto updateP<GRID, DATA>(typename GRID::template Field<DATA, 0>&, const GRID::template Field<DATA, 0>&, const DATA&, const DATA&)->Neon::set::Container;                                                                      \
    extern template auto printField<GRID, DATA>(typename GRID::template Field<DATA, 0>&)->Neon::set::Container;

CG_EXTERN_TEMPLATE(Neon::dGrid, double);
CG_EXTERN_TEMPLATE(Neon::domain::bGrid, double);
CG_EXTERN_TEMPLATE(Neon::domain::eGrid, double);
CG_EXTERN_TEMPLATE(Neon::dGrid, float);
CG_EXTERN_TEMPLATE(Neon::domain::eGrid, float);
CG_EXTERN_TEMPLATE(Neon::domain::bGrid, float);
#undef CG_EXTERN_TEMPLATE

}  // namespace solver
}  // namespace Neon
