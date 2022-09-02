#pragma once

#include <functional>

#include "Neon/domain/aGrid.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/set/Containter.h"
#include "Neon/set/container/ContainerAPI.h"
#include "Neon/skeleton/Skeleton.h"
#include "gtest/gtest.h"
#include "setUt_containerGraph_runHelper.h"

namespace UserTools {
template <typename Field>
auto xpy(const Field& x,
         Field&       y) -> Neon::set::Container;

template <typename Field, typename T>
auto aInvXpY(const Neon::template PatternScalar<T>& fR,
             const Field&                           x,
             Field&                                 y) -> Neon::set::Container;

template <typename Field, typename T>
auto axpy(const Neon::template PatternScalar<T>& fR,
          const Field&                           x,
          Field&                                 y,
          const std::string&                     name) -> Neon::set::Container;

template <typename Field>
auto laplace(const Field& x,
             Field&       y,
             size_t       sharedMem = 0) -> Neon::set::Container;


using eField32_t = Neon::domain::internal::eGrid::eGrid::Field<int32_t>;

extern template auto xpy<eField32_t>(const eField32_t& x, eField32_t& y) -> Neon::set::Container;

extern template auto axpy<eField32_t, int32_t>(const Neon::template PatternScalar<int32_t>& fR,
                                               const eField32_t&                            x,
                                               eField32_t&                                  y,
                                               const std::string&                           name) -> Neon::set::Container;

extern template auto laplace<eField32_t>(const eField32_t& x,
                                         eField32_t&       y,
                                         size_t            sharedMem) -> Neon::set::Container;

}  // namespace UserTools