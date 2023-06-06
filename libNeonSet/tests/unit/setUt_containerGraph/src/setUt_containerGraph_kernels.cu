#include "setUt_containerGraph_kernels.h"

namespace UserTools {
template <typename Field>
auto xpy(const Field& x,
         Field&       y) -> Neon::set::Container
{
    auto container = x.getGrid().newContainer(
        "xpy", [&](Neon::set::Loader & L) -> auto{
            auto& xLocal = L.load(x);
            auto& yLocal = L.load(y);
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                for (int i = 0; i < yLocal.cardinality(); i++) {
                    yLocal(e, i) += xLocal(e, i);
                }
            };
        });
    return container;
}

template <typename Field, typename T>
auto aInvXpY(const Neon::template PatternScalar<T>& fR,
             const Field&                           x,
             Field&                                 y) -> Neon::set::Container
{
    auto container = x.getGrid().newContainer(
        "AXPY", [&](Neon::set::Loader & L) -> auto{
            auto&      xLocal = L.load(x);
            auto&      yLocal = L.load(y);
            auto       fRLocal = L.load(fR);
            const auto fRVal = fRLocal();
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                // printf("%d yLocal.cardinality()\n", yLocal.cardinality());

                for (int i = 0; i < yLocal.cardinality(); i++) {
                    // printf("%d %d (%d) x\n", e, xLocal(e, i), i);
                    yLocal(e, i) += (1.0 / fRVal) * xLocal(e, i);
                }
            };
        },
        Neon::Execution::device);
    return container;
}

template <typename Field, typename T>
auto axpy(const Neon::template PatternScalar<T>& fR,
          const Field&                           x,
          Field&                                 y,
          const std::string&                     name) -> Neon::set::Container
{
    auto container = x.getGrid().newContainer(
        name + "-AXPY", [&](Neon::set::Loader & L) -> auto{
            auto&      xLocal = L.load(x);
            auto&      yLocal = L.load(y);
            auto       fRLocal = L.load(fR);
            const auto fRVal = fRLocal();
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                // printf("%d yLocal.cardinality()\n", yLocal.cardinality());

                for (int i = 0; i < yLocal.cardinality(); i++) {
                    // printf("%d %d (%d) x\n", e, xLocal(e, i), i);
                    yLocal(e, i) += fRVal * xLocal(e, i);
                }
            };
        });
    return container;
}

template <typename Field>
auto laplace(const Field& x,
             Field&       y,
             size_t       sharedMem ) -> Neon::set::Container
{
    auto container = x.getGrid().newContainer(
        "Laplace", [&](Neon::set::Loader & L) -> auto{
            auto& xLocal = L.load(x, Neon::Pattern::STENCIL);
            auto& yLocal = L.load(y);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& cell) mutable {
                using Type = typename Field::Type;
                for (int card = 0; card < xLocal.cardinality(); card++) {
                    typename Field::Type res = 0;


                    auto checkNeighbor = [&res](Neon::domain::NghData<Type>& neighbor) {
                        if (neighbor.isValid()) {
                            res += neighbor.value;
                        }
                    };

                    // Laplacian stencil operates on 6 neighbors (assuming 3D)
                    if constexpr (std::is_same<typename Field::Grid, Neon::domain::details::eGrid::eGrid>::value) {
                        for (int8_t nghIdx = 0; nghIdx < 6; ++nghIdx) {
                            auto neighbor = xLocal.getNghData(cell, nghIdx, card, Type(0));
                            checkNeighbor(neighbor);
                        }
                    } else {
                        typename Field::Partition::nghIdx_t ngh(0, 0, 0);

                        //+x
                        ngh.x = 1;
                        ngh.y = 0;
                        ngh.z = 0;
                        auto neighbor = xLocal.getNghData(cell, ngh, card, Type(0));
                        checkNeighbor(neighbor);

                        //-x
                        ngh.x = -1;
                        ngh.y = 0;
                        ngh.z = 0;
                        neighbor = xLocal.getNghData(cell, ngh, card, Type(0));
                        checkNeighbor(neighbor);

                        //+y
                        ngh.x = 0;
                        ngh.y = 1;
                        ngh.z = 0;
                        neighbor = xLocal.getNghData(cell, ngh, card, Type(0));
                        checkNeighbor(neighbor);

                        //-y
                        ngh.x = 0;
                        ngh.y = -1;
                        ngh.z = 0;
                        neighbor = xLocal.getNghData(cell, ngh, card, Type(0));
                        checkNeighbor(neighbor);

                        //+z
                        ngh.x = 0;
                        ngh.y = 0;
                        ngh.z = 1;
                        neighbor = xLocal.getNghData(cell, ngh, card, Type(0));
                        checkNeighbor(neighbor);

                        //-z
                        ngh.x = 0;
                        ngh.y = 0;
                        ngh.z = -1;
                        neighbor = xLocal.getNghData(cell, ngh, card, Type(0));
                        checkNeighbor(neighbor);
                    }


                    yLocal(cell, card) = -6 * res;
                }
            };
        });
    return container;
}


template auto xpy<eField32_t>(const eField32_t& x, eField32_t& y) -> Neon::set::Container;

template auto axpy<eField32_t, int32_t>(const Neon::template PatternScalar<int32_t>& fR,
                                        const eField32_t&                            x,
                                        eField32_t&                                  y,
                                        const std::string&                           name) -> Neon::set::Container;

}  // namespace UserTools