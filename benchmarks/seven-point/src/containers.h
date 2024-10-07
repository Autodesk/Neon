#pragma once

#include "Neon/Neon.h"
#include "Neon/set/Containter.h"
#include "parameters.h"
/**
 * Specialization for D3Q19
 */
template <typename Parameters>
struct ContainerFactory
{
    using Type = typename Parameters::Type;
    using Field = typename Parameters::Field;
    using Grid = typename Parameters::Grid;
    static constexpr int spaceDim = Parameters::spaceDim;
    static constexpr int fieldCard = Parameters::fieldCard;
    using S = SevenPoint<Parameters>;

    static auto iteration(Field& fin, Field& fout, double stepSize) -> Neon::set::Container
    {
        auto g = fin.getGrid();
        auto c = g.newContainer("iteration",
                                [=](Neon::set::Loader& loader) {
                                    auto       pin = loader.load(fin, Neon::Pattern::STENCIL);
                                    auto       pout = loader.load(fout, Neon::Pattern::MAP);
                                    const Type invh2 = Type(1.0) / (stepSize * stepSize);


                                    return
                                        [pin, invh2, pout] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& gidx) mutable -> void {
                                            auto partial = Type(0);
                                            //printf(".");
                                            int  numNeighb = 0;
                                            int  card = 0;
                                            Type sum = 0;
                                            for (; card < fieldCard; card++) {
                                                const Type center = pin(gidx, card);

                                                Neon::ConstexprFor<0, S::numNgh, 1>(
                                                    [&](auto q) -> void {
                                                        Neon::domain::NghData<Type> neighbor =
                                                            pin.template getNghData<S::template getOffset<q, 0>(),
                                                                                    S::template getOffset<q, 1>(),
                                                                                    S::template getOffset<q, 2>()>(gidx, card, Type(0));

                                                        if (neighbor.isValid()) {
                                                            ++numNeighb;
                                                            sum += neighbor.getData();
                                                        }

                                                        return;
                                                    });
                                                pout(gidx, card) = (-sum + static_cast<Type>(numNeighb) * center) * invh2;
                                            };
                                        };
                                });
        return c;
    }
};