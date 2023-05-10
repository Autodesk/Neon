#pragma once
#include "lattice.h"

template <typename T, int Q>
inline Neon::set::Container coalescencePull(Neon::domain::mGrid&                   grid,
                                            const bool                             fineInitStore,
                                            const int                              level,
                                            const Neon::domain::mGrid::Field<int>& sumStore,
                                            const Neon::domain::mGrid::Field<T>&   postCollision,
                                            Neon::domain::mGrid::Field<T>&         postStreaming)
{
    // Initiated by the coarse level (hence "pull"), this function simply read the missing population
    // across the interface between coarse<->fine boundary by reading the population prepare during the store()

    return grid.getContainer(
        "Coalescence" + std::to_string(level), level,
        [&, level, fineInitStore](Neon::set::Loader& loader) {
            const auto& fpost_col = postCollision.load(loader, level, Neon::MultiResCompute::STENCIL);
            const auto& ss = sumStore.load(loader, level, Neon::MultiResCompute::STENCIL);
            auto&       fpost_stm = postStreaming.load(loader, level, Neon::MultiResCompute::MAP);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                //If this cell has children i.e., it is been refined, than we should not work on it
                //because this cell is only there to allow query and not to operate on
                const int refFactor = fpost_col.getRefFactor(level);
                if (!fpost_stm.hasChildren(cell)) {

                    for (int q = 1; q < Q; ++q) {
                        const Neon::int8_3d dir = -getDir(q);
                        //if we have a neighbor at the same level that has been refined, then cell is on
                        //the interface and this is where we should do the coalescence
                        if (fpost_stm.hasChildren(cell, dir)) {
                            auto neighbor = fpost_col.nghVal(cell, dir, q, T(0));
                            if (neighbor.isValid) {
                                if (fineInitStore) {
                                    fpost_stm(cell, q) = neighbor.value / static_cast<T>(ss(cell, q) * refFactor);
                                } else {
                                    fpost_stm(cell, q) = neighbor.value / static_cast<T>(refFactor);
                                }
                            }
                        }
                    }
                }
            };
        });
}