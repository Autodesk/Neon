#pragma once
#include "lattice.h"

template <typename T, int Q>
inline Neon::set::Container coalescence(Neon::domain::mGrid&                   grid,
                                        const bool                             fineInitStore,
                                        const int                              level,
                                        const Neon::domain::mGrid::Field<int>& sumStore,
                                        const Neon::domain::mGrid::Field<T>&   fout,
                                        Neon::domain::mGrid::Field<T>&         fin)
{
    // Initiated by the coarse level (hence "pull"), this function simply read the missing population
    // across the interface between coarse<->fine boundary by reading the population prepare during the store()

    return grid.newContainer(
        "O" + std::to_string(level), level,
        [&, level, fineInitStore](Neon::set::Loader& loader) {
            const auto& pout = fout.load(loader, level, Neon::MultiResCompute::STENCIL);
            const auto& ss = sumStore.load(loader, level, Neon::MultiResCompute::STENCIL);
            auto&       pin = fin.load(loader, level, Neon::MultiResCompute::MAP);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::mGrid::Idx& cell) mutable {
                //If this cell has children i.e., it is been refined, than we should not work on it
                //because this cell is only there to allow query and not to operate on
                const int refFactor = pout.getRefFactor(level);
                if (!pin.hasChildren(cell)) {

                    for (int q = 0; q < Q; ++q) {
                        const Neon::int8_3d dir = -getDir(q);
                        if (dir.x == 0 && dir.y == 0 && dir.z == 0) {
                            continue;
                        }
                        //if we have a neighbor at the same level that has been refined, then cell is on
                        //the interface and this is where we should do the coalescence
                        if (pin.hasChildren(cell, dir)) {
                            auto neighbor = pout.getNghData(cell, dir, q);
                            if (neighbor.mIsValid) {
                                if (fineInitStore) {
                                    auto ssVal = ss.getNghData(cell, dir, q);
                                    assert(ssVal.mData != 0);
                                    pin(cell, q) = neighbor.mData / static_cast<T>(ssVal.mData * refFactor);
                                } else {
                                    pin(cell, q) = neighbor.mData / static_cast<T>(refFactor);
                                }
                            }
                        }
                    }
                }
            };
        });
}