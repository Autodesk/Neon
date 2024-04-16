#pragma once

template <typename T, int Q, bool atInterface>
inline Neon::set::Container streamFusedCoalescenceExplosion(Neon::domain::mGrid&                        grid,
                                                            const int                                   level,
                                                            const int                                   numLevels,
                                                            const Neon::domain::mGrid::Field<float>&    sumStore,
                                                            const Neon::domain::mGrid::Field<CellType>& cellType,
                                                            const Neon::domain::mGrid::Field<T>&        fout,
                                                            Neon::domain::mGrid::Field<T>&              fin)
{
    return grid.newContainer(
        "SOE" + std::to_string(level), level, !atInterface,
        [&, level, numLevels](Neon::set::Loader& loader) {
            const auto& type = cellType.load(loader, level, Neon::MultiResCompute::STENCIL);
            auto        pin = fin.load(loader, level, Neon::MultiResCompute::MAP);
            const auto& pout = fout.load(loader, level, Neon::MultiResCompute::STENCIL);
            const auto& ss = sumStore.load(loader, level, Neon::MultiResCompute::STENCIL);
            constexpr T repRefFactor = 0.5;
            if (level != numLevels - 1) {
                fout.load(loader, level, Neon::MultiResCompute::STENCIL_UP);
            }

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::mGrid::Idx& cell) mutable {
                if (type(cell, 0) == CellType::bulk) {
                    
                    (void)ss;

                    if (!pin.hasChildren(cell)) {

                        for (int8_t q = 0; q < Q; ++q) {
                            const Neon::int8_3d dir = -getDir(q);

                            //if the neighbor cell has children, then this 'cell' is interfacing with L-1 (fine) along q direction
                            auto nghType = type.getNghData(cell, dir, 0);
                            if (!pin.hasChildren(cell, dir)) {
                                if (nghType.mIsValid) {
                                    if (nghType.mData == CellType::bulk) {
                                        pin(cell, q) = pout.getNghData(cell, dir, q).mData;
                                    } else {
                                        const int8_t opposte_q = latticeOppositeID[q];
                                        pin(cell, q) = pout(cell, opposte_q) + pout.getNghData(cell, dir, opposte_q).mData;
                                    }
                                } else {
                                    if constexpr (atInterface) {

                                        if (pin.hasParent(cell) && !(dir.x == 0 && dir.y == 0 && dir.z == 0)) {
                                            Neon::int8_3d uncleDir = uncleOffset(cell.mInDataBlockIdx, dir);
                                            auto          uncle = pout.uncleVal(cell, uncleDir, q, T(0));
                                            if (uncle.mIsValid) {
                                                pin(cell, q) = uncle.mData;
                                            }
                                        }
                                    }
                                }
                            } else {

                                if constexpr (atInterface) {
                                    if (!(dir.x == 0 && dir.y == 0 && dir.z == 0)) {
                                        if (nghType.mIsValid) {
                                            auto neighbor = pout.getNghData(cell, dir, q);
                                            auto ssVal = ss.getNghData(cell, dir, q);
                                            assert(ssVal.mData != 0);
                                            pin(cell, q) = neighbor.mData * ssVal.mData;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            };
        });
}

template <typename T, int Q, bool atInterface>
inline void streamFusedCoalescenceExplosion(Neon::domain::mGrid&                        grid,
                                            const int                                   level,
                                            const int                                   numLevels,
                                            const Neon::domain::mGrid::Field<CellType>& cellType,
                                            const Neon::domain::mGrid::Field<float>&    sumStore,
                                            const Neon::domain::mGrid::Field<T>&        fout,
                                            Neon::domain::mGrid::Field<T>&              fin,
                                            std::vector<Neon::set::Container>&          containers)
{
    containers.push_back(streamFusedCoalescenceExplosion<T, Q, atInterface>(grid, level, numLevels, sumStore, cellType, fout, fin));
}