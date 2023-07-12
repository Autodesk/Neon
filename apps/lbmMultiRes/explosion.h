#pragma once
template <typename T, int Q>
inline Neon::set::Container explosion(Neon::domain::mGrid&                 grid,
                                      int                                  level,
                                      const Neon::domain::mGrid::Field<T>& fout,
                                      Neon::domain::mGrid::Field<T>&       fin)
{
    // Initiated by the fine level (hence "pull"), this function performs a coarse (level+1) to
    // fine (level) communication or "explosion" by simply distributing copies of coarse grid onto the fine grid.
    // In other words, this function updates the "halo" cells of the fine level by making copies of the coarse cell
    // values.


    return grid.newContainer(
        "E" + std::to_string(level), level,
        [&, level](Neon::set::Loader& loader) {
            const auto& pout = fout.load(loader, level, Neon::MultiResCompute::STENCIL_UP);
            auto        pin = fin.load(loader, level, Neon::MultiResCompute::MAP);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::mGrid::Idx& cell) mutable {
                //If this cell has children i.e., it is been refined, then we should not work on it
                //because this cell is only there to allow query and not to operate on
                if (!pin.hasChildren(cell)) {
                    for (int8_t q = 0; q < Q; ++q) {
                        const Neon::int8_3d dir = -getDir(q);
                        if (dir.x == 0 && dir.y == 0 && dir.z == 0) {
                            continue;
                        }

                        //if the neighbor cell has children, then this 'cell' is interfacing with L-1 (fine) along q direction
                        //we want to only work on cells that interface with L+1 (coarse) cell along q
                        if (!pin.hasChildren(cell, dir)) {

                            //try to query the cell along this direction (opposite of the population direction) as we do
                            //in 'normal' streaming
                            auto neighborCell = pout.helpGetNghIdx(cell, dir);
                            if (!neighborCell.isActive()) {
                                //only if we can not do normal streaming, then we may have a coarser neighbor from which
                                //we can read this pop

                                //get the uncle direction/offset i.e., the neighbor of the cell's parent
                                //this direction/offset is wrt to the cell's parent
                                Neon::int8_3d uncleDir = uncleOffset(cell.mInDataBlockIdx, dir);

                                auto uncleLoc = pout.getUncle(cell, uncleDir);

                                auto uncle = pout.uncleVal(cell, uncleDir, q, T(0));
                                if (uncle.mIsValid) {
                                    pin(cell, q) = uncle.mData;
                                }
                            }
                        }
                    }
                }
            };
        });
}