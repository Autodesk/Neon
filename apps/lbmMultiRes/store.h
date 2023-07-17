#pragma once

template <typename T, int Q>
inline Neon::set::Container storeCoarse(Neon::domain::mGrid&           grid,
                                        int                            level,
                                        Neon::domain::mGrid::Field<T>& fout)
{
    //Initiated by the coarse level (level), this function prepares and stores the fine (level - 1)
    // information for further pulling initiated by the coarse (this) level invoked by coalescence_pull
    //
    //Where a coarse cell stores its information? at itself i.e., pull
    //Where a coarse cell reads the needed info? from its children and neighbor cell's children (level -1)
    //This function only operates on a coarse cell that has children.
    //For such cell, we check its neighbor cells at the same level. If any of these neighbor has NO
    //children, then we need to prepare something for them to be read during coalescence. What
    //we prepare is some sort of averaged the data from the children (the cell's children and/or
    //its neighbor's children)

    return grid.newContainer(
        "H" + std::to_string(level), level,
        [&, level](Neon::set::Loader& loader) {
            auto& pout = fout.load(loader, level, Neon::MultiResCompute::STENCIL_DOWN);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::mGrid::Idx& cell) mutable {
                //if the cell is refined, we might need to store something in it for its neighbor
                if (pout.hasChildren(cell)) {

                    const int refFactor = pout.getRefFactor(level);

                    //for each direction aka for each neighbor
                    //we skip the center here
                    for (int8_t q = 0; q < Q; ++q) {
                        const Neon::int8_3d qDir = getDir(q);
                        if (qDir.x == 0 && qDir.y == 0 && qDir.z == 0) {
                            continue;
                        }
                        //check if the neighbor in this direction has children
                        auto neighborCell = pout.helpGetNghIdx(cell, qDir);
                        if (neighborCell.isActive()) {

                            if (!pout.hasChildren(neighborCell)) {
                                //now, we know that there is actually something we need to store for this neighbor
                                //in cell along q (qDir) direction
                                int num = 0;
                                T   sum = 0;


                                //for every neighbor cell including the center cell (i.e., cell)
                                for (int8_t p = 0; p < Q; ++p) {
                                    const Neon::int8_3d pDir = getDir(p);

                                    const auto p_cell = pout.helpGetNghIdx(cell, pDir);
                                    //relative direction of q w.r.t p
                                    //i.e., in which direction we should move starting from p to land on q
                                    const Neon::int8_3d r_dir = qDir - pDir;

                                    //if this neighbor is refined
                                    if (pout.hasChildren(cell, pDir)) {

                                        //for each children of p
                                        for (int8_t i = 0; i < refFactor; ++i) {
                                            for (int8_t j = 0; j < refFactor; ++j) {
                                                for (int8_t k = 0; k < refFactor; ++k) {
                                                    const Neon::int8_3d c(i, j, k);

                                                    //cq is coarse neighbor (i.e., uncle) that we need to go in order to read q
                                                    //for c (this is what we do for explosion but here we do this just for the check)
                                                    const Neon::int8_3d cq = uncleOffset(c, qDir);
                                                    if (cq == r_dir) {
                                                        auto childVal = pout.childVal(p_cell, c, q, 0);
                                                        auto childCell = pout.getChild(p_cell, c);
                                                        if (childVal.mIsValid) {
                                                            num++;
                                                            sum += childVal.mData;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                pout(cell, q) += sum / static_cast<T>(num);
                            }
                        }
                    }
                }
            };
        });
}


template <typename T, int Q>
inline Neon::set::Container storeFine(Neon::domain::mGrid&           grid,
                                      int                            level,
                                      Neon::domain::mGrid::Field<T>& fout)
{
    //Initiated by the fine level (level), this function prepares and stores the fine (level)
    // information for further pulling initiated by the coarse (level+1) level invoked by coalescence
    //
    //This function only access a fine cell (Cf) that on the interface with a coarse cell.
    // For such fine cell, we figure out where to add its pop along certain direction in level + 1
    // such that another coarse cell (in level +1) will read this pop during coalescence


    return grid.newContainer(
        "H" + std::to_string(level), level,
        [&, level](Neon::set::Loader& loader) {
            //auto& pout = fout.load(loader, level, Neon::MultiResCompute::STENCIL_UP);

            //load this level as a map but not const since we will use pout to modify the next level
            auto& pout = fout.load(loader, level, Neon::MultiResCompute::MAP);

            //reload the next level as a map to indicate that we will (remote) write to it
            fout.load(loader, level + 1, Neon::MultiResCompute::MAP);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::mGrid::Idx& cell) mutable {
                assert(pout.hasParent(cell));

                if (!pout.hasChildren(cell)) {
                    for (int8_t q = 0; q < Q; ++q) {

                        const Neon::int8_3d qDir = getDir(q);
                        if (qDir.x == 0 && qDir.y == 0 && qDir.z == 0) {
                            continue;
                        }

                        const Neon::int8_3d uncleDir = uncleOffset(cell.mInDataBlockIdx, qDir);

                        //we try to access a cell on the same level (i.e., the refined level) along the same
                        //direction as the uncle and we use this a proxy to check if there is an unrefined uncle
                        const auto cn = pout.helpGetNghIdx(cell, uncleDir);

                        //cn may not be active because 1. it is outside the domain, or 2. this location is occupied by a coarse cell
                        //we are interested in 2.
                        if (!cn.isActive()) {

                            //now, we can get the uncle but we need to make sure it is active i.e.,
                            //it is not out side the domain boundary
                            const auto uncle = pout.getUncle(cell, uncleDir);
                            if (uncle.isActive()) {

                                //locate the coarse cell where we should store this cell info
                                const Neon::int8_3d CsDir = uncleDir - qDir;

                                const auto cs = pout.getUncle(cell, CsDir);

                                if (cs.isActive()) {

                                    const T cellVal = pout(cell, q);

#ifdef NEON_PLACE_CUDA_DEVICE
                                    atomicAdd(&pout.uncleVal(cell, CsDir, q), cellVal);
#else
#pragma omp atomic
                                    pout.uncleVal(cell, CsDir, q) += cellVal;
#endif
                                }
                            }
                        }
                    }
                }
            };
        });
}