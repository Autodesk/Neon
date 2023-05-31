#pragma once
#include "coalescence.h"
#include "explosion.h"

template <typename T, int Q>
inline Neon::set::Container stream(Neon::domain::mGrid&                        grid,
                                   const int                                   level,
                                   const Neon::domain::mGrid::Field<CellType>& cellType,
                                   const Neon::domain::mGrid::Field<T>&        postCollision,
                                   Neon::domain::mGrid::Field<T>&              postStreaming)
{
    //regular Streaming of the normal voxels at level L which are not interfaced with L+1 and L-1 levels.
    //This is "pull" stream

    return grid.getContainer(
        "stream_" + std::to_string(level), level,
        [&, level](Neon::set::Loader& loader) {
            const auto& type = cellType.load(loader, level, Neon::MultiResCompute::STENCIL);
            const auto& fpost_col = postCollision.load(loader, level, Neon::MultiResCompute::STENCIL);
            auto        fpost_stm = postStreaming.load(loader, level, Neon::MultiResCompute::MAP);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                if (type(cell, 0) == CellType::bulk) {
                    //If this cell has children i.e., it is been refined, than we should not work on it
                    //because this cell is only there to allow query and not to operate on
                    if (!fpost_stm.hasChildren(cell)) {

                        for (int8_t q = 0; q < Q; ++q) {
                            const Neon::int8_3d dir = -getDir(q);

                            //if the neighbor cell has children, then this 'cell' is interfacing with L-1 (fine) along q direction
                            if (!fpost_stm.hasChildren(cell, dir)) {
                                auto nghType = type.nghVal(cell, dir, 0, CellType::undefined);

                                if (nghType.isValid) {
                                    if (nghType.value == CellType::bulk) {
                                        fpost_stm(cell, q) = fpost_col.nghVal(cell, dir, q, T(0)).value;
                                    } else {
                                        const int8_t opposte_q = latticeOppositeID[q];
                                        fpost_stm(cell, q) = fpost_col(cell, opposte_q) + fpost_col.nghVal(cell, dir, opposte_q, T(0)).value;
                                    }
                                }
                            }
                        }
                    }
                }
            };
        });
}


template <typename T, int Q>
inline Neon::set::Container streamFusedExplosion(Neon::domain::mGrid&                        grid,
                                                 const int                                   level,
                                                 const int                                   numLevels,
                                                 const Neon::domain::mGrid::Field<CellType>& cellType,
                                                 const Neon::domain::mGrid::Field<T>&        postCollision,
                                                 Neon::domain::mGrid::Field<T>&              postStreaming)
{
    //regular Streaming of the normal voxels at level L which are not interfaced with L+1 and L-1 levels.
    //This is "pull" stream

    return grid.getContainer(
        "streamFusedExplosion_" + std::to_string(level), level,
        [&, level, numLevels](Neon::set::Loader& loader) {
            const auto& type = cellType.load(loader, level, Neon::MultiResCompute::STENCIL);

            auto fpost_stm = postStreaming.load(loader, level, Neon::MultiResCompute::MAP);

            const auto& fpost_col = postCollision.load(loader, level, Neon::MultiResCompute::STENCIL);

            if (level != numLevels - 1) {
                postCollision.load(loader, level, Neon::MultiResCompute::STENCIL_UP);
            }

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                //We only do streaming in the bulk i.e., non-boundary condition voxels. Since we only allow grid
                //transition on bulk, then it is okay to do explosion inside this condition because
                //if the voxel is not bulk then all its neighbours are on the same level and no explosion is needed
                if (type(cell, 0) == CellType::bulk) {
                    //If this cell has children i.e., it is been refined, than we should not work on it
                    //because this cell is only there to allow query and not to operate on
                    if (!fpost_stm.hasChildren(cell)) {

                        for (int8_t q = 0; q < Q; ++q) {
                            const Neon::int8_3d dir = -getDir(q);

                            //if the neighbor cell has children, then this 'cell' is interfacing with L-1 (fine) along q direction
                            if (!fpost_stm.hasChildren(cell, dir)) {
                                auto neighborCell = fpost_col.getNghCell(cell, dir);

                                if (neighborCell.isActive()) {
                                    auto nghType = type.nghVal(cell, dir, 0, CellType::undefined);
                                    assert(nghType.isValid);
                                    if (nghType.value == CellType::bulk) {
                                        fpost_stm(cell, q) = fpost_col.nghVal(cell, dir, q, T(0)).value;
                                    } else {
                                        const int8_t opposte_q = latticeOppositeID[q];
                                        fpost_stm(cell, q) = fpost_col(cell, opposte_q) + fpost_col.nghVal(cell, dir, opposte_q, T(0)).value;
                                    }
                                } else if (level != numLevels - 1) {
                                    //only if we are not on the coarsest level and
                                    //only if we can not do normal streaming, then we may have a coarser neighbor from which
                                    //we can read this pop

                                    //get the uncle direction/offset i.e., the neighbor of the cell's parent
                                    //this direction/offset is wrt to the cell's parent
                                    Neon::int8_3d uncleDir = uncleOffset(cell.mLocation, dir);

                                    auto uncleLoc = fpost_col.getUncle(cell, uncleDir);

                                    auto uncle = fpost_col.uncleVal(cell, uncleDir, q, T(0));
                                    if (uncle.isValid) {
                                        fpost_stm(cell, q) = uncle.value;
                                    }
                                }
                            }
                        }
                    }
                }
            };
        });
}

template <typename T, int Q>
inline void stream(Neon::domain::mGrid&                        grid,
                   const bool                                  fineInitStore,
                   const int                                   level,
                   const int                                   numLevels,
                   const Neon::domain::mGrid::Field<CellType>& cellType,
                   const Neon::domain::mGrid::Field<int>&      sumStore,
                   const Neon::domain::mGrid::Field<T>&        postCollision,  //fout
                   Neon::domain::mGrid::Field<T>&              postStreaming,  //fin
                   std::vector<Neon::set::Container>&          containers)
{
    containers.push_back(stream<T, Q>(grid, level, cellType, postCollision, postStreaming));

    /*
    * Streaming for interface voxels that have
    *  (i) coarser or (ii) finer neighbors at level+1 and level-1 and hence require
    *  (i) "explosion" or (ii) coalescence
    */
    if (level != numLevels - 1) {
        /* Explosion: pull missing populations from coarser neighbors by copying coarse (level+1) to fine (level) 
        * neighbors, initiated by the fine level ("Pull").
        */
        containers.push_back(explosion<T, Q>(grid, level, postCollision, postStreaming));
    }

    if (level != 0) {
        /* Coalescence: pull missing populations from finer neighbors by "smart" averaging fine (level-1) 
        * to coarse (level) communication, initiated by the coarse level ("Pull").
        */
        containers.push_back(coalescence<T, Q>(grid, fineInitStore, level, sumStore, postCollision, postStreaming));
    }
}


template <typename T, int Q>
inline void streamFusedExplosion(Neon::domain::mGrid&                        grid,
                                 const bool                                  fineInitStore,
                                 const int                                   level,
                                 const int                                   numLevels,
                                 const Neon::domain::mGrid::Field<CellType>& cellType,
                                 const Neon::domain::mGrid::Field<int>&      sumStore,
                                 const Neon::domain::mGrid::Field<T>&        postCollision,  //fout
                                 Neon::domain::mGrid::Field<T>&              postStreaming,  //fin
                                 std::vector<Neon::set::Container>&          containers)
{
    /*
    * Streaming for interface voxels that have
    *  (i) coarser or (ii) finer neighbors at level+1 and level-1 and hence require
    *  (i) "explosion" or (ii) coalescence
    * Explosion: pull missing populations from coarser neighbors by copying coarse (level+1) to fine (level) 
    * neighbors, initiated by the fine level ("Pull").
    * Here we fuse the explosion with stream and do the check if we need explosion (since we don't 
    * need explosion on the coarsest level) inside the container. 
    */
    containers.push_back(streamFusedExplosion<T, Q>(grid, level, numLevels, cellType, postCollision, postStreaming));

    if (level != 0) {
        /* Coalescence: pull missing populations from finer neighbors by "smart" averaging fine (level-1) 
        * to coarse (level) communication, initiated by the coarse level ("Pull").
        */
        containers.push_back(coalescence<T, Q>(grid, fineInitStore, level, sumStore, postCollision, postStreaming));
    }
}
