#pragma once

#include "Neon/domain/internal/bGrid/bGrid.h"
#include "Neon/domain/internal/bGrid/bPartitionIndexSpace.h"

namespace Neon::domain::internal::bGrid {

template <typename T, int C>
bPartition<T, C>::bPartition(Neon::DataView  dataView,
                             T*              mem,
                             Neon::index_3d  dim,
                             int             cardinality,
                             uint32_t*       neighbourBlocks,
                             Neon::int32_3d* origin,
                             uint32_t*       mask,
                             T               outsideValue,
                             nghIdx_t*       stencilNghIndex)
    : mDataView(dataView),
      mMem(mem),
      mDim(dim),
      mCardinality(cardinality),
      mNeighbourBlocks(neighbourBlocks),
      mOrigin(origin),
      mMask(mask),
      mOutsideValue(outsideValue),
      mStencilNghIndex(stencilNghIndex),
      mIsInSharedMem(false),
      mMemSharedMem(nullptr),
      mStencilRadius(0)
{
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::cardinality() const -> int
{
    return mCardinality;
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::dim() const -> Neon::index_3d
{
    return mDim;
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::operator()(const bCell& cell,
                                                               int          card) -> T&
{
    if (mIsInSharedMem) {
        return mMemSharedMem[shmemPitch(cell, card)];
    } else {
        return mMem[pitch(cell, card)];
    }
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::operator()(const bCell& cell,
                                                               int          card) const -> const T&
{
    if (!cell.mIsActive) {
        return mOutsideValue;
    }
    if (mIsInSharedMem) {
        return mMemSharedMem[shmemPitch(cell, card)];
    } else {
        return mMem[pitch(cell, card)];
    }
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::pitch(const Cell& cell, int card) const -> uint32_t
{
    //assumes SoA within the block i.e., AoSoA
    return
        //stride across all block before cell's block
        cell.mBlockID * Cell::sBlockSizeX * Cell::sBlockSizeY * Cell::sBlockSizeZ * mCardinality +
        //stride within the block
        cell.pitch(card);
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::setNghCell(const Cell&     cell,
                                                               const nghIdx_t& offset) const -> Cell
{
    Cell ngh_cell(cell.mLocation.x + offset.x,
                  cell.mLocation.y + offset.y,
                  cell.mLocation.z + offset.z);

    if (ngh_cell.mLocation.x < 0 || ngh_cell.mLocation.y < 0 || ngh_cell.mLocation.z < 0 ||
        ngh_cell.mLocation.x >= Cell::sBlockSizeX || ngh_cell.mLocation.y >= Cell::sBlockSizeY || ngh_cell.mLocation.z >= Cell::sBlockSizeZ) {

        //The neighbor is not in this block

        //Calculate the neighbor block ID and the local index within the neighbor block
        int16_3d block_offset(0, 0, 0);

        if (ngh_cell.mLocation.x < 0) {
            block_offset.x = -1;
            ngh_cell.mLocation.x += Cell::sBlockSizeX;
        } else if (ngh_cell.mLocation.x >= Cell::sBlockSizeX) {
            block_offset.x = 1;
            ngh_cell.mLocation.x -= Cell::sBlockSizeX;
        }

        if (ngh_cell.mLocation.y < 0) {
            block_offset.y = -1;
            ngh_cell.mLocation.y += Cell::sBlockSizeY;
        } else if (ngh_cell.mLocation.y >= Cell::sBlockSizeY) {
            block_offset.y = 1;
            ngh_cell.mLocation.y -= Cell::sBlockSizeY;
        }

        if (ngh_cell.mLocation.z < 0) {
            block_offset.z = -1;
            ngh_cell.mLocation.z += Cell::sBlockSizeZ;
        } else if (ngh_cell.mLocation.z >= Cell::sBlockSizeZ) {
            block_offset.z = 1;
            ngh_cell.mLocation.z -= Cell::sBlockSizeZ;
        }

        ngh_cell.mBlockID = mNeighbourBlocks[26 * cell.mBlockID + Cell::getNeighbourBlockID(block_offset)];

    } else {
        ngh_cell.mBlockID = cell.mBlockID;
    }
    return ngh_cell;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::nghVal(const Cell& eId,
                                                           uint8_t     nghID,
                                                           int         card,
                                                           const T&    alternativeVal) const -> NghInfo<T>
{
    nghIdx_t nghOffset = mStencilNghIndex[nghID];
    return nghVal(eId, nghOffset, card, alternativeVal);
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::nghVal(const Cell&     cell,
                                                           const nghIdx_t& offset,
                                                           const int       card,
                                                           const T         alternativeVal) const -> NghInfo<T>
{
    NghInfo<T> ret;
    ret.value = alternativeVal;
    ret.isValid = false;
    if (!cell.mIsActive) {
        return ret;
    }
    Cell ngh_cell = setNghCell(cell, offset);
    if (ngh_cell.mBlockID != std::numeric_limits<uint32_t>::max()) {
        ret.isValid = ngh_cell.computeIsActive(mMask);
        if (ret.isValid) {
            if (mIsInSharedMem) {
                ngh_cell.mLocation.x = cell.mLocation.x + offset.x;
                ngh_cell.mLocation.y = cell.mLocation.y + offset.y;
                ngh_cell.mLocation.z = cell.mLocation.z + offset.z;
            }
            ret.value = this->operator()(ngh_cell, card);
        }
    }

    return ret;
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::shmemPitch(
    const Cell& cell,
    const int   card) const -> Cell::Location::Integer
{
    return cell.pitch(card, mStencilRadius);
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::loadInSharedMemory(
    [[maybe_unused]] const Cell&                cell,
    [[maybe_unused]] const nghIdx_t::Integer    stencilRadius,
    [[maybe_unused]] Neon::sys::ShmemAllocator& shmemAlloc) const -> void
{
#ifdef NEON_PLACE_CUDA_DEVICE
    mStencilRadius = stencilRadius;

    mMemSharedMem = shmemAlloc.alloc<T>((Cell::sBlockSizeX + 2 * mStencilRadius) *
                                        (Cell::sBlockSizeY + 2 * mStencilRadius) *
                                        (Cell::sBlockSizeZ + 2 * mStencilRadius) * mCardinality);
    //load the block itself in shared memory
    Cell shmemcell(cell.mLocation.x, cell.mLocation.y, cell.mLocation.z);


    //load the 6 faces from neighbor blocks in shared memory
    auto load_ngh = [&](const nghIdx_t& offset, int card) {
        NghInfo<T> ngh = nghVal(cell, offset, card, mOutsideValue);
        shmemcell.mLocation.x = cell.mLocation.x + offset.x;
        shmemcell.mLocation.y = cell.mLocation.y + offset.y;
        shmemcell.mLocation.z = cell.mLocation.z + offset.z;
        mMemSharedMem[shmemPitch(shmemcell, card)] = ngh.value;
    };


    nghIdx_t offset;
#pragma unroll 2
    for (int card = 0; card < mCardinality; ++card) {
        mMemSharedMem[shmemPitch(shmemcell, card)] = this->operator()(cell, card);

        for (nghIdx_t::Integer r = 1; r <= mStencilRadius; ++r) {
            //face (x, y, -z)
            if (cell.mLocation.z == 0) {
                offset.x = 0;
                offset.y = 0;
                offset.z = -r;
                load_ngh(offset, card);
            }

            //face (x, y, +z)
            if (cell.mLocation.z == Cell::sBlockSizeZ - 1) {
                offset.x = 0;
                offset.y = 0;
                offset.z = r;
                load_ngh(offset, card);
            }

            //face (x, -y, z)
            if (cell.mLocation.y == 0) {
                offset.x = 0;
                offset.y = -r;
                offset.z = 0;
                load_ngh(offset, card);
            }

            //face (x, +y, z)
            if (cell.mLocation.y == Cell::sBlockSizeY - 1) {
                offset.x = 0;
                offset.y = r;
                offset.z = 0;
                load_ngh(offset, card);
            }

            //face (-x, y, z)
            if (cell.mLocation.x == 0) {
                offset.x = -r;
                offset.y = 0;
                offset.z = 0;
                load_ngh(offset, card);
            }

            //face (+x, y, z)
            if (cell.mLocation.x == Cell::sBlockSizeX - 1) {
                offset.x = r;
                offset.y = 0;
                offset.z = 0;
                load_ngh(offset, card);
            }


            //load the 12 edges

            //edges along x-axis
            // edge (x, -y, -z)
            if (cell.mLocation.y == 0 && cell.mLocation.z == 0) {
                offset.x = 0;
                offset.y = -r;
                offset.z = -r;
                load_ngh(offset, card);
            }
            // edge (x, +y, -z)
            if (cell.mLocation.y == Cell::sBlockSizeY - 1 && cell.mLocation.z == 0) {
                offset.x = 0;
                offset.y = r;
                offset.z = -r;
                load_ngh(offset, card);
            }
            // edge (x, -y, +z)
            if (cell.mLocation.y == 0 && cell.mLocation.z == Cell::sBlockSizeZ - 1) {
                offset.x = 0;
                offset.y = -r;
                offset.z = r;
                load_ngh(offset, card);
            }
            // edge (x, +y, +z)
            if (cell.mLocation.y == Cell::sBlockSizeY - 1 && cell.mLocation.z == Cell::sBlockSizeZ - 1) {
                offset.x = 0;
                offset.y = r;
                offset.z = r;
                load_ngh(offset, card);
            }


            //edges along y-axis
            // edge (-x, y, -z)
            if (cell.mLocation.x == 0 && cell.mLocation.z == 0) {
                offset.x = -r;
                offset.y = 0;
                offset.z = -r;
                load_ngh(offset, card);
            }
            // edge (-x, y, +z)
            if (cell.mLocation.x == 0 && cell.mLocation.z == Cell::sBlockSizeZ - 1) {
                offset.x = -r;
                offset.y = 0;
                offset.z = r;
                load_ngh(offset, card);
            }
            // edge (+x, y, -z)
            if (cell.mLocation.x == Cell::sBlockSizeX - 1 && cell.mLocation.z == 0) {
                offset.x = r;
                offset.y = 0;
                offset.z = -r;
                load_ngh(offset, card);
            }
            // edge (+x, y, -z)
            if (cell.mLocation.x == Cell::sBlockSizeX - 1 && cell.mLocation.z == Cell::sBlockSizeZ - 1) {
                offset.x = r;
                offset.y = 0;
                offset.z = r;
                load_ngh(offset, card);
            }


            //edges along z-axis
            // edge (-x, -y, z)
            if (cell.mLocation.x == 0 && cell.mLocation.y == 0) {
                offset.x = -r;
                offset.y = -r;
                offset.z = 0;
                load_ngh(offset, card);
            }
            // edge (-x, +y, z)
            if (cell.mLocation.x == 0 && cell.mLocation.y == Cell::sBlockSizeY - 1) {
                offset.x = -r;
                offset.y = r;
                offset.z = 0;
                load_ngh(offset, card);
            }
            // edge (+x, -y, z)
            if (cell.mLocation.x == Cell::sBlockSizeX - 1 && cell.mLocation.y == 0) {
                offset.x = r;
                offset.y = -r;
                offset.z = 0;
                load_ngh(offset, card);
            }
            // edge (+x, +y, z)
            if (cell.mLocation.x == Cell::sBlockSizeX - 1 && cell.mLocation.y == Cell::sBlockSizeY - 1) {
                offset.x = r;
                offset.y = r;
                offset.z = 0;
                load_ngh(offset, card);
            }
        }

        //load the 8 corner

        //0,0,0
        for (Cell::Location::Integer z = -mStencilRadius; z <= -1; ++z) {
            for (Cell::Location::Integer y = -mStencilRadius; y <= -1; ++y) {
                for (Cell::Location::Integer x = -mStencilRadius; x <= -1; ++x) {

                    //0,0,0
                    if (cell.mLocation.x == 0 && cell.mLocation.y == 0 && cell.mLocation.z == 0) {
                        offset.x = static_cast<nghIdx_t::Integer>(x);
                        offset.y = static_cast<nghIdx_t::Integer>(y);
                        offset.z = static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //1,0,0
                    if (cell.mLocation.x == Cell::sBlockSizeX - 1 && cell.mLocation.y == 0 && cell.mLocation.z == 0) {
                        offset.x = -1 * static_cast<nghIdx_t::Integer>(x);
                        offset.y = static_cast<nghIdx_t::Integer>(y);
                        offset.z = static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //0,1,0
                    if (cell.mLocation.x == 0 && cell.mLocation.y == Cell::sBlockSizeY - 1 && cell.mLocation.z == 0) {
                        offset.x = static_cast<nghIdx_t::Integer>(x);
                        offset.y = -1 * static_cast<nghIdx_t::Integer>(y);
                        offset.z = static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //1,1,0
                    if (cell.mLocation.x == Cell::sBlockSizeX - 1 && cell.mLocation.y == Cell::sBlockSizeY - 1 && cell.mLocation.z == 0) {
                        offset.x = -1 * static_cast<nghIdx_t::Integer>(x);
                        offset.y = -1 * static_cast<nghIdx_t::Integer>(y);
                        offset.z = static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }


                    //0,0,1
                    if (cell.mLocation.x == 0 && cell.mLocation.y == 0 && cell.mLocation.z == Cell::sBlockSizeZ - 1) {
                        offset.x = static_cast<nghIdx_t::Integer>(x);
                        offset.y = static_cast<nghIdx_t::Integer>(y);
                        offset.z = -1 * static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //1,0,1
                    if (cell.mLocation.x == Cell::sBlockSizeX - 1 && cell.mLocation.y == 0 && cell.mLocation.z == Cell::sBlockSizeZ - 1) {
                        offset.x = -1 * static_cast<nghIdx_t::Integer>(x);
                        offset.y = static_cast<nghIdx_t::Integer>(y);
                        offset.z = -1 * static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //0,1,1
                    if (cell.mLocation.x == 0 && cell.mLocation.y == Cell::sBlockSizeY - 1 && cell.mLocation.z == Cell::sBlockSizeZ - 1) {
                        offset.x = static_cast<nghIdx_t::Integer>(x);
                        offset.y = -1 * static_cast<nghIdx_t::Integer>(y);
                        offset.z = -1 * static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //1,1,1
                    if (cell.mLocation.x == Cell::sBlockSizeX - 1 && cell.mLocation.y == Cell::sBlockSizeY - 1 && cell.mLocation.z == Cell::sBlockSizeZ - 1) {
                        offset.x = -1 * static_cast<nghIdx_t::Integer>(x);
                        offset.y = -1 * static_cast<nghIdx_t::Integer>(y);
                        offset.z = -1 * static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }
                }
            }
        }
    }


    __syncthreads();
    mIsInSharedMem = true;
#endif
};

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::loadInSharedMemoryAsync(
    [[maybe_unused]] const Cell&                cell,
    [[maybe_unused]] const nghIdx_t::Integer    stencilRadius,
    [[maybe_unused]] Neon::sys::ShmemAllocator& shmemAlloc) const -> void
{
#ifdef NEON_PLACE_CUDA_DEVICE
    //TODO only works on cardinality 1 for now
    assert(mCardinality == 1);

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    mStencilRadius = stencilRadius;

    mMemSharedMem = shmemAlloc.alloc<T>((Cell::sBlockSizeX + 2 * mStencilRadius) *
                                        (Cell::sBlockSizeY + 2 * mStencilRadius) *
                                        (Cell::sBlockSizeZ + 2 * mStencilRadius) * mCardinality);
    //load the interior
    Neon::sys::loadSharedMemAsync(block,
                                  mMem + cell.mBlockID * Cell::sBlockSizeX * Cell::sBlockSizeY * Cell::sBlockSizeZ * mCardinality,
                                  Cell::sBlockSizeX * Cell::sBlockSizeY * Cell::sBlockSizeZ,
                                  mMemSharedMem, false);


    //wait for all memory to arrive
    cg::wait(block);
    //do we really need a sync here or wait() is enough
    block.sync();
    mIsInSharedMem = true;
#endif
};


}  // namespace Neon::domain::internal::bGrid