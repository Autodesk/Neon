#pragma once

#include "Neon/domain/internal/bGrid/bGrid.h"
#include "Neon/domain/internal/bGrid/bPartitionIndexSpace.h"

namespace Neon::domain::internal::bGrid {

template <typename T, int C>
bPartition<T, C>::bPartition()
    : mDataView(Neon::DataView::STANDARD),
      mMem(nullptr),
      mCardinality(0),
      mNeighbourBlocks(nullptr),
      mOrigin(nullptr),
      mMask(nullptr),
      mOutsideValue(T()),
      mStencilNghIndex(nullptr),
      mIsInSharedMem(false),
      mMemSharedMem(nullptr),
      mSharedNeighbourBlocks(nullptr),
      mStencilRadius(0)
{
}

template <typename T, int C>
bPartition<T, C>::bPartition(Neon::DataView  dataView,
                             T*              mem,
                             int             cardinality,
                             uint32_t*       neighbourBlocks,
                             Neon::int32_3d* origin,
                             uint32_t*       mask,
                             T               outsideValue,
                             nghIdx_t*       stencilNghIndex)
    : mDataView(dataView),
      mMem(mem),
      mCardinality(cardinality),
      mNeighbourBlocks(neighbourBlocks),
      mOrigin(origin),
      mMask(mask),
      mOutsideValue(outsideValue),
      mStencilNghIndex(stencilNghIndex),
      mIsInSharedMem(false),
      mMemSharedMem(nullptr),
      mSharedNeighbourBlocks(nullptr),
      mStencilRadius(0)
{
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::mapToGlobal(const Cell& cell) const -> Neon::index_3d
{
    Neon::index_3d ret = mOrigin[cell.mBlockID];
#ifdef NEON_PLACE_CUDA_DEVICE
    if constexpr (Cell::sUseSwirlIndex) {
        auto swirl = cell.toSwirl();
        ret.x += swirl.mLocation.x;
        ret.y += swirl.mLocation.y;
        ret.z += swirl.mLocation.z;
    } else {
#endif
        ret.x += cell.mLocation.x;
        ret.y += cell.mLocation.y;
        ret.z += cell.mLocation.z;
#ifdef NEON_PLACE_CUDA_DEVICE
    }
#endif
    return ret;
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::cardinality() const -> int
{
    return mCardinality;
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
        cell.mBlockID * cell.mBlockSize * cell.mBlockSize * cell.mBlockSize * mCardinality +
        //stride within the block
        cell.pitch(card);
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::getneighbourBlocksPtr(const Cell& cell) const -> const uint32_t*
{
    if (mSharedNeighbourBlocks != nullptr) {
        return mSharedNeighbourBlocks;
    } else {
        return mNeighbourBlocks + (26 * cell.mBlockID);
    }
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::getNghCell(const Cell& cell, const nghIdx_t& offset) const -> Cell
{
    return getNghCell(cell, offset, getneighbourBlocksPtr(cell));
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::getNghCell(const Cell&     cell,
                                                               const nghIdx_t& offset,
                                                               const uint32_t* neighbourBlocks) const -> Cell
{
    Cell nghCell(cell.mLocation.x + offset.x,
                 cell.mLocation.y + offset.y,
                 cell.mLocation.z + offset.z);
    nghCell.mBlockSize = cell.mBlockSize;


    assert(nghCell.mBlockSize <= Cell::sBlockAllocGranularity);

    auto isNghInDifferentBlock = [&]() {
        //check if the neighbor cell is in a different block then cell

        return nghCell.mLocation.x < 0 || nghCell.mLocation.y < 0 || nghCell.mLocation.z < 0 ||
               nghCell.mLocation.x >= cell.mBlockSize || nghCell.mLocation.y >= cell.mBlockSize || nghCell.mLocation.z >= cell.mBlockSize;
    };

    auto calcOffsetBlock = [&]() {
        //calc the relative offset of the block in which nghCell resides
        //this should only be called when nghCell's block is different than cell's block
        int16_3d blockOffset(0, 0, 0);

        if (nghCell.mLocation.x < 0) {
            blockOffset.x = -1;
        } else if (nghCell.mLocation.x >= cell.mBlockSize) {
            blockOffset.x = 1;
        }

        if (nghCell.mLocation.y < 0) {
            blockOffset.y = -1;
        } else if (nghCell.mLocation.y >= cell.mBlockSize) {
            blockOffset.y = 1;
        }

        if (nghCell.mLocation.z < 0) {
            blockOffset.z = -1;
        } else if (nghCell.mLocation.z >= cell.mBlockSize) {
            blockOffset.z = 1;
        }

        assert(!(blockOffset.x == 0 && blockOffset.y == 0 && blockOffset.z == 0));

        return Cell::getNeighbourBlockID(blockOffset);
    };


    auto updateNghCell = [&]() {
        //update the nghCell local index to be the local index in the neighbor block
        //the addition is done to avoid module operations on negative numbers
        nghCell.mLocation.x = (nghCell.mLocation.x + cell.mBlockSize) % cell.mBlockSize;
        nghCell.mLocation.y = (nghCell.mLocation.y + cell.mBlockSize) % cell.mBlockSize;
        nghCell.mLocation.z = (nghCell.mLocation.z + cell.mBlockSize) % cell.mBlockSize;
    };


    //if the block size is less the block allocation granularity, then this means we can implicitly get the neighbor cell implicitly
    if (nghCell.mBlockSize < Cell::sBlockAllocGranularity) {
        const Neon::int32_3d cellOrigin = mOrigin[cell.mBlockID] + cell.mLocation.newType<int32_t>();
        const Neon::int32_3d cellTray(cellOrigin.x / Cell::sBlockAllocGranularity,
                                      cellOrigin.y / Cell::sBlockAllocGranularity,
                                      cellOrigin.z / Cell::sBlockAllocGranularity);

        Neon::int32_3d nghOrigin(cellOrigin.x + offset.x, cellOrigin.y + offset.y, cellOrigin.z + offset.z);
        if (nghOrigin.x < 0 || nghOrigin.y < 0 || nghOrigin.z < 0) {
            nghCell.mBlockID = std::numeric_limits<uint32_t>::max();

        } else {
            Neon::int32_3d nghTray(nghOrigin.x / Cell::sBlockAllocGranularity,
                                   nghOrigin.y / Cell::sBlockAllocGranularity,
                                   nghOrigin.z / Cell::sBlockAllocGranularity);

            //if the neighbor does not live in the same tray, then we do the usual nghCell
            //where we try to find the neighbor using neighbor block ID
            if (cellTray != nghTray) {
                nghCell.mBlockID = neighbourBlocks[calcOffsetBlock()];

            } else {
                //otherwise, nghCell may resides on a different block in the same tray
                if (isNghInDifferentBlock()) {
                    const int numBlockPerTray = Cell::sBlockAllocGranularity / cell.mBlockSize;

                    nghCell.mBlockID = cell.mBlockID + offset.mPitch(Neon::index_3d(numBlockPerTray,
                                                                                    numBlockPerTray,
                                                                                    numBlockPerTray));
                } else {
                    //or nghCell may reside on the same block
                    nghCell.mBlockID = cell.mBlockID;
                }
            }
            updateNghCell();
        }

    } else {
        if (isNghInDifferentBlock()) {
            nghCell.mBlockID = neighbourBlocks[calcOffsetBlock()];
        } else {
            nghCell.mBlockID = cell.mBlockID;
        }
        updateNghCell();
    }

    nghCell.mIsActive = (nghCell.mBlockID != std::numeric_limits<uint32_t>::max());
    return nghCell;
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


    if constexpr (Cell::sUseSwirlIndex) {
        Cell swirl_cell = cell.toSwirl();
        swirl_cell.mBlockSize = cell.mBlockSize;

        Cell ngh_cell = getNghCell(swirl_cell, offset, getneighbourBlocksPtr(swirl_cell));
        ngh_cell.mBlockSize = cell.mBlockSize;
        if (ngh_cell.isActive()) {
            //TODO maybe ngh_cell should be mapped to its memory layout
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

    } else {
        Cell ngh_cell = getNghCell(cell, offset, getneighbourBlocksPtr(cell));
        if (ngh_cell.isActive()) {
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
    }

    return ret;
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::shmemPitch(
    Cell      cell,
    const int card) const -> Cell::Location::Integer
{
    if constexpr (Cell::sUseSwirlIndex) {
        if (cell.mLocation.x >= 0 && cell.mLocation.x < cell.mBlockSize &&
            cell.mLocation.y >= 0 && cell.mLocation.y < cell.mBlockSize &&
            cell.mLocation.z >= -1 && cell.mLocation.z <= cell.mBlockSize) {

            if (cell.mLocation.z == -1) {
                cell.mLocation.z = cell.mBlockSize;
            } else if (cell.mLocation.z == 1) {
                cell.mLocation.z = cell.mBlockSize + 1;
            }
        } else {
            cell.mLocation.z += (cell.mLocation.z / 2) + 10;
        }


        return (2 * mStencilRadius + cell.mBlockSize) * (2 * mStencilRadius + cell.mBlockSize) * (2 * mStencilRadius + cell.mBlockSize) * static_cast<Cell::Location::Integer>(card) +
               cell.mLocation.x +
               cell.mLocation.y * cell.mBlockSize +
               cell.mLocation.z * cell.mBlockSize * cell.mBlockSize;

    } else {
        return (2 * mStencilRadius + Cell::Location::Integer(cell.mBlockSize)) * (2 * mStencilRadius + Cell::Location::Integer(cell.mBlockSize)) * (2 * mStencilRadius + Cell::Location::Integer(cell.mBlockSize)) * static_cast<Cell::Location::Integer>(card) +
               //offset to this cell's data
               (cell.mLocation.x + mStencilRadius) + (cell.mLocation.y + mStencilRadius) * (2 * mStencilRadius + Cell::Location::Integer(cell.mBlockSize)) + (cell.mLocation.z + mStencilRadius) * (2 * mStencilRadius + Cell::Location::Integer(cell.mBlockSize)) * (2 * mStencilRadius + Cell::Location::Integer(cell.mBlockSize));
    }
}


template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::loadInSharedMemory(
    [[maybe_unused]] const Cell&                cell,
    [[maybe_unused]] const nghIdx_t::Integer    stencilRadius,
    [[maybe_unused]] Neon::sys::ShmemAllocator& shmemAlloc) const -> void
{
#ifdef NEON_PLACE_CUDA_DEVICE
    mStencilRadius = stencilRadius;

    mMemSharedMem = shmemAlloc.alloc<T>((cell.mBlockSize + 2 * mStencilRadius) *
                                        (cell.mBlockSize + 2 * mStencilRadius) *
                                        (cell.mBlockSize + 2 * mStencilRadius) * mCardinality);
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


    /*__shared__ uint32_t sNeighbour[26];
    mSharedNeighbourBlocks = sNeighbour;
    {
        Cell::Location::Integer tid = cell.getLocal1DID();
        for (int i = 0; i < 26; i += cell.mBlockSize * cell.mBlockSize * cell.mBlockSize) {
            mSharedNeighbourBlocks[i] = mNeighbourBlocks[26 * cell.mBlockID + i];
        }
    }*/

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
            if (cell.mLocation.z == cell.mBlockSize - 1) {
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
            if (cell.mLocation.y == cell.mBlockSize - 1) {
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
            if (cell.mLocation.x == cell.mBlockSize - 1) {
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
            if (cell.mLocation.y == cell.mBlockSize - 1 && cell.mLocation.z == 0) {
                offset.x = 0;
                offset.y = r;
                offset.z = -r;
                load_ngh(offset, card);
            }
            // edge (x, -y, +z)
            if (cell.mLocation.y == 0 && cell.mLocation.z == cell.mBlockSize - 1) {
                offset.x = 0;
                offset.y = -r;
                offset.z = r;
                load_ngh(offset, card);
            }
            // edge (x, +y, +z)
            if (cell.mLocation.y == cell.mBlockSize - 1 && cell.mLocation.z == cell.mBlockSize - 1) {
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
            if (cell.mLocation.x == 0 && cell.mLocation.z == cell.mBlockSize - 1) {
                offset.x = -r;
                offset.y = 0;
                offset.z = r;
                load_ngh(offset, card);
            }
            // edge (+x, y, -z)
            if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.z == 0) {
                offset.x = r;
                offset.y = 0;
                offset.z = -r;
                load_ngh(offset, card);
            }
            // edge (+x, y, -z)
            if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.z == cell.mBlockSize - 1) {
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
            if (cell.mLocation.x == 0 && cell.mLocation.y == cell.mBlockSize - 1) {
                offset.x = -r;
                offset.y = r;
                offset.z = 0;
                load_ngh(offset, card);
            }
            // edge (+x, -y, z)
            if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.y == 0) {
                offset.x = r;
                offset.y = -r;
                offset.z = 0;
                load_ngh(offset, card);
            }
            // edge (+x, +y, z)
            if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.y == cell.mBlockSize - 1) {
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
                    if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.y == 0 && cell.mLocation.z == 0) {
                        offset.x = -1 * static_cast<nghIdx_t::Integer>(x);
                        offset.y = static_cast<nghIdx_t::Integer>(y);
                        offset.z = static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //0,1,0
                    if (cell.mLocation.x == 0 && cell.mLocation.y == cell.mBlockSize - 1 && cell.mLocation.z == 0) {
                        offset.x = static_cast<nghIdx_t::Integer>(x);
                        offset.y = -1 * static_cast<nghIdx_t::Integer>(y);
                        offset.z = static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //1,1,0
                    if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.y == cell.mBlockSize - 1 && cell.mLocation.z == 0) {
                        offset.x = -1 * static_cast<nghIdx_t::Integer>(x);
                        offset.y = -1 * static_cast<nghIdx_t::Integer>(y);
                        offset.z = static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }


                    //0,0,1
                    if (cell.mLocation.x == 0 && cell.mLocation.y == 0 && cell.mLocation.z == cell.mBlockSize - 1) {
                        offset.x = static_cast<nghIdx_t::Integer>(x);
                        offset.y = static_cast<nghIdx_t::Integer>(y);
                        offset.z = -1 * static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //1,0,1
                    if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.y == 0 && cell.mLocation.z == cell.mBlockSize - 1) {
                        offset.x = -1 * static_cast<nghIdx_t::Integer>(x);
                        offset.y = static_cast<nghIdx_t::Integer>(y);
                        offset.z = -1 * static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //0,1,1
                    if (cell.mLocation.x == 0 && cell.mLocation.y == cell.mBlockSize - 1 && cell.mLocation.z == cell.mBlockSize - 1) {
                        offset.x = static_cast<nghIdx_t::Integer>(x);
                        offset.y = -1 * static_cast<nghIdx_t::Integer>(y);
                        offset.z = -1 * static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //1,1,1
                    if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.y == cell.mBlockSize - 1 && cell.mLocation.z == cell.mBlockSize - 1) {
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
    //TODO only works on stencil 1 for now
    assert(stencilRadius == 1);

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    mStencilRadius = stencilRadius;

    __shared__ uint32_t sNeighbour[26];
    mSharedNeighbourBlocks = sNeighbour;

    Neon::sys::loadSharedMemAsync(
        block,
        mNeighbourBlocks + 26 * cell.mBlockID,
        26,
        mSharedNeighbourBlocks,
        true);


    mMemSharedMem = shmemAlloc.alloc<T>((cell.mBlockSize + 2 * mStencilRadius) *
                                        (cell.mBlockSize + 2 * mStencilRadius) *
                                        (cell.mBlockSize + 2 * mStencilRadius) * mCardinality);


    Cell::Location::Integer shmem_offset = 0;

    auto load = [&](const int16_3d block_offset,
                    const uint32_t src_offset,
                    const uint32_t size) {
        uint32_t ngh_block_id = mSharedNeighbourBlocks[Cell::getNeighbourBlockID(block_offset)];
        assert(ngh_block_id != cell.mBlockID);
        if (ngh_block_id != std::numeric_limits<uint32_t>::max()) {
            Neon::sys::loadSharedMemAsync(
                block,
                mMem + ngh_block_id * cell.mBlockSize * cell.mBlockSize * cell.mBlockSize * mCardinality + src_offset,
                //     ^^start of this block                                                                ^^ offset from the start
                size,
                mMemSharedMem + shmem_offset,
                false);
            shmem_offset += size;
        }
    };


    //load the interior
    Neon::sys::loadSharedMemAsync(
        block,
        mMem + cell.mBlockID * cell.mBlockSize * cell.mBlockSize * cell.mBlockSize * mCardinality,
        cell.mBlockSize * cell.mBlockSize * cell.mBlockSize,
        mMemSharedMem,
        false);
    shmem_offset += cell.mBlockSize * cell.mBlockSize * cell.mBlockSize;


    if (mStencilRadius > 0) {
        int16_3d block_offset(0, 0, 0);

        //load -Z faces
        block_offset.x = 0;
        block_offset.y = 0;
        block_offset.z = -1;
        load(block_offset,
             cell.mBlockSize * cell.mBlockSize * (cell.mBlockSize - 1),
             cell.mBlockSize * cell.mBlockSize);


        //load +Z faces
        block_offset.x = 0;
        block_offset.y = 0;
        block_offset.z = 1;
        load(block_offset,
             0,
             cell.mBlockSize * cell.mBlockSize);


        for (int z = 0; z < cell.mBlockSize; ++z) {
            // load strips from -Y
            block_offset.x = 0;
            block_offset.y = -1;
            block_offset.z = 0;
            load(block_offset,
                 z * cell.mBlockSize * cell.mBlockSize + 14,
                 cell.mBlockSize);

            // load strips from +Y
            block_offset.x = 0;
            block_offset.y = 1;
            block_offset.z = 0;
            load(block_offset,
                 z * cell.mBlockSize * cell.mBlockSize + 0,
                 cell.mBlockSize);

            // load strips from -X
            block_offset.x = -1;
            block_offset.y = 0;
            block_offset.z = 0;
            load(block_offset,
                 z * cell.mBlockSize * cell.mBlockSize + 7,
                 cell.mBlockSize);


            // load strips from +X
            block_offset.x = 1;
            block_offset.y = 0;
            block_offset.z = 0;
            load(block_offset,
                 z * cell.mBlockSize * cell.mBlockSize + 21,
                 cell.mBlockSize - 1);
            load(block_offset,
                 z * cell.mBlockSize * cell.mBlockSize + 0,
                 1);
        }
    }

    //wait for all memory to arrive
    cg::wait(block);
    //do we really need a sync here or wait() is enough
    block.sync();
    mIsInSharedMem = true;
#endif
};

}  // namespace Neon::domain::internal::bGrid