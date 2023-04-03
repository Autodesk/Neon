#pragma once

#include "Neon/domain/details/bGrid/bGrid.h"
#include "Neon/domain/details/bGrid/bSpan.h"

namespace Neon::domain::details::bGrid {

template <typename T, int C>
bPartition<T, C>::bPartition()
    : mDataView(Neon::DataView::STANDARD),
      mMem(nullptr),
      mCardinality(0),
      mNeighbourBlocks(nullptr),
      mOrigin(nullptr),
      mMask(nullptr),
      mOutsideValue(0),
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
                             NghIdx*       stencilNghIndex)
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
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::mapToGlobal(const Index& cell) const -> Neon::index_3d
{
    Neon::index_3d ret = mOrigin[cell.mBlockID];
#ifdef NEON_PLACE_CUDA_DEVICE
    if constexpr (Index::sUseSwirlIndex) {
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
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::operator()(const bIndex& cell,
                                                                                 int           card) -> T&
{

    if (mIsInSharedMem) {
        return mMemSharedMem[shmemPitch(cell, card)];
    } else {
        return mMem[pitch(cell, card)];
    }
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::operator()(const bIndex& cell,
                                                                                 int           card) const -> const T&
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
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::pitch(const Index& cell, int card) const -> uint32_t
{
    // assumes SoA within the block i.e., AoSoA
    return
        // stride across all block before cell's block
        cell.mBlockID * cell.mBlockSize * cell.mBlockSize * cell.mBlockSize * mCardinality +
        // stride within the block
        cell.pitch(card);
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::setNghCell(const Index&     cell,
                                                                                 const NghIdx& offset) const -> Index
{
    Index ngh_cell(cell.mLocation.x + offset.x,
                  cell.mLocation.y + offset.y,
                  cell.mLocation.z + offset.z);
    ngh_cell.mBlockSize = cell.mBlockSize;

    if (ngh_cell.mLocation.x < 0 || ngh_cell.mLocation.y < 0 || ngh_cell.mLocation.z < 0 ||
        ngh_cell.mLocation.x >= cell.mBlockSize || ngh_cell.mLocation.y >= cell.mBlockSize || ngh_cell.mLocation.z >= cell.mBlockSize) {

        // The neighbor is not in this block

        // Calculate the neighbor block ID and the local index within the neighbor block
        int16_3d block_offset(0, 0, 0);

        if (ngh_cell.mLocation.x < 0) {
            block_offset.x = -1;
            ngh_cell.mLocation.x += cell.mBlockSize;
        } else if (ngh_cell.mLocation.x >= cell.mBlockSize) {
            block_offset.x = 1;
            ngh_cell.mLocation.x -= cell.mBlockSize;
        }

        if (ngh_cell.mLocation.y < 0) {
            block_offset.y = -1;
            ngh_cell.mLocation.y += cell.mBlockSize;
        } else if (ngh_cell.mLocation.y >= cell.mBlockSize) {
            block_offset.y = 1;
            ngh_cell.mLocation.y -= cell.mBlockSize;
        }

        if (ngh_cell.mLocation.z < 0) {
            block_offset.z = -1;
            ngh_cell.mLocation.z += cell.mBlockSize;
        } else if (ngh_cell.mLocation.z >= cell.mBlockSize) {
            block_offset.z = 1;
            ngh_cell.mLocation.z -= cell.mBlockSize;
        }

        if (mSharedNeighbourBlocks != nullptr) {
            ngh_cell.mBlockID = mSharedNeighbourBlocks[Index::getNeighbourBlockID(block_offset)];
        } else {
            ngh_cell.mBlockID = mNeighbourBlocks[26 * cell.mBlockID + Index::getNeighbourBlockID(block_offset)];
        }

    } else {
        ngh_cell.mBlockID = cell.mBlockID;
    }
    return ngh_cell;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::nghVal(const Index& eId,
                                                                             uint8_t     nghID,
                                                                             int         card,
                                                                             const T&    alternativeVal) const -> NghData<T>
{
    NghIdx nghOffset = mStencilNghIndex[nghID];
    return nghVal(eId, nghOffset, card, alternativeVal);
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::nghVal(const Index&     cell,
                                                                             const NghIdx& offset,
                                                                             const int       card,
                                                                             const T         alternativeVal) const -> NghData<T>
{
    NghData<T> ret;
    ret.value = alternativeVal;
    ret.mIsValid = false;
    if (!cell.mIsActive) {
        return ret;
    }


    Index ngh_cell = setNghCell(cell, offset);
    ngh_cell.mBlockSize = cell.mBlockSize;
    if (ngh_cell.mBlockID != std::numeric_limits<uint32_t>::max()) {
        ret.mIsValid = ngh_cell.computeIsActive(mMask);
        if (ret.mIsValid) {
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
    Index     cell,
    const int card) const -> Index::Location::Integer
{

    return (2 * mStencilRadius + Index::Location::Integer(cell.mBlockSize)) * (2 * mStencilRadius + Index::Location::Integer(cell.mBlockSize)) * (2 * mStencilRadius + Index::Location::Integer(cell.mBlockSize)) * static_cast<Index::Location::Integer>(card) +
           // offset to this cell's data
           (cell.mLocation.x + mStencilRadius) + (cell.mLocation.y + mStencilRadius) * (2 * mStencilRadius + Index::Location::Integer(cell.mBlockSize)) + (cell.mLocation.z + mStencilRadius) * (2 * mStencilRadius + Index::Location::Integer(cell.mBlockSize)) * (2 * mStencilRadius + Index::Location::Integer(cell.mBlockSize));
}


template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::loadInSharedMemory(
    [[maybe_unused]] const Index&                cell,
    [[maybe_unused]] const NghIdx::Integer    stencilRadius,
    [[maybe_unused]] Neon::sys::ShmemAllocator& shmemAlloc) const -> void
{
#ifdef NEON_PLACE_CUDA_DEVICE
    mStencilRadius = stencilRadius;

    mMemSharedMem = shmemAlloc.alloc<T>((cell.mBlockSize + 2 * mStencilRadius) *
                                        (cell.mBlockSize + 2 * mStencilRadius) *
                                        (cell.mBlockSize + 2 * mStencilRadius) * mCardinality);
    // load the block itself in shared memory
    Index shmemcell(cell.mLocation.x, cell.mLocation.y, cell.mLocation.z);


    // load the 6 faces from neighbor blocks in shared memory
    auto load_ngh = [&](const NghIdx& offset, int card) {
        NghInfo<T> ngh = nghVal(cell, offset, card, mOutsideValue);
        shmemcell.mLocation.x = cell.mLocation.x + offset.x;
        shmemcell.mLocation.y = cell.mLocation.y + offset.y;
        shmemcell.mLocation.z = cell.mLocation.z + offset.z;
        mMemSharedMem[shmemPitch(shmemcell, card)] = ngh.value;
    };


    /*__shared__ uint32_t sNeighbour[26];
    mSharedNeighbourBlocks = sNeighbour;
    {
        Index::Location::Integer tid = cell.getLocal1DID();
        for (int i = 0; i < 26; i += cell.mBlockSize * cell.mBlockSize * cell.mBlockSize) {
            mSharedNeighbourBlocks[i] = mNeighbourBlocks[26 * cell.mBlockID + i];
        }
    }*/

    NghIdx offset;
#pragma unroll 2
    for (int card = 0; card < mCardinality; ++card) {
        mMemSharedMem[shmemPitch(shmemcell, card)] = this->operator()(cell, card);

        for (NghIdx::Integer r = 1; r <= mStencilRadius; ++r) {
            // face (x, y, -z)
            if (cell.mLocation.z == 0) {
                offset.x = 0;
                offset.y = 0;
                offset.z = -r;
                load_ngh(offset, card);
            }

            // face (x, y, +z)
            if (cell.mLocation.z == cell.mBlockSize - 1) {
                offset.x = 0;
                offset.y = 0;
                offset.z = r;
                load_ngh(offset, card);
            }

            // face (x, -y, z)
            if (cell.mLocation.y == 0) {
                offset.x = 0;
                offset.y = -r;
                offset.z = 0;
                load_ngh(offset, card);
            }

            // face (x, +y, z)
            if (cell.mLocation.y == cell.mBlockSize - 1) {
                offset.x = 0;
                offset.y = r;
                offset.z = 0;
                load_ngh(offset, card);
            }

            // face (-x, y, z)
            if (cell.mLocation.x == 0) {
                offset.x = -r;
                offset.y = 0;
                offset.z = 0;
                load_ngh(offset, card);
            }

            // face (+x, y, z)
            if (cell.mLocation.x == cell.mBlockSize - 1) {
                offset.x = r;
                offset.y = 0;
                offset.z = 0;
                load_ngh(offset, card);
            }


            // load the 12 edges

            // edges along x-axis
            //  edge (x, -y, -z)
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


            // edges along y-axis
            //  edge (-x, y, -z)
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


            // edges along z-axis
            //  edge (-x, -y, z)
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

        // load the 8 corner

        // 0,0,0
        for (Index::Location::Integer z = -mStencilRadius; z <= -1; ++z) {
            for (Index::Location::Integer y = -mStencilRadius; y <= -1; ++y) {
                for (Index::Location::Integer x = -mStencilRadius; x <= -1; ++x) {

                    // 0,0,0
                    if (cell.mLocation.x == 0 && cell.mLocation.y == 0 && cell.mLocation.z == 0) {
                        offset.x = static_cast<NghIdx::Integer>(x);
                        offset.y = static_cast<NghIdx::Integer>(y);
                        offset.z = static_cast<NghIdx::Integer>(z);
                        load_ngh(offset, card);
                    }

                    // 1,0,0
                    if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.y == 0 && cell.mLocation.z == 0) {
                        offset.x = -1 * static_cast<NghIdx::Integer>(x);
                        offset.y = static_cast<NghIdx::Integer>(y);
                        offset.z = static_cast<NghIdx::Integer>(z);
                        load_ngh(offset, card);
                    }

                    // 0,1,0
                    if (cell.mLocation.x == 0 && cell.mLocation.y == cell.mBlockSize - 1 && cell.mLocation.z == 0) {
                        offset.x = static_cast<NghIdx::Integer>(x);
                        offset.y = -1 * static_cast<NghIdx::Integer>(y);
                        offset.z = static_cast<NghIdx::Integer>(z);
                        load_ngh(offset, card);
                    }

                    // 1,1,0
                    if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.y == cell.mBlockSize - 1 && cell.mLocation.z == 0) {
                        offset.x = -1 * static_cast<NghIdx::Integer>(x);
                        offset.y = -1 * static_cast<NghIdx::Integer>(y);
                        offset.z = static_cast<NghIdx::Integer>(z);
                        load_ngh(offset, card);
                    }


                    // 0,0,1
                    if (cell.mLocation.x == 0 && cell.mLocation.y == 0 && cell.mLocation.z == cell.mBlockSize - 1) {
                        offset.x = static_cast<NghIdx::Integer>(x);
                        offset.y = static_cast<NghIdx::Integer>(y);
                        offset.z = -1 * static_cast<NghIdx::Integer>(z);
                        load_ngh(offset, card);
                    }

                    // 1,0,1
                    if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.y == 0 && cell.mLocation.z == cell.mBlockSize - 1) {
                        offset.x = -1 * static_cast<NghIdx::Integer>(x);
                        offset.y = static_cast<NghIdx::Integer>(y);
                        offset.z = -1 * static_cast<NghIdx::Integer>(z);
                        load_ngh(offset, card);
                    }

                    // 0,1,1
                    if (cell.mLocation.x == 0 && cell.mLocation.y == cell.mBlockSize - 1 && cell.mLocation.z == cell.mBlockSize - 1) {
                        offset.x = static_cast<NghIdx::Integer>(x);
                        offset.y = -1 * static_cast<NghIdx::Integer>(y);
                        offset.z = -1 * static_cast<NghIdx::Integer>(z);
                        load_ngh(offset, card);
                    }

                    // 1,1,1
                    if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.y == cell.mBlockSize - 1 && cell.mLocation.z == cell.mBlockSize - 1) {
                        offset.x = -1 * static_cast<NghIdx::Integer>(x);
                        offset.y = -1 * static_cast<NghIdx::Integer>(y);
                        offset.z = -1 * static_cast<NghIdx::Integer>(z);
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
    [[maybe_unused]] const Index&                cell,
    [[maybe_unused]] const NghIdx::Integer    stencilRadius,
    [[maybe_unused]] Neon::sys::ShmemAllocator& shmemAlloc) const -> void
{
#ifdef NEON_PLACE_CUDA_DEVICE
    // TODO only works on cardinality 1 for now
    assert(mCardinality == 1);
    // TODO only works on stencil 1 for now
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


    Index::Location::Integer shmem_offset = 0;

    auto load = [&](const int16_3d block_offset,
                    const uint32_t src_offset,
                    const uint32_t size) {
        uint32_t ngh_block_id = mSharedNeighbourBlocks[Index::getNeighbourBlockID(block_offset)];
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


    // load the interior
    Neon::sys::loadSharedMemAsync(
        block,
        mMem + cell.mBlockID * cell.mBlockSize * cell.mBlockSize * cell.mBlockSize * mCardinality,
        cell.mBlockSize * cell.mBlockSize * cell.mBlockSize,
        mMemSharedMem,
        false);
    shmem_offset += cell.mBlockSize * cell.mBlockSize * cell.mBlockSize;


    if (mStencilRadius > 0) {
        int16_3d block_offset(0, 0, 0);

        // load -Z faces
        block_offset.x = 0;
        block_offset.y = 0;
        block_offset.z = -1;
        load(block_offset,
             cell.mBlockSize * cell.mBlockSize * (cell.mBlockSize - 1),
             cell.mBlockSize * cell.mBlockSize);


        // load +Z faces
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

    // wait for all memory to arrive
    cg::wait(block);
    // do we really need a sync here or wait() is enough
    block.sync();
    mIsInSharedMem = true;
#endif
};

}  // namespace Neon::domain::details::bGrid