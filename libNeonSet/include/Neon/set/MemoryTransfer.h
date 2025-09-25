#pragma once
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

#include "Neon/core/core.h"
#include "Neon/set/Backend.h"

namespace Neon {
namespace set {

class MemoryTransfer
{
   public:
    class Endpoint
    {
       public:
        Neon::SetIdx setIdx{-1};
        void*        mem{nullptr};
        Neon::size_4d logicalId;
        bool hasLogicalId= false;
        Endpoint() = default;

        Endpoint(int devId, void* mem)
            : setIdx(devId), mem(mem), hasLogicalId(true)
        {
        }

        Endpoint(int devId, void* mem,   Neon::size_4d id)
            : setIdx(devId), mem(mem), logicalId(id), hasLogicalId(true)
        {
        }

        auto set(Neon::SetIdx devId, void* mem) -> void;
        auto toString(const std::string& prefix = "") const -> std::string{
            if(!hasLogicalId) {
                std::stringstream s;
                s << prefix;
                s << "SetId: " << setIdx.idx() << " Mem: " << mem;
                return s.str();
            }
            std::stringstream s;
            s << prefix;
            s << "SetId: " << setIdx.idx() << " Mem: " << mem << " Id " << logicalId;
            return s.str();
        }
    };

    Endpoint dst;
    Endpoint src;
    size_t   size{0};

   public:
    MemoryTransfer() = default;

    MemoryTransfer(const Endpoint& dst,
                   const Endpoint& src,
                   size_t          size)
        : dst(dst), src(src), size(size)
    {
    }

    auto toString() const ->std::string{
       std::stringstream s;
       s << "Dst: {"<< dst.toString() << "} Src: {"<<src.toString()<<"} Size: {"<<size<<"}";
       return s.str();
    }
};

struct NcclPtoP
{
   public:
    enum Operation
    {
        send,
        receive
    };
    int            mPeerRank;
    cudaStream_t   mStream;
    ncclComm_t     mNcclComm;
    size_t         mNumElements;
    void*          mLocalBuffer;
    ncclDataType_t mNcclDataType;
    Operation      mOperation;

    NcclPtoP() = default;

    template <typename T>
    auto static init(Neon::Backend const& bk,
             int            peerRank,
             size_t numElements,
             void*  localBuffer,
             Operation op) -> NcclPtoP
    {
        NcclPtoP res;
        // NOTE: for now NCCL manages only one device
        // Neon::SetIdx setIdx = 0;
        res.mPeerRank = peerRank;
        // mStream = bk.streamSet(streamID)[setIdx].stream();
        res.mNcclComm = bk.getNccl().getNcclComm();
        res.mNumElements = numElements;
        res.mLocalBuffer = localBuffer;
        res.mNcclDataType = setDataType<T>();
        res.mOperation = op;
        return res;
    }

    auto execute(Neon::Backend const& bk, int streamIdx) -> void
    {
        Neon::SetIdx const setIdx = 0;
        auto         cudaStream = bk.streamSet(streamIdx)[setIdx].stream();
        ncclResult_t res;
        if (mOperation == Operation::send) {
            res = ncclSend(mLocalBuffer, mNumElements, mNcclDataType, mPeerRank, mNcclComm, cudaStream);
        } else {
            res = ncclRecv(mLocalBuffer, mNumElements, mNcclDataType, mPeerRank, mNcclComm, cudaStream);
        }
        if (res != ncclSuccess) {
            Neon::NeonException exp("NCCL");
            exp << "Error in NCCL operation: " << ncclGetErrorString(res);
            NEON_THROW(exp);
        }
    }

    template <typename T>
    static auto setDataType()
    {
        if constexpr (sizeof(int8_t) == sizeof(T)) {
            return ncclInt32;
        } else if (sizeof(int32_t) == sizeof(T)) {
            return ncclFloat32;
        } else if (sizeof(int64_t) == sizeof(T)) {
            return ncclFloat64;
        } else {
            NEON_THROW_UNSUPPORTED_OPTION("Unsupported data type");
        }
    }
};
}  // namespace set
}  // namespace Neon
