#pragma once
#include <string>
#include <vector>

#include "Neon/core/core.h"

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

        Endpoint() = default;

        Endpoint(int devId, void* mem)
            : setIdx(devId), mem(mem)
        {
        }

        auto set(Neon::SetIdx devId, void* mem) -> void;
        auto toString(const std::string& prefix = "") const -> std::string{
            std::stringstream s;
            s<< prefix;
            s << "SetId: "<< setIdx.idx() << " Mem: "<<mem;
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


}  // namespace set
}  // namespace Neon
