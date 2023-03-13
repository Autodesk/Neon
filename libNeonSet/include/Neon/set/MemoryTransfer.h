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
        auto toString(const std::string& prefix = "") const -> std::string;
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
};


}  // namespace set
}  // namespace Neon
