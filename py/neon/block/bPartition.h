#pragma once

#include "Neon/domain/details/bGrid/bPartition.h"
#include "../Index_3d.h"
#include "./bIndex.h"
#include "../ngh_idx.h"

// NOTE: we need this header to avoid errors about missing copy constructor for Pitch (Vec_4d)
//#include "Neon/core/types/vec/vec4d_integer.timp.h"

namespace wp
{
// NOTE: We create a subclass so that we can add a custom constructor
template <typename T>
class NeonBlockPartition : public ::Neon::domain::details::bGrid::bPartition<T,0, Neon::domain::details::bGrid::BlockDefault>
{
public:

   // initialize from bytes
   NeonBlockPartition(const char* bytes, size_t n)
   {
      assert(n == sizeof(*this));
      memcpy(this, bytes, n);
   }

   // NOTE: need default constructor for adjoint vars
   NeonBlockPartition()
   {
   }
};

using NeonBlockPartition_int8 = NeonBlockPartition<int8_t>;
using NeonBlockPartition_uint8 = NeonBlockPartition<uint8_t>;

using NeonBlockPartition_int32 = NeonBlockPartition<int32_t>;
using NeonBlockPartition_uint32 = NeonBlockPartition<uint32_t>;

using NeonBlockPartition_int64 = NeonBlockPartition<int64_t>;
using NeonBlockPartition_uint64 = NeonBlockPartition<uint64_t>;

using NeonBlockPartition_float32 = NeonBlockPartition<float>;
using NeonBlockPartition_float64 = NeonBlockPartition<double>;


template<typename T>
CUDA_CALLABLE inline auto neon_read(
   NeonBlockPartition<T>& p,
   NeonBlockIdx const & idx,
   int card)
 -> T
{
   return p(idx, card);
}

template<typename T>
CUDA_CALLABLE inline auto neon_write(
   NeonBlockPartition<T>& p,
   NeonBlockIdx const & idx,
   int card,
   T  const& value)
 -> void
{
    p(idx, card) = value;
}

template<typename T>
CUDA_CALLABLE inline auto neon_cardinality(
   NeonBlockPartition<T>& p)
 -> int
{
    return p.cardinality();
}

template<typename T>
CUDA_CALLABLE inline auto neon_ngh_data(
    const NeonBlockPartition<T>& p,
    NeonBlockIdx const & idx,
    NeonNghIdx const & ngh,
     int card,
     T alternative,
      bool& valid
    ) -> T
{
    typename NeonBlockPartition<T>::NghData nghData =
     p.getNghData(idx, ngh, card, alternative);
    valid = nghData.isValid();
    return nghData.getData();
}

template<typename T>
CUDA_CALLABLE inline auto neon_partition_id(
   NeonBlockPartition<T>& p)
 -> int
{
   return p.prtID();
}

// print
template<typename T>
CUDA_CALLABLE inline auto neon_print_dbg(const NeonBlockPartition<T>& p) -> void
{
   const Neon::index_3d& dim = p.dim();
   const Neon::index_3d& halo = p.halo();
   const Neon::index_3d& origin = p.origin();
   const int prtID = p.prtID();
   printf("NeonBlockPartition(dim={%d, %d, %d}, halo={%d, %d, %d}, origin={%d, %d, %d}, mem=%p prtID %d)\n",
      dim.x, dim.y, dim.z,
      halo.x, halo.y, halo.z,
      origin.x, origin.y, origin.z,
      p.mem(),
      prtID
   );
}

template<typename T>
CUDA_CALLABLE inline auto neon_global_idx(
   NeonBlockPartition<T>& p,
   NeonBlockIdx const & idx)
     -> Neon::index_3d
{
     Neon::index_3d globalIdx = p.getGlobalIndex(idx);
     return globalIdx;
}

}