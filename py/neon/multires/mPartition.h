#pragma once

#include "../Index_3d.h"
#include "Neon/domain/details/mGrid/mPartition.h"
// #include "./bIndex.h"
#include "../ngh_idx.h"

// NOTE: we need this header to avoid errors about missing copy constructor for Pitch (Vec_4d)
// #include "Neon/core/types/vec/vec4d_integer.timp.h"

namespace wp {
// NOTE: We create a subclass so that we can add a custom constructor
template <typename T>
class NeonMultiresPartition : public ::Neon::domain::details::mGrid::mPartition<T, 0>
{
   public:
    // initialize from bytes
    NeonMultiresPartition(const char* bytes, size_t n)
    {
        // printf("NeonMultiresPartition(const char* bytes, size_t n) n %ld sizeof(*this) %ld\n", n, sizeof(*this));
        assert(n == sizeof(*this));
        memcpy(this, bytes, n);
    }

    // NOTE: need default constructor for adjoint vars
    NeonMultiresPartition()
    {
    }
};

using NeonMultiresPartition_int8 = NeonMultiresPartition<int8_t>;
using NeonMultiresPartition_uint8 = NeonMultiresPartition<uint8_t>;

using NeonMultiresPartition_int32 = NeonMultiresPartition<int32_t>;
using NeonMultiresPartition_uint32 = NeonMultiresPartition<uint32_t>;

using NeonMultiresPartition_int64 = NeonMultiresPartition<int64_t>;
using NeonMultiresPartition_uint64 = NeonMultiresPartition<uint64_t>;

using NeonMultiresPartition_float32 = NeonMultiresPartition<float>;
using NeonMultiresPartition_float64 = NeonMultiresPartition<double>;


template <typename T>
CUDA_CALLABLE inline auto neon_read(
    NeonMultiresPartition<T>& p,
    NeonBlockIdx const&       idx,
    int                       card)
    -> T
{
    return p(idx, card);
}

template <typename T>
CUDA_CALLABLE inline auto neon_write(
    NeonMultiresPartition<T>& p,
    NeonBlockIdx const&       idx,
    int                       card,
    T const&                  value)
    -> void
{
    p(idx, card) = value;
}

template <typename T>
CUDA_CALLABLE inline auto neon_cardinality(
    NeonMultiresPartition<T>& p)
    -> int
{
    return p.cardinality();
}

template <typename T>
CUDA_CALLABLE inline auto neon_ngh_data(
    const NeonMultiresPartition<T>& p,
    NeonBlockIdx const&             idx,
    NeonNghIdx const&               ngh,
    int                             card,
    T                               alternative,
    bool&                           valid) -> T
{
    typename NeonMultiresPartition<T>::NghData nghData =
        p.getNghData(idx, ngh, card, alternative);
    valid = nghData.isValid();
    return nghData.getData();
}

template <typename T>
CUDA_CALLABLE inline auto neon_partition_id(
    NeonMultiresPartition<T>& p)
    -> int
{
    return p.prtID();
}

// print
// template<typename T>
// CUDA_CALLABLE inline auto neon_print_dbg(const NeonMultiresPartition<T>& p) -> void
//{
//   const Neon::index_3d& dim = p.dim();
//   const Neon::index_3d& halo = p.halo();
//   const Neon::index_3d& origin = p.origin();
//   const int prtID = p.prtID();
//   printf("NeonMultiresPartition(dim={%d, %d, %d}, halo={%d, %d, %d}, origin={%d, %d, %d}, mem=%p prtID %d)\n",
//      dim.x, dim.y, dim.z,
//      halo.x, halo.y, halo.z,
//      origin.x, origin.y, origin.z,
//      p.mem(),
//      prtID
//   );
//}

template <typename T>
CUDA_CALLABLE inline auto neon_global_idx(
    NeonMultiresPartition<T>& p,
    NeonBlockIdx const&       idx)
    -> Neon::index_3d
{
    Neon::index_3d globalIdx = p.getGlobalIndex(idx);
    return globalIdx;
}
//// Multi-res Capabilities

template <typename T>
CUDA_CALLABLE inline auto neon_childValue(
    NeonMultiresPartition<T>& p,
    const NeonBlockIdx&       parentCell,
    const NeonNghIdx          child,
    int                       card,
    const T&                  alternativeVal,
    bool&                     isValid) -> T
{

    typename NeonMultiresPartition<T>::NghData nghData =
        p.childVal(parentCell, child, card, alternativeVal);
    isValid = nghData.isValid();
    return nghData.getData();
}

template <typename T>
CUDA_CALLABLE inline auto neon_getChild(
    NeonMultiresPartition<T>& p,
    const NeonBlockIdx&       parentCell,
    const NeonNghIdx          child) -> NeonNghIdx
{
    return p.getChild(parentCell, child);
}

template <typename T>
CUDA_CALLABLE inline auto neon_childValue(
    NeonMultiresPartition<T>& p,
    const NeonBlockIdx&       childIdx,
    int                       card) -> T
{
    return p.childVal(childIdx, card);
}

template <typename T>
CUDA_CALLABLE inline auto neon_hasChildren(
    NeonMultiresPartition<T>& p,
    const NeonBlockIdx&       idx) -> bool
{
    return p.hasChildren(idx);
}

template <typename T>
CUDA_CALLABLE inline auto neon_hasChildren(
    NeonMultiresPartition<T>& p,
    const NeonBlockIdx&       cell,
    const NeonNghIdx          nghDir) -> bool
{
    return p.hasChildren(cell, nghDir);
}

template <typename T>
CUDA_CALLABLE inline auto neon_getParent(
    NeonMultiresPartition<T>& p,
    const NeonBlockIdx&       cell) -> NeonBlockIdx
{
    return p.getParent(cell);
}

template <typename T>
CUDA_CALLABLE inline auto neon_parentVal_read(
    NeonMultiresPartition<T> const& p,
    const NeonBlockIdx&             cell,
    int                             card) -> T
{
    return p.parentVal(cell, card);
}


template <typename T>
CUDA_CALLABLE inline auto neon_parentVal_write(
    NeonMultiresPartition<T> const& p,
    const NeonBlockIdx&             cell,
    int                             card,
    T value) -> void
{
    return p.parentVal(cell, card) = value;
}

template <typename T>
CUDA_CALLABLE inline auto neon_parentVal_atomic_write(
    NeonMultiresPartition<T> const& p,
    const NeonBlockIdx&             cell,
    int                             card,
    T value) -> void
{
    atomicAdd(&p.parentVal(cell, card), value);
}

template <typename T>
CUDA_CALLABLE inline auto neon_hasParent(
    NeonMultiresPartition<T> const& p,
    const NeonBlockIdx&             cell) -> bool
{
    return p.hasParent(cell);
}

template <typename T>
CUDA_CALLABLE inline auto neon_getUncle(
    NeonMultiresPartition<T> const& p,
    const NeonBlockIdx&             cell,
    const NeonNghIdx                direction) -> NeonBlockIdx
{
    return p.getUncle(cell, direction);
}

template <typename T>
CUDA_CALLABLE inline auto neon_uncleVal(
    NeonMultiresPartition<T> const& p,
    const NeonBlockIdx&             cell,
    const NeonNghIdx                direction,
    int                             card,
    T                               alternativeValue,
    bool&                           isValid) -> T
{
    typename NeonMultiresPartition<T>::NghData nghData =
        p.uncleVal(cell, direction, card, alternativeValue);
    isValid = nghData.isValid();
    return nghData.getData();
}

template <typename T>
CUDA_CALLABLE inline auto neon_uncleVal(
    NeonMultiresPartition<T> const& p,
    const NeonBlockIdx&             cell,
    const NeonNghIdx                direction,
    int                             card) -> T
{
    return p.uncleVal(cell, direction, card);
}

template <typename T>
CUDA_CALLABLE inline auto neon_getRefFactor(int level) -> int
{
    return p.getRefFactor(level);
}

template <typename T>
CUDA_CALLABLE inline auto neon_getSpacing(int level) -> int
{
    return p.neon_getSpacing(level);
}

}  // namespace wp