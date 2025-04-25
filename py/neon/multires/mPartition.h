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
CUDA_CALLABLE inline auto neon_read_ngh(
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
CUDA_CALLABLE inline auto neon_read_uncle(
    const NeonMultiresPartition<T>& p,
    NeonBlockIdx const&             idx,
    NeonNghIdx const&               ngh,
    int                             card,
    T                               alternative,
    bool&                           valid) -> T
{
    auto getUncleOffset  = [] (const Neon::int8_3d& cell,
                               const Neon::int8_3d& q)->Neon::int8_3d{//uncleOffset
         //given a local index within a cell and a population direction (q)
        //find the uncle's (the parent neighbor) offset from which the desired population (q) should be read
        //this offset is wrt the cell containing the localID (i.e., the parent of localID)
        auto off = [](const int8_t i, const int8_t j) ->int8_t {
            //0, -1 --> -1
            //1, -1 --> 0
            //0, 0 --> 0
            //0, 1 --> 0
            //1, 1 --> 1
            const int8_t s = i + j;
            return (s <= 0) ? s : s - 1;
        };

        Neon::int8_3d offset(off(cell.x % Neon::domain::details::mGrid::kUserBlockSizeX, q.x),
                             off(cell.y % Neon::domain::details::mGrid::kUserBlockSizeY, q.y),
                             off(cell.z % Neon::domain::details::mGrid::kUserBlockSizeZ, q.z));
        return offset;
    };
    Neon::int8_3d uncleDir = getUncleOffset(idx.mInDataBlockIdx, ngh);
    const auto   uncleData = p.uncleVal(idx, uncleDir, card, alternative);
    valid = uncleData.isValid();
    return uncleData.getData();
}


template <typename T>
CUDA_CALLABLE inline auto neon_mres_lbm_store_op( const NeonMultiresPartition<T>& pout,
                                         const NeonBlockIdx& cell,
                                        const int8_t                             q,
                                        NeonNghIdx const&               qDir,
                                        const T                                  cellVal) -> void
{

    auto uncleOffset =[](const auto& cell, const Neon::int8_3d& q)->Neon::int8_3d
    {
        //given a local index within a cell and a population direction (q)
        //find the uncle's (the parent neighbor) offset from which the desired population (q) should be read
        //this offset is wrt the cell containing the localID (i.e., the parent of localID)
        auto off = [](const int8_t i, const int8_t j) {
            //0, -1 --> -1
            //1, -1 --> 0
            //0, 0 --> 0
            //0, 1 --> 0
            //1, 1 --> 1
            const int8_t s = i + j;
            return (s <= 0) ? s : s - 1;
        };
        Neon::int8_3d offset(off(cell.x % Neon::domain::details::mGrid::kUserBlockSizeX, q.x),
                             off(cell.y % Neon::domain::details::mGrid::kUserBlockSizeY, q.y),
                             off(cell.z % Neon::domain::details::mGrid::kUserBlockSizeZ, q.z));
        return offset;
    };

    if (qDir.x == 0 && qDir.y == 0 && qDir.z == 0) {
        return;
    }

    const Neon::int8_3d uncleDir = uncleOffset(cell.mInDataBlockIdx, qDir);

    //we try to access a cell on the same level (i.e., the refined level) along the same
    //direction as the uncle and we use this a proxy to check if there is an unrefined uncle
    const auto cn = pout.helpGetNghIdx(cell, uncleDir);

    //cn may not be active because 1. it is outside the domain, or 2. this location is occupied by a coarse cell
    //we are interested in 2.
    if (!pout.isActive(cn)) {

        //now, we can get the uncle but we need to make sure it is active i.e.,
        //it is not out side the domain boundary
        const auto uncle = pout.getUncle(cell, uncleDir);
        if (uncle.isActive()) {

            //locate the coarse cell where we should store this cell info
            const Neon::int8_3d CsDir = uncleDir - qDir;

            const auto cs = pout.getUncle(cell, CsDir);

            const auto csChild = pout.helpGetNghIdx(cell, CsDir);

            if (cs.isActive() && pout.isActive(csChild)) {
                atomicAdd(&pout.uncleVal(cell, CsDir, q), cellVal);
            }
        }
    }
}


template <typename T>
CUDA_CALLABLE inline auto neon_is_active(
    const NeonMultiresPartition<T>& p,
    NeonBlockIdx const&             idx,
    NeonNghIdx const&               ngh) -> bool
{
    bool isValid status = p.isValid(idx, ngh);
    return status;
}

template <typename T>
CUDA_CALLABLE inline auto neon_ngh_idx(
    const NeonMultiresPartition<T>& p,
    NeonBlockIdx const&             idx,
    NeonNghIdx const&               ngh) -> NeonBlockIdx
{
    NeonBlockIdx const ngh_idx = p.helpGetNghIdx(idx, ngh);
    return ngh_idx;
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
CUDA_CALLABLE inline auto neon_read_child(
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
CUDA_CALLABLE inline auto neon_get_child(
    NeonMultiresPartition<T>& p,
    const NeonBlockIdx&       parentCell,
    const NeonNghIdx          child) -> NeonNghIdx
{
    return p.getChild(parentCell, child);
}

template <typename T>
CUDA_CALLABLE inline auto neon_read_child(
    NeonMultiresPartition<T>& p,
    const NeonBlockIdx&       childIdx,
    int                       card) -> T
{
    return p.childVal(childIdx, card);
}

template <typename T>
CUDA_CALLABLE inline auto neon_has_child(
    NeonMultiresPartition<T>& p,
    const NeonBlockIdx&       idx) -> bool
{
    return p.hasChildren(idx);
}

template <typename T>
CUDA_CALLABLE inline auto neon_has_child(
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
CUDA_CALLABLE inline auto neon_read_parent(
    NeonMultiresPartition<T> const& p,
    const NeonBlockIdx&             cell,
    int                             card) -> T
{
    return p.parentVal(cell, card);
}


template <typename T>
CUDA_CALLABLE inline auto neon_write_parent(
    NeonMultiresPartition<T> const& p,
    const NeonBlockIdx&             cell,
    int                             card,
    T value) -> void
{
    return p.parentVal(cell, card) = value;
}

//template <typename T>
//CUDA_CALLABLE inline auto neon_parentVal_atomic_write(
//    NeonMultiresPartition<T> const& p,
//    const NeonBlockIdx&             cell,
//    int                             card,
//    T value) -> void
//{
//    atomicAdd(&p.parentVal(cell, card), value);
//}

template <typename T>
CUDA_CALLABLE inline auto neon_has_parent(
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
CUDA_CALLABLE inline auto neon_read_uncle(
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
CUDA_CALLABLE inline auto neon_read_uncle(
    NeonMultiresPartition<T> const& p,
    const NeonBlockIdx&             cell,
    const NeonNghIdx                direction,
    int                             card) -> T
{
    return p.uncleVal(cell, direction, card);
}

template <typename T>
CUDA_CALLABLE inline auto neon_refinement_factor(int level) -> int
{
    return p.getRefFactor(level);
}

template <typename T>
CUDA_CALLABLE inline auto neon_spacing(int level) -> int
{
    return p.neon_getSpacing(level);
}

}  // namespace wp
