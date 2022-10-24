#include <omp.h>
#include "IODense.h"

namespace Neon {

template <typename ExportType,
          typename IntType>
IODense<ExportType, IntType>::IODense()
    : mMemSharedPtr(),
      mMem(nullptr),
      mSpace({0, 0, 0}),
      mCardinality(0),
      mOrder(),
      mRepresentation(Representation::EXPLICIT)
{
    initPitch();
}

template <typename ExportType,
          typename IntType>
IODense<ExportType, IntType>::IODense(const Integer_3d<IntType>&     d,
                                      int                            c,
                                      std::shared_ptr<ExportType[]>& m,
                                      Neon::memLayout_et::order_e    order)
    : mMemSharedPtr(m), mMem(m.get()), mSpace(d), mCardinality(c), mOrder(order)
{
    initPitch();
}

template <typename ExportType,
          typename IntType>
IODense<ExportType, IntType>::IODense(const Integer_3d<IntType>&  d,
                                      int                         c,
                                      ExportType*                 m,
                                      Neon::memLayout_et::order_e order)
    : mMemSharedPtr(), mMem(m), mSpace(d), mCardinality(c), mOrder(order), mRepresentation(Representation::EXPLICIT)

{
    initPitch();
}

template <typename ExportType,
          typename IntType>
IODense<ExportType, IntType>::IODense(const Integer_3d<IntType>&  d,
                                      int                         c,
                                      Neon::memLayout_et::order_e order)
    : mSpace(d), mCardinality(c), mOrder(order), mRepresentation(Representation::EXPLICIT)

{
    const size_t                  cardJump = d.template rMulTyped<size_t>();
    std::shared_ptr<ExportType[]> mem(new ExportType[cardJump * c]);
    mMemSharedPtr = mem;
    mMem = mem.get();
    initPitch();
}

template <typename ExportType,
          typename IntType>
IODense<ExportType, IntType>::IODense(const Integer_3d<IntType>&                                             d,
                                      int                                                                    c,
                                      const std::function<ExportType(const Integer_3d<IntType>&, int cardinality)>& fun)
    : mSpace(d), mCardinality(c), mRepresentation(Representation::IMPLICIT), mImplicitFun(fun)

{
    initPitch();
}

template <typename ExportType,
          typename IntType>
template <typename Lambda_ta>
auto IODense<ExportType, IntType>::densify(const Lambda_ta&           fun,
                                           const Integer_3d<IntType>& space,
                                           int                        cardinality,
                                           Representation             representation)
    -> IODense<ExportType, IntType>
{
    if (representation == Representation::EXPLICIT) {
        const size_t                  cardJump = space.template rMulTyped<size_t>();
        std::shared_ptr<ExportType[]> m_mem(new ExportType[cardJump * cardinality]);
        IODense<ExportType, IntType>  dense(space, cardinality, m_mem);
        dense.forEach([&](const Integer_3d<IntType>& idx, int c, ExportType& val) {
            val = ExportType(fun(idx, c));
        });
        return dense;
    } else {
        IODense<ExportType, IntType> dense(space, cardinality, fun);
    }
}

template <typename ExportType,
          typename IntType>
auto IODense<ExportType, IntType>::makeLinear(ExportType                 offset,
                                              const Integer_3d<IntType>& space /*!          dense grid dimension */,
                                              int                        cardinality /*!    Field cardinality */)
    -> IODense<ExportType>
{
    IODense<ExportType, IntType> dense(space, cardinality);
    dense.resetValuesToLinear(offset);
    return dense;
}

template <typename ExportType,
          typename IntType>
auto IODense<ExportType, IntType>::makeRandom(int                        min,
                                              int                        max,
                                              const Integer_3d<IntType>& space,
                                              int                        cardinality)
    -> IODense<ExportType>
{
    IODense<ExportType, IntType> dense(space, cardinality);
    dense.resetValuesToRandom(min, max);
    return dense;
}

template <typename ExportType,
          typename IntType>
auto IODense<ExportType, IntType>::makeMasked(ExportType                 offset,
                                              const Integer_3d<IntType>& space,
                                              int                        cardinality,
                                              int                        digit)
    -> IODense<ExportType>
{
    IODense<ExportType, IntType> dense(space, cardinality);
    dense.resetValuesToMasked(offset, digit);
    return dense;
}

template <typename ExportType,
          typename IntType>
auto IODense<ExportType, IntType>::getDimension() const
    -> Integer_3d<IntType>
{
    return mSpace;
}

template <typename ExportType,
          typename IntType>
auto IODense<ExportType, IntType>::getCardinality() const -> int
{
    return mCardinality;
}

template <typename ExportType,
          typename IntType>
auto IODense<ExportType, IntType>::getMemory() -> ExportType*
{
    if (mRepresentation == Representation::IMPLICIT) {
        NEON_THROW_UNSUPPORTED_OPERATION("A IODense configure as IMPLICIT does not support such operation");
    }
    return mMem;
}

template <typename ExportType,
          typename IntType>
auto IODense<ExportType, IntType>::getSharedPtr()
    -> std::shared_ptr<ExportType[]>
{
    if (mRepresentation == Representation::IMPLICIT) {
        NEON_THROW_UNSUPPORTED_OPERATION("A IODense configure as IMPLICIT does not support such operation");
    }
    return mMemSharedPtr;
}


template <typename ExportType,
          typename IntType>
auto IODense<ExportType, IntType>::operator()(const Integer_3d<IntType>& xyz,
                                              int                        card) const
    -> ExportType
{
    if (mRepresentation == Representation::IMPLICIT) {
        return mImplicitFun(xyz, card);
    }
    const size_t pitch =
        mPitch.mXpitch * xyz.x +
        mPitch.mYpitch * xyz.y +
        mPitch.mZpitch * xyz.z +
        mPitch.mCpitch * card;
    return mMem[pitch];
}

template <typename ExportType,
          typename IntType>
auto IODense<ExportType, IntType>::getReference(const Integer_3d<IntType>& xyz,
                                                int                        card)
    -> ExportType&
{
    if (mRepresentation == Representation::IMPLICIT) {
        NEON_THROW_UNSUPPORTED_OPERATION("A IODense configure as IMPLICIT does not support such operation");
    }
    const size_t pitch =
        mPitch.mXpitch * xyz.x +
        mPitch.mYpitch * xyz.y +
        mPitch.mZpitch * xyz.z +
        mPitch.mCpitch * card;
    return mMem[pitch];
}

template <typename ExportType,
          typename IntType>
template <typename Lambda_ta,
          typename... ExportTypeVariadic_ta>
auto IODense<ExportType, IntType>::forEach(const Lambda_ta& lambda,
                                           IODense<ExportTypeVariadic_ta>&... otherDense)
    -> void
{
    if (mRepresentation == Representation::IMPLICIT) {
        NEON_THROW_UNSUPPORTED_OPERATION("A IODense configure as IMPLICIT does not support such operation");
    }
#pragma omp parallel for collapse(3)
    for (int z = 0; z < mSpace.z; z++) {
        for (int y = 0; y < mSpace.y; y++) {
            for (int x = 0; x < mSpace.x; x++) {
                for (int c = 0; c < mCardinality; c++) {
                    index_3d xyz(x, y, z);
                    lambda(xyz,
                           c,
                           getReference(xyz, c),
                           otherDense.getReference(xyz, c)...);
                }
            }
        }
    }
}

template <typename ExportType,
          typename IntType>
template <typename Lambda_ta,
          typename... ExportTypeVariadic_ta>
auto IODense<ExportType, IntType>::forEach(const Lambda_ta& lambda,
                                           const IODense<ExportTypeVariadic_ta>&... otherDense)
    const -> void
{
#pragma omp parallel for collapse(3)
    for (int z = 0; z < mSpace.z; z++) {
        for (int y = 0; y < mSpace.y; y++) {
            for (int x = 0; x < mSpace.x; x++) {
                for (int c = 0; c < mCardinality; c++) {
                    index_3d xyz(x, y, z);
                    lambda(xyz, c, operator()(xyz, c), otherDense(xyz, c)...);
                }
            }
        }
    }
}

template <typename ExportType,
          typename IntType>
auto IODense<ExportType, IntType>::maxDiff(const IODense<ExportType, IntType>& a,
                                           const IODense<ExportType, IntType>& b)
    -> std::tuple<ExportType,
                  Neon::index_3d,
                  int>
{
    if (a.getCardinality() != b.getCardinality()) {
        Neon::NeonException exception("IoDense");
        exception << "Incompatible cardinalities";
        NEON_THROW(exception);
    }

    constexpr int valueId = 0;
    constexpr int idx3dId = 1;
    constexpr int cardId = 2;

    const int nThreads = omp_get_max_threads();
    // Adding 128 byte to avoid false sharing
    using Values = std::tuple<ExportType, Neon::index_3d, int, std::array<char, 128>>;

    std::vector<Values> max_diff(nThreads);
    for (auto& val : max_diff) {
        std::get<valueId>(val) = -1;
        std::get<idx3dId>(val) = -1;
        std::get<cardId>(val) = -1;
    }

    a.forEach([&](const Integer_3d<IntType>& idx,
                  int                        c,
                  const ExportType&          valA,
                  const ExportType&          valB) {
        const auto newDiff = std::abs(valA - valB);
        const int  threadId = omp_get_thread_num();
        if (newDiff > std::get<0>(max_diff[threadId])) {
#pragma omp critical
            if (newDiff > std::get<0>(max_diff[threadId])) {
            Values& md = max_diff[threadId];
            std::get<0>(md) = newDiff;
            std::get<1>(md) = idx;
            std::get<2>(md) = c;
        }
        }
    },
              b);

    int target = 0;
    for (auto i = 1; i < nThreads; i++) {
        if (std::get<valueId>(max_diff[i]) > std::get<valueId>(max_diff[target])) {
            target = i;
        }
    }
    return std::make_tuple(std::get<valueId>(max_diff[target]),
                           std::get<idx3dId>(max_diff[target]),
                           std::get<cardId>(max_diff[target]));
}
template <typename ExportType,
          typename IntType>
template <typename ExportTypeVTK_ta>
auto IODense<ExportType, IntType>::ioVtk(const std::string&       filename,
                                         const std::string&       fieldName,
                                         ioToVTKns::VtiDataType_e nodeOrVoxel,
                                         const Vec_3d<double>&    spacingData,
                                         const Vec_3d<double>&    origin,
                                         IoFileType               vtiIOe) -> void
{
    IoToVTK<IntType, ExportTypeVTK_ta> ioVtk(filename,
                                             nodeOrVoxel == ioToVTKns::VtiDataType_e::node ? getDimension() : getDimension() + 1,
                                             spacingData,
                                             origin,
                                             vtiIOe);

    ioVtk.addField([&](index_3d idx, int card) -> ExportTypeVTK_ta {
        return operator()(idx, card);
    },
                   getCardinality(), fieldName, nodeOrVoxel);
    ioVtk.flush();
}
template <typename ExportType,
          typename IntType>
auto IODense<ExportType, IntType>::initPitch() -> void
{
    if (mOrder == Neon::memLayout_et::order_e::structOfArrays) {
        mPitch.mXpitch = 1;
        mPitch.mYpitch = static_cast<size_t>(mSpace.x);

        mPitch.mZpitch = static_cast<size_t>(mSpace.x) *
                         static_cast<size_t>(mSpace.y);

        mPitch.mCpitch = static_cast<size_t>(mSpace.x) *
                         static_cast<size_t>(mSpace.y) *
                         static_cast<size_t>(mSpace.z);
    } else {
        mPitch.mXpitch = mCardinality;
        mPitch.mYpitch = mCardinality *
                         static_cast<size_t>(mSpace.x);
        mPitch.mZpitch = mCardinality *
                         static_cast<size_t>(mSpace.x) *
                         static_cast<size_t>(mSpace.y);
        mPitch.mCpitch = 1;
    }
}
template <typename ExportType, typename IntType>
auto IODense<ExportType, IntType>::resetValuesToLinear(ExportType offset) -> void
{
    if (mRepresentation == Representation::IMPLICIT) {
        NEON_THROW_UNSUPPORTED_OPERATION("A IODense configure as IMPLICIT does not support such operation");
    }
    auto&        field = *this;
    const auto   space = field.getDimension();
    const size_t cardJump = space.template rMulTyped<size_t>();
    field.forEach([&](const Integer_3d<IntType>& idx, int c, ExportType& val) {
        val = offset + ExportType(idx.mPitch(space)) + ExportType(c * cardJump);
    });
}
template <typename ExportType, typename IntType>
auto IODense<ExportType, IntType>::resetValuesToRandom(int min, int max) -> void
{
    if (mRepresentation == Representation::IMPLICIT) {
        NEON_THROW_UNSUPPORTED_OPERATION("A IODense configure as IMPLICIT does not support such operation");
    }
    auto& field = *this;

    std::random_device                 rd;
    std::mt19937                       mt(rd());
    std::uniform_int_distribution<int> dist(min, max);

    field.forEach([&](const Integer_3d<IntType>&, int, ExportType& val) {
        val = ExportType(dist(mt));
    });
}
template <typename ExportType, typename IntType>
auto IODense<ExportType, IntType>::resetValuesToMasked(ExportType offset, int digit)
    -> void
{
    if (mRepresentation == Representation::IMPLICIT) {
        NEON_THROW_UNSUPPORTED_OPERATION("A IODense configure as IMPLICIT does not support such operation");
    }
    auto& field = *this;

    const auto space = field.getDimension();
    const int  multiplier = int(std::pow(10, digit));

    field.forEach([&](const Integer_3d<IntType>& idx, int c, ExportType& val) {
        val = idx.z +
              idx.y * multiplier +
              idx.x * multiplier * multiplier +
              c * multiplier * multiplier * multiplier +
              offset * multiplier * multiplier * multiplier * 10;
    });
}
template <typename ExportType, typename IntType>
auto IODense<ExportType, IntType>::resetValuesToConst(ExportType offset) -> void
{
    if (mRepresentation == Representation::IMPLICIT) {
        NEON_THROW_UNSUPPORTED_OPERATION("A IODense configure as IMPLICIT does not support such operation");
    }
    auto& field = *this;

    field.forEach([&](const Integer_3d<IntType>& idx, int c, ExportType& val) {
        val = offset;
    });
}

}  // namespace Neon
