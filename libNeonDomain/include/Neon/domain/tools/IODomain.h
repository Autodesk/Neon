/**
 * This file contains a set of tools to manage dense grids on the CPU
 * These are useful to convert any grid into a dense representation,
 * to easily convert external data to a format that Neon grids can load
 * and store. The dense representation also includes capabilities to compare
 * the values of two different grids.
 */
#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include "Neon/core/core.h"
#include "Neon/core/tools/io/IODense.h"
#include "Neon/core/tools/io/ioToVTK.h"
#include "Neon/core/types/vec.h"


namespace Neon::domain::tool::testing {

/**
 * An abstraction for fields over a dense 3D grid.
 * @tparam ExportType
 * @tparam intType_ta
 */
template <typename ExportType,
          typename intType_ta = int>
struct IODomain
{
    using Index = intType_ta /** type used to index elements */;
    using Type = ExportType /** Type of the data stored in the grid cells */;

    using FlagType = uint8_t;
    static constexpr FlagType InsideFlag = 1;
    static constexpr FlagType OutsideFlag = 0;

    IODomain();

    IODomain(const IODense<Type, Index>&    values,
             const IODense<uint8_t, Index>& mask,
             Type                           outsideValue);

    IODomain(const Integer_3d<intType_ta>&  d,
             int                            c,
             const IODense<uint8_t, Index>& mask,
             Type                           outsideValue = Type(0),
             Neon::MemoryLayout             order = Neon::MemoryLayout::structOfArrays);

    auto setMask(const IODense<uint8_t, Index>& mask)
        -> void;

    auto getMask()
        -> const IODense<uint8_t, Index>&;

    auto getData() -> Neon::IODense<Type, Index>;

    auto resetValuesToLinear(ExportType offset)
        -> void;

    auto resetValuesToRandom(int min,
                             int max)
        -> void;

    auto resetValuesToMasked(ExportType offset,
                             int        digit = 2)
        -> void;

    auto resetValuesToConst(ExportType offset)
        -> void;


    /**
     * The space of the gris
     */
    auto getDimension() const
        -> Integer_3d<intType_ta>;

    /**
     * Cardinality of the data over the grid
     * @return
     */
    auto getCardinality() const -> int;

    auto getOutsideValue() const
        -> Type;

    auto setValue(ExportType&                   val,
                  const Integer_3d<intType_ta>& xyz /**< Point in the grid        */,
                  int                           card /**< Cardinality of the field */)
        -> bool;

    /**
     * Accessing a point in the field. Read only mode.
     */
    auto getValue(const Integer_3d<intType_ta>& xyz /**< Point in the grid        */,
                  int                           card /**< Cardinality of the field */,
                  bool*                         wasInside = nullptr)
        const -> Type;

    /**
     * For each operator to visit all field elements in parallel
     */
    template <typename Lambda_ta, typename... ExportTypeVariadic_ta>
    auto forEachActive(const Lambda_ta& lambda /**< User function                                                    */,
                       IODomain<ExportTypeVariadic_ta>&... otherDense /**< Optional. Other fields that may be needed during the field visit */)
        -> void;

    /**
     * For each operator to visit all field elements in parallel.
     * Read only mode
     */
    template <typename Lambda_ta, typename... ExportTypeVariadic_ta>
    auto forEachActive(const Lambda_ta& lambda /**< User function                                                    */,
                       const IODomain<ExportTypeVariadic_ta>&... otherDense /**< Optional. Other fields that may be needed during the field visit */)
        const -> void;

    /**
     * Computing the max different component by component.
     */
    static auto maxDiff(const IODomain<ExportType, intType_ta>& a,
                        const IODomain<ExportType, intType_ta>& b)
        -> std::tuple<ExportType /**< the max difference value */,
                      Neon::index_3d /**< the location of the max difference */,
                      int /**< the cardinality of the max difference */>;

    auto isNghActive(const Integer_3d<intType_ta>& xyz,
                     const Neon::int8_3d&          offset) -> bool;

    auto nghVal(const Integer_3d<intType_ta>& xyz,
                const Neon::int8_3d&          offset,
                int                           card,
                bool*                         isValid = nullptr) -> Type;

    auto isActive(const Integer_3d<intType_ta>& xyz) const -> bool;

    auto isInBox(const Integer_3d<intType_ta>& xyz) const -> bool;

    /**
     * Exporting to vtk
     * @return
     */
    template <typename ExportTypeVTK_ta = ExportType>
    auto ioToVti(const std::string&       filename /*!                              File name */,
                 const std::string&       fieldName /*!                              Field name */,
                 ioToVTKns::VtiDataType_e nodeOrVoxel = ioToVTKns::VtiDataType_e::voxel,
                 const Vec_3d<double>&    spacingData = Vec_3d<double>(1, 1, 1) /*! Spacing, i.e. size of a voxel */,
                 const Vec_3d<double>&    origin = Vec_3d<double>(0, 0, 0) /*!      Origin  */,
                 IoFileType               vtiIOe = IoFileType::ASCII /*!            Binary or ASCII file  */);

   private:
    auto getReference(const Integer_3d<intType_ta>& xyz /**< Point in the grid        */,
                      int                           card /**< Cardinality of the field */)
        -> Type&;

    auto getReference(const Integer_3d<intType_ta>& xyz /**< Point in the grid        */,
                      int                           card /**< Cardinality of the field */)
        const -> const Type&;

    Type                          mOutsideValue;
    Neon::IODense<Type, Index>    mField;
    Neon::IODense<uint8_t, Index> mMask;
};


template <typename ExportType, typename intType_ta>
IODomain<ExportType, intType_ta>::IODomain() = default;

template <typename ExportType, typename intType_ta>
IODomain<ExportType, intType_ta>::IODomain(const IODense<Type, Index>&    values,
                                           const IODense<uint8_t, Index>& mask,
                                           Type                           outsideValue)
{
    mOutsideValue = outsideValue;
    mField = values;
    mMask = mask;
}

template <typename ExportType, typename intType_ta>
IODomain<ExportType, intType_ta>::IODomain(const Integer_3d<intType_ta>&  d,
                                           int                            c,
                                           const IODense<uint8_t, Index>& mask,
                                           Type                           outsideValue,
                                           Neon::MemoryLayout             order)
{
    mOutsideValue = outsideValue;
    mField = Neon::IODense<Type, Index>(d, c, order);
    mMask = mask;
}

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::setMask(const IODense<uint8_t, Index>& mask) -> void
{
    mMask = mask;
}

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::getMask() -> const IODense<uint8_t, Index>&
{
    return mMask;
}

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::resetValuesToLinear(ExportType offset) -> void
{
    const auto   space = mField.getDimension();
    const size_t cardJump = space.template rMulTyped<size_t>();
    mField.forEach([&](const Integer_3d<intType_ta>& idx, int c, ExportType& val) {
        const bool active = isActive(idx);
        if (active) {
            val = offset + ExportType(idx.mPitch(space)) + ExportType(c * cardJump);
        } else {
            val = getOutsideValue();
        }
    });
}

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::resetValuesToRandom(int min,
                                                           int max) -> void
{

    std::random_device                 rd;
    std::mt19937                       mt(rd());
    std::uniform_int_distribution<int> dist(min, max);

    mField.forEach([&](const Integer_3d<intType_ta>&, int, ExportType& val) {
        val = ExportType(dist(mt));
    });
}

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::resetValuesToMasked(ExportType offset,
                                                           int        digit) -> void
{
    const auto space = mField.getDimension();
    const int  cardinality = mField.getCardinality();
    const int  multiplier = int(std::pow(10, digit));

    mField.forEach([&](const Integer_3d<intType_ta>& idx, int c, ExportType& val) {
        val = idx.z +
              idx.y * multiplier +
              idx.x * multiplier * multiplier +
              c * multiplier * multiplier * multiplier +
              offset * multiplier * multiplier * multiplier * 10;
    });
}

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::resetValuesToConst(ExportType offset) -> void
{
    mField.forEach([&](const Integer_3d<intType_ta>&, int, ExportType& val) {
        val = offset;
    });
}

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::getDimension() const -> Integer_3d<intType_ta>
{
    return mMask.getDimension();
}

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::getCardinality() const -> int
{
    return mField.getCardinality();
}

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::setValue(ExportType&                   val,
                                                const Integer_3d<intType_ta>& xyz,
                                                int                           card) -> bool
{
    const bool isValid = isActive(xyz, card);
    if (!isValid) {
        return false;
    }
    mField(xyz, card) = val;
    return true;
}

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::getValue(const Integer_3d<intType_ta>& xyz,
                                                int                           card,
                                                bool*                         wasInside) const -> Type
{
    const bool isValid = isActive(xyz);
    if (wasInside != nullptr)
        *wasInside = isValid;
    if (!isValid) {
        return mOutsideValue;
    }
    auto val = mField(xyz, card);

    return val;
}

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::getReference(const Integer_3d<intType_ta>& xyz,
                                                    int                           card) -> Type&
{
    const bool isValid = isActive(xyz);
    if (!isValid) {
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
    auto& val = mField.getReference(xyz, card);
    return val;
}

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::getReference(const Integer_3d<intType_ta>& xyz,
                                                    int                           card) const -> const Type&
{
    const bool isValid = isActive(xyz);
    if (!isValid) {
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
    auto& val = mField.getReference(xyz, card);
    return val;
}

template <typename ExportType, typename intType_ta>
template <typename Lambda_ta, typename... ExportTypeVariadic_ta>
auto IODomain<ExportType, intType_ta>::forEachActive(const Lambda_ta& userLambda,
                                                     IODomain<ExportTypeVariadic_ta>&... otherDense) -> void
{
    mMask.forEach([&, this](const Neon::index_3d& idx, int /*card*/, typename decltype(mMask)::Type& val) -> void {
        const bool isA = val == InsideFlag;
        if (!isA) {
            return;
        }
        for (int cc = 0; cc < this->getCardinality(); cc++) {
            userLambda(idx, cc, getReference(idx, cc), otherDense.getReference(idx, cc)...);
        }
    });
}

template <typename ExportType, typename intType_ta>
template <typename Lambda_ta, typename... ExportTypeVariadic_ta>
auto IODomain<ExportType, intType_ta>::forEachActive(const Lambda_ta& lambda,
                                                     const IODomain<ExportTypeVariadic_ta>&... otherDense) const -> void
{
    mMask.forEach([&, this](const Neon::index_3d& idx, int card, Type& val) -> void {
        const bool isA = isActive(idx);
        if (!isA) {
            return;
        }
        userLambda((*this)(idx, card), otherDense()(idx, card)...);
    });
}

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::maxDiff(const IODomain<ExportType, intType_ta>& a,
                                               const IODomain<ExportType, intType_ta>& b)
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

    a.forEachActive([&](const Integer_3d<intType_ta>& idx,
                        int                           c,
                        const ExportType&             valA,
                        const ExportType&             valB) {
        const auto newDiff = std::abs(valA - valB);
        const int  threadId = omp_get_thread_num();
        if (newDiff > std::get<0>(max_diff[threadId])) {
            Values& md = max_diff[threadId];
            std::get<0>(md) = newDiff;
            std::get<1>(md) = idx;
            std::get<2>(md) = c;
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

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::isNghActive(const Integer_3d<intType_ta>& xyz,
                                                   const int8_3d&                offset)
    -> bool
{
    auto nghIDX = xyz + offset.template newType<Neon::index_1d>();

    const bool inBox = isInBox(nghIDX);
    if (!inBox) {
        return false;
    }
    const bool active = isActive(nghIDX);
    if (!active) {
        return false;
    }
    return true;
}

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::nghVal(const Integer_3d<intType_ta>& xyz,
                                              const int8_3d&                offset,
                                              int                           card,
                                              bool*                         isValidPtr)
    -> Type
{
    const bool nghActive = isNghActive(xyz, offset);
    if (!nghActive) {
        if (isValidPtr != nullptr)
            (*isValidPtr) = false;
        return mOutsideValue;
    }
    if (isValidPtr != nullptr)
        (*isValidPtr) = true;
    auto nghIDX = xyz + offset.template newType<Neon::index_1d>();

    auto val = getValue(nghIDX, card);
    return val;
}

template <typename ExportType, typename intType_ta>
template <typename ExportTypeVTK_ta>
auto IODomain<ExportType, intType_ta>::ioToVti(const std::string&       filename,
                                               const std::string&       fieldName,
                                               ioToVTKns::VtiDataType_e nodeOrVoxel,
                                               const Vec_3d<double>&    spacingData,
                                               const Vec_3d<double>&    origin,
                                               IoFileType               vtiIOe)
{
    mField.ioVtk(filename, fieldName, nodeOrVoxel, spacingData, origin, vtiIOe);
}

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::isActive(const Integer_3d<intType_ta>& xyz) const -> bool
{
    if (isInBox(xyz)) {
        bool const active = mMask(xyz, 0) == InsideFlag;
        return active;
    }
    return false;
}

template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::isInBox(const Integer_3d<intType_ta>& xyz) const -> bool
{
    bool testA = mMask.getDimension() > xyz;
    bool testB = Neon::index_3d(0, 0, 0) <= xyz;
    bool andAB = testA && testB;

    return andAB;
}
template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::getOutsideValue() const -> Type
{
    return mOutsideValue;
}
template <typename ExportType, typename intType_ta>
auto IODomain<ExportType, intType_ta>::getData() -> IODense<Type, Index>
{
    return mField;
}


}  // namespace Neon::domain::tool::testing
