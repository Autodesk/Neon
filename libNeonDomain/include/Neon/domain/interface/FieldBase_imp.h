#pragma once

#include "Neon/domain/interface/FieldBase.h"
#include "Neon/domain/tools/IOGridVTK.h"

namespace Neon::domain::interface {


template <typename T, int C>
FieldBase<T, C>::FieldBase()
{
    mStorage = std::make_shared<Storage>();
}

template <typename T, int C>
FieldBase<T, C>::FieldBase(const std::string&             FieldBaseUserName,
                           const std::string&             fieldClassName,
                           const Neon::index_3d&          dimension,
                           int                            cardinality,
                           T                              outsideVal,
                           Neon::DataUse                  dataUse,
                           Neon::MemoryOptions            memoryOptions,
                           Neon::domain::haloStatus_et::e haloStatus,
                           const Vec_3d<double>&          spacing,
                           const Vec_3d<double>&          origin)
{
    mStorage = std::make_shared<Storage>(FieldBaseUserName,
                                         fieldClassName,
                                         dimension,
                                         cardinality,
                                         outsideVal,
                                         dataUse,
                                         memoryOptions,
                                         haloStatus,
                                         spacing,
                                         origin);
}

template <typename T, int C>
auto FieldBase<T, C>::getDimension() const
    -> const Neon::index_3d&
{
    return mStorage->dimension;
}

template <typename T, int C>
auto FieldBase<T, C>::getCardinality() const
    -> int
{
    return mStorage->cardinality;
}

template <typename T, int C>
auto FieldBase<T, C>::getOutsideValue() const
    -> T&
{
    return mStorage->outsideVal;
}

template <typename T, int C>
auto FieldBase<T, C>::getOutsideValue()
    -> T&
{
    return mStorage->outsideVal;
}


template <typename T, int C>
auto FieldBase<T, C>::getDataUse() const
    -> Neon::DataUse
{
    return mStorage->dataUse;
}

template <typename T, int C>
auto FieldBase<T, C>::getMemoryOptions() const
    -> const Neon::MemoryOptions&
{
    return mStorage->memoryOptions;
}

template <typename T, int C>
auto FieldBase<T, C>::getHaloStatus() const
    -> Neon::domain::haloStatus_et::e
{
    return mStorage->haloStatus;
}

template <typename T, int C>
auto FieldBase<T, C>::getName() const
    -> const std::string&
{
    return mStorage->name;
}

template <typename T, int C>
auto FieldBase<T, C>::forEachActiveCell(const std::function<void(const Neon::index_3d&,
                                                                 const int& cardinality,
                                                                 T&)>&     fun,
                                        Neon::computeMode_t::computeMode_e mode)
    -> void
{
    const auto& dim = getDimension();
    if (mode == Neon::computeMode_t::computeMode_e::par) {
#ifdef _MSC_VER
#pragma omp parallel for
#else
#pragma omp parallel for collapse(1) schedule(guided)
#endif
        for (int z = 0; z < dim.z; z++) {
            for (int y = 0; y < dim.y; y++) {
                for (int x = 0; x < dim.x; x++) {
                    for (int c = 0; c < getCardinality(); c++) {
                        Neon::index_3d index3D(x, y, z);
                        const bool     isInside = this->isInsideDomain(index3D);
                        if (isInside) {
                            auto& ref = this->getReference(index3D, c);
                            fun(index3D, c, ref);
                        }
                    }
                }
            }
        }
    } else {
        for (int z = 0; z < dim.z; z++) {
            for (int y = 0; y < dim.y; y++) {
                for (int x = 0; x < dim.x; x++) {
                    for (int c = 0; c < getCardinality(); c++) {
                        Neon::index_3d index3D(x, y, z);
                        const bool     isInside = this->isInsideDomain(index3D);
                        if (isInside) {
                            auto& ref = this->getReference(index3D, c);
                            fun(index3D, c, ref);
                        }
                    }
                }
            }
        }
    }
}


template <typename T, int C>
auto FieldBase<T, C>::forEachActiveCell(const std::function<void(const Neon::index_3d&,
                                                                 std::vector<T*>&)>& fun,
                                        Neon::computeMode_t::computeMode_e           mode)
    -> void
{
    const auto& dim = getDimension();
    std::vector<T*> vec(getCardinality(), nullptr);

    if (mode == Neon::computeMode_t::computeMode_e::par) {
#ifdef _MSC_VER
#pragma omp parallel for
#else
#pragma omp parallel for collapse(3) default(shared) firstprivate(vec)
#endif
        for (int z = 0; z < dim.z; z++) {
            for (int y = 0; y < dim.y; y++) {
                for (int x = 0; x < dim.x; x++) {
                    Neon::index_3d index3D(x, y, z);
                    const bool     isInside = this->isInsideDomain(index3D);
                    if (isInside) {
                        for (int c = 0; c < getCardinality(); c++) {
                            auto& ref = this->getReference(index3D, c);
                            vec.push_back(&ref);
                        }
                        fun(index3D, vec);
                    }
                }
            }
        }
    } else {
        for (int z = 0; z < dim.z; z++) {
            for (int y = 0; y < dim.y; y++) {
                for (int x = 0; x < dim.x; x++) {
                    Neon::index_3d index3D(x, y, z);
                    const bool     isInside = this->isInsideDomain(index3D);
                    if (isInside) {
                        for (int c = 0; c < getCardinality(); c++) {
                            auto& ref = this->getReference(index3D, c);
                            vec.push_back(&ref);
                        }
                        fun(index3D, vec);
                    }
                }
            }
        }
    }
}


template <typename T, int C>
template <Neon::computeMode_t::computeMode_e mode>
auto FieldBase<T, C>::forEachCell(const std::function<void(const Neon::index_3d&,
                                                           const int& cardinality,
                                                           T)>& fun) const
    -> void
{
    const auto& dim = getDimension();
    if constexpr (mode == Neon::computeMode_t::computeMode_e::par) {
#ifdef _MSC_VER
#pragma omp parallel for
#else
#pragma omp parallel for collapse(3)
#endif
        for (int z = 0; z < dim.z; z++) {
            for (int y = 0; y < dim.y; y++) {
                for (int x = 0; x < dim.x; x++) {
                    for (int c = 0; c < getCardinality(); c++) {
                        Neon::index_3d index3D(x, y, z);
                        auto           val = this->operator()(index3D, c);
                        fun(index3D, c, val);
                    }
                }
            }
        }
    } else {
        for (int z = 0; z < dim.z; z++) {
            for (int y = 0; y < dim.y; y++) {
                for (int x = 0; x < dim.x; x++) {
                    for (int c = 0; c < getCardinality(); c++) {
                        Neon::index_3d index3D(x, y, z);
                        auto           val = this->operator()(index3D, c);
                        fun(index3D, c, val);
                    }
                }
            }
        }
    }
}

template <typename T, int C>
template <typename ExportType,
          typename ExportIndex>
auto FieldBase<T, C>::ioToDense(NEON_OUT Neon::IODense<ExportType, ExportIndex>& ioDense) const
    -> void
{
    forEachCell<Neon::computeMode_t::par>([&](const Neon::index_3d& idx,
                                              const int&            c,
                                              T                     cellValue) mutable {
        ioDense.getReference(idx.template newType<ExportIndex>(), c) = cellValue;
    });
}

template <typename T, int C>
template <typename ExportType,
          typename ExportIndex>
auto FieldBase<T, C>::ioToDense() const
    -> Neon::IODense<ExportType, ExportIndex>
{
    return this->ioToDense<ExportType, ExportIndex>(getMemoryOptions().getOrder());
}

template <typename T, int C>
template <typename ExportType,
          typename ExportIndex>
auto FieldBase<T, C>::ioToDense(Neon::MemoryLayout order) const
    -> Neon::IODense<ExportType, ExportIndex>
{
    Neon::IODense<ExportType, ExportIndex> ioDense(getDimension(),
                                                   getCardinality(),
                                                   order);
    this->ioToDense(ioDense);
    return ioDense;
}

template <typename T, int C>
template <typename ImportType,
          typename ImportIndex>
auto FieldBase<T, C>::ioFromDense(const Neon::IODense<ImportType, ImportIndex>& ioDense)
    -> void
{
    {
        const int cardDense = ioDense.getCardinality();
        const int cardGrid = getCardinality();

        const auto dimensionDense = ioDense.getDimension().template newType<size_t>();
        const auto dimensionGrid = getDimension().template newType<size_t>();

        const bool cardinalityCheckFailed = (cardDense != cardGrid);
        const bool dimensionCheckFailed = dimensionDense != dimensionGrid;

        if (cardinalityCheckFailed || dimensionCheckFailed) {
            NeonException exp("FieldBase Interface - ioFromDense");
            exp << "FieldBase and ioDense are not compatible";
            NEON_THROW(exp);
        }
    }
    forEachActiveCell([&](const Neon::index_3d& point,
                          const int&            cardinality,
                          T&                    value) {
        value = ioDense(point.template newType<ImportIndex>(), cardinality);
    });
}

template <typename T, int C>
template <typename VtiExportType>
auto FieldBase<T, C>::ioToVtk(const std::string& fileName,
                              const std::string& FieldName,
                              bool               includeDomain,
                              Neon::IoFileType   ioFileType,
                              bool               isNodeSpace) const -> void
{

    auto iovtk = Neon::domain::IOGridVTK<VtiExportType>(this->getBaseGridTool(), fileName, isNodeSpace, ioFileType);
    iovtk.addField(*this, FieldName);

    Neon::IODense<VtiExportType, int32_t> domain(getDimension(), 1, [&](const Neon::index_3d& idx, int) {
        VtiExportType setIdx = VtiExportType(getBaseGridTool().getSetIdx(idx));
        return setIdx;
    });

    if (includeDomain) {
        iovtk.addIODenseField(domain, "Domain");
    }
    iovtk.flushAndClear();
    return;
}

template <typename T, int C>
auto FieldBase<T, C>::getClassName() const -> const std::string&
{
    return mStorage->className;
}

template <typename T, int C>
FieldBase<T, C>::Storage::Storage(const std::string              FieldBaseUserName,
                                  const std::string              fieldClassName,
                                  const Neon::index_3d&          dimension,
                                  int                            cardinality,
                                  T                              outsideVal,
                                  Neon::DataUse                  dataUse,
                                  Neon::MemoryOptions            memoryOptions,
                                  Neon::domain::haloStatus_et::e haloStatus,
                                  const Vec_3d<double>&          spacing,
                                  const Vec_3d<double>&          origin)
    : name(FieldBaseUserName),
      className(fieldClassName),
      dimension(dimension),
      cardinality(cardinality),
      outsideVal(outsideVal),
      dataUse(dataUse),
      memoryOptions(memoryOptions),
      haloStatus(haloStatus),
      spacing(spacing),
      origin(origin)
{
    if (name == "") {
        name = "Anonymous";
    }
}

#if defined(NEON_OS_WINDOWS)
#pragma warning(push)
#pragma warning(disable : 4244)
#endif
template <typename T, int C>
FieldBase<T, C>::Storage::Storage()
    : dimension(0),
      cardinality(0),
      outsideVal(T()),
      dataUse(),
      memoryOptions(),
      haloStatus(),
      spacing(0.0),
      origin(0.0)
{
}
#if defined(NEON_OS_WINDOWS)
#pragma warning(pop)
#endif

}  // namespace Neon::domain::interface
