#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/HuOptions.h"

#include "VoxelPartition.h"

#include "VoxelField.h"

namespace Neon::domain::internal::experimental::staggeredGrid::details {


template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::self()
    -> VoxelField::Self&
{
    return *this;
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::
    self()
        const -> const VoxelField::Self&
{
    return *this;
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::
     operator()(const index_3d& idx,
           const int&      cardinality)
    const -> Type
{
    return this->getStorage().getBuildingBlockField().operator()(idx, cardinality);
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::
    haloUpdate(set::HuOptions& opt)
        const -> void
{
    return this->getStorage().getBuildingBlockField().haloUpdate(opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::
    haloUpdate(SetIdx setIdx, set::HuOptions& opt)
        const -> void
{
    return this->getStorage().getBuildingBlockField().haloUpdate(setIdx, opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::
    haloUpdate(set::HuOptions& opt)
        -> void
{
    return this->getStorage().getBuildingBlockField().haloUpdate(opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::
    haloUpdate(SetIdx setIdx, set::HuOptions& opt)
        -> void
{
    return this->getStorage().getBuildingBlockField().haloUpdate(setIdx, opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::
    getReference(const index_3d& idx, const int& cardinality)
        -> Type&
{
    return this->getStorage().getBuildingBlockField().getReference(idx, cardinality);
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::
    updateCompute(int streamSetId)
        -> void
{
    return this->getStorage().getBuildingBlockField().updateDeviceData(streamSetId);
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::
    updateIO(int streamSetId)
        -> void
{
    return this->getStorage().getBuildingBlockField().updateHostData(streamSetId);
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::
    getPartition(const DeviceType& devType,
                 const SetIdx&     idx,
                 const DataView&   dataView)
        const -> const VoxelField::Partition&
{
    const Neon::Execution execution = DeviceTypeUtil::getExecution(devType);
    const auto&           partition = getPartition(execution, idx, dataView);
    return partition;
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::
    getPartition(const DeviceType& devType,
                 const SetIdx&     idx,
                 const DataView&   dataView)
        -> VoxelField::Partition&
{
    const Neon::Execution execution = DeviceTypeUtil::getExecution(devType);
    auto&                 partition = getPartition(execution, idx, dataView);
    return partition;
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::
    getPartition(Neon::Execution execution,
                 Neon::SetIdx    setIdx,
                 const DataView& dataView)
        const -> const VoxelField::Partition&
{
    auto& s = this->getStorage();
    if (!s.isSupported(execution)) {
        std::stringstream message;
        message << "An execution of type " << execution << " is not supported by a " << s.getDataUse();
        NEON_THROW_UNSUPPORTED_OPERATION(message.str())
    }

    const auto& partition = s.getPartition(execution, dataView, setIdx);

    return partition;
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::
    getPartition(Neon::Execution execution,
                 Neon::SetIdx    setIdx,
                 const DataView& dataView)
        -> VoxelField::Partition&
{
    auto& s = this->getStorage();
    if (!s.isSupported(execution)) {
        std::stringstream message;
        message << "An execution of type " << execution << " is not supported by a " << s.getDataUse();
        NEON_THROW_UNSUPPORTED_OPERATION(message.str())
    }

    auto& partition = s.getPartition(execution, dataView, setIdx);

    return partition;
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::swap(VoxelField& A, VoxelField& B)
    -> void
{
    (void)A;
    (void)B;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
VoxelField<BuildingBlockGridT, T, C>::
    VoxelField(const std::string&                                   fieldUserName,
               Neon::DataUse                                        dataUse,
               const Neon::MemoryOptions&                           memoryOptions,
               const Grid&                                          grid,
               const typename BuildingBlocks::FieldNodeToVoxelMask& nodeToVoxelMaskField,
               const typename BuildingBlocks::Grid&                 buildingBlockGrid,
               int                                                  cardinality,
               T                                                    outsideVal,
               Neon::domain::haloStatus_et::e                       haloStatus)
    : Neon::domain::interface::FieldBaseTemplate<T, C, typename Self::Grid, typename Self::Partition, Storage>(&grid,
                                                                                                               fieldUserName,
                                                                                                               std::string("Fea-") + buildingBlockGrid.getImplementationName(),
                                                                                                               cardinality,
                                                                                                               outsideVal,
                                                                                                               dataUse,
                                                                                                               memoryOptions,
                                                                                                               haloStatus) {
    auto buildingBlockField = buildingBlockGrid.template newField<T, C>(fieldUserName,
                                                                        cardinality,
                                                                        outsideVal,
                                                                        dataUse,
                                                                        memoryOptions);
    this->getStorage() = Storage(buildingBlockField,
                                 nodeToVoxelMaskField,
                                 dataUse);

}

template <typename BuildingBlockGridT, typename T, int C>
template <typename VtiExportType>
auto VoxelField<BuildingBlockGridT, T, C>::ioToVtk(const std::string& fileName,
                                                   const std::string& FieldName,
                                                   bool               includeDomain,
                                                   Neon::IoFileType   ioFileType,
                                                   bool               isNodeSpace)
    const -> void
{
    auto&      buildingBlockField = this->getStorage().getBuildingBlockField();
    const auto spacing = buildingBlockField.getGrid().getSpacing();

    auto iovtk = [&] {
        if (!isNodeSpace) {
            return Neon::domain::IOGridVTK<VtiExportType>(this->getBaseGridTool(), fileName, isNodeSpace, ioFileType);
        } else {
            return Neon::domain::IOGridVTK<VtiExportType>(this->getBaseGridTool(), (spacing / 2) * (1), fileName, isNodeSpace, ioFileType);
        }
    }();
    iovtk.addField(*this, FieldName);

    Neon::IODense<VtiExportType, int32_t> domain(buildingBlockField.getDimension(), 1, [&](const Neon::index_3d& idx, int) {
        VtiExportType isIn = VtiExportType(buildingBlockField.isInsideDomain(idx));
        return isIn;
    });

    if (includeDomain) {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }
    iovtk.flushAndClear();
}


}  // namespace Neon::domain::internal::experimental::staggeredGrid::details
