#pragma once
#include <iostream>
#include <sstream>

#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/domain/tools/IOGridVTK.h"

namespace Neon::domain::interface {


template <typename T, int C, typename G, typename P, typename S>
FieldBaseTemplate<T, C, G, P, S>::FieldBaseTemplate()
    : mGridPrt(nullptr)
{
}

template <typename T, int C, typename G, typename P, typename S>
FieldBaseTemplate<T, C, G, P, S>::FieldBaseTemplate(const Grid*                    gridPtr,
                                                    const std::string              fieldUserName,
                                                    const std::string              fieldClassName,
                                                    int                            cardinality,
                                                    T                              outsideVal,
                                                    Neon::DataUse                  dataUse,
                                                    Neon::MemoryOptions            memoryOptions,
                                                    Neon::domain::haloStatus_et::e haloStatus)
    : Neon::domain::interface::FieldBase<T, C>(fieldUserName,
                                               fieldClassName,
                                               gridPtr->getDimension(),
                                               cardinality,
                                               outsideVal,
                                               dataUse,
                                               memoryOptions,
                                               haloStatus,
                                               gridPtr->getSpacing(),
                                               gridPtr->getOrigin()),
      mGridPrt(gridPtr)
{
}

template <typename T, int C, typename G, typename P, typename S>
auto FieldBaseTemplate<T, C, G, P, S>::getGrid() const
    -> const Grid&
{
    return *mGridPrt;
}

template <typename T, int C, typename G, typename P, typename S>
auto FieldBaseTemplate<T, C, G, P, S>::getBaseGridTool() const
    -> const Neon::domain::interface::GridBase&
{
    return *mGridPrt;
}

template <typename T, int C, typename G, typename P, typename S>
auto FieldBaseTemplate<T, C, G, P, S>::getBackend() const
    -> const Neon::Backend&
{
    return mGridPrt->getBackend();
}

template <typename T, int C, typename G, typename P, typename S>
auto FieldBaseTemplate<T, C, G, P, S>::getDevSet() const
    -> const Neon::set::DevSet&
{
    return mGridPrt->getDevSet();
}

template <typename T, int C, typename G, typename P, typename S>
auto FieldBaseTemplate<T, C, G, P, S>::toString() const
    -> std::string
{
    std::string fieldClassName = this->getClassName();
    size_t      uid = this->getUid();

    std::stringstream s;
    s << "[" << fieldClassName << "]\n";
    s << "| Name            " << this->getName() << "\n";
    s << "| Backend         " << this->getBaseGridTool().getBackend().toString() << "\n";
    s << "| Dimensions      " << this->getDimension().to_string() << "\n";
    s << "| Cardinality     " << this->getCardinality() << "\n";
    s << "| DataUse         " << std::string(Neon::DataUseUtils::toString(this->getDataUse())) << "\n";
    s << "| OutsideValue    " << this->getOutsideValue() << "\n";
    s << "| HaloStatus      " << this->getHaloStatus() << "\n";
    s << "| Field Uid       " << uid << "\n";
    s << "| Active Cells    " << this->getBaseGridTool().getNumActiveCells() << "-> [";
    bool firstTime = true;
    for (auto a : this->getBaseGridTool().getNumActiveCellsPerPartition()) {
        if (!firstTime) {
            s << " ";
        }
        s << a;
        firstTime = false;
    }
    s << "]\n";
    for (auto dw : {Neon::DataView::STANDARD,
                    Neon::DataView::INTERNAL,
                    Neon::DataView::BOUNDARY}) {
        try {
            [[maybe_unused]] const auto& tmp = this->getPartition(Neon::Execution::device, 0, dw);

            const auto& launchParameters = this->getBaseGridTool().getDefaultLaunchParameters(dw);
            s << "| DW::" << Neon::DataViewUtil::toString(dw);

            s << "\n|   Partition     ";
            for (int i = 0; i < launchParameters.cardinality(); i++) {
                s << launchParameters[i].domainGrid() << "\t";
            }
            s << "\n|   Blocks       ";
            for (int i = 0; i < launchParameters.cardinality(); i++) {
                auto     val = launchParameters[i].cudaBlock();
                index_3d dim(val.x, val.y, val.z);
                s << " " << dim.to_string() << "\t";
            }
            s << "\n|   Girds        ";
            for (int i = 0; i < launchParameters.cardinality(); i++) {
                auto     val = launchParameters[i].cudaGrid();
                index_3d dim(val.x, val.y, val.z);
                s << " " << dim.to_string() << "\t";
            }
            s << "\n|   Accelerator first pointers";
            for (int i = 0; i < launchParameters.cardinality(); i++) {
                const auto& partitionCompute = this->getPartition(Neon::Execution::device, i, dw);
                s << " " << partitionCompute.mem() << " ";
            }
            s << "\n|   Host first pointers       ";
            for (int i = 0; i < launchParameters.cardinality(); i++) {
                const auto& partitionCompute = this->getPartition(Neon::Execution::host, i, dw);
                s << " " << partitionCompute.mem() << " ";
            }
            s << "\n";
        } catch (...) {
            s << "| DW::" << Neon::DataViewUtil::toString(dw);
            s << "\n|            NotSupported\n";
        }
    }
    s << "\\_____________|"
      << "\n";

    return s.str();
}

template <typename T, int C, typename G, typename P, typename S>
auto FieldBaseTemplate<T, C, G, P, S>::isInsideDomain(const index_3d& idx) const -> bool
{
    return getGrid().isInsideDomain(idx);
}
template <typename T, int C, typename G, typename P, typename S>
auto FieldBaseTemplate<T, C, G, P, S>::swapUIDBeforeFullSwap(FieldBaseTemplate::Self& A, FieldBaseTemplate::Self& B) -> void
{
    const bool isCardinalityCompatible = A.getCardinality() == B.getCardinality();
    const bool isFromSameGrid = A.getGrid().getGridUID() == B.getGrid().getGridUID();
    if (!isCardinalityCompatible ||
        !isFromSameGrid) {

        Neon::NeonException exp(A.getClassName());
        exp << "Provided fields " << A.getName() << " and " << B.getName()
            << " are incompatible for a swap operation.";
        NEON_THROW(exp);
    }
    Neon::set::interface::MultiXpuDataInterface<P, S>::swapUIDs(A,B);
}


}  // namespace Neon::domain::interface
