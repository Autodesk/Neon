#pragma once
#include "Neon/domain/details/sGrid/sField.h"

namespace Neon::domain::details::sGrid {

template <typename OuterGridT, typename T, int C>
sField<OuterGridT, T, C>::sField(std::string const&                                            fieldUserName,
                                 sGrid<OuterGridT> const&                                      grid,
                                 int                                                           cardinality,
                                 T                                                             outsideVal,
                                 Neon::domain::haloStatus_et::e                                haloStatus,
                                 Neon::DataUse                                                 dataUse,
                                 Neon::MemoryOptions const&                                    memoryOptions,
                                 Neon::set::MemSet<typename OuterGridT::Cell::OuterIdx> const& tabelSCellToOuterIdx)
    : Neon::domain::interface::FieldBaseTemplate<Type,
                                                 Cardinality,
                                                 typename Self::Grid,
                                                 typename Self::Partition,
                                                 typename BaseTemplate::Storage>(&grid,
                                                                                 fieldUserName,
                                                                                 "sField",
                                                                                 cardinality,
                                                                                 outsideVal,
                                                                                 dataUse,
                                                                                 memoryOptions,
                                                                                 haloStatus)
{

    self().getStorage() = sFieldStorage<OuterGridT, T, C>(grid);

    initMemory();
    initPartitions(tabelSCellToOuterIdx);
}

template <typename OuterGridT, typename T, int C>
sField<OuterGridT, T, C>::sField()
{
}

template <typename OuterGridT, typename T, int C>
auto sField<OuterGridT, T, C>::self() -> Self&
{
    return *this;
}

template <typename OuterGridT, typename T, int C>
auto sField<OuterGridT, T, C>::self() const -> const Self&
{
    return *this;
}

template <typename OuterGridT, typename T, int C>
auto sField<OuterGridT, T, C>::initMemory() -> void
{
    this->getStorage().rawMem = self().getDevSet().template newMemSet<T>(self().getDataUse(),
                                                                         self().getCardinality(),
                                                                         self().getMemoryOptions(),
                                                                         self().getGrid().getNumActiveCellsPerPartition().template newType<uint64_t>());
}

template <typename OuterGridT, typename T, int C>
auto sField<OuterGridT, T, C>::initPartitions(Neon::set::MemSet<typename OuterGridT::Cell::OuterIdx> const& mapToOuterIdx) -> void
{

    auto computePitch = [&](int setIdx) {
        int nelements = int(this->getGrid().getNumActiveCellsPerPartition()[setIdx]);

        typename Self::Partition::Pitch pitch;
        if (Neon::MemoryLayout::structOfArrays == this->getMemoryOptions().getOrder()) {
            pitch.pMain = 1;
            pitch.pCardinality = nelements;
        } else {
            pitch.pMain = this->getCardinality();
            pitch.pCardinality = 1;
        }
        return pitch;
    };

    for (int setIdx = 0; setIdx < self().getDevSet().setCardinality(); ++setIdx) {
        for (auto execution : {Neon::Execution::device, Neon::Execution::host}) {
            for (auto& dw : DataViewUtil::validOptions()) {

                T* mem = nullptr;
                if (execution == Neon::Execution::host) {
                    mem = self().getStorage().rawMem.rawMem(setIdx, Neon::DeviceType::CPU);
                } else {
                    if (self().getDevSet().type() == Neon::DeviceType::CPU) {
                        mem = self().getStorage().rawMem.rawMem(setIdx, Neon::DeviceType::CPU);
                    } else {
                        mem = self().getStorage().rawMem.rawMem(setIdx, Neon::DeviceType::CUDA);
                    }
                }

                auto& partition = self().getStorage().getPartition(execution, dw, setIdx);

                partition = Partition(dw,
                                      setIdx,
                                      mem,
                                      self().getCardinality(),
                                      computePitch(setIdx),
                                      mapToOuterIdx.get(setIdx).rawMem(execution));
            }
        }
    }
}

template <typename OuterGridT, typename T, int C>

auto sField<OuterGridT, T, C>::getReference(const Neon::index_3d& idx,
                                            const int&            cardinality)
    -> Type&
{
    const Grid& grid = this->getGrid();
    SetIdx      setIdx;
    DataView    dw;
    auto* const meta = grid.mStorage->map.getMetadata(idx, setIdx, dw);

    if (meta != nullptr) {
        auto  cellOffset = meta->cellOffset;
        Type& val = self().getStorage().rawMem.eRef(setIdx,
                                                    cellOffset,
                                                    cardinality);
        return val;
    }
    return this->getOutsideValue();
}

template <typename OuterGridT, typename T, int C>

auto sField<OuterGridT, T, C>::operator()(const Neon::index_3d& idx,
                                          const int&            cardinality) const
    -> Type
{
    Grid const& grid = this->getGrid();
    SetIdx      setIdx;
    DataView    dw;
    auto* const meta = grid.mStorage->map.getMetadata(idx, setIdx, dw);

    if (meta != nullptr) {
        auto        cellOffset = meta->cellOffset;
        const Type& val = self().getStorage().rawMem.eRef(setIdx,
                                                          cellOffset,
                                                          cardinality);
        return val;
    }
    return this->getOutsideValue();
}


template <typename OuterGridT, typename T, int C>
auto sField<OuterGridT, T, C>::updateHostData(int streamIdx)
    -> void
{
    if (self().getDevSet().type() == Neon::DeviceType::CPU) {
        return;
    }
    self().getStorage().rawMem.updateHostData(this->getGrid().getBackend(), streamIdx);
}

template <typename OuterGridT, typename T, int C>

auto sField<OuterGridT, T, C>::updateDeviceData(int streamIdx)
    -> void
{
    if (self().getDevSet().type() == Neon::DeviceType::CPU) {
        return;
    }
    self().getStorage().rawMem.updateDeviceData(this->getGrid().getBackend(), streamIdx);
}

template <typename OuterGridT, typename T, int C>

auto sField<OuterGridT, T, C>::getPartition(Neon::Execution       execution,
                                            Neon::SetIdx          setIdx,
                                            const Neon::DataView& dataView) const -> const Partition&
{
    if (dataView != Neon::DataView::STANDARD) {
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
    return self().getStorage().getPartition(execution, dataView, setIdx);
}

template <typename OuterGridT, typename T, int C>
auto sField<OuterGridT, T, C>::getPartition(Neon::Execution       execution,
                                            Neon::SetIdx          setIdx,
                                            const Neon::DataView& dataView) -> Partition&
{
    if (dataView != Neon::DataView::STANDARD) {
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
    return self().getStorage().getPartition(execution, dataView, setIdx);
}

template <typename OuterGridT, typename T, int C>
auto sField<OuterGridT, T, C>::swap(Field& A, Field& B) -> void
{

    Neon::domain::interface::FieldBaseTemplate<Type,
                                               Cardinality,
                                               typename Self::Grid,
                                               typename Self::Partition,
                                               typename BaseTemplate::Storage>::swapUIDBeforeFullSwap(A, B);
    std::swap(A, B);
}

template <typename OuterGridT, typename T, int C>
auto sField<OuterGridT, T, C>::newHaloUpdate(Neon::set::StencilSemantic semantic,
                                             Neon::set::TransferMode    transferMode,
                                             Neon::Execution            execution)
    -> Neon::set::Container
{
    NEON_THROW_UNSUPPORTED_OPERATION("sField");
}

template <typename OuterGridT, typename T, int C>
auto sField<OuterGridT, T, C>::newHaloUpdate(Neon::set::StencilSemantic semantic,
                                             Neon::set::TransferMode    transferMode,
                                             Neon::Execution            execution)
    const -> Neon::set::Container
{
    NEON_THROW_UNSUPPORTED_OPERATION("sField");
}
}  // namespace Neon::domain::details::sGrid