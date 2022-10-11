#pragma once
#include "Neon/domain/internal/aGrid/aField.h"

namespace Neon::domain::internal::aGrid {

template <typename T, int C>
aField<T, C>::aField(const std::string              fieldUserName,
                     const aGrid&                   grid,
                     int                            cardinality,
                     T                              outsideVal,
                     Neon::domain::haloStatus_et::e haloStatus,
                     Neon::DataUse                  dataUse,
                     const Neon::MemoryOptions&     memoryOptions)
    : Neon::domain::interface::FieldBaseTemplate<Type,
                                                 Cardinality,
                                                 typename Self::Grid,
                                                 typename Self::Partition,
                                                 typename BaseTemplate::Storage>(&grid,
                                                                                 fieldUserName,
                                                                                 "aField",
                                                                                 cardinality,
                                                                                 outsideVal,
                                                                                 dataUse,
                                                                                 memoryOptions,
                                                                                 haloStatus)
{

    self().getStorage() = Neon::domain::internal::aGrid::Storage<T, C>();

    iniMemory();
    initPartitions();
}

template <typename T, int C>
aField<T, C>::aField()
{
}

template <typename T, int C>
auto aField<T, C>::self() -> Self&
{
    return *this;
}

template <typename T, int C>
auto aField<T, C>::self() const -> const Self&
{
    return *this;
}

template <typename T, int C>
auto aField<T, C>::iniMemory() -> void
{
    self().getStorage().rawMem = self().getDevSet().template newMemSet<T>(self().getDataUse(),
                                                                          self().getCardinality(),
                                                                          self().getMemoryOptions(),
                                                                          self().getGrid().getNumActiveCellsPerPartition().template newType<uint64_t>());
}

template <typename T, int C>
auto aField<T, C>::initPartitions() -> void
{
    auto initPartitionSet = [&](Neon::Execution execution) {
        auto& partitionSet = self().getStorage().getPartitionSet(execution, Neon::DataView::STANDARD);
        partitionSet = self().getDevSet().template newDataSet<Partition>();
    };

    initPartitionSet(Neon::Execution::device);
    initPartitionSet(Neon::Execution::host);

    auto computePitch = [&](int setIdx) {
        int nelements = int(this->getGrid().getNumActiveCellsPerPartition()[setIdx]);

        typename Self::Partition::pitch_t pitch;
        if (Neon::MemoryLayout::structOfArrays == this->getMemoryOptions().getOrder()) {
            pitch.pMain = 1;
            pitch.pCardinality = nelements;
        } else {
            pitch.pMain = this->getCardinality();
            pitch.pCardinality = 1;
        }
        return pitch;
    };

    for (int i = 0; i < self().getDevSet().setCardinality(); ++i) {
        for (auto execution : {Neon::Execution::device, Neon::Execution::host}) {
            auto& partition = self().getStorage().getPartition(execution, Neon::DataView::STANDARD, i);
            /**
             * This is how it should be after the refactoring of devSet and Mem_t
             */
            //        partition = Partition(i,
            //                              mStorage->rawMem.rawMem(execution, i),
            //                              computePitch(i),
            //                              getGrid().getNumActiveCellsPerPartition()[i],
            //                              this->getCardinality());

            // TODO The following implementation will be changed once
            // the refactoring on the deFvSet is completed.
            T* mem = nullptr;
            if (execution == Neon::Execution::host) {
                mem = self().getStorage().rawMem.rawMem(i, Neon::DeviceType::CPU);
            } else {
                if (self().getDevSet().type() == Neon::DeviceType::CPU) {
                    mem = self().getStorage().rawMem.rawMem(i, Neon::DeviceType::CPU);
                } else {
                    mem = self().getStorage().rawMem.rawMem(i, Neon::DeviceType::CUDA);
                }
            }
            partition = Partition(i,
                                  mem,
                                  computePitch(i),
                                  typename Partition::count_t(this->getGrid().getNumActiveCellsPerPartition()[i]),
                                  this->getCardinality());
        }
    }
}

template <typename T, int C>
auto aField<T, C>::getReference(const Neon::index_3d&      idx,
                                const int&                 cardinality)
    -> Type&
{
    if (idx.y != 0 || idx.z != 0) {
        NEON_THROW_UNSUPPORTED_OPERATION("aGrid accepts only 1D indexes");
    }

    const auto& firstIdxPerPartition = this->getGrid().getFirstIdxPerPartition();
    const auto& activeCellsPerPartition = this->getGrid().getNumActiveCellsPerPartition();

    size_t counter = 0;
    for (int i = 0; i < firstIdxPerPartition.cardinality(); i++) {
        size_t firstID = firstIdxPerPartition[i];
        size_t counterJump = activeCellsPerPartition[i];
        counter += counterJump;
        size_t lastID = counter - 1;
        if (static_cast<size_t>(idx.x) >= firstID && static_cast<size_t>(idx.x) <= lastID) {
            auto& value = self().getStorage().rawMem.eRef(i,
                                                          static_cast<size_t>(idx.x) - firstID,
                                                          cardinality);
            return value;
        }
    }
    return this->getOutsideValue();
}

template <typename T, int C>
auto aField<T, C>::operator()(const Neon::index_3d&      idx,
                              const int&                 cardinality) const
    -> Type
{
    if (idx.y != 0 || idx.z != 0) {
        NEON_THROW_UNSUPPORTED_OPERATION("aGrid accepts only 1D indexes");
    }

    const auto& firstIdxPerPartition = this->getGrid().getFirstIdxPerPartition();
    const auto& activeCellsPerPartition = this->getGrid().getNumActiveCellsPerPartition();

    size_t counter = 0;
    for (int i = 0; i < firstIdxPerPartition.cardinality(); i++) {
        size_t firstID = firstIdxPerPartition[i];
        size_t counterJump = activeCellsPerPartition[i];
        counter += counterJump;
        size_t lastID = counter - 1;

        if (static_cast<size_t>(idx.x) >= firstID && static_cast<size_t>(idx.x) <= lastID) {
            return self().getStorage().rawMem.eRef(i,
                                                   static_cast<size_t>(idx.x) - firstID,
                                                   cardinality);
        }
    }
    return this->getOutsideValue();
}


template <typename T, int C>
auto aField<T, C>::updateIO(int streamIdx)
    -> void
{
    if (self().getDevSet().type() == Neon::DeviceType::CPU) {
        return;
    }
    self().getStorage().rawMem.updateIO(this->getGrid().getBackend(), streamIdx);
}

template <typename T, int C>
auto aField<T, C>::updateCompute(int streamIdx)
    -> void
{
    if (self().getDevSet().type() == Neon::DeviceType::CPU) {
        return;
    }
    self().getStorage().rawMem.updateCompute(this->getGrid().getBackend(), streamIdx);
}

template <typename T, int C>
auto aField<T, C>::getPartition(Neon::DeviceType      devEt,
                                Neon::SetIdx          setIdx,
                                const Neon::DataView& dataView)
    const -> const Partition&
{
    switch (devEt) {
        case Neon::DeviceType::CPU: {
            return self().getStorage().getPartition(Neon::Execution::host, dataView, setIdx);
        }
        case Neon::DeviceType::CUDA: {
            return self().getStorage().getPartition(Neon::Execution::device, dataView, setIdx);
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION();
        }
    }
}

template <typename T, int C>
auto aField<T, C>::getPartition(Neon::DeviceType      devEt,
                                Neon::SetIdx          setIdx,
                                const Neon::DataView& dataView)
    -> Partition&
{
    switch (devEt) {
        case Neon::DeviceType::CPU: {
            return self().getStorage().getPartition(Neon::Execution::host, dataView, setIdx);
        }
        case Neon::DeviceType::CUDA: {
            return self().getStorage().getPartition(Neon::Execution::device, dataView, setIdx);
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION();
        }
    }
}


template <typename T, int C>
auto aField<T, C>::getPartition(Neon::Execution       execution,
                                Neon::SetIdx          setIdx,
                                const Neon::DataView& dataView) const -> const Partition&
{
    if (dataView != Neon::DataView::STANDARD) {
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
    return self().getStorage().getPartition(execution, dataView, setIdx);
}

template <typename T, int C>
auto aField<T, C>::getPartition(Neon::Execution       execution,
                                Neon::SetIdx          setIdx,
                                const Neon::DataView& dataView) -> Partition&
{
    if (dataView != Neon::DataView::STANDARD) {
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
    return self().getStorage().getPartition(execution, dataView, setIdx);
}
template <typename T, int C>
auto aField<T, C>::haloUpdate(set::HuOptions&) const -> void
{
    // Nothing to do
}
template <typename T, int C>
auto aField<T, C>::haloUpdate(set::HuOptions&) -> void
{
    // Nothing to do
}

template <typename T, int C>
auto aField<T, C>::swap(aField::Field& A, aField::Field& B) -> void
{
    Neon::domain::interface::FieldBaseTemplate<Type,
                                               Cardinality,
                                               typename Self::Grid,
                                               typename Self::Partition,
                                               typename BaseTemplate::Storage>::swapUIDBeforeFullSwap(A, B);
    std::swap(A, B);
}

}  // namespace Neon::domain::internal::aGrid