#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Execution.h"
#include "Neon/core/types/Macros.h"

#include "Neon/set/DevSet.h"
#include "Neon/set/memory/memSet.h"

#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/domain/interface/common.h"
#include "Neon/domain/details/aGrid/aFieldStorage.h"
#include "Neon/domain/details/aGrid/aPartition.h"
#include "Neon/set/MemoryOptions.h"

namespace Neon::domain::details::aGrid {

class aGrid /** Forward declaration for aField */;

template <typename T, int C = 0>
class aField : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                 C,
                                                                 aGrid,
                                                                 aPartition<T, C>,
                                                                 Storage<T, C>>
{
    friend aGrid;

   public:
    // New Naming:
    using Partition = aPartition<T, C>; /**< Type of the associated fieldCompute */
    using Type = typename Partition::Type /**< Type of the information stored in one element */;
    using Cell = typename Partition::Cell /**< Internal type that represent the location in memory of a element */;
    using Field = aField<T, C>;
    static constexpr int Cardinality = C;

    // ALIAS
    using Self = aField<Type, Cardinality>;

    using count_t = typename Partition::count_t;  // TODO: remove at the end of the refactoring
    using index_t = typename Partition::index_t;  // TODO: remove at the end of the refactoring


   public:
    aField();

    auto self() -> Self&;

    auto self() const -> const Self&;

    /**
     * Returns the metadata associated with the element in location idx.
     * If the element is not active (it does not belong to the voxelized domain),
     * then the default outside value is returned.
     */
    auto operator()(const Neon::index_3d& idx,
                    const int&            cardinality) const
        -> Type final;

    virtual auto getReference(const Neon::index_3d& idx,
                              const int&            cardinality)
        -> Type& final;


    /**
     * Move the field metadata from host to the accelerators.
     * The operation is asynchronous.
     */
    auto updateDeviceData(int streamIdx = 0)
        -> void;

    /**
     * Move the field metadata from the accelerators to the host space.
     * The operation is asynchronous.
     */
    auto updateHostData(int streamIdx = 0)
        -> void;

    [[deprecated("Will be replace by the getPartition method")]] auto
    getPartition(Neon::DeviceType      devEt,
                 Neon::SetIdx          setIdx,
                 const Neon::DataView& dataView = Neon::DataView::STANDARD) const -> const Partition&;

    [[deprecated("Will be replace by the getPartition method")]] auto
    getPartition(Neon::DeviceType      devEt,
                 Neon::SetIdx          setIdx,
                 const Neon::DataView& dataView = Neon::DataView::STANDARD) -> Partition&;

    /**
     * Return a constant reference to a specific partition based on a set of parameters:
     * execution type, target device, dataView
     */
    auto getPartition(Neon::Execution       execution,
                      Neon::SetIdx          setIdx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD) const -> const Partition&;
    /**
     * Return a reference to a specific partition based on a set of parameters:
     * execution type, target device, dataView
     */
    auto getPartition(Neon::Execution       execution,
                      Neon::SetIdx          setIdx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD) -> Partition&;

    static auto swap(Field& A, Field& B) -> void;

   private:
    using BaseTemplate = Neon::domain::interface::FieldBaseTemplate<T,
                                                                    C,
                                                                    aGrid,
                                                                    aPartition<T, C>,
                                                                    Neon::domain::details::aGrid::Storage<T, C>>;
    /**
     * Private constructor used by aGrid
     */
    aField(const std::string              fieldUserName,
           const aGrid&                   grid,
           int                            cardinality,
           T                              outsideVal,
           Neon::domain::haloStatus_et::e haloStatus,
           Neon::DataUse                  dataUse,
           const Neon::MemoryOptions&     memoryOptions);

    /**
     * Internal helper function to allocate and initialized memory
     */
    auto iniMemory() -> void;

    /**
     * Internal helper function to initialize the partition structures
     */
    auto initPartitions() -> void;
};

extern template class aField<int, 0>;
extern template class aField<double, 0>;

}  // namespace Neon::domain::details::aGrid