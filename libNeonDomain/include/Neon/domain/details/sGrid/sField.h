#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Execution.h"
#include "Neon/core/types/Macros.h"


#include "Neon/set/DevSet.h"
#include "Neon/set/memory/memSet.h"

#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/domain/interface/common.h"
#include "Neon/domain/details/sGrid/sFieldStorage.h"
#include "Neon/domain/details/sGrid/sPartition.h"
#include "Neon/set/MemoryOptions.h"

namespace Neon::domain::details::sGrid {

template <typename OuterGridT>
class sGrid /** Forward declaration for sField */;

/**
 * Definition of a Field abstraction  for sGrid.
 * The class extends FieldBaseTemplate to get some capabilities for free.
 */
template <typename OuterGridT, typename T, int C>
class sField : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                 C,
                                                                 sGrid<OuterGridT>,
                                                                 sPartition<OuterGridT, T, C>,
                                                                 sFieldStorage<OuterGridT, T, C>>
{
   public:
    friend sGrid<OuterGridT>;

    // New Naming:
    using Partition = sPartition<OuterGridT, T, C>; /**< Type of the associated fieldCompute */
    using Type = typename Partition::Type /**< Type of the information stored in one element */;
    using Idx = typename Partition::Idx /**< Internal type that represent the location in memory of a element */;
    using Grid = sGrid<OuterGridT>;
    using Field = sField<OuterGridT, T, C>;
    static constexpr int Cardinality = C;

    // ALIAS
    using Self = sField<OuterGridT, Type, Cardinality>;

    using Count = typename Partition::Count;
    using Index = typename Partition::Index;

    sField();

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

    auto newHaloUpdate(Neon::set::StencilSemantic semantic,
                       Neon::set::TransferMode    transferMode,
                       Neon::Execution            execution)
         -> Neon::set::Container;

    auto newHaloUpdate(Neon::set::StencilSemantic semantic,
                       Neon::set::TransferMode    transferMode,
                       Neon::Execution            execution)
        const -> Neon::set::Container;

    /**
     * Move the field metadata from host to the accelerators.
     * The operation is asynchronous.
     */
    auto updateDeviceData(int streamIdx)
        -> void;

    /**
     * Move the field metadata from the accelerators to the host space.
     * The operation is asynchronous.
     */
    auto updateHostData(int streamIdx )
        -> void;

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
                                                                    sGrid<OuterGridT>,
                                                                    sPartition<OuterGridT, T, C>,
                                                                    sFieldStorage<OuterGridT, T, C>>;
    /**
     * Private constructor used by sGrid
     */
    sField(std::string const&                                               fieldUserName,
           sGrid<OuterGridT> const&                                         grid,
           int                                                              cardinality,
           T                                                                outsideVal,
           Neon::domain::haloStatus_et::e                                   haloStatus,
           Neon::DataUse                                                    dataUse,
           Neon::MemoryOptions const&                                       memoryOptions,
           Neon::set::MemSet<typename OuterGridT::Cell::OuterIdx> const& tabelSCellToOuterIdx);

    /**
     * Internal helper function to allocate and initialized memory
     */
    auto initMemory() -> void;

    /**
     * Internal helper function to initialize the partition structures
     */
    auto initPartitions(Neon::set::MemSet<typename OuterGridT::Cell::OuterIdx> const&) -> void;
};


}  // namespace Neon::domain::details::sGrid