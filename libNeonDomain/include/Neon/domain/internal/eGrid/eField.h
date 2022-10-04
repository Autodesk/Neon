#pragma once

#include "Neon/core/core.h"
#include "Neon/core/tools/io/ioToVTK.h"
#include "Neon/core/types/Macros.h"

#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/memory/memSet.h"

#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/domain/internal/eGrid/eInternals/builder/dsBuilderCommon.h"
#include "Neon/domain/internal/eGrid/eInternals/builder/dsFrame.h"
#include "eFieldDev.h"
#include "ePartition.h"
#include "help.h"

namespace Neon::domain::internal::eGrid {

class eGrid;

template <typename T, int C = 0>
class eField : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                 C,
                                                                 eGrid,
                                                                 ePartition<T, C>,
                                                                 int /** as storage type we pass an int as eField manages the storage manually*/>
{
    friend eGrid;

   private:
    // ALIAS
    using LocalIndexingInfo_t = internals::LocalIndexingInfo_t;

   public:
    // Neon aliases
    using Self = eField<T, C>; /** Self */
    using Type = T;            /** type of the elements contained by the field */
    static const int Cardinality = C;

    // GRID, FIELDS and LOCAL aliases
    using Grid = eGrid;                    /**< Type of the associated grid */
    using Field = Self;                    /**< Type of the associated field */
    using Partition = ePartition<T, C>;    /**< Type of the associated local_t */
    using FieldDev = eFieldDevice_t<T, C>; /**< Type of the associated fieldMirror */
    using Cell = typename Partition::Cell;
    using ngh_idx = typename Partition::nghIdx_t;  //<- type of an index to identify a neighbour
    template <typename TT>
    using GenericSelf = eField<TT, 0>;
    template <typename TT, int Card>
    using SpecificSelf = eField<TT, Card>;

    /**
     * Default constructor
     */
    eField() = default;


    /**
     * Default destructor
     */
    ~eField() = default;

    /**
     * Return a const reference to this object.
     * @return
     */
    auto cSelf() const -> const Self&;
    /**
     * Returns a reference to this object where cardinality has been converted to be dynamic parameter.
     * @return
     */
    auto genericSelf() -> GenericSelf<T>&;

    /**
     * Returns a reference to this object where getCardinality has been converted to be a template parameter.
     * @return
     */
    template <int Card>
    auto specificSelf() -> SpecificSelf<T, Card>&;

    /**
     * Returns a reference to this object where getCardinality has been converted to be a template parameter.
     * @return
     */
    template <int Card>
    auto specificSelf() const -> const SpecificSelf<T, Card>&;

    /**
     * Returns the metadata associated with the element in location idx.
     * If the element is not active (it does not belong to the voxelized domain),
     * then the default outside value is returned.
     */
    auto operator()(const Neon::index_3d& idx,
                    const int&            cardinality,
                    const int             level = 0) const
        -> Type final;

    auto getReference(const Neon::index_3d& idx,
                      const int&            cardinality,
                      const int             level = 0)
        -> Type& final;

    /**
     *
     * @param streamSet
     */
    auto updateIO(int streamSetId)
        -> void;

    /**
     *
     * @param streamSet
     */
    auto updateCompute(int streamSetId)
        -> void;

    auto getPartition(const Neon::DeviceType& devType,
                      Neon::SetIdx            idx,
                      const Neon::DataView&   dataView = Neon::DataView::STANDARD)
        const
        -> const Partition&;

    auto getPartition(const Neon::DeviceType& devType,
                      Neon::SetIdx            idx,
                      const Neon::DataView&   dataView = Neon::DataView::STANDARD)
        -> Partition&;

    auto getPartition(Neon::Execution,
                      Neon::SetIdx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD)
        const
        -> const Partition& final;

    auto getPartition(Neon::Execution,
                      Neon::SetIdx          idx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD)
        -> Partition& final;


    auto haloUpdate(Neon::set::HuOptions& opt) const
        -> void final;

    auto haloUpdate(Neon::set::HuOptions& opt)
        -> void final;

    static auto swap(Field& A, Field& B) -> void;

   private:
    /**
     * Construction by composition
     * @param CPU
     * @param GPU
     */
    eField(const std::string&             fieldUserName,
           int                            cardinality,
           T                              outsideVal,
           Neon::DataUse                  dataUse,
           Neon::MemoryOptions            memoryOptions,
           Neon::domain::haloStatus_et::e haloStatus,
           const Neon::set::DataConfig&   dataConfig,
           FieldDev&                      CPU,
           FieldDev&                      GPU);

    /**
     * Return a mutable reference to this object.
     * @return
     */
    auto self() -> Self&;

    /**
     * Return a const reference to this object.
     * @return
     */
    auto self() const -> const Self&;


    /**
     * Returns the cardinality of the field, i.e. the number of components associated to each grid points
     * @return
     */
    auto cardinality() const -> int;

    /**
     * linking operator
     * @param field
     */
    auto helpLink(FieldDev& field) -> void;

    /**
     * Returns whether the voxel is active
     * @param idx 3D index of the voxel
     * @return Whether active
     */
    auto helpIsActive(const Neon::index_3d& idx) const
        -> bool;

    /**
     * Update by device
     * @param streamSet
     * @param devEt
     */
    auto helpUpdate(const Neon::set::StreamSet& streamSet,
                    const Neon::DeviceType&     devEt)
        -> void;
    /**
     *
     * @param streamSet
     */
    auto updateCompute(const Neon::set::StreamSet& streamSet)
        -> void;
    /**
     *
     * @param streamSet
     */
    auto updateIO(const Neon::set::StreamSet& streamSet)
        -> void;


   private:
    FieldDev              mCpu;
    FieldDev              mGpu;
    bool                  mCpuLink{false};
    bool                  mGpuLink{false};
    Neon::set::DataConfig m_dataConfig;
};

}  // namespace Neon::domain::internal::eGrid
