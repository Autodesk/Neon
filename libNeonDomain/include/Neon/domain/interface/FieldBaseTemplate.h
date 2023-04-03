#pragma once

#include "Neon/core/core.h"
#include "Neon/core/tools/io/IODense.h"
#include "Neon/core/types/Macros.h"

#include "Neon/set/DevSet.h"
#include "Neon/set/MultiXpuDataInterface.h"
#include "Neon/set/memory/memSet.h"

#include "Neon/domain/interface/FieldBase.h"
#include "Neon/domain/interface/common.h"


namespace Neon::domain::interface {

template <typename T /** Field's cell metadata type */,
          int C /**      Field cardinality */,
          typename G /** Grid type */,
          typename P /** Type of a Partition */,
          typename S /** Storage type */>
class FieldBaseTemplate : public FieldBase<T, C>,
                          public Neon::set::interface::MultiXpuDataInterface<P, S>
{
   public:
    using Partition = P;
    using Storage = S;
    using Grid = G;
    using Type = T;

    static constexpr int Cardinality = C;

    using Self = FieldBaseTemplate<Type, Cardinality, Grid, Partition, Storage>;

    virtual ~FieldBaseTemplate() = default;

    FieldBaseTemplate();

    FieldBaseTemplate(const Grid*                    gridPtr,
                      const std::string              fieldUserName,
                      const std::string              fieldClassName,
                      int                            cardinality,
                      T                              outsideVal,
                      Neon::DataUse                  dataUse,
                      Neon::MemoryOptions            memoryOptions,
                      Neon::domain::haloStatus_et::e haloStatus);

    /**
     * Return a partition based on a set of parameters: execution type, target device, dataView
     */
    virtual auto getPartition(Neon::Execution       execution,
                              Neon::SetIdx          setIdx,
                              const Neon::DataView& dataView = Neon::DataView::STANDARD) const
        -> const Partition& = 0;

    virtual auto getPartition(Neon::Execution       execution,
                              Neon::SetIdx          setIdx,
                              const Neon::DataView& dataView = Neon::DataView::STANDARD)
        -> Partition& = 0;

    auto isInsideDomain(const Neon::index_3d& idx) const
        -> bool;

    auto getGrid() const
        -> const Grid&;

    auto getBackend() const
        -> const Neon::Backend&;

    auto getDevSet() const
        -> const Neon::set::DevSet&;

    auto getBaseGridTool() const
        -> const Neon::domain::interface::GridBase& final;

    auto toString() const -> std::string;

   protected:
    /**
     * This function should be called before executing an std::swap
     * in the swap method of the derived field.
     *
     * The ideas is that on a derived field, we can use std::swap
     * directly, we only need to call this method first.
     *
     * The method does two things:
     *
     * a. do some correctness checks
     * b. swap the UID of the fields
     *
     * Because the UID are swapped twice (one from the std::swap call
     * and the other from the swapUIDBeforeFullSwap) the UID do not change
     *
     * Typical implementation of a derived field would be:
     *
     * FieldBaseTemplate::swapUIDBeforeFullSwap(A,B);
     * std::swap(A, B);
     */
    static auto swapUIDBeforeFullSwap(Self& A, Self& B) -> void;
    const Grid* mGridPrt;
};

}  // namespace Neon::domain::interface

#include "Neon/domain/interface/FieldBaseTemplate_imp.h"