#pragma once

#include "Neon/core/core.h"
#include "Neon/core/tools/io/IODense.h"
#include "Neon/core/types/Macros.h"

#include "Neon/set/Containter.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/HuOptions.h"
#include "Neon/set/memory/memSet.h"

#include "GridBase.h"
#include "Neon/domain/interface/common.h"

namespace Neon::domain::interface {

template <typename T, int C>
class FieldBase
{
   public:
    using Self = FieldBase<T, C>;
    using Type = T;
    virtual ~FieldBase() = default;

    FieldBase();

    FieldBase(const std::string&             fieldUserName,
              const std::string&             fieldClassName,
              const Neon::index_3d&          dimension,
              int                            cardinality,
              T                              outsideVal,
              Neon::DataUse                  dataUse,
              Neon::MemoryOptions            memoryOptions,
              Neon::domain::haloStatus_et::e haloStatus,
              const Vec_3d<double>&          spacing, /*!             Spacing, i.e. size of a voxel            */
              const Vec_3d<double>&          origin);


    virtual auto isInsideDomain(const Neon::index_3d& idx) const
        -> bool = 0;

    virtual auto operator()(const Neon::index_3d& idx,
                            const int&            cardinality) const
        -> T = 0;

    virtual auto getReference(const Neon::index_3d& idx,
                              const int&            cardinality)
        -> T& = 0;

    virtual auto getBaseGridTool() const
        -> const Neon::domain::interface::GridBase& = 0;

    virtual auto newHaloUpdate(Neon::set::StencilSemantic /*semantic*/,
                               Neon::set::TransferMode    /*transferMode*/,
                               Neon::Execution            /*execution*/)
        const -> Neon::set::Container
    {
        NEON_THROW_UNSUPPORTED_OPERATION("");
    }

    auto getDimension() const
        -> const Neon::index_3d&;

    auto getCardinality() const
        -> int;

    auto getOutsideValue() const
        -> T&;

    auto getOutsideValue()
        -> T&;

    auto getDataUse() const
        -> Neon::DataUse;

    auto getMemoryOptions() const
        -> const Neon::MemoryOptions&;

    auto getHaloStatus() const
        -> Neon::domain::haloStatus_et::e;

    auto getName() const
        -> const std::string&;

    auto getClassName() const
        -> const std::string&;

    /**
     * For each operator that target active cells.
     * Index values are provided in RW mode.
     *
     * @tparam mode
     * @param fun
     */
    virtual auto forEachActiveCell(const std::function<void(const Neon::index_3d&,
                                                            const int& cardinality,
                                                            T&)>&     fun,
                                   Neon::computeMode_t::computeMode_e mode = Neon::computeMode_t::computeMode_e::par) -> void;

    virtual auto forEachActiveCell(const std::function<void(const Neon::index_3d&,
                                                            std::vector<T*>&)>& fun,
                                   Neon::computeMode_t::computeMode_e           mode = Neon::computeMode_t::computeMode_e::par) -> void;
    /**
     * For each operator that target all cells in the cubic domain.
     * Index values are provided in read only mode.
     *
     * @tparam mode
     * @param fun
     */
    template <Neon::computeMode_t::computeMode_e mode = Neon::computeMode_t::computeMode_e::par>
    auto forEachCell(const std::function<void(const Neon::index_3d&,
                                              const int& cardinality,
                                              T)>& fun) const
        -> void;

    template <typename ExportType = T,
              typename ExportIndex = int>
    auto ioToDense(NEON_OUT Neon::IODense<ExportType, ExportIndex>& dense) const
        -> void;

    template <typename ExportType = T,
              typename ExportIndex = int>
    auto ioToDense(Neon::MemoryLayout order) const
        -> Neon::IODense<ExportType, ExportIndex>;

    template <typename ExportType = T,
              typename ExportIndex = int>
    auto ioToDense() const
        -> Neon::IODense<ExportType, ExportIndex>;

    template <typename ImportType = T,
              typename ImportIndex = int>
    auto ioFromDense(const Neon::IODense<ImportType, ImportIndex>&)
        -> void;

    template <typename VtiExportType = T>
    auto ioToVtk(const std::string& fileName,
                 const std::string& fieldName,
                 bool               includeDomain = false,
                 Neon::IoFileType   ioFileType = Neon::IoFileType::ASCII,
                 bool               isNodeSpace = false) const -> void;


   private:
    struct Storage
    {
        std::string                    name /**< Name the user associate to the field */;
        const std::string              className;
        Neon::index_3d                 dimension;
        int                            cardinality;
        T                              outsideVal;
        Neon::DataUse                  dataUse;
        Neon::MemoryOptions            memoryOptions;
        Neon::domain::haloStatus_et::e haloStatus;
        Vec_3d<double>                 spacing /**< Spacing, i.e. size of a voxel */;
        Vec_3d<double>                 origin /**< Origin */;

        Storage();

        Storage(const std::string              fieldUserName,
                const std::string              fieldClassName,
                const Neon::index_3d&          dimension,
                int                            cardinality,
                T                              outsideVal,
                Neon::DataUse                  dataUse,
                Neon::MemoryOptions            memoryOptions,
                Neon::domain::haloStatus_et::e haloStatus,
                const Vec_3d<double>&          spacing,
                const Vec_3d<double>&          origin);
    };

    std::shared_ptr<Storage> mStorage;
};

}  // namespace Neon::domain::interface

#include "Neon/domain/interface/FieldBase_imp.h"