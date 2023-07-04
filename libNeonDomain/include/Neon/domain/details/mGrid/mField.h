#pragma once
#include "Neon/set/container/Loader.h"


#include "Neon/domain/details/bGrid/bField.h"
#include "Neon/domain/details/mGrid/mPartition.h"
#include "Neon/domain/interface/FieldBaseTemplate.h"

#include "Neon/domain/details/mGrid/xField.h"

namespace Neon {

enum struct MultiResCompute
{
    MAP /**< Map operation */,
    STENCIL /**< Stencil that reads the neighbor at the same level */,
    STENCIL_UP /**< Stencil that reads the parent */,
    STENCIL_DOWN /**< Stencil that reads the children */,
};
}

namespace Neon::domain::details::mGrid {
class mGrid;


template <typename T, int C = 0>
class mField
{
    friend mGrid;

   public:
    using Type = T;
    using Grid = Neon::domain::details::mGrid::mGrid;
    using Partition = Neon::domain::details::mGrid::mPartition<T, C>;
    using Idx = typename Partition::Idx;

    mField() = default;

    virtual ~mField() = default;


    auto isInsideDomain(const Neon::index_3d& idx, const int level = 0) const -> bool;


    auto operator()(int level) -> xField<T, C>&;


    auto operator()(int level) const -> const xField<T, C>&;


    auto operator()(const Neon::index_3d& idx,
                    const int&            cardinality,
                    const int             level) -> T&;


    auto operator()(const Neon::index_3d& idx,
                    const int&            cardinality,
                    const int             level) const -> const T&;


    auto getReference(const Neon::index_3d& idx,
                      const int&            cardinality,
                      const int             level) -> T&;

    auto getReference(const Neon::index_3d& idx,
                      const int&            cardinality,
                      const int             level) const -> const T&;

    auto haloUpdate(Neon::set::HuOptions& opt) const -> void;

    auto haloUpdate(Neon::set::HuOptions& opt) -> void;

    auto updateHostData(int streamId = 0) -> void;

    auto updateDeviceData(int streamId = 0) -> void;

    auto getSharedMemoryBytes(const int32_t stencilRadius, int level = 0) const -> size_t;


    auto forEachActiveCell(int                                                                           level,
                           const std::function<void(const Neon::index_3d&, const int& cardinality, T&)>& fun,
                           bool                                                                          filterOverlaps = true,
                           Neon::computeMode_t::computeMode_e                                            mode = Neon::computeMode_t::computeMode_e::par) -> void;


    auto ioToVtk(std::string fileName,
                 bool        outputLevels = true,
                 bool        outputBlockID = true,
                 bool        outputVoxelID = true,
                 bool        filterOverlaps = true) const -> void;

    auto load(Neon::set::Loader loader, int level, Neon::MultiResCompute compute) -> typename xField<T, C>::Partition&;

    auto load(Neon::set::Loader loader, int level, Neon::MultiResCompute compute) const -> const typename xField<T, C>::Partition&;

    auto getBackend() const -> const Backend&
    {
        return mData->grid->getBackend();
    }

   private:
    mField(const std::string&         name,
           const mGrid&               grid,
           int                        cardinality,
           T                          outsideVal,
           Neon::DataUse              dataUse,
           const Neon::MemoryOptions& memoryOptions);

    auto getRef(const Neon::index_3d& idx, const int& cardinality, const int level = 0) const -> T&;


    enum PartitionBackend
    {
        cpu = 0,
        gpu = 1,
    };
    struct Data
    {
        std::shared_ptr<Grid>     grid;
        std::vector<xField<T, C>> fields;
    };
    std::shared_ptr<Data> mData;
};
}  // namespace Neon::domain::details::mGrid


#include "Neon/domain/details/mGrid/mField_imp.h"
