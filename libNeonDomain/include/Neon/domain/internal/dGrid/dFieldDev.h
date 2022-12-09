#pragma once

#include "Neon/core/core.h"
#include "Neon/core/tools/io/ioToVti.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/interface/common.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/memory/memSet.h"
#include "Neon/set/patterns/BlasSet.h"
#include "Neon/sys/memory/memConf.h"
#include "dPartition.h"

namespace Neon::domain::internal::dGrid {

class dGrid;

template <typename T, int C>
class dField;

/**
 * Global representation for the dGrid for on all devices of certain type (CPU/GPU)
 * works as a wrapper for the mem3dSet which represents the allocated memory on
 * all devices. For CPU, we have only one device. For GPU, we could have as many devices
 * as passed to the constructor.
 **/
template <typename T, int C = 0>
class dFieldDev
{
   public:
    friend dGrid;
    friend dField<T, C>;

   public:
    using self_t = dFieldDev<T, C>;
    using element_t = T;

    using grid_t = dGrid;
    using field_t = dField<T, C>;
    using fieldDev_t = dFieldDev<T, C>;
    using local_t = dPartition<T, C>;

   private:
    struct data_t
    {
        Neon::DeviceType                                devType = {Neon::DeviceType::NONE};
        int                                             cardinality;
        Neon::memLayout_et::order_e                     memOrder;
        Neon::Allocator                                 memAlloc;
        Neon::set::MemDevSet<element_t>                 memory;
        Neon::set::MemDevSet<typename field_t::ngh_idx> stencilNghIndex;
        Neon::set::DataSet<element_t*>                  userPointersSet;
        Neon::set::DataSet<Neon::size_4d>               pitch;

        std::array<Neon::set::DataSet<local_t>, Neon::DataViewUtil::nConfig> dFieldComputeSetByView;

        std::array<std::vector<Neon::set::DataSet<int>>, Neon::DataViewUtil::nConfig> startIDByView;
        std::array<std::vector<Neon::set::DataSet<int>>, Neon::DataViewUtil::nConfig> nElementsByView;

        std::shared_ptr<grid_t>        grid;
        int                            zHaloDim;
        Neon::domain::haloStatus_et::e haloStatus;
        bool                           periodic_z;
    };
    std::shared_ptr<data_t> m_data;


   private:
    dFieldDev(const grid_t&                             grid,
              const Neon::set::DataSet<Neon::index_3d>& dims,
              int                                       zHaloDim,
              Neon::domain::haloStatus_et::e            haloStatus,
              Neon::DeviceType                          deviceType,
              Neon::memLayout_et::order_e               memOrder,
              Neon::Allocator                           allocator,
              int                                       cardinality);

   public:
    dFieldDev();

    dFieldDev(const dFieldDev& other);

    dFieldDev(dFieldDev&& other);

    dFieldDev& operator=(const dFieldDev& other);

    dFieldDev& operator=(dFieldDev&& other);

    ~dFieldDev() = default;

    auto uid() const -> Neon::set::dataDependency::MultiXpuDataUid;

    auto grid() -> grid_t&;

    auto grid() const -> const grid_t&;

    auto cardinality() const -> int;

    auto devType() const -> Neon::DeviceType;

    auto dot(
        Neon::set::patterns::BlasSet<T>& blasSet,
        const dFieldDev<T>&              input,
        Neon::set::MemDevSet<T>&         output,
        const Neon::DataView&            dataView)
        -> T;

    auto dotCUB(
        Neon::set::patterns::BlasSet<T>& blasSet,
        const dFieldDev<T>&              input,
        Neon::set::MemDevSet<T>&         output,
        const Neon::DataView&            dataView) -> void;

    auto norm2(
        Neon::set::patterns::BlasSet<T>& blasSet,
        Neon::set::MemDevSet<T>&         output,
        const Neon::DataView&            dataView)
        -> T;

    auto norm2CUB(
        Neon::set::patterns::BlasSet<T>& blasSet,
        Neon::set::MemDevSet<T>&         output,
        const Neon::DataView&            dataView)
        -> void;


    template <Neon::DataView Indexing_ta = Neon::DataView::STANDARD>
    auto eRef(const Neon::index_3d& idx,
              const int             cardinality)
        -> T&;

    template <Neon::DataView Indexing_ta = Neon::DataView::STANDARD>
    auto eRef(const Neon::index_3d& idx,
              const int&            cardinality) const
        -> const T&;


    /**
     * Halo update for all cardinalities
     */
    template <Neon::set::TransferMode transferMode_ta>
    auto haloUpdate(const Neon::Backend& bk,
                    bool                 startWithBarrier = true,
                    int                  streamSetIdx = 0)
        -> void;

    template <Neon::set::TransferMode transferMode_ta>
    auto haloUpdate(Neon::SetIdx         setIdx,
                    const Neon::Backend& bk,
                    bool                 startWithBarrier = true,
                    int                  streamSetIdx = 0)
        -> void;
    /**
     * Halo update for one cardinality
     */
    template <Neon::set::TransferMode transferMode_ta>
    auto haloUpdate(const Neon::Backend& bk,
                    const int            cardIdx = -1,
                    bool                 startWithBarrier = true,
                    int                  streamSetIdx = 0)
        -> void;

    template <Neon::set::TransferMode transferMode_ta>
    auto haloUpdate(Neon::SetIdx         setIdx,
                    const Neon::Backend& bk,
                    const int            cardIdx = -1,
                    bool                 startWithBarrier = true,
                    int                  streamSetIdx = 0)
        -> void;

    auto forEach(std::function<void(bool,
                                    const Neon::index_3d&,
                                    const int32_t&,
                                    T&)> fun) -> void;

    auto forEach(std::function<void(bool,
                                    const Neon::index_3d&,
                                    const int32_t&,
                                    const T&)> fun) const
        -> void;

    auto forEachActive(std::function<void(const Neon::index_3d&,
                                          const int&,
                                          T&)> fun)
        -> void;

    template <typename exportReal_ta = T>
    auto ioToDense(Neon::memLayout_et::order_e order,
                   exportReal_ta*              dense) const
        -> void;

    template <typename exportReal_ta = T>
    auto ioToDense(Neon::memLayout_et::order_e order) const
        -> std::shared_ptr<exportReal_ta>;

    template <typename exportReal_ta = T>
    auto ioFromDense(Neon::memLayout_et::order_e order,
                     const exportReal_ta*        dense,
                     exportReal_ta /* inactiveValue*/)
        -> void;


    auto getPartition(Neon::DeviceType,
                      Neon::SetIdx   idx,
                      Neon::DataView dataView)
        -> local_t&;


    auto getPartition(Neon::DeviceType,
                      Neon::SetIdx   idx,
                      Neon::DataView dataView) const
        -> const local_t&;

    auto haloStatus() const
        -> Neon::domain::haloStatus_et::e;

   private:
    void h_init(const Neon::set::DataSet<Neon::index_3d>& dims, const grid_t& grid);


    int32_t convert_to_local(Neon::index_3d& index,
                             Neon::DataView  dataView = Neon::DataView::STANDARD) const;
};


}  // namespace Neon::domain::internal::dGrid