#pragma once

#include <memory>
#include "Neon/set/Containter.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/MultiXpuDataInterface.h"


namespace Neon::set::internal::datum {

template <typename Obj>
class Partition
{
   public:
    auto NEON_CUDA_HOST_DEVICE operator()() -> Obj&
    {
        return *objPrt;
    }
    auto NEON_CUDA_HOST_DEVICE operator()() const -> const Obj&
    {
        return *objPrt;
    }
    Obj* objPrt;
};

template <typename Obj>
class PartitionIndexSpace
{
   public:
    using Cell = void*;
    static constexpr int SpaceDim = 1;

    NEON_CUDA_HOST_DEVICE
    inline auto setAndValidate(Cell&,
                               const size_t& x,
                               const size_t& y,
                               const size_t& z)
        const
        -> bool
    {
        if (x == 0 && y == 0 && z == 0) {
            bool const isValid = true;
            return isValid;
        }
        bool const isValid = false;
        return isValid;
    }
};

template <typename Obj>
struct Storage
{
    PartitionIndexSpace<Obj> indexSpace;
    std::array<Neon::set::DataSet<Partition<Obj>>,
               Neon::PlaceUtils::numConfigurations>
        partitionByView;

    Neon::set::MemSet_t<Obj> obj;
    Neon::Backend            bk;
    Neon::MemoryOptions      memoryOptions;
    Neon::DataUse            dataUse;
};
}  // namespace Neon::set::internal::datum

namespace Neon::set {

/**
 * Abstracting  an object replicated on each device memory.
 * @tparam Obj
 */
template <typename Obj>
class Replica : public Neon::set::interface::MultiXpuDataInterface<Neon::set::internal::datum::Partition<Obj>,
                                                                        Neon::set::internal::datum::Storage<Obj>>
{
   public:
    using Partition = Neon::set::internal::datum::Partition<Obj>;
    using PartitionIndexSpace = Neon::set::internal::datum::PartitionIndexSpace<Obj>;
    using Storage = Neon::set::internal::datum::Storage<Obj>;
    using Cell = typename PartitionIndexSpace::Cell;

    using Self = Replica<Obj>;

    virtual ~Replica() = default;

    Replica() = default;

    explicit Replica(Neon::Backend&      bk,
                     Neon::DataUse       dataUse = Neon::DataUse::IO_COMPUTE,
                     Neon::MemoryOptions memoryOptions = Neon::MemoryOptions());

    virtual auto updateIO(int streamId = 0)
        -> void;

    virtual auto updateCompute(int streamId = 0)
        -> void;

    virtual auto getPartition(Neon::Place  execution,
                              Neon::SetIdx          setIdx,
                              const Neon::DataView& dataView = Neon::DataView::STANDARD) const
        -> const Self::Partition&;

    virtual auto getPartition(Neon::Place  execution,
                              Neon::SetIdx          setIdx,
                              const Neon::DataView& dataView = Neon::DataView::STANDARD)
        -> Self::Partition&;

    virtual auto getPartition(Neon::DeviceType      execution,
                              Neon::SetIdx          setIdx,
                              const Neon::DataView& dataView = Neon::DataView::STANDARD) const
        -> const Self::Partition&;

    virtual auto getPartition(Neon::DeviceType      execution,
                              Neon::SetIdx          setIdx,
                              const Neon::DataView& dataView = Neon::DataView::STANDARD)
        -> Self::Partition&;

    virtual auto getPartitionIndexSpace(Neon::DeviceType      execution,
                                        Neon::SetIdx          setIdx,
                                        const Neon::DataView& dataView = Neon::DataView::STANDARD) const
        -> const Self::PartitionIndexSpace&;

    virtual auto getPartitionIndexSpace(Neon::DeviceType      execution,
                                        Neon::SetIdx          setIdx,
                                        const Neon::DataView& dataView = Neon::DataView::STANDARD)
        -> Self::PartitionIndexSpace&;

    auto getBackend() -> Neon::Backend&;

    auto getBackend() const -> const Neon::Backend&;

    auto operator()(Neon::SetIdx setIdx) -> Obj&;

    auto operator()(Neon::SetIdx setIdx) const -> const Obj&;

    template <typename LoadingLambda>
    auto getContainer(const std::string& name, LoadingLambda lambda)
        const
        -> Neon::set::Container;

    auto getLaunchParameters(Neon::DataView  dataView,
                             const index_3d& blockDim,
                             const size_t&   shareMem) const -> Neon::set::LaunchParameters;
};


}  // namespace Neon::set
#include "Neon/set/Replica_imp.h"