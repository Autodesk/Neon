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
class Span
{
   public:
    using Idx = void*;
    static constexpr int SpaceDim = 1;

    NEON_CUDA_HOST_DEVICE
    inline auto setAndValidate(Idx&,
                               const size_t& x)
        const
        -> bool
    {
        if (x == 0) {
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
    Span<Obj> indexSpace;
    std::array<Neon::set::DataSet<Partition<Obj>>,
               Neon::ExecutionUtils::numConfigurations>
        partitionByView;

    Neon::set::MemSet<Obj> obj;
    Neon::Backend          bk;
    Neon::MemoryOptions    memoryOptions;
    Neon::DataUse          dataUse;
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
    using Span = Neon::set::internal::datum::Span<Obj>;
    using Storage = Neon::set::internal::datum::Storage<Obj>;
    using Idx = typename Span::Idx;

    static constexpr Neon::set::details::ExecutionThreadSpan executionThreadSpan = Neon::set::details::ExecutionThreadSpan::d1;
    using ExecutionThreadSpanIndexType = uint32_t;

    using Self = Replica<Obj>;

    virtual ~Replica() = default;

    Replica() = default;

    explicit Replica(Neon::Backend&      bk,
                     Neon::DataUse       dataUse = Neon::DataUse::HOST_DEVICE,
                     Neon::MemoryOptions memoryOptions = Neon::MemoryOptions());

    virtual auto updateHostData(int streamId = 0)
        -> void;

    virtual auto updateDeviceData(int streamId = 0)
        -> void;

    virtual auto getPartition(Neon::Execution       execution,
                              Neon::SetIdx          setIdx,
                              const Neon::DataView& dataView = Neon::DataView::STANDARD) const
        -> const Self::Partition&;

    virtual auto getPartition(Neon::Execution       execution,
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

    virtual auto getSpan(Neon::Execution       execution,
                         Neon::SetIdx          setIdx,
                         const Neon::DataView& dataView = Neon::DataView::STANDARD) const
        -> const Self::Span&;

    virtual auto getSpan(Neon::Execution       execution,
                         Neon::SetIdx          setIdx,
                         const Neon::DataView& dataView = Neon::DataView::STANDARD)
        -> Self::Span&;

    auto getBackend() -> Neon::Backend&;

    auto getBackend() const -> const Neon::Backend&;

    auto operator()(Neon::SetIdx setIdx) -> Obj&;

    auto operator()(Neon::SetIdx setIdx) const -> const Obj&;

    template <Neon::Execution execution = Neon::Execution::device,
              typename LoadingLambda = void*>
    auto newContainer(const std::string& name, LoadingLambda lambda)
        const
        -> Neon::set::Container;

    auto getLaunchParameters(Neon::DataView  dataView,
                             const index_3d& blockDim,
                             const size_t&   shareMem) const -> Neon::set::LaunchParameters;
};


}  // namespace Neon::set
#include "Neon/set/Replica_imp.h"