#pragma once

#include <memory>

#include "Neon/core/core.h"
#include "Neon/core/types/Place.h"
#include "Neon/set/MultiXpuDataUid.h"

namespace Neon::set::interface {

template <typename P, typename S>
class MultiXpuDataInterface
{
   public:
    using Partition = P;
    using Storage = S;

    using Self = MultiXpuDataInterface<Partition, Storage>;

    virtual ~MultiXpuDataInterface() = default;

    MultiXpuDataInterface();

    virtual auto updateIO(int streamId = 0)
        -> void = 0;

    virtual auto updateCompute(int streamId = 0)
        -> void = 0;

    /**
     * Return a partition based on a set of parameters: execution type, target device, dataView
     */
    virtual auto getPartition(Neon::Place  execution,
                              Neon::SetIdx          setIdx,
                              const Neon::DataView& dataView = Neon::DataView::STANDARD) const
        -> const Self::Partition& = 0;

    virtual auto getPartition(Neon::Place  execution,
                              Neon::SetIdx          setIdx,
                              const Neon::DataView& dataView = Neon::DataView::STANDARD)
        -> Self::Partition& = 0;

    auto getStorage() -> Storage&;

    auto getStorage() const -> const Storage&;

    auto getUid() const -> Neon::set::dataDependency::MultiXpuDataUid;

   protected:
    static auto swapUIDs(Self& A, Self& B) -> void;

   private:

    std::shared_ptr<int>     mUid;
    std::shared_ptr<Storage> mStorage;
};

template <typename P, typename S>
auto MultiXpuDataInterface<P, S>::getStorage() -> Storage&
{
    return *(mStorage.get());
}
template <typename P, typename S>
auto MultiXpuDataInterface<P, S>::getStorage() const -> const Storage&
{
    return *(mStorage.get());
}

template <typename P, typename S>
auto MultiXpuDataInterface<P, S>::getUid() const -> Neon::set::dataDependency::MultiXpuDataUid
{
    void*                           addr = static_cast<void*>(mUid.get());
    Neon::set::dataDependency::MultiXpuDataUid uidRes = (size_t)addr;
    return uidRes;
}

template <typename P, typename S>
MultiXpuDataInterface<P, S>::MultiXpuDataInterface()
{
    mStorage = std::make_shared<Storage>();
    mUid = std::make_shared<int>();
}

template <typename P, typename S>
auto MultiXpuDataInterface<P, S>::swapUIDs(MultiXpuDataInterface::Self& A, MultiXpuDataInterface::Self& B) -> void
{
    std::swap(A.mUid,B.mUid);
}

}  // namespace Neon::set::interface
