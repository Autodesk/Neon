#pragma once
#include "Neon/core/core.h"

#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Loader.h"

namespace Neon {
namespace set {
namespace internal {

/**
 * Specialized implementation of KContainer_i
 *
 *
 * @tparam DataIteratorContainerT
 * @tparam UserComputeLambdaT
 */
struct AnchorContainer : ContainerAPI
{
   public:
    virtual ~AnchorContainer() override = default;

   public:
    AnchorContainer(const std::string& name);

    auto parse() -> const std::vector<Neon::set::internal::dependencyTools::DataToken>& override;

    auto getHostContainer() -> std::shared_ptr<ContainerAPI> final;

    virtual auto getDeviceContainer() -> std::shared_ptr<ContainerAPI> final;

    /**
     * Run container over streams
     * @param streamIdx
     * @param dataView
     */
    virtual auto run(int streamIdx = 0, Neon::DataView dataView = Neon::DataView::STANDARD) -> void override;

    /**
     * Run container over streams
     * @param streamIdx
     * @param dataView
     */
    virtual auto run(Neon::SetIdx setIdx, int streamIdx, Neon::DataView dataView) -> void override;

   private:
    std::vector<Neon::set::internal::dependencyTools::DataToken> mEmtpy;

};

}  // namespace internal
}  // namespace set
}  // namespace Neon
