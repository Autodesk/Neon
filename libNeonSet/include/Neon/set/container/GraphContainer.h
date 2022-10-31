#pragma once

#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Loader.h"

namespace Neon::set::internal {
struct Graph;
/**
 * Specialized implementation of KContainer_i
 *
 *
 * @tparam DataContainer
 * @tparam ComputeLambdaT
 */
struct GraphContainer : ContainerAPI
{
   public:
    ~GraphContainer() override = default;

    GraphContainer(const std::string&                         name,
                   const Neon::set::container::Graph&         containerGraph,
                   std::function<void(Neon::SetIdx, Loader&)> loadingLambda);

    auto newParser()
        -> Loader;

    auto parse()
        -> const std::vector<Neon::set::dataDependency::Token>& override;

    auto getGraph()
      const  -> const Neon::set::container::Graph& override;

    auto run(int            streamIdx = 0,
             Neon::DataView dataView = Neon::DataView::STANDARD)
        -> void override;

    auto run(Neon::SetIdx   setIdx,
             int            streamIdx = 0,
             Neon::DataView dataView = Neon::DataView::STANDARD)
        -> void override;

   private:
    std::function<void(Neon::SetIdx, Loader&)>   mLoadingLambda;
    std::shared_ptr<Neon::set::container::Graph> mGraph;
};

}  // namespace Neon::set::internal
