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
struct SequenceContainer : ContainerAPI
{
   public:
    ~SequenceContainer() override = default;

    SequenceContainer(const std::string&                       name,
                      std::vector<Neon::set::Container> const& containerGraph);

    auto parse()
        -> const std::vector<Neon::set::dataDependency::Token>& override;

    auto run(int            streamIdx = 0,
             Neon::DataView dataView = Neon::DataView::STANDARD)
        -> void override;

    auto run(Neon::SetIdx   setIdx,
             int            streamIdx = 0,
             Neon::DataView dataView = Neon::DataView::STANDARD)
        -> void override;

    auto getSequence() const -> const std::vector<Neon::set::Container>&;

   private:
    std::vector<Neon::set::Container> mSequence;
};

}  // namespace Neon::set::internal
