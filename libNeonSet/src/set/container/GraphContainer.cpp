#include "Neon/set/container/GraphContainer.h"
#include "Neon/set/container/Graph.h"

namespace Neon::set::internal {

GraphContainer::GraphContainer(const std::string&                         name,
                               const Neon::set::container::Graph&         containerGraph,
                               std::function<void(Neon::SetIdx, Loader&)> loadingLambda)
    : mLoadingLambda(loadingLambda)
{
    mGraph = std::make_shared<Neon::set::container::Graph>(containerGraph);
    setContainerExecutionType(ContainerExecutionType::graph);
    setDataViewSupport(DataViewSupport::off);
    setName(name);

    this->parse();
}

auto GraphContainer::newParser() -> Loader
{
    auto parser = Loader(*this,
                         Neon::DeviceType::CPU,
                         Neon::SetIdx(0),
                         Neon::DataView::STANDARD,
                         Neon::set::internal::LoadingMode_e::PARSE_AND_EXTRACT_LAMBDA);
    return parser;
}

auto GraphContainer::parse() -> const std::vector<Neon::set::internal::dependencyTools::DataToken>&
{
    Neon::SetIdx setIdx(0);
    if (!this->isParsingDataUpdated()) {
        auto parser = newParser();
        this->mLoadingLambda(setIdx, parser);
        this->setParsingDataUpdated(true);

        setContainerPattern(ContainerPatternType::complex);
    }
    return getTokens();
}

auto GraphContainer::getGraph() -> const Neon::set::container::Graph&
{
    return *mGraph;
}

auto GraphContainer::getHostContainer() -> std::shared_ptr<internal::ContainerAPI>
{
    NEON_THROW_UNSUPPORTED_OPTION("A managed Container Container is not associated with any host operation.");
}

auto GraphContainer::getDeviceContainer() -> std::shared_ptr<internal::ContainerAPI>
{
    NEON_THROW_UNSUPPORTED_OPTION("A managed Container Container is not associated with any host operation.");
}

/**
 * Run container over streams
 * @param streamIdx
 * @param dataView
 */
auto GraphContainer::run(int /*streamIdx*/,
                         Neon::DataView /*dataView*/) -> void
{
    ///  mGraph->run(streamIdx, dataView);
}

auto GraphContainer::run(Neon::SetIdx   /*setIdx*/,
                         int            /*streamIdx*/,
                         Neon::DataView /*dataView*/) -> void
{
//    if (ContainerExecutionType::graph == this->getContainerExecutionType()) {
//        mGraph->run(setIdx, streamIdx, dataView);
//    }
//    NEON_THROW_UNSUPPORTED_OPTION("");
}

}  // namespace Neon::set::internal
