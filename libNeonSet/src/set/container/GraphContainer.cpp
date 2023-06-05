#include "Neon/set/container/GraphContainer.h"
#include "Neon/set/container/Graph.h"

namespace Neon::set::internal {

GraphContainer::
    GraphContainer(const std::string&                         name,
                   const Neon::set::container::Graph&         containerGraph,
                   std::function<void(Neon::SetIdx, Loader&)> loadingLambda)
    : mLoadingLambda(loadingLambda)
{
    mGraph = std::make_shared<Neon::set::container::Graph>(containerGraph);
    setContainerExecutionType(ContainerExecutionType::graph);
    setContainerOperationType(ContainerOperationType::graph);
    setDataViewSupport(DataViewSupport::off);
    setName(name);

    this->parse();
}

auto GraphContainer::
    newParser()
        -> Loader
{
    auto parser = Loader(*this,
                         Execution::host,
                         Neon::SetIdx(0),
                         Neon::DataView::STANDARD,
                         Neon::set::internal::LoadingMode_e::PARSE_AND_EXTRACT_LAMBDA);
    return parser;
}

auto GraphContainer::
    parse()
        -> const std::vector<Neon::set::dataDependency::Token>&
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

auto GraphContainer::
    getGraph()
        const -> const Neon::set::container::Graph&
{
    return *mGraph;
}

/**
 * Run container over streams
 * @param streamIdx
 * @param dataView
 */
auto GraphContainer::
    run(int            streamIdx,
        Neon::DataView dataView) -> void
{
    mGraph->run(streamIdx, dataView);
}

auto GraphContainer::
    run(Neon::SetIdx   setIdx,
        int            streamIdx,
        Neon::DataView dataView) -> void
{
    mGraph->run(setIdx, streamIdx, dataView);
}

}  // namespace Neon::set::internal
