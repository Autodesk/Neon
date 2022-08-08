#include "Neon/set/ContainerTools/graph/GraphNodeScheduling.h"

namespace Neon::set::container {

GraphNodeScheduling::GraphNodeScheduling()
{
    mStream = -1;
    mEvent = -1;
    mDataView = Neon::DataView::STANDARD;
}

auto GraphNodeScheduling::getStream() const
    -> int
{
    return mStream;
}

auto GraphNodeScheduling::getEvent() const
    -> int
{
    return mEvent;
}

auto GraphNodeScheduling::getDependentEvents() const
    -> const std::vector<int>&
{
    return mDependentEvents;
}

auto GraphNodeScheduling::setStream(int stream)
    -> void
{
    mStream = stream;
}

auto GraphNodeScheduling::setEvent(int event)
    -> void
{
    mEvent = event;
}

auto GraphNodeScheduling::setDependentEvents(const std::vector<int>& dependentEvents)
    -> void
{
    mDependentEvents = dependentEvents;
}

auto GraphNodeScheduling::setDataView(Neon::DataView dataView) -> void
{
    mDataView = dataView;
}

auto GraphNodeScheduling::getDataView() const -> Neon::DataView
{
    return mDataView;
}

}  // namespace Neon::set::container
