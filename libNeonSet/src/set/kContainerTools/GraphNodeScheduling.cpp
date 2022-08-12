#include "Neon/set/ContainerTools/graph/GraphNodeScheduling.h"

namespace Neon::set::container {

GraphNodeScheduling::GraphNodeScheduling()
{
    reset();
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

auto GraphNodeScheduling::getDependentEvents()
    -> std::vector<int>&
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

auto GraphNodeScheduling::setDataView(Neon::DataView dataView) -> void
{
    mDataView = dataView;
}

auto GraphNodeScheduling::getDataView() const -> Neon::DataView
{
    return mDataView;
}

auto GraphNodeScheduling::reset() -> void
{
    mStream = -1;
    mEvent = -1;
    mDataView = DataView::STANDARD;
    mDependentEvents.clear();
}

}  // namespace Neon::set::container
