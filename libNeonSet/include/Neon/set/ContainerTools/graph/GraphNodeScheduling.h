#pragma once

#include "Neon/core/types/DataView.h"
#include "vector"

namespace Neon::set::container {

class GraphNodeScheduling
{

    /**
     *  Get the stream to execute the Container
     */
    auto getStream() const
        -> int;

    /**
     *  Get the event to asynchronously signal that the execution of the Container is completed.
     */
    auto getEvent() const
        -> int;

    /**
     *  Get list of events to wait the completion of.
     */
    auto getDependentEvents() const
        -> const std::vector<int>&;

    /**
     * Set the stream for the Container execution
     */
    auto setStream(int stream /**< stream for the Container execution */) -> void;

    /**
     * Set the event to asynchronously signal the completion of the Container.
     */
    auto setEvent(int event /**< Event to be used to signal the completion of the Container */) -> void;

    /**
     * Set the list of events that needed to be waited for before running the Container.
     */
    auto setDependentEvents(const std::vector<int>&) -> void;

    /**
     * Set the data view for the node
     * @param dataView
     */
    auto setDataView(Neon::DataView dataView) -> void;

   public:
    GraphNodeScheduling();

    /**
     * Returns data view associated to this node;
     */
    auto getDataView() const -> Neon::DataView;

   private:
    int              mStream /**< Stream for each operation for the node */;
    int              mEvent /**< Event to be used to signal the completion of the node container */;
    std::vector<int> mDependentEvents /**< Events to be waited for before running the Container */;
    Neon::DataView   mDataView;
};

}  // namespace Neon::set::container
