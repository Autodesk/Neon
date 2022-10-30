#pragma once

#include "Neon/core/types/DataView.h"
#include "vector"

namespace Neon::set::container {

class GraphNodeScheduling
{
   public:
    /**
     *  Get list of events to wait the completion of.
     */
    auto getDependentEvents()
        -> std::vector<int>&;

    auto getDependentEvents()
        const -> const std::vector<int>&;

    /**
     * Set the data view for the node
     * @param dataView
     */
    auto setDataView(Neon::DataView dataView)
        -> void;

    GraphNodeScheduling();

    /**
     * Returns data view associated to this node;
     */
    auto getDataView()
        const -> Neon::DataView;

    /**
     *  Get the stream to execute the Container
     */
    auto getStream()
        const -> int;

    /**
     * Set the stream for the Container execution
     */
    auto setStream(int stream /**< stream for the Container execution */)
        -> void;

    /**
     *  Get the event to asynchronously signal that the execution of the Container is completed.
     */
    auto getEvent() const
        -> int;

    /**
     * Set the event to asynchronously signal the completion of the Container.
     */
    auto setEvent(int event /**< Event to be used to signal the completion of the Container */)
        -> void;

    /**
     * Reset all scheduling data
     */
    auto reset()
        -> void;

   private:
    int              mStream{-1} /**< Stream for each operation for the node */;
    int              mEvent{-1} /**< Event to be used to signal the completion of the node container */;
    std::vector<int> mDependentEvents /**< Events to be waited for before running the Container */;
    Neon::DataView   mDataView{DataView::STANDARD};
};

}  // namespace Neon::set::container
