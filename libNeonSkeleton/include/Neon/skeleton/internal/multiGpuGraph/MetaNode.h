#pragma once
#include "Neon/domain/internal/haloUpdateType.h"
#include "Neon/set/DevSet.h"
#include "Neon/skeleton/internal/dependencyTools/Alias.h"
namespace Neon {
namespace skeleton {
namespace internal {

struct MetaNodeType_te
{
    MetaNodeType_te() = delete;
    MetaNodeType_te(const MetaNodeType_te&) = delete;
    enum e
    {
        SYNC_LEFT_RIGHT,
        BARRIER,
        HALO_UPDATE,
        CONTAINER,
        HELPER,
        UNDEFINED
    };

    static auto
    toString(MetaNodeType_te::e type) -> std::string;
};
using MetaNodeType_e = MetaNodeType_te::e;


struct MetaNode
{
   private:
    MetaNode(size_t                nodeId,
             const std::string&    name,
             MetaNodeType_e        nodeType,
             ContainerIdx          kernelContainerIdx,
             const Neon::DataView& dataView = Neon::DataView::STANDARD);

   private:
    size_t         m_nodeId;
    std::string    m_name;
    MetaNodeType_e m_nodeType;
    ContainerIdx   m_kernelContainerIdx;
    Neon::DataView m_dataView{Neon::DataView::STANDARD};
    Neon::Compute  m_compute{Neon::Compute::MAP};

    Neon::set::TransferMode         m_transferMode{Neon::set::TransferMode::get};
    Neon::set::MultiDeviceObjectUid m_uid;

    size_t m_linearContinuousIndex = 0; /** this index is created when any change to the graph has been completed.
                                         * It provides a way to associate data to nodes by the means of vectors
                                         */

    bool m_hasCoherentInput = false;

    std::function<void(Neon::set::HuOptions& opt)>               m_hu;
    std::function<void(Neon::SetIdx, Neon::set::HuOptions& opt)> m_huPerDevice;

    struct cudaGraphHandles_t
    {
        std::vector<Neon::set::DataSet<cudaGraphNode_t>> m_data;
        int                                              m_first = 3;
        int                                              m_last = 3;

        auto init(MetaNodeType_e m_nodeType, int numDevices) -> void;
        auto first() -> Neon::set::DataSet<cudaGraphNode_t>&;
        auto last() -> Neon::set::DataSet<cudaGraphNode_t>&;

    } m_cudaGraphHandles;

   public:
    /**
     *
     * @return
     */
    static auto
    factory(MetaNodeType_e        nodeType,
            const std::string&    name,
            ContainerIdx          kernelContainerIdx,
            const Neon::DataView& dataView = Neon::DataView::STANDARD) -> MetaNode;

    /**
     * Create a clone of the node.
     * The new node will have a different id
     * @return
     */
    auto clone() const -> MetaNode;

    static auto
    haloUpdateFactory(Neon::set::TransferMode,
                      const Neon::set::MultiDeviceObjectUid&                              dataUids,
                      std::function<void(Neon::set::HuOptions& opt)>                      hu,
                      const std::function<void(Neon::SetIdx, Neon::set::HuOptions& opt)>& huPerDevice) -> MetaNode;

    static auto
    syncLeftRightFactory(const Neon::set::MultiDeviceObjectUid& dataUids) -> MetaNode;

    auto
    nodeId() const -> NodeId;


    auto
    name() const -> const std::string&;

    auto setAsCoherent() -> void;

    auto nodeType() const -> MetaNodeType_e;

    auto getContainerId() const -> ContainerIdx;

    auto toString() const -> std::string;

    auto setDataView(const Neon::DataView&) -> void;

    auto getDataView() const -> Neon::DataView;

    auto setCompute(const Neon::Compute&) -> void;

    auto getCompute() const -> Neon::Compute;

    auto isStencil() const -> bool;

    auto isReduce() const -> bool;

    auto isMap() const -> bool;

    auto isSync() const -> bool;

    auto isHu() const -> bool;

    auto hu(Neon::set::HuOptions& opt) -> void;

    auto hu(Neon::SetIdx setIdx, Neon::set::HuOptions& opt) -> void;

    auto transferMode() const -> Neon::set::TransferMode;

    auto setLinearContinuousIndex(size_t id) -> void;

    auto getLinearContinuousIndex() const -> size_t;
};

}  // namespace internal
}  // namespace skeleton
}  // namespace Neon