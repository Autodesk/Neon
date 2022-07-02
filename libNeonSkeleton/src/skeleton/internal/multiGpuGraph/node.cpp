#include "Neon/skeleton/internal/multiGpuGraph/MetaNode.h"

namespace Neon {
namespace skeleton {
namespace internal {

std::atomic<uint64_t> nodeCounter_g{0};

auto MetaNodeType_te::toString(MetaNodeType_te::e type) -> std::string
{
    switch (type) {
        case SYNC_LEFT_RIGHT: {
            return "SYNC_LEFT_RIGHT";
        }
        case BARRIER: {
            return "BARRIER";
        }
        case HALO_UPDATE: {
            return "HALO_UPDATE";
        }
        case CONTAINER: {
            return "KERNEL_CONTAINER";
        }
        case HELPER: {
            return "HELPER";
        }
        case UNDEFINED: {
            return "UNDEFINED";
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
}

namespace depTools = Neon::set::internal::dependencyTools;
MetaNode::MetaNode(size_t                         nodeId,
                   const std::string&             name,
                   MetaNodeType_e                 nodeType,
                   depTools::KernelContainerIdx_t kernelContainerIdx,
                   const Neon::DataView&          dataView)
    : m_nodeId(nodeId),
      m_name(name),
      m_nodeType(nodeType),
      m_kernelContainerIdx(kernelContainerIdx),
      m_dataView(dataView)
{
}

auto MetaNode::factory(MetaNodeType_e                 nodeType,
                       const std::string&             name,
                       depTools::KernelContainerIdx_t kernelContainerIdx,
                       const Neon::DataView&          dataView) -> MetaNode
{
    MetaNode node(++nodeCounter_g, name, nodeType, kernelContainerIdx, dataView);
    return node;
}

auto MetaNode::nodeId()
    const -> NodeId
{
    return m_nodeId;
}

auto MetaNode::name()
    const -> const std::string&
{
    return m_name;
}

auto MetaNode::nodeType()
    const -> MetaNodeType_e
{
    return m_nodeType;
}

auto MetaNode::getContainerId()
    const -> depTools::KernelContainerIdx_t
{
    return m_kernelContainerIdx;
}

auto MetaNode::toString() const -> std::string
{
    switch (m_nodeType) {
        case MetaNodeType_e::SYNC_LEFT_RIGHT: {
            std::string res =  //"Node ID " + std::to_string(nodeId()) +
                std::string("Left-Right Sync\n\n");
            //            std::string ret = "LR_SYNC";
            //            ret += " " + std::to_string(m_uid);
            //            ret += " DEBUG " + std::to_string(m_nodeId);
            return res;
        }
        case MetaNodeType_e::BARRIER: {
            return "BARRIER";
        }

        case MetaNodeType_e::HALO_UPDATE: {
            std::string res =  //"Node ID " + std::to_string(nodeId()) +
                std::string("Halo Update") +
                "\n\nTransferMode: " + Neon::set::TransferModeUtils::toString(m_transferMode) + "\\l";
            //                              std::string ret = "HU " +
            //                                                Neon::set::Transfer_t::toString(m_transferMode) +
            //                                                " " + std::to_string(m_uid);
            //            ret += " DEBUG " + std::to_string(m_nodeId);

            return res;
        }
        case MetaNodeType_e::CONTAINER: {
            std::string res = "" + name() +
                              //"\n\nNode ID " + std::to_string(nodeId()) +
                              "\\n\\nPattern: " + Neon::ComputeUtils::toString(m_compute) +
                              "\\lDataView: " + Neon::DataViewUtil::toString(m_dataView) + "\\l";
            //            if (isStencil() && m_hasCoherentInput) {
            //                res += "\\l coherent";
            //            }
            //            if (isStencil() && !m_hasCoherentInput) {
            //                res += "\\l NOT coherent";
            //            }
            //            res += "\\l";
            //            res += " DEBUG " + std::to_string(m_nodeId);

            return res;
        }
        case MetaNodeType_e::HELPER: {
            return "HELPER";
        }
        case MetaNodeType_e::UNDEFINED: {
            return "UNDEFINED";
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
}  // namespace userGraph

auto MetaNode::setDataView(const Neon::DataView& dv) -> void
{
    m_dataView = dv;
}

auto MetaNode::getDataView() const -> Neon::DataView
{
    return m_dataView;
}

auto MetaNode::setCompute(const Neon::Compute& c) -> void
{
    m_compute = c;
}

auto MetaNode::getCompute() const -> Neon::Compute
{
    return m_compute;
}

auto MetaNode::haloUpdateFactory(Neon::set::TransferMode                                             tm,
                                 const Neon::set::MultiDeviceObjectUid&                              dataUids,
                                 std::function<void(Neon::set::HuOptions& opt)>                      hu,
                                 const std::function<void(Neon::SetIdx, Neon::set::HuOptions& opt)>& huPerDevice) -> MetaNode
{
    MetaNode node(++nodeCounter_g, "HaloUpdate", MetaNodeType_e::HALO_UPDATE, -1);
    node.m_transferMode = tm;
    node.m_uid = dataUids;
    node.m_hu = hu;
    node.m_huPerDevice = huPerDevice;
    return node;
}

auto MetaNode::syncLeftRightFactory(const Neon::set::MultiDeviceObjectUid& dataUids) -> MetaNode
{
    MetaNode node(++nodeCounter_g, "syncLeftRigh", MetaNodeType_e::SYNC_LEFT_RIGHT, -1);
    node.m_uid = dataUids;
    return node;
}

auto MetaNode::setAsCoherent() -> void
{
    m_hasCoherentInput = true;
}
auto MetaNode::isStencil() const -> bool
{
    return m_nodeType == MetaNodeType_e::CONTAINER && m_compute == Neon::Compute::STENCIL;
}
auto MetaNode::isMap() const -> bool
{
    return m_nodeType == MetaNodeType_e::CONTAINER && m_compute == Neon::Compute::MAP;
}

auto MetaNode::isSync() const -> bool
{
    return m_nodeType == MetaNodeType_e::SYNC_LEFT_RIGHT;
}

auto MetaNode::isHu() const -> bool
{
    return m_nodeType == MetaNodeType_e::HALO_UPDATE;
}

auto MetaNode::isReduce() const -> bool
{
    return m_nodeType == MetaNodeType_e::CONTAINER && m_compute == Neon::Compute::REDUCE;
}

auto MetaNode::clone() const -> MetaNode
{
    MetaNode clone(*this);
    clone.m_nodeId = ++nodeCounter_g;
    return clone;
}

auto MetaNode::cudaGraphHandles_t::init(MetaNodeType_e nodeType, int numDevices) -> void
{
    m_data = std::vector<Neon::set::DataSet<cudaGraphNode_t>>(2);

    m_data[0] = Neon::set::DataSet<cudaGraphNode_t>(numDevices);
    m_data[1] = Neon::set::DataSet<cudaGraphNode_t>(numDevices);

    switch (nodeType) {
        case MetaNodeType_e::CONTAINER:
        case MetaNodeType_e::HELPER:
            // Node that is papped to only one CUDA graph node
            m_first = 0;
            m_last = 0;
            return;
        case MetaNodeType_e::HALO_UPDATE:
        case MetaNodeType_e::SYNC_LEFT_RIGHT:
            // Node that is papped to a sequence of nodes.
            // We track only the first and the last.
            // first() is used to creates dependencies BEFORE this node
            // last() is used to create dependencies AFTER this node
            m_first = 0;
            m_last = 1;
            return;
        default: {
            return;
        }
    }
}

auto MetaNode::cudaGraphHandles_t::first() -> Neon::set::DataSet<cudaGraphNode_t>&
{
    return m_data.at(m_first);
}

auto MetaNode::cudaGraphHandles_t::last() -> Neon::set::DataSet<cudaGraphNode_t>&
{
    return m_data.at(m_last);
}


auto MetaNode::hu(Neon::set::HuOptions& opt) -> void
{
    m_hu(opt);
}

auto MetaNode::hu(Neon::SetIdx setIdx, Neon::set::HuOptions& opt) -> void
{
    m_huPerDevice(setIdx, opt);
}

auto MetaNode::transferMode() const
    -> Neon::set::TransferMode
{
    return m_transferMode;
}

auto MetaNode::setLinearContinuousIndex(size_t id) -> void
{
    m_linearContinuousIndex = id;
}
auto MetaNode::getLinearContinuousIndex() const -> size_t
{
    return m_linearContinuousIndex;
}

}  // namespace internal
}  // namespace skeleton
}  // namespace Neon