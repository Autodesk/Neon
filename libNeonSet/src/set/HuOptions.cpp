#include "Neon/set/HuOptions.h"
#include "Neon/set/Transfer.h"

namespace Neon {
namespace set {

HuOptions::HuOptions(Neon::set::TransferMode     transferMode,
                     bool                        startWithBarrier,
                     int                         streamSetIdx,
                     Neon::set::TransferSemantic structure)
    : m_startWithBarrier(startWithBarrier),
      m_streamSetIdx(streamSetIdx),
      m_peerTransferOpt(transferMode, structure),
      m_structure(structure)
{
    // EMPTY
}

HuOptions::HuOptions(Neon::set::TransferMode transferMode,
                     NEON_OUT std::vector<Neon::set::Transfer>& transfers,
                     Neon::set::TransferSemantic                structure)
    : m_startWithBarrier(false),
      m_streamSetIdx(0),
      m_peerTransferOpt(transferMode, transfers, structure),
      m_structure(structure)
{
    // EMPTY
}

auto HuOptions::getPeerTransferOpt(const Neon::Backend& bk) -> Neon::set::PeerTransferOption&
{
    if (m_peerTransferOpt.operationMode() ==
        Neon::set::PeerTransferOption::operationMode_e::storeInfo) {
    } else {
        m_peerTransferOpt.setStreamSet(bk.streamSet(m_streamSetIdx));
    }
    return m_peerTransferOpt;
}

auto HuOptions::startWithBarrier() const
    -> bool
{
    if (operationMode() == Neon::set::PeerTransferOption::operationMode_e::storeInfo) {
        return false;
    }
    return m_startWithBarrier;
}

auto HuOptions::streamSetIdx() const
    -> int
{
    return m_streamSetIdx;
}

auto HuOptions::transfers() -> std::vector<Neon::set::Transfer>&
{
    return m_peerTransferOpt.transfers();
}

auto HuOptions::transferMode() const
    -> Neon::set::TransferMode
{
    return m_peerTransferOpt.transferMode();
}

auto HuOptions::operationMode() const
    -> Neon::set::PeerTransferOption::operationMode_e
{
    return m_peerTransferOpt.operationMode();
}

auto HuOptions::isExecuteMode() const
    -> bool
{
    return operationMode() == Neon::set::PeerTransferOption::operationMode_e::execute;
}

auto HuOptions::structure() -> Neon::set::TransferSemantic
{
    return m_structure;
}

}  // namespace set
}  // namespace Neon
