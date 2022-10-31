#pragma once
#include <vector>
#include "Neon/set/Backend.h"
#include "Neon/set/Transfer.h"

namespace Neon {
namespace set {
struct HuOptions
{
   private:
    bool                          m_startWithBarrier = true;
    int                           m_streamSetIdx = 0;
    Neon::set::PeerTransferOption m_peerTransferOpt;
    Neon::set::StencilSemantic    m_structure;

   public:
    HuOptions(Neon::set::TransferMode     transferMode /*<                                          Mode of the transfer: put or get                                 */,
              bool                        startWithBarrier /*<                                      If true a barrier is executed before initiating the halo update */,
              int                         streamSetIdx = Neon::Backend::mainStreamIdx /*<                                      Target stream for the halo update                               */,
              Neon::set::StencilSemantic structure = Neon::set::StencilSemantic::standard /*<    Structure on top of which the transfer is one: grid or lattice  */);

    HuOptions(Neon::set::TransferMode transferMode,
              NEON_OUT std::vector<Neon::set::Transfer>& transfers,
              Neon::set::StencilSemantic                 structure = Neon::set::StencilSemantic::standard);

    auto getPeerTransferOpt(const Neon::Backend& bk) -> Neon::set::PeerTransferOption&;
    auto startWithBarrier() const -> bool;
    auto streamSetIdx() const -> int;
    auto transfers() -> std::vector<Neon::set::Transfer>&;
    auto operationMode() const -> Neon::set::PeerTransferOption::operationMode_e;
    auto transferMode() const -> Neon::set::TransferMode;
    auto isExecuteMode() const -> bool;
    auto getSemantic() const -> Neon::set::StencilSemantic;
};
}  // namespace set
}  // namespace Neon
