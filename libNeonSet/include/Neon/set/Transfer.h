#pragma once
#include <string>
#include <vector>

#include "Neon/core/core.h"
#include "Neon/set/TransferMode.h"
#include "Neon/set/TransferSemantic.h"

namespace Neon {
namespace set {

struct Transfer
{

    struct Endpoint_t
    {
        int   devId{-1};
        void* mem{nullptr};

        Endpoint_t() = default;

        Endpoint_t(int devId, void* mem)
            : devId(devId), mem(mem)
        {
        }
        auto toString(const std::string& prefix = "") const -> std::string;
    };

   private:
    TransferMode     m_mode;
    Endpoint_t       m_dst;
    Endpoint_t       m_src;
    size_t           m_size{0};
    TransferSemantic m_structure;

   public:
    Transfer(TransferMode      mode,
             const Endpoint_t& dst,
             const Endpoint_t& src,
             size_t            size,
             TransferSemantic  structure = TransferSemantic::grid)
        : m_mode(mode), m_dst(dst), m_src(src), m_size(size), m_structure(structure)
    {
    }

    /**
     * return information on the source endpoint
     * @return
     */
    auto src() const -> const Endpoint_t&;

    /**
     * return information on the destination endpoint
     * @return
     */
    auto dst() const -> const Endpoint_t&;

    /**
     * return the size of the transfer
     * @return
     */
    auto size() const -> const size_t&;

    /**
     * Print to string this object
     * @return
     */
    auto toString() const -> std::string;

    /**
     * Returns the mode of the trusfer: put or get
     * @return
     */
    auto mode() const -> TransferMode;

    auto activeDevice() const -> Neon::SetIdx;
};

// Forward declaration
class StreamSet;

/**
 * Options for a peer transfer
 */
struct PeerTransferOption
{
    enum operationMode_e
    {
        execute /** execute the transfer */,
        storeInfo /** just stores the transfer info */
    };

   private:
    std::vector<Transfer>* m_transfers = nullptr;
    StreamSet const*       m_streamSet = nullptr;
    operationMode_e        m_operationMode = Neon::set::PeerTransferOption::execute;
    TransferMode           m_transferMode = Neon::set::TransferMode::get;

    Neon::set::TransferSemantic m_structure;

   public:
    /**
     * Constructor with stream parameter
     */
    explicit PeerTransferOption(TransferMode                tranferMode,
                                Neon::set::TransferSemantic structure)
        : m_transfers(nullptr),
          m_streamSet(nullptr),
          m_operationMode(operationMode_e::execute),
          m_transferMode(tranferMode),
          m_structure(structure)
    {
        // EMPTY
    }

    /**
     * Construction with vector of transaction
     * transfers is going to be used as output parameter
     * @param transfers
     */
    PeerTransferOption(TransferMode                    tranferMode,
                       std::vector<Transfer>& NEON_OUT transfers,
                       Neon::set::TransferSemantic     structure)
        : m_transfers(&transfers),
          m_streamSet(nullptr),
          m_operationMode(operationMode_e::storeInfo),
          m_transferMode(tranferMode),
          m_structure(structure)
    {
    }
    /**
     *
     */
    auto setStreamSet(const StreamSet& streamSet) -> void
    {
        if (m_operationMode != operationMode_e::execute) {
            // Stream can be used only when the operation mode is set to execute
            NEON_THROW_UNSUPPORTED_OPTION();
        }
        m_streamSet = &streamSet;
        return;
    }


    /**
     *
     * @return
     */
    auto operationMode() const -> operationMode_e
    {
        return m_operationMode;
    }

    /**
     *
     * @return
     */
    auto transferMode() const -> TransferMode
    {
        return m_transferMode;
    }

    auto streamSet() const -> const StreamSet&
    {
        if (m_streamSet == nullptr) {
            Neon::NeonException exp("PeerTransfer_opt");
            exp << "streamSet was not set.";
            NEON_THROW(exp);
        }
        return *m_streamSet;
    }

    auto transfers() const -> std::vector<Transfer>&
    {
        if (m_transfers == nullptr) {
            Neon::NeonException exp("PeerTransfer_opt");
            exp << "transfers vector was not set.";
            NEON_THROW(exp);
        }
        return *m_transfers;
    }

    auto structure() const -> const Neon::set::TransferSemantic&
    {
        return m_structure;
    }
};

}  // namespace set
}  // namespace Neon
