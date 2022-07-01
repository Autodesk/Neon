#include "Neon/set/Transfer.h"
#include "Neon/set/Backend.h"

namespace Neon {
namespace set {

auto Transfer::Endpoint_t::toString(const std::string& prefix) const -> std::string
{
    return prefix + "[dev: " + std::to_string(devId) +
           " addr " + std::to_string((size_t)(mem)) + "]";
}

auto Transfer::src() const -> const Endpoint_t&
{
    return m_src;
}

auto Transfer::dst() const -> const Endpoint_t&
{
    return m_dst;
}

auto Transfer::size() const -> const size_t&
{
    return m_size;
}

auto Transfer::toString() const -> std::string
{
    return m_dst.toString("Transfer_t - dst ") + m_src.toString(" src ") + "size " + std::to_string(m_size);
}

auto Transfer::mode() const -> TransferMode
{
    return m_mode;
}

auto Transfer::activeDevice() const -> Neon::SetIdx
{
    int activeDev = -1;
    switch (m_mode) {
        case TransferMode::put: {
            // The src copy (PUT) the memory into dst
            activeDev = m_src.devId;
            break;
        }
        case TransferMode::get: {
            // The dst copy (GET) the memory from the src
            activeDev = m_dst.devId;
            break;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
    return activeDev;
}

}  // namespace set
}  // namespace Neon
