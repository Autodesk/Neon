#include "Neon/set//dependencyTools/DataParsing.h"

namespace Neon {
namespace set {
namespace internal {
namespace dependencyTools {


DataToken::DataToken(DataUId_t uid,
                     Access_e  access,
                     Compute   compute)
{
    update(uid, access, compute);
    setHaloUpdate(
        [&](Neon::set::HuOptions&) -> void {
            NeonException exp("DataToken_t");
            exp << "This is not a stencil token and this is not a valid operation";
            NEON_THROW(exp);
        },
        [&](Neon::SetIdx, Neon::set::HuOptions&) -> void {
            NeonException exp("DataToken_t");
            exp << "This is not a stencil token and this is not a valid operation";
            NEON_THROW(exp);
        });
}

auto DataToken::update(DataUId_t uid,
                       Access_e  access,
                       Compute   compute) -> void
{
    m_uid = uid;
    m_access = access;
    m_compute = compute;
    setHaloUpdate(
        [&](Neon::set::HuOptions&) -> void {
        NeonException exp("DataToken_t");
        exp << "This is not a stencil token and this is not a valid operation";
        NEON_THROW(exp); },
        [&](Neon::SetIdx, Neon::set::HuOptions&) -> void {
            NeonException exp("DataToken_t");
            exp << "This is not a stencil token and this is not a valid operation";
            NEON_THROW(exp);
        });
}

auto DataToken::access() const -> Access_e
{
    return m_access;
}

auto DataToken::compute() const -> Compute
{
    return m_compute;
}
auto DataToken::uid() const -> DataUId_t
{
    return m_uid;
}
auto DataToken::toString() const -> std::string
{
    return " uid " + std::to_string(m_uid) +
           " [Op " + Access_et::toString(m_access) +
           " Model " + Neon::ComputeUtils::toString(m_compute) + "]";
}


auto DataToken::setHaloUpdate(std::function<void(Neon::set::HuOptions& opt)>               hu,
                              std::function<void(Neon::SetIdx, Neon::set::HuOptions& opt)> huPerDevice) -> void
{
    m_hu = hu;
    m_huPerDevice = huPerDevice;
}

auto DataToken::getHaloUpdate() const
    -> const std::function<void(Neon::set::HuOptions& opt)>&
{
    return m_hu;
}

auto DataToken::getHaloUpdatePerDevice() const
    -> const std::function<void(Neon::SetIdx, Neon::set::HuOptions& opt)>&
{
    return m_huPerDevice;
}
auto DataToken::mergeAccess(Access_et::e tomerge) -> void
{
    m_access = Access_et::merge(m_access, tomerge);
}
}  // namespace dependencyTools
}  // namespace internal
}  // namespace set
}  // namespace Neon