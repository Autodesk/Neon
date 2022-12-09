#include <utility>

#include "Neon/set//dependency/Token.h"
#include "Neon/set/Containter.h"

namespace Neon::set::dataDependency {

Token::Token(MultiXpuDataUid uid,
             AccessType      access,
             Compute         compute)
{
    update(uid, access, compute);
}

auto Token::update(MultiXpuDataUid uid,
                   AccessType      access,
                   Compute         compute) -> void
{
    mUid = uid;
    mAccess = access;
    mCompute = compute;

    setDataTransferContainer(
        [&](Neon::set::TransferMode)
            -> Neon::set::Container {
        NeonException exp("Token");
        exp << "This is not a stencil token and this is not a valid operation";
        NEON_THROW(exp); });
}

auto Token::access() const -> AccessType
{
    return mAccess;
}

auto Token::compute() const -> Compute
{
    return mCompute;
}

auto Token::uid() const -> MultiXpuDataUid
{
    return mUid;
}

auto Token::
    toString() const -> std::string
{
    return " uid " + std::to_string(mUid) +
           " [Op " + AccessTypeUtils::toString(mAccess) +
           " Model " + Neon::ComputeUtils::toString(mCompute) + "]";
}

auto Token::
    setDataTransferContainer(std::function<Neon::set::Container(Neon::set::TransferMode transferMode)> huPerDevice)
        -> void
{
    mHaloUpdateExtractor = std::move(huPerDevice);
}

auto Token::
    getDataTransferContainer(Neon::set::TransferMode transferMode)
        const -> Neon::set::Container
{
    return mHaloUpdateExtractor(transferMode);
}

auto Token::
    mergeAccess(AccessType tomerge) -> void
{
    mAccess = AccessTypeUtils::merge(mAccess, tomerge);
}

}  // namespace Neon::set::dataDependency