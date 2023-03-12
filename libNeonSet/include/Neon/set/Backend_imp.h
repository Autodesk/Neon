#pragma once

namespace Neon {

template <typename T>
auto Backend::newDataSet()
    const -> Neon::set::DataSet<T>
{
    int  nDevs = getDeviceCount();
    auto result = Neon::set::DataSet<T>(nDevs);
    return result;
}

template <typename T>
auto Backend::newDataSet(T const& val)
    const -> Neon::set::DataSet<T>
{
    int  nDevs = getDeviceCount();
    auto result = Neon::set::DataSet<T>(nDevs, val);
    return result;
}

template <typename T, typename Lambda>
auto Backend::newDataSet(Lambda lambda)
    const -> Neon::set::DataSet<T>
{
    int  nDevs = getDeviceCount();
    auto result = Neon::set::DataSet<T>(nDevs);
    result.forEachSeq(lambda);
    return result;
}

template <typename Lambda>
auto Backend::forEachDeviceSeq(const Lambda& lambda)
    const -> void
{
    int const nDevs = getDeviceCount();
    for (int i = 0; i < nDevs; i++) {
        lambda(Neon::SetIdx(i));
    }
}

template <typename T>
auto Backend::deviceToDeviceTransfer(int                     streamId,
                                     size_t                  nItems,
                                     Neon::set::TransferMode transferMode,
                                     Neon::SetIdx            dstSet,
                                     T*                      dstAddr,
                                     Neon::SetIdx            srcSet,
                                     T const*                srcAddr) -> void
{
    helpDeviceToDeviceTransferByte(streamId,
                                   sizeof(T) * nItems,
                                   transferMode,
                                   dstSet,
                                   dstAddr,
                                   srcSet,
                                   srcAddr);
}
}  // namespace Neon