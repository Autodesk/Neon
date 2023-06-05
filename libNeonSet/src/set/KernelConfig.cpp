#include "Neon/set/KernelConfig.h"

namespace Neon {
namespace set {


KernelConfig::KernelConfig(Neon::DataView         dataView,
                               const Neon::Backend&   bk,
                               int                    streamIdx,
                               const LaunchParameters& launchInfoSet)
{
    mDataView = dataView;
    mBackend = bk;
    mStreamIdx = streamIdx;
    mLaunchInfoSet = launchInfoSet;
}

KernelConfig::KernelConfig(const Neon::Backend&   bk,
                               int                    streamIdx,
                               const LaunchParameters& launchInfoSet)
{
    mBackend = bk;
    mStreamIdx = streamIdx;
    mLaunchInfoSet = launchInfoSet;
}


auto KernelConfig::dataView()
    const
    -> const Neon::DataView&
{
    return mDataView;
}

auto KernelConfig::backend()
    const
    -> const Neon::Backend&
{
    return mBackend;
}

auto KernelConfig::stream()
    const
    -> const int&
{
    return mStreamIdx;
}

auto KernelConfig::launchInfoSet()
    const
    -> const LaunchParameters&
{
    return mLaunchInfoSet;
}

auto KernelConfig::expertSetDataView(const Neon::DataView& dataview)
    -> void
{
    mDataView = dataview;
}

auto KernelConfig::expertSetBackend(const Neon::Backend& bk) -> void
{
    mBackend = bk;
}

auto KernelConfig::expertSetStream(const int& s) -> void
{
    mStreamIdx = s;
}

auto KernelConfig::expertSetLaunchParameters(const LaunchParameters& l) -> void
{
    mLaunchInfoSet = l;
}
auto KernelConfig::expertSetLaunchParameters(std::function<void(LaunchParameters&)> f) -> void
{
    f(mLaunchInfoSet);
}

auto KernelConfig::streamSet() const -> const StreamSet&
{
    return mBackend.streamSet(this->stream());
}
auto KernelConfig::runMode() const -> Neon::run_et::et
{
    return mBackend.runMode();
}

}  // namespace set
}  // namespace Neon
