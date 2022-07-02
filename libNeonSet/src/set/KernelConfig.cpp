#include "Neon/set/KernelConfig.h"

namespace Neon {
namespace set {


KernelConfig::KernelConfig(Neon::DataView         dataView,
                               const Neon::Backend&   bk,
                               int                    streamIdx,
                               const LaunchParameters& launchInfoSet)
{
    m_dataView = dataView;
    m_bk = bk;
    m_streamIdx = streamIdx;
    m_launchInfoSet = launchInfoSet;
}

KernelConfig::KernelConfig(const Neon::Backend&   bk,
                               int                    streamIdx,
                               const LaunchParameters& launchInfoSet)
{
    m_bk = bk;
    m_streamIdx = streamIdx;
    m_launchInfoSet = launchInfoSet;
}


auto KernelConfig::dataView()
    const
    -> const Neon::DataView&
{
    return m_dataView;
}

auto KernelConfig::backend()
    const
    -> const Neon::Backend&
{
    return m_bk;
}

auto KernelConfig::stream()
    const
    -> const int&
{
    return m_streamIdx;
}

auto KernelConfig::launchInfoSet()
    const
    -> const LaunchParameters&
{
    return m_launchInfoSet;
}

auto KernelConfig::expertSetDataView(const Neon::DataView& dataview)
    -> void
{
    m_dataView = dataview;
}

auto KernelConfig::expertSetBackend(const Neon::Backend& bk) -> void
{
    m_bk = bk;
}

auto KernelConfig::expertSetStream(const int& s) -> void
{
    m_streamIdx = s;
}

auto KernelConfig::expertSetLaunchParameters(const LaunchParameters& l) -> void
{
    m_launchInfoSet = l;
}
auto KernelConfig::expertSetLaunchParameters(std::function<void(LaunchParameters&)> f) -> void
{
    f(m_launchInfoSet);
}

auto KernelConfig::streamSet() const -> const StreamSet&
{
    return m_bk.streamSet(this->stream());
}
auto KernelConfig::runMode() const -> Neon::run_et::et
{
    return m_bk.runMode();
}

}  // namespace set
}  // namespace Neon
