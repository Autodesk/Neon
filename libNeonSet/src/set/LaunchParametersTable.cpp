#include "Neon/set/LaunchParametersTable.h"
#include "Neon/set/Backend.h"
#include "Neon/set/DevSet.h"

namespace Neon::set {


LaunchParametersTable::LaunchParametersTable(Neon::Backend const& bk)
{
    init(bk);
}

auto LaunchParametersTable::init(const Neon::Backend& bk)
->void
{
    mSetSize = bk.getDeviceCount();
    for (auto& tableRow : mLaunchParameters) {
        tableRow = bk.devSet().newLaunchParameters();
    }
}

auto LaunchParametersTable::get(Neon::DataView dw) -> Neon::set::LaunchParameters const&
{
    return mLaunchParameters[Neon::DataViewUtil::toInt(dw)];
}

}  // namespace Neon::set
