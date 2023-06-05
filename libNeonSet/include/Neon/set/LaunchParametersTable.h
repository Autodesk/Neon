#pragma once


#include "Neon/core/core.h"
#include "Neon/set/LaunchParameters.h"

namespace Neon {
class Backend;
}
namespace Neon::set {

class DevSet;

class LaunchParametersTable
{
   public:
    LaunchParametersTable() = default;
    LaunchParametersTable(Neon::Backend const& bk);

    template <typename Lambda>
    auto forEachSeq(Lambda fun)
        -> void
    {
        for (auto dw : Neon::DataViewUtil::validOptions()) {
            fun(dw, mLaunchParameters[Neon::DataViewUtil::toInt(dw)]);
        }
    }

    auto init(Neon::Backend const& bk)
        -> void;

    auto get(Neon::DataView dw)
        -> Neon::set::LaunchParameters const&;

   private:
    int                                                                  mSetSize = 0;
    std::array<Neon::set::LaunchParameters, Neon::DataViewUtil::nConfig> mLaunchParameters;
};
}  // namespace Neon::set
