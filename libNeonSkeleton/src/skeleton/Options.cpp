#include "Neon/skeleton/Options.h"

namespace Neon {
namespace skeleton {

Options::Options(Occ                       occEOpt,
                 Neon::set::TransferMode trensferModeOpt)
{
    mOcc = occEOpt;
    mTransferMode = trensferModeOpt;
}

void Options::reportStore(Neon::Report& report)
{

    auto subdoc = report.getSubdoc();
    report.addMember("OCC", OccUtils::toString(mOcc), &subdoc);
    report.addMember("TransferMode", Neon::set::TransferModeUtils::toString(mTransferMode), &subdoc);
    report.addSubdoc("SkeletonOptions", subdoc);
}

auto Options::occ() const -> Occ
{
    return mOcc;
}

auto Options::transferMode() const -> Neon::set::TransferMode
{
    return mTransferMode;
}
auto Options::executor() const -> Neon::skeleton::Executor
{
    return mExecutor;
}

}  // namespace skeleton
}  // namespace Neon