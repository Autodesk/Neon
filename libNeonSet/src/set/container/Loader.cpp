#include "type_traits"

#include "Neon/set/Containter.h"
#include "Neon/set/container/Loader.h"


namespace Neon::set {

auto Loader::
    computeMode() -> bool
{
    return m_loadingMode == Neon::set::internal::LoadingMode_e::EXTRACT_LAMBDA;
}

auto Loader::getExecution() const -> Neon::Execution
{
    return mExecution;
}
auto Loader::getSetIdx() const -> Neon::SetIdx
{
    return m_setIdx;
}

auto Loader::getDataView() const -> Neon::DataView
{
    return m_dataView;
}

}  // namespace Neon::set