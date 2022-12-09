#include "type_traits"

#include "Neon/set/Containter.h"

namespace Neon::set {

auto Loader::
    computeMode() -> bool
{
    return m_loadingMode == Neon::set::internal::LoadingMode_e::EXTRACT_LAMBDA;
}

}  // namespace Neon::set