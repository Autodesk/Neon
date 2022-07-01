#include "Neon/domain/internal/haloUpdateType.h"
#include "Neon/core/core.h"

namespace Neon {
namespace domain {
auto HaloUpdateMode_e::toString(e val)
{
    switch (val) {
        case STANDARD:
            return "STANDARD";
        case LATTICE:
            return "LATTICE";
        case HALOUPDATEMODE_LEN:
            return "HALOUPDATEMODE_LEN";
        default:
            NEON_THROW_UNSUPPORTED_OPTION("");
    }
}


}  // namespace grids
}  // namespace Neon