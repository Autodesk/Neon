#include <vector>
#include "Neon/core//core.h"
#include "Neon/core/types/memOptions.h"


namespace Neon {


auto memPadding_e::toString(int config) -> const char*
{
    switch (config) {
        case memPadding_e::OFF: {
            return "OFF";
        }
        case memPadding_e::ON: {
            return "ON";
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
}

auto memAlignment_e::toString(int config) -> const char*
{
    switch (config) {
        case memAlignment_e::SYSTEM: {
            return "SYSTEM";
        }
        case memAlignment_e::L1: {
            return "L1";
        }
        case memAlignment_e::L2: {
            return "L2";
        }
        case memAlignment_e::PAGE: {
            return "PAGE";
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
}

static std::vector<std::string> mem3dLayoutOrderNames{std::string("structOfArrays"), std::string("arrayOfStructs")};
static std::vector<std::string> mem3dLayoutPaddingNames{std::string("PADDING_ON"), std::string("PADDING_OFF")};

}  // namespace Neon
