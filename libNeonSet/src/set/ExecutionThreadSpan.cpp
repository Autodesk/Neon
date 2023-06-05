#include "Neon/set/ExecutionThreadSpan.h"

namespace Neon::set::details {

auto ExecutionThreadSpanUtils::toString(ExecutionThreadSpan runtime) -> std::string
{
    switch (runtime) {
        case ExecutionThreadSpan::d1: {
            return "d1";
        }
        case ExecutionThreadSpan::d2: {
            return "d2";
        }
        case ExecutionThreadSpan::d3: {
            return "d3";
        }
        case ExecutionThreadSpan::d1b3: {
            return "d1b3";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

}  // namespace Neon::set::details