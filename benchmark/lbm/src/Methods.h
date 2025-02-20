#pragma once
#include "Neon/core/core.h"

namespace lbm {
enum class Method
{
    push = 0,
    pull = 1,
    aa = 2
};

struct MethodUtils
{
    template <lbm::Method method>
    static auto getNumberOfPFields() -> int
    {
        switch (method) {
            case Method::pull:
                return 2;
            case Method::push:
                return 2;
            case Method::aa:
                return 1;
        }
        std::stringstream msg;
        msg << "The following LBM method is not recognized" << lbm::MethodUtils::toString(method) << std::endl;
        NEON_THROW_UNSUPPORTED_OPERATION(msg.str());
    }

    static auto toString(lbm::Method method) -> std::string
    {
        switch (method) {
            case Method::pull:
                return "pull";
            case Method::push:
                return "push";
            case Method::aa:
                return "aa";
        }
        std::stringstream msg;
        msg << "The following LBM method is not recognized" << lbm::MethodUtils::toString(method) << std::endl;
        NEON_THROW_UNSUPPORTED_OPERATION(msg.str());
    }

    static auto formInt(int method) -> Method
    {
        if (method == int(Method::pull))
            return Method::pull;
        if (method == int(Method::push))
            return Method::push;
        if (method == int(Method::aa))
            return Method::aa;

        std::stringstream msg;
        msg << "The following LBM method is not recognized" << method << std::endl;
        NEON_THROW_UNSUPPORTED_OPERATION(msg.str());
    }
};
}  // namespace lbm