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
    template <int method>
    static auto getNumberOfPFields() -> int
    {
        Method m = formInt(method);
        switch (m) {
            case Method::pull:
                return 2;
            case Method::push:
                return 2;
            case Method::aa:
                return 1    ;
        }
        std::stringstream msg;
        msg << "The following LBM method is not recognized" << method << std::endl;
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
}