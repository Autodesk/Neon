#pragma once
#include "Neon/core/types/Exceptions.h"


/**
 * Macro to be used as place holder for features whose development is incomplete.
 */
#define NEON_DEV_UNDER_CONSTRUCTION(componentName)                            \
    {                                                                         \
        Neon::NeonException exp(componentName);                             \
        exp << "The requested feature is still under construction, sorry!!!"; \
        (exp).setEnvironmentInfo(__FILE__, __LINE__, NEON_FUNCTION_NAME());   \
        exp.logThrow();                                                       \
        throw exp;                                                            \
    }
