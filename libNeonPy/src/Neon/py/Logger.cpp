#include "Neon/core/tools/Logger.h"

extern "C" {

/**
 * @brief Enable or disable INFO level logging at runtime.
 * @param enabled 1 to enable, 0 to disable
 */
auto neon_set_info_enabled(int enabled) -> void
{
    Neon::globalSpace::LoggerObj.setInfoEnabled(enabled != 0);
}

/**
 * @brief Check if INFO level logging is enabled.
 * @return 1 if enabled, 0 if disabled
 */
auto neon_is_info_enabled() -> int
{
    return Neon::globalSpace::LoggerObj.isInfoEnabled() ? 1 : 0;
}

}  // extern "C"
