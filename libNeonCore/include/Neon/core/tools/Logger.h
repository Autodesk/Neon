#pragma once

#include <vector>
#include "libneoncore_export.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace Neon {

/**
 * @class Logger
 * @brief Wraps and configures a spdlog::logger with console and file sinks.
 *
 * The Logger sets up a colored console sink (trace level) and a file sink
 * (info level and above), with customizable patterns and automatic flushing.
 */
class Logger
{
public:
    /**
     * @brief Construct a new Logger object.
     *
     * Initializes console and file sinks, sets patterns, colors, and
     * logging levels. Registers the logger under the name "Neon".
     */
    Logger();

    /**
     * @brief Access the underlying spdlog logger.
     * @return Shared pointer to the spdlog::logger instance.
     */
    inline std::shared_ptr<spdlog::logger>& getLogger() { return mLogger; }

    /**
     * @brief Change the logging level of the logger.
     * @param level The new logging level to apply (e.g., spdlog::level::info).
     *
     * Also configures the logger to flush on the specified level.
     */
    inline void set_level(spdlog::level::level_enum level)
    {
        mLogger->set_level(level);
        mLogger->flush_on(level);
    }

private:
    std::shared_ptr<spdlog::logger> mLogger;    ///< Underlying spdlog logger.
};

namespace globalSpace {
/**
 * @brief Global Logger instance for application-wide logging.
 */
LIBNEONCORE_EXPORT extern Logger LoggerObj;
} // namespace globalSpace

} // namespace Neon

/**
 * @file Logger.hpp
 * @brief Defines logging macros for Neon namespace.
 */

#if defined(NEON_ACTIVATE_TRACING)
/**
 * @brief Trace-level logging macro.
 * @param cat Category or module name.
 * @param fmt printf-style format string.
 * @param ... Format arguments.
 *
 * Logs a trace message with category prepended.
 */
#define NEON_TRACE(cat, fmt, ...)                                     \
    ::Neon::globalSpace::LoggerObj.getLogger()                       \
        ->trace("[{}] " fmt, cat, ##__VA_ARGS__)
#else
#define NEON_TRACE(...)
#endif

/**
 * @brief Info-level logging macro.
 * @param cat Category or module name.
 * @param fmt printf-style format string.
 * @param ... Format arguments.
 *
 * Logs an info message with category prepended.
 */
#define NEON_INFO(cat, fmt, ...)                                      \
    ::Neon::globalSpace::LoggerObj.getLogger()                       \
        ->info("[{}] " fmt, cat, ##__VA_ARGS__)

/**
 * @brief Warning-level logging macro.
 * @param ... printf-style format string and arguments.
 *
 * Logs a warning, including source file and line information.
 */
#define NEON_WARNING(...)                                             \
    ::Neon::globalSpace::LoggerObj.getLogger()->warn("Line {} File {}", __LINE__, __FILE__); \
    ::Neon::globalSpace::LoggerObj.getLogger()->warn(__VA_ARGS__)

/**
 * @brief Error-level logging macro.
 * @param ... printf-style format string and arguments.
 *
 * Logs an error, including source file and line information.
 */
#define NEON_ERROR(...)                                               \
    ::Neon::globalSpace::LoggerObj.getLogger()->error("Line {} File {}", __LINE__, __FILE__); \
    ::Neon::globalSpace::LoggerObj.getLogger()->error(__VA_ARGS__)

/**
 * @brief Critical-level logging macro.
 * @param ... printf-style format string and arguments.
 *
 * Logs a critical error, including source file and line information.
 */
#define NEON_CRITICAL(...)                                            \
    ::Neon::globalSpace::LoggerObj.getLogger()->critical("Line {} File {}", __LINE__, __FILE__); \
    ::Neon::globalSpace::LoggerObj.getLogger()->critical(__VA_ARGS__)
