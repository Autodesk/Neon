#pragma once

#include <carbon/export.h>

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

namespace carbon::log::detail {

/** Thread-safe, idempotent: ensure spdlog has a default "carbon" console logger. */
CARBON_API auto
ensure_default_logger() -> void;

}  // namespace carbon::log::detail

namespace carbon::log {

/** Register the default color console logger (safe to call multiple times). */
CARBON_API auto
init_default() -> void;

/** Returns the default spdlog logger (creates the "carbon" console sink on first use). */
CARBON_API auto
default_logger() -> std::shared_ptr<spdlog::logger>;

/** Adjust level on the default logger (initializes it if needed). */
CARBON_API auto
set_level(spdlog::level::level_enum level) -> void;

/**
 * Log a formatted message at error level on the default @c carbon logger, then throw an
 * exception of type @p E constructed from the same message. Centralises the recurring
 * "log then throw" pattern:
 *
 *   carbon::log::throw_with_log<std::invalid_argument>("expected {} got {}", a, b);
 *
 * is shorthand for
 *
 *   CARBON_LOG_ERROR("expected {} got {}", a, b);
 *   throw std::invalid_argument(fmt::format("expected {} got {}", a, b));
 *
 * If @p E is not constructible from @c std::string (e.g. @c std::bad_alloc), the message is
 * still logged and a default-constructed @p E is thrown.
 */
template <class E,
          class... Args>
[[noreturn]] inline auto
throw_with_log(fmt::format_string<Args...> fmt_str,
               Args&&... args) -> void {
  std::string msg = fmt::format(fmt_str, std::forward<Args>(args)...);
  ::carbon::log::detail::ensure_default_logger();
  ::carbon::log::default_logger()->error("{}", msg);
  if constexpr (std::is_constructible_v<E, std::string>) {
    throw E(std::move(msg));
  } else {
    throw E();
  }
}

}  // namespace carbon::log

/* Route through carbon::log::default_logger() (defined in logging.cpp) so host code and
 * libcarbon.so share one spdlog registry. Using SPDLOG_* here inlined default_logger_raw()
 * from headers in each TU, which duplicated the registry when spdlog is linked into the shared
 * library — logs from the executable then missed the "[carbon]" name and init_default(). */

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE
#define CARBON_LOG_TRACE(...)                                                                     \
  do {                                                                                            \
    ::carbon::log::detail::ensure_default_logger();                                               \
    ::carbon::log::default_logger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, \
                                         spdlog::level::trace, __VA_ARGS__);                      \
  } while (0)
#else
#define CARBON_LOG_TRACE(...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
#define CARBON_LOG_DEBUG(...)                                                                     \
  do {                                                                                            \
    ::carbon::log::detail::ensure_default_logger();                                               \
    ::carbon::log::default_logger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, \
                                         spdlog::level::debug, __VA_ARGS__);                      \
  } while (0)
#else
#define CARBON_LOG_DEBUG(...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_INFO
#define CARBON_LOG_INFO(...)                                                                      \
  do {                                                                                            \
    ::carbon::log::detail::ensure_default_logger();                                               \
    ::carbon::log::default_logger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, \
                                         spdlog::level::info, __VA_ARGS__);                       \
  } while (0)
#else
#define CARBON_LOG_INFO(...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_WARN
#define CARBON_LOG_WARN(...)                                                                      \
  do {                                                                                            \
    ::carbon::log::detail::ensure_default_logger();                                               \
    ::carbon::log::default_logger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, \
                                         spdlog::level::warn, __VA_ARGS__);                       \
  } while (0)
#else
#define CARBON_LOG_WARN(...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_ERROR
#define CARBON_LOG_ERROR(...)                                                                     \
  do {                                                                                            \
    ::carbon::log::detail::ensure_default_logger();                                               \
    ::carbon::log::default_logger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, \
                                         spdlog::level::err, __VA_ARGS__);                        \
  } while (0)
#else
#define CARBON_LOG_ERROR(...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_CRITICAL
#define CARBON_LOG_CRITICAL(...)                                                                  \
  do {                                                                                            \
    ::carbon::log::detail::ensure_default_logger();                                               \
    ::carbon::log::default_logger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, \
                                         spdlog::level::critical, __VA_ARGS__);                   \
  } while (0)
#else
#define CARBON_LOG_CRITICAL(...) (void)0
#endif
