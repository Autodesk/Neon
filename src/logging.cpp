#include <carbon/logging.h>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <mutex>

namespace carbon::log::detail {

auto
ensure_default_logger() -> void {
  static std::once_flag once;
  std::call_once(once, []() {
    auto logger = spdlog::stdout_color_mt("carbon");
    logger->set_level(spdlog::level::info);
    spdlog::set_default_logger(std::move(logger));
  });
}

}  // namespace carbon::log::detail

namespace carbon::log {

auto
init_default() -> void {
  detail::ensure_default_logger();
}

auto
default_logger() -> std::shared_ptr<spdlog::logger> {
  detail::ensure_default_logger();
  return spdlog::default_logger();
}

auto
set_level(spdlog::level::level_enum level) -> void {
  detail::ensure_default_logger();
  spdlog::default_logger()->set_level(level);
}

}  // namespace carbon::log
