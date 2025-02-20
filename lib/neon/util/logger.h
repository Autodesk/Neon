#pragma once
#include <vector>
#include "neon_export.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
namespace neon::util {

class Logger
{
   public:
    Logger()
    {
        std::vector<spdlog::sink_ptr> sinks;
        sinks.emplace_back(
            std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
        sinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(
            "Neon.log", true));

        sinks[0]->set_pattern("%^[%T] %n: %v%$");
        sinks[1]->set_pattern("[%T] [%l] %n: %v");

        mLogger = std::make_shared<spdlog::logger>(
            "Neon", begin(sinks), end(sinks));
        spdlog::register_logger(mLogger);

        //set level to trace by default
        mLogger->set_level(spdlog::level::trace);
        mLogger->flush_on(spdlog::level::trace);
    }

    inline std::shared_ptr<spdlog::logger>& getLogger()
    {
        return mLogger;
    }

    inline void set_level(spdlog::level::level_enum level)
    {
        mLogger->set_level(level);
        mLogger->flush_on(level);
    }


   private:
    std::shared_ptr<spdlog::logger> mLogger;
};


namespace global {
NEON_EXPORT extern Logger LoggerObj;
}  // namespace globalSpace


}  // namespace Neon

#if defined( NEON_ACTIVETE_TRACING)
#define NEON_TRACE(...) ::neon::util::global::LoggerObj.getLogger()->trace(__VA_ARGS__)
#else
#define NEON_TRACE(...)
#endif

#define NEON_INFO(...) ::neon::util::global::LoggerObj.getLogger()->info(__VA_ARGS__)
#define NEON_WARNING(...)                                                                         \
    ::neon::util::global::LoggerObj.getLogger()->warn("Line {} File {}", __LINE__, __FILE__); \
    ::neon::util::global::LoggerObj.getLogger()->warn(__VA_ARGS__)
#define NEON_ERROR(...)                                                                            \
    ::neon::util::global::LoggerObj.getLogger()->error("Line {} File {}", __LINE__, __FILE__); \
    ::neon::util::global::LoggerObj.getLogger()->error(__VA_ARGS__)
#define NEON_CRITICAL(...)                                                                            \
    ::neon::util::global::LoggerObj.getLogger()->critical("Line {} File {}", __LINE__, __FILE__); \
    ::neon::util::global::LoggerObj.getLogger()->critical(__VA_ARGS__)
