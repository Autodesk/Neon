#include "Neon/core/tools/Logger.h"
#include "libneoncore_export.h"

namespace Neon {

Logger::Logger()
{
    // Console sink with colored output
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    // Pattern includes timestamp, left-aligned level, logger name, and message
    console_sink->set_pattern("%^[%T] [%-5l] [%n] %v%$");
    // Custom colors per level
    console_sink->set_color(spdlog::level::trace, console_sink->cyan);
    console_sink->set_color(spdlog::level::info, console_sink->green);
    console_sink->set_color(spdlog::level::warn, console_sink->yellow);
    console_sink->set_color(spdlog::level::err, console_sink->red);

    // File sink (only log info and above) with aligned level
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("Neon.log", true);
    file_sink->set_pattern("[%T] [%-5l] [%n] %v");
    file_sink->set_level(spdlog::level::info);

    // Combine sinks into the logger
    std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
    mLogger = std::make_shared<spdlog::logger>("Neon", sinks.begin(), sinks.end());
    spdlog::register_logger(mLogger);

    // Enable all levels on the logger and flush on trace
    mLogger->set_level(spdlog::level::trace);
    mLogger->flush_on(spdlog::level::trace);
}

namespace globalSpace {
LIBNEONCORE_EXPORT Logger LoggerObj;  // Definition of the extern LoggerObj
}

}  // namespace Neon
