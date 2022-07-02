#pragma once
#include "Neon/Report.h"
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"

namespace Cli {

enum class Apps
{
    map,
    mapMapMap,
};

struct AppsUtils
{
    static constexpr int nOptions = 2;

    static auto toString(Apps option) -> std::string;
    static auto toInt(Apps option) -> int;
    static auto fromString(const std::string& option) -> Apps;
    static auto fromInt(int option) -> Apps;

    static auto getOptions() -> std::array<Apps, nOptions>;

    struct Cli
    {
        explicit Cli(std::string);
        explicit Cli(Apps model);
        Cli();

        auto getOption() -> Apps;
        auto getStringOptions() -> std::string;
        auto set(const std::string& opt) -> void;

        auto addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) -> void;
        auto addToReport(Neon::Report& report) -> void;

       private:
        bool mSet = false;
        Apps mOption;
    };
};


}  // namespace Cli