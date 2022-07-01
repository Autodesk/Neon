#pragma once
#include "Neon/Report.h"
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"

namespace Cli {

enum class GridType
{
    eGrid,
    dGrid,
    bGrid
};

struct GridTypeUtils
{
    static constexpr int nOptions = 3;

    static auto toString(GridType option) -> std::string;
    static auto toInt(GridType option) -> int;
    static auto fromString(const std::string& option) -> GridType;
    static auto fromInt(int option) -> GridType;

    static auto getOptions() -> std::array<GridType, nOptions>;

    struct Cli
    {
        explicit Cli(std::string);
        explicit Cli(GridType model);
        Cli();

        auto getOption() -> GridType;
        auto getStringOptions() -> std::string;
        auto set(const std::string& opt) -> void;

        auto addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) -> void;
        auto addToReport(Neon::Report& report) -> void;

       private:
        bool mSet = false;
        GridType mOption;
    };
};


}  // namespace Cli