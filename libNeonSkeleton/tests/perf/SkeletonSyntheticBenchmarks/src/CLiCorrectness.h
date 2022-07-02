#pragma once
#include "Neon/Report.h"
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"

namespace Cli {

enum class Correctnesss
{
    on,
    off,
};

struct CorrectnesssUtils
{
    static constexpr int nOptions = 2;

    static auto toString(Correctnesss option) -> std::string;
    static auto toInt(Correctnesss option) -> int;
    static auto fromString(const std::string& option) -> Correctnesss;
    static auto fromInt(int option) -> Correctnesss;

    static auto getOptions() -> std::array<Correctnesss, nOptions>;

    struct Cli
    {
        explicit Cli(std::string);
        explicit Cli(Correctnesss model);
        Cli();

        auto getOption() -> Correctnesss;
        auto getStringOptions() -> std::string;
        auto set(const std::string& opt) -> void;

        auto addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) -> void;
        auto addToReport(Neon::Report& report) -> void;

       private:
        bool         mSet = false;
        Correctnesss mOption;
    };
};


}  // namespace Cli