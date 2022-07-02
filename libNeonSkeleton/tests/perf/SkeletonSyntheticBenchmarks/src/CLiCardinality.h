#pragma once
#include "Neon/Report.h"
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"

namespace Cli {

enum class Cardinality
{
    one,
    three,
    nineteen
};

struct CardinalityUtils
{
    static constexpr int nOptions = 3;

    static auto toString(Cardinality option) -> std::string;
    static auto toInt(Cardinality option) -> int;
    static auto fromString(const std::string& option) -> Cardinality;
    static auto fromInt(int option) -> Cardinality;

    static auto getOptions() -> std::array<Cardinality, nOptions>;

    struct Cli
    {
        explicit Cli(std::string);
        explicit Cli(Cardinality model);
        Cli();

        auto getOption() -> Cardinality;
        auto getStringOptions() -> std::string;
        auto set(const std::string& opt) -> void;

        auto addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) -> void;
        auto addToReport(Neon::Report& report) -> void;

       private:
        bool mSet = false;
        Cardinality mOption;
    };
};


}  // namespace Cli