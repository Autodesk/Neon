#pragma once
#include "Neon/Report.h"
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"
#include "Neon/Report.h"

namespace Cli {

enum class Type
{
    INT64,
    DOUBLE
};

struct TypeUtils
{
    static constexpr int nOptions = 2;

    static auto toString(Type type) -> std::string;
    static auto toInt(Type type) -> int;
    static auto fromString(const std::string& type) -> Type;
    static auto fromInt(int type) -> Type;
    static auto getOptions() -> std::array<Type, nOptions>;

    struct Cli
    {
        explicit Cli(std::string);
        explicit Cli(Type model);
        Cli();

        auto getOption() -> Type;
        auto getStringOptions() -> std::string;
        auto set(const std::string& opt) -> void;

        auto addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock)->void;
        auto addToReport(Neon::Report& report)->void;

       private:
        bool mSet = false;
        Type mOption;
    };
};


}  // namespace Cli