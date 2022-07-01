#pragma once
#include "Neon/Report.h"
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"

namespace Neon::skeleton {

enum class Executor
{
    ompAtNodeLevel,
    ompAtGraphLevel
};

struct ExecutorUtils
{
    static constexpr int nOptions = 2;

    static auto toString(Executor forkJoin) -> std::string;
    static auto toInt(Executor forkJoin) -> int;
    static auto fromString(const std::string& forkJoin) -> Executor;
    static auto fromInt(int forkJoin) -> Executor;

    static auto getOptions() -> std::array<Executor, nOptions>;

    struct Cli
    {
        explicit Cli(std::string);
        explicit Cli(Executor model);
        Cli();

        auto getOption() -> Executor;
        auto getStringOptions() -> std::string;
        auto set(const std::string& opt) -> void;

        auto addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock)->void;
        auto addToReport(Neon::Report& report)->void;

       private:
        bool     mSet = false;
        Executor mOption;
    };
};


}  // namespace Neon::skeleton