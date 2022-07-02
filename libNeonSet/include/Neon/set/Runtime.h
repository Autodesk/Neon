#pragma once

#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>

#include "Neon/Report.h"
#include "Neon/core/core.h"
#include "Neon/set/MemoryOptions.h"
//#include "Neon/core/types/mode.h"
//#include "Neon/core/types/devType.h"

namespace Neon {

enum struct Runtime
{
    none = 0,
    system = 0,
    stream = 1,
    openmp = 2
};


struct RuntimeUtils
{
    static constexpr int nOptions = 2;

    static auto toString(Runtime runtime) -> std::string;
    static auto fromString(const std::string& runtime) -> Runtime;
    static auto getOptions() -> std::array<Runtime, nOptions>;

    struct Cli
    {
        explicit Cli(std::string);
        explicit Cli(Runtime model);
        Cli();

        auto getOption() -> Runtime;
        auto set(const std::string& opt) -> void;
        auto getStringOptions() -> std::string;

       private:
        bool    mSet = false;
        Runtime mOption;
    };
};

}  // namespace Neon::set
