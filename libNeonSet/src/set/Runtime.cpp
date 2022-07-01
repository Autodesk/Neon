#include "Neon/set/Runtime.h"

namespace Neon {

auto RuntimeUtils::toString(Runtime runtime) -> std::string
{
    switch (runtime) {
        case Runtime::none: {
            return "none";
        }
        case Runtime::stream: {
            return "stream";
        }
        case Runtime::openmp: {
            return "openmp";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto RuntimeUtils::fromString(const std::string& runtime) -> Runtime
{
    std::array<Runtime, nOptions + 1> runtimes{Runtime::stream, Runtime::openmp, Runtime::none};
    for (auto a : runtimes) {
        if (toString(a) == runtime) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto RuntimeUtils::getOptions() -> std::array<Runtime, nOptions>
{
    std::array<Runtime, nOptions> opts = {Runtime::stream, Runtime::openmp};
    return opts;
}

RuntimeUtils::Cli::Cli()
{
    mSet = false;
}

RuntimeUtils::Cli::Cli(std::string s)
{
    set(s);
}

RuntimeUtils::Cli::Cli(Runtime model)
{
    mOption = model;
}

auto RuntimeUtils::Cli::getOption() -> Runtime
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "Runtime model was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return mOption;
}

auto RuntimeUtils::Cli::set(const std::string& opt)
    -> void
{
    try {
        mOption = RuntimeUtils::fromString(opt);
    } catch (...) {
        std::stringstream errorMsg;
        errorMsg << "Runtime: " << opt << " is not a valid option (valid options are {";
        auto options = RuntimeUtils::getOptions();
        int  i = 0;
        for (auto o : options) {
            if (i != 0) {
                errorMsg << ", " << RuntimeUtils::toString(o);
            }
            errorMsg << RuntimeUtils::toString(o);
            i = 1;
        }
        errorMsg << "})";
        NEON_ERROR(errorMsg.str());
    }
    mSet = true;
}

auto RuntimeUtils::Cli::getStringOptions() -> std::string
{
    std::stringstream s;
    auto              options = RuntimeUtils::getOptions();
    int               i = 0;
    for (auto o : options) {
        if (i != 0) {
            s << ", ";
        }
        s << RuntimeUtils::toString(o);
        i = 1;
    }
    std::string msg = s.str();
    return msg;
}
}  // namespace Neon