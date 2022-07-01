#include "Neon/skeleton/Executor.h"

namespace Neon::skeleton {

auto ExecutorUtils::toString(Executor option) -> std::string
{
    switch (option) {
        case Executor::ompAtNodeLevel: {
            return "ompAtNodeLevel";
        }
        case Executor::ompAtGraphLevel: {
            return "ompAtGraphLevel";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto ExecutorUtils::fromString(const std::string& forkJoin) -> Executor
{
    auto options = ExecutorUtils::getOptions();
    for (auto a : options) {
        if (ExecutorUtils::toString(a) == forkJoin) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto ExecutorUtils::getOptions() -> std::array<Executor, nOptions>
{
    std::array<Executor, nOptions> options{Executor::ompAtGraphLevel, Executor::ompAtNodeLevel};
    return options;
}

auto ExecutorUtils::toInt(Executor forkJoin) -> int
{
    return static_cast<int>(forkJoin);
}

auto ExecutorUtils::fromInt(int forkJoin) -> Executor
{
    auto options = ExecutorUtils::getOptions();
    for (auto a : options) {
        if (ExecutorUtils::toInt(a) == forkJoin) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

ExecutorUtils::Cli::Cli(std::string str)
{
    set(str);
}

auto ExecutorUtils::Cli::getOption() -> Executor
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "Executor model was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return mOption;
}

ExecutorUtils::Cli::Cli()
{
    mSet = false;
}
ExecutorUtils::Cli::Cli(Executor model)
{
    mSet = true;
    mOption = model;
}

auto ExecutorUtils::Cli::set(const std::string& opt)
    -> void
{
    try {
        mOption = ExecutorUtils::fromString(opt);
    } catch (...) {
        std::stringstream errorMsg;
        errorMsg << "Executor: " << opt << " is not a valid option (valid options are {";
        auto options = ExecutorUtils::getOptions();
        int  i = 0;
        for (auto o : options) {
            if (i != 0) {
                errorMsg << ", " << ExecutorUtils::toString(o);
            }
            errorMsg << ExecutorUtils::toString(o);
            i = 1;
        }
        errorMsg << "})";
        NEON_ERROR(errorMsg.str());
    }
    mSet = true;
}
auto ExecutorUtils::Cli::getStringOptions() -> std::string
{
    std::stringstream s;
    auto              options = ExecutorUtils::getOptions();
    int               i = 0;
    for (auto o : options) {
        if (i != 0) {
            s << ", ";
        }
        s << ExecutorUtils::toString(o);
        i = 1;
    }
    std::string msg = s.str();
    return msg;
}

auto ExecutorUtils::Cli::addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) -> void
{
    report.addMember("Executor", ExecutorUtils::toString(this->getOption()), &subBlock);
}

auto ExecutorUtils::Cli::addToReport(Neon::Report& report) -> void
{
    report.addMember("Executor", ExecutorUtils::toString(this->getOption()));
}
}  // namespace Neon::skeleton