#include "CLiCorrectness.h"

namespace Cli {

auto CorrectnesssUtils::toString(Correctnesss option) -> std::string
{
    switch (option) {
        case Correctnesss::on: {
            return "on";
        }
        case Correctnesss::off: {
            return "off";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto CorrectnesssUtils::fromString(const std::string& forkJoin) -> Correctnesss
{
    auto options = CorrectnesssUtils::getOptions();
    for (auto a : options) {
        if (CorrectnesssUtils::toString(a) == forkJoin) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto CorrectnesssUtils::getOptions() -> std::array<Correctnesss, nOptions>
{
    std::array<Correctnesss, nOptions> options{Correctnesss::on, Correctnesss::off};
    return options;
}

auto CorrectnesssUtils::toInt(Correctnesss forkJoin) -> int
{
    return static_cast<int>(forkJoin);
}

auto CorrectnesssUtils::fromInt(int forkJoin) -> Correctnesss
{
    auto options = CorrectnesssUtils::getOptions();
    for (auto a : options) {
        if (CorrectnesssUtils::toInt(a) == forkJoin) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

CorrectnesssUtils::Cli::Cli(std::string str)
{
    set(str);
}

auto CorrectnesssUtils::Cli::getOption() -> Correctnesss
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "Correctness was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return mOption;
}

CorrectnesssUtils::Cli::Cli()
{
    mSet = false;
}
CorrectnesssUtils::Cli::Cli(Correctnesss model)
{
    mSet = true;
    mOption = model;
}

auto CorrectnesssUtils::Cli::set(const std::string& opt)
    -> void
{
    try {
        mOption = CorrectnesssUtils::fromString(opt);
    } catch (...) {
        std::stringstream errorMsg;
        errorMsg << "Correctnesss: " << opt << " is not a valid option (valid options are {";
        auto options = CorrectnesssUtils::getOptions();
        int  i = 0;
        for (auto o : options) {
            if (i != 0) {
                errorMsg << ", " << CorrectnesssUtils::toString(o);
            }
            errorMsg << CorrectnesssUtils::toString(o);
            i = 1;
        }
        errorMsg << "})";
        NEON_ERROR(errorMsg.str());
    }
    mSet = true;
}
auto CorrectnesssUtils::Cli::getStringOptions() -> std::string
{
    std::stringstream s;
    auto              options = CorrectnesssUtils::getOptions();
    int               i = 0;
    for (auto o : options) {
        if (i != 0) {
            s << ", ";
        }
        s << CorrectnesssUtils::toString(o);
        i = 1;
    }
    std::string msg = s.str();
    return msg;
}

auto CorrectnesssUtils::Cli::addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) -> void
{
    report.addMember("Executor", CorrectnesssUtils::toString(this->getOption()), &subBlock);
}

auto CorrectnesssUtils::Cli::addToReport(Neon::Report& report) -> void
{
    report.addMember("Executor", CorrectnesssUtils::toString(this->getOption()));
}
}  // namespace Cli