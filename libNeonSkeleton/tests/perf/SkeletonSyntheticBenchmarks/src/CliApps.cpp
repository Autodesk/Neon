#include "CLiApps.h"

namespace Cli {

auto AppsUtils::toString(Apps option) -> std::string
{
    switch (option) {
        case Apps::map: {
            return "map";
        }
        case Apps::mapMapMap: {
            return "mapMapMap";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto AppsUtils::fromString(const std::string& forkJoin) -> Apps
{
    auto options = AppsUtils::getOptions();
    for (auto a : options) {
        if (AppsUtils::toString(a) == forkJoin) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto AppsUtils::getOptions() -> std::array<Apps, nOptions>
{
    std::array<Apps, nOptions> options{Apps::map, Apps::mapMapMap};
    return options;
}

auto AppsUtils::toInt(Apps forkJoin) -> int
{
    return static_cast<int>(forkJoin);
}

auto AppsUtils::fromInt(int forkJoin) -> Apps
{
    auto options = AppsUtils::getOptions();
    for (auto a : options) {
        if (AppsUtils::toInt(a) == forkJoin) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

AppsUtils::Cli::Cli(std::string str)
{
    set(str);
}

auto AppsUtils::Cli::getOption() -> Apps
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "App was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return mOption;
}

AppsUtils::Cli::Cli()
{
    mSet = false;
}
AppsUtils::Cli::Cli(Apps model)
{
    mSet = true;
    mOption = model;
}

auto AppsUtils::Cli::set(const std::string& opt)
    -> void
{
    try {
        mOption = AppsUtils::fromString(opt);
    } catch (...) {
        std::stringstream errorMsg;
        errorMsg << "Apps: " << opt << " is not a valid option (valid options are {";
        auto options = AppsUtils::getOptions();
        int  i = 0;
        for (auto o : options) {
            if (i != 0) {
                errorMsg << ", " << AppsUtils::toString(o);
            }
            errorMsg << AppsUtils::toString(o);
            i = 1;
        }
        errorMsg << "})";
        NEON_ERROR(errorMsg.str());
    }
    mSet = true;
}
auto AppsUtils::Cli::getStringOptions() -> std::string
{
    std::stringstream s;
    auto              options = AppsUtils::getOptions();
    int               i = 0;
    for (auto o : options) {
        if (i != 0) {
            s << ", ";
        }
        s << AppsUtils::toString(o);
        i = 1;
    }
    std::string msg = s.str();
    return msg;
}

auto AppsUtils::Cli::addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) -> void
{
    report.addMember("Executor", AppsUtils::toString(this->getOption()), &subBlock);
}

auto AppsUtils::Cli::addToReport(Neon::Report& report) -> void
{
    report.addMember("Executor", AppsUtils::toString(this->getOption()));
}
}  // namespace Cli