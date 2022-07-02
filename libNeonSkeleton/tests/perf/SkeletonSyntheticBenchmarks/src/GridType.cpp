#include "GridType.h"

namespace Cli {

auto GridTypeUtils::toString(GridType option) -> std::string
{
    switch (option) {
        case GridType::eGrid: {
            return "eGrid";
        }
        case GridType::dGrid: {
            return "dGrid";
        }
        case GridType::bGrid: {
            return "bGrid";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto GridTypeUtils::fromString(const std::string& option) -> GridType
{
    auto options = GridTypeUtils::getOptions();
    for (auto a : options) {
        if (GridTypeUtils::toString(a) == option) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto GridTypeUtils::getOptions() -> std::array<GridType, nOptions>
{
    std::array<GridType, nOptions> options{GridType::dGrid, GridType::eGrid, GridType::bGrid};
    return options;
}

auto GridTypeUtils::toInt(GridType option) -> int
{
    return static_cast<int>(option);
}

auto GridTypeUtils::fromInt(int option) -> GridType
{
    auto options = GridTypeUtils::getOptions();
    for (auto a : options) {
        if (GridTypeUtils::toInt(a) == option) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

GridTypeUtils::Cli::Cli(std::string str)
{
    set(str);
}

auto GridTypeUtils::Cli::getOption() -> GridType
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "GridType was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return mOption;
}

GridTypeUtils::Cli::Cli()
{
    mSet = false;
}
GridTypeUtils::Cli::Cli(GridType model)
{
    mSet = true;
    mOption = model;
}

auto GridTypeUtils::Cli::set(const std::string& opt)
    -> void
{
    try {
        mOption = GridTypeUtils::fromString(opt);
    } catch (...) {
        std::stringstream errorMsg;
        errorMsg << "GridType: " << opt << " is not a valid option (valid options are {";
        auto options = GridTypeUtils::getOptions();
        int  i = 0;
        for (auto o : options) {
            if (i != 0) {
                errorMsg << ", " << GridTypeUtils::toString(o);
            }
            errorMsg << GridTypeUtils::toString(o);
            i = 1;
        }
        errorMsg << "})";
        NEON_ERROR(errorMsg.str());
    }
    mSet = true;
}
auto GridTypeUtils::Cli::getStringOptions() -> std::string
{
    std::stringstream s;
    auto              options = GridTypeUtils::getOptions();
    int               i = 0;
    for (auto o : options) {
        if (i != 0) {
            s << ", ";
        }
        s << GridTypeUtils::toString(o);
        i = 1;
    }
    std::string msg = s.str();
    return msg;
}

auto GridTypeUtils::Cli::addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) -> void
{
    report.addMember("GridType", GridTypeUtils::toString(this->getOption()), &subBlock);
}

auto GridTypeUtils::Cli::addToReport(Neon::Report& report) -> void
{
    report.addMember("GridType", GridTypeUtils::toString(this->getOption()));
}
}  // namespace Cli