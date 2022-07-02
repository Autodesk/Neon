#include "CLiType.h"
#include "Neon/Report.h"

namespace Cli {

auto TypeUtils::toString(Type option) -> std::string
{
    switch (option) {
        case Type::INT64: {
            return "INT64";
        }
        case Type::DOUBLE: {
            return "DOUBLE";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto TypeUtils::fromString(const std::string& forkJoin) -> Type
{
    auto options = TypeUtils::getOptions();
    for (auto a : options) {
        if (TypeUtils::toString(a) == forkJoin) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto TypeUtils::getOptions() -> std::array<Type, nOptions>
{
    std::array<Type, nOptions> options{Type::INT64, Type::DOUBLE};
    return options;
}

auto TypeUtils::toInt(Type forkJoin) -> int
{
    return static_cast<int>(forkJoin);
}

auto TypeUtils::fromInt(int forkJoin) -> Type
{
    auto options = TypeUtils::getOptions();
    for (auto a : options) {
        if (TypeUtils::toInt(a) == forkJoin) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

TypeUtils::Cli::Cli(std::string str)
{
    set(str);
}

auto TypeUtils::Cli::getOption() -> Type
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "Option was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return mOption;
}

TypeUtils::Cli::Cli()
{
    mSet = false;
}
TypeUtils::Cli::Cli(Type model)
{
    mSet = true;
    mOption = model;
}

auto TypeUtils::Cli::set(const std::string& opt)
    -> void
{
    try {
        mOption = TypeUtils::fromString(opt);
    } catch (...) {
        std::stringstream errorMsg;
        errorMsg << "ForkJoin: " << opt << " is not a valid option (valid options are {";
        auto options = TypeUtils::getOptions();
        int  i = 0;
        for (auto o : options) {
            if (i != 0) {
                errorMsg << ", " << TypeUtils::toString(o);
            }
            errorMsg << TypeUtils::toString(o);
            i = 1;
        }
        errorMsg << "})";
        NEON_ERROR(errorMsg.str());
    }
    mSet = true;
}
auto TypeUtils::Cli::getStringOptions() -> std::string
{
    std::stringstream s;
    auto              options = TypeUtils::getOptions();
    int               i = 0;
    for (auto o : options) {
        if (i != 0) {
            s << ", ";
        }
        s << TypeUtils::toString(o);
        i = 1;
    }
    std::string msg = s.str();
    return msg;
}

auto TypeUtils::Cli::addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) -> void
{
    report.addMember("Type", TypeUtils::toString(this->getOption()), &subBlock);
}

auto TypeUtils::Cli::addToReport(Neon::Report& report) -> void
{
    report.addMember("Type", TypeUtils::toString(this->getOption()));
}
}  // namespace Cli