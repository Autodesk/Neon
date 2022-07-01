#include "CLiCardinality.h"

namespace Cli {

auto CardinalityUtils::toString(Cardinality option) -> std::string
{
    switch (option) {
        case Cardinality::one: {
            return "one";
        }
        case Cardinality::three: {
            return "three";
        }
        case Cardinality::nineteen: {
            return "nineteen";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto CardinalityUtils::fromString(const std::string& option) -> Cardinality
{
    auto options = CardinalityUtils::getOptions();
    for (auto a : options) {
        if (CardinalityUtils::toString(a) == option) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto CardinalityUtils::getOptions() -> std::array<Cardinality, nOptions>
{
    std::array<Cardinality, nOptions> options{Cardinality::one, Cardinality::three, Cardinality::nineteen};
    return options;
}

auto CardinalityUtils::toInt(Cardinality option) -> int
{
    return static_cast<int>(option);
}

auto CardinalityUtils::fromInt(int option) -> Cardinality
{
    auto options = CardinalityUtils::getOptions();
    for (auto a : options) {
        if (CardinalityUtils::toInt(a) == option) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

CardinalityUtils::Cli::Cli(std::string str)
{
    set(str);
}

auto CardinalityUtils::Cli::getOption() -> Cardinality
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "Cardinality was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return mOption;
}

CardinalityUtils::Cli::Cli()
{
    mSet = false;
}
CardinalityUtils::Cli::Cli(Cardinality model)
{
    mSet = true;
    mOption = model;
}

auto CardinalityUtils::Cli::set(const std::string& opt)
    -> void
{
    try {
        mOption = CardinalityUtils::fromString(opt);
    } catch (...) {
        std::stringstream errorMsg;
        errorMsg << "Cardinality: " << opt << " is not a valid option (valid options are {";
        auto options = CardinalityUtils::getOptions();
        int  i = 0;
        for (auto o : options) {
            if (i != 0) {
                errorMsg << ", " << CardinalityUtils::toString(o);
            }
            errorMsg << CardinalityUtils::toString(o);
            i = 1;
        }
        errorMsg << "})";
        NEON_ERROR(errorMsg.str());
    }
    mSet = true;
}
auto CardinalityUtils::Cli::getStringOptions() -> std::string
{
    std::stringstream s;
    auto              options = CardinalityUtils::getOptions();
    int               i = 0;
    for (auto o : options) {
        if (i != 0) {
            s << ", ";
        }
        s << CardinalityUtils::toString(o);
        i = 1;
    }
    std::string msg = s.str();
    return msg;
}

auto CardinalityUtils::Cli::addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) -> void
{
    report.addMember("Cardinality", CardinalityUtils::toString(this->getOption()), &subBlock);
}

auto CardinalityUtils::Cli::addToReport(Neon::Report& report) -> void
{
    report.addMember("Cardinality", CardinalityUtils::toString(this->getOption()));
}
}  // namespace Cli