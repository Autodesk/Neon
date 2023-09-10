#include "Neon/set/TransferMode.h"

namespace Neon::set {

auto TransferModeUtils::toString(TransferMode occ) -> std::string
{
    switch (occ) {
        case TransferMode::get: {
            return "get";
        }
        case TransferMode::put: {
            return "put";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto TransferModeUtils::fromString(const std::string& occ) -> TransferMode
{
    std::array<TransferMode, nOptions> opts{TransferMode::get, TransferMode::put};
    for (auto a : opts) {
        if (toString(a) == occ) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto TransferModeUtils::getOptions() -> std::array<TransferMode, nOptions>
{
    std::array<TransferMode, nOptions> opts = {TransferMode::get, TransferMode::put};
    return opts;
}

TransferModeUtils::Cli::Cli()
{
    mSet = false;
}

TransferModeUtils::Cli::Cli(std::string s)
{
    set(s);
}

TransferModeUtils::Cli::Cli(TransferMode model)
{
    mOption = model;
}

auto TransferModeUtils::Cli::getOption() const -> TransferMode
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "TransferMode was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return mOption;
}

auto TransferModeUtils::Cli::set(const std::string& opt)
    -> void
{
    try {
        mOption = TransferModeUtils::fromString(opt);
    } catch (...) {
        std::stringstream errorMsg;
        errorMsg << "Transfer: " << opt << " is not a valid option (valid options are {";
        auto options = TransferModeUtils::getOptions();
        int  i = 0;
        for (auto o : options) {
            if (i != 0) {
                errorMsg << ", " << TransferModeUtils::toString(o);
            }
            errorMsg << TransferModeUtils::toString(o);
            i = 1;
        }
        errorMsg << "})";
        NEON_ERROR(errorMsg.str());
    }
    mSet = true;
}

auto TransferModeUtils::Cli::getStringOptions() const -> std::string
{
    std::stringstream s;
    auto              options = TransferModeUtils::getOptions();
    int               i = 0;
    for (auto o : options) {
        if (i != 0) {
            s << ", ";
        }
        s << TransferModeUtils::toString(o);
        i = 1;
    }
    std::string msg = s.str();
    return msg;
}

auto TransferModeUtils::Cli::getStringOption() const -> std::string
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "TransferMode was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return TransferModeUtils::toString(mOption);
}

auto TransferModeUtils::Cli::getDoc() const -> std::string
{
    std::stringstream s;
    s << getStringOptions();
    s << " default: " << getStringOptions();
    return s.str();
}

auto TransferModeUtils::Cli::addToReport(Neon::Report& report) const -> void
{
    report.addMember("TransferMode", TransferModeUtils::toString(this->getOption()));
}

auto TransferModeUtils::Cli::addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) const -> void
{
    report.addMember("TransferMode", TransferModeUtils::toString(this->getOption()), &subBlock);
}
}  // namespace Neon::set
