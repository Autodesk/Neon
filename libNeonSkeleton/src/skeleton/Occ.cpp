#include "Neon/skeleton/Occ.h"

namespace Neon::skeleton {

auto OccUtils::toString(Occ occ) -> std::string
{
    switch (occ) {
        case Occ::standard: {
            return "standard";
        }
        case Occ::extended: {
            return "extended";
        }
        case Occ::twoWayExtended: {
            return "twoWayExtended";
        }
        case Occ::none: {
            return "none";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto OccUtils::fromString(const std::string& occ) -> Occ
{
    std::array<Occ, nOptions> occs{Occ::standard, Occ::extended, Occ::twoWayExtended, Occ::none};
    for (auto a : occs) {
        if (toString(a) == occ) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto OccUtils::getOptions() -> std::array<Occ, nOptions>
{
    std::array<Occ, nOptions> opts = {Occ::standard, Occ::extended, Occ::twoWayExtended, Occ::none};
    return opts;
}

OccUtils::Cli::Cli()
{
    mSet = false;
}

OccUtils::Cli::Cli(std::string s)
{
    set(s);
}

OccUtils::Cli::Cli(Occ model)
{
    mOption = model;
}

auto OccUtils::Cli::getOption() const -> Occ
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "Occ model was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return mOption;
}

auto OccUtils::Cli::set(const std::string& opt)
    -> void
{
    try {
        mOption = OccUtils::fromString(opt);
    } catch (...) {
        std::stringstream errorMsg;
        errorMsg << "Occ: " << opt << " is not a valid option (valid options are {";
        auto options = OccUtils::getOptions();
        int  i = 0;
        for (auto o : options) {
            if (i != 0) {
                errorMsg << ", " << OccUtils::toString(o);
            }
            errorMsg << OccUtils::toString(o);
            i = 1;
        }
        errorMsg << "})";
        NEON_ERROR(errorMsg.str());
    }
    mSet = true;
}

auto OccUtils::Cli::getStringOptions() const -> std::string
{
    std::stringstream s;
    auto              options = OccUtils::getOptions();
    int               i = 0;
    for (auto o : options) {
        if (i != 0) {
            s << ", ";
        }
        s << OccUtils::toString(o);
        i = 1;
    }
    std::string msg = s.str();
    return msg;
}

auto OccUtils::Cli::getDoc() const -> std::string
{
    std::stringstream s;
    s << getStringOptions();
    s << " default: " << getStringOptions();
    return s.str();
}

auto OccUtils::Cli::addToReport(Neon::Report& report) const -> void
{
    report.addMember("Occ", OccUtils::toString(this->getOption()));
}

auto OccUtils::Cli::addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) const -> void
{
    report.addMember("Occ", OccUtils::toString(this->getOption()), &subBlock);
}

}  // namespace Neon::skeleton