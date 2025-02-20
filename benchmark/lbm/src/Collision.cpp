#include "Collision.h"


auto CollisionUtils::toString(Collision occ) -> std::string
{
    switch (occ) {
        case Collision::bgk: {
            return "bgk";
        }
        case Collision::kbc: {
            return "kbc";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto CollisionUtils::fromString(const std::string& occ) -> Collision
{
    std::array<Collision, nOptions> occs{Collision::bgk, Collision::kbc};
    for (auto a : occs) {
        if (toString(a) == occ) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto CollisionUtils::getOptions() -> std::array<Collision, nOptions>
{
    std::array<Collision, nOptions> opts = {Collision::bgk, Collision::kbc};
    return opts;
}

CollisionUtils::Cli::Cli()
{
    mSet = false;
}

CollisionUtils::Cli::Cli(std::string s)
{
    set(s);
}

CollisionUtils::Cli::Cli(Collision model)
{
    mOption = model;
    mSet = true;
}

auto CollisionUtils::Cli::getOption() const -> Collision
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "Collision model was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return mOption;
}

auto CollisionUtils::Cli::getOptionStr() const -> std::string
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "Collision model was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return CollisionUtils::toString(mOption);
}

auto CollisionUtils::Cli::set(const std::string& opt)
    -> void
{
    try {
        mOption = CollisionUtils::fromString(opt);
    } catch (...) {
        std::stringstream errorMsg;
        errorMsg << "Collision: " << opt << " is not a valid option (valid options are {";
        auto options = CollisionUtils::getOptions();
        int  i = 0;
        for (auto o : options) {
            if (i != 0) {
                errorMsg << ", " << CollisionUtils::toString(o);
            }
            errorMsg << CollisionUtils::toString(o);
            i = 1;
        }
        errorMsg << "})";
        NEON_ERROR(errorMsg.str());
    }
    mSet = true;
}

auto CollisionUtils::Cli::getAllOptionsStr() const -> std::string
{
    std::stringstream s;
    auto              options = CollisionUtils::getOptions();
    int               i = 0;
    for (auto o : options) {
        if (i != 0) {
            s << ", ";
        }
        s << CollisionUtils::toString(o);
        i = 1;
    }
    std::string msg = s.str();
    return msg;
}


auto CollisionUtils::Cli::getDoc() const -> std::string
{
    std::stringstream s;
    s << getAllOptionsStr();
    s << " default: " << CollisionUtils::toString(getOption());
    return s.str();
}

auto CollisionUtils::Cli::addToReport(Neon::Report& report) const -> void
{
    report.addMember("Collision", CollisionUtils::toString(this->getOption()));
}

auto CollisionUtils::Cli::addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) const -> void
{
    report.addMember("Collision", CollisionUtils::toString(this->getOption()), &subBlock);
}

