#include "Neon/domain/tools/SpaceCurves.h"
#include "Neon/core/types/Exceptions.h"

namespace Neon::domain::tool::spaceCurves {

auto EncoderTypeUtil::getOptions() -> std::array<EncoderType, EncoderTypeUtil::nConfig>
{
    std::array<EncoderType, EncoderTypeUtil::nConfig> options = {EncoderType::sweep,
                                                                 EncoderType::morton,
                                                                 EncoderType::hilbert};
    return options;
}

auto EncoderTypeUtil::toString(EncoderType e) -> std::string
{
    switch (e) {
        case EncoderType::sweep: {
            return "sweep";
        }
        case EncoderType::morton: {
            return "morton";
        }
        case EncoderType::hilbert: {
            return "hilbert";
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("EncoderTypeUtil");
        }
    }
}

auto EncoderTypeUtil::fromInt(int val) -> EncoderType
{
    switch (val) {
        case static_cast<int>(EncoderType::sweep): {
            return EncoderType::sweep;
        }
        case static_cast<int>(EncoderType::morton): {
            return EncoderType::morton;
        }
        case static_cast<int>(EncoderType::hilbert): {
            return EncoderType::hilbert;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("EncoderTypeUtil");
        }
    }
}

auto EncoderTypeUtil::fromString(const std::string& occ) -> EncoderType
{
    std::array<EncoderType, 3> opts = getOptions();
    for (auto a : opts) {
        if (toString(a) == occ) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto EncoderTypeUtil::toInt(EncoderType dataView) -> int
{
    return static_cast<int>(dataView);
}

std::ostream& operator<<(std::ostream& os, EncoderType const& m)
{
    return os << std::string(EncoderTypeUtil::toString(m));
}


EncoderTypeUtil::Cli::Cli()
{
    mSet = false;
}

EncoderTypeUtil::Cli::Cli(std::string s)
{
    set(s);
}

EncoderTypeUtil::Cli::Cli(EncoderType model)
{
    mOption = model;
}

auto EncoderTypeUtil::Cli::getOption() const -> EncoderType
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "TransferSemantic was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return mOption;
}

auto EncoderTypeUtil::Cli::set(const std::string& opt)
    -> void
{
    try {
        mOption = EncoderTypeUtil::fromString(opt);
    } catch (...) {
        std::stringstream errorMsg;
        errorMsg << "TransferSemantic: " << opt << " is not a valid option (valid options are {";
        auto options = EncoderTypeUtil::getOptions();
        int  i = 0;
        for (auto o : options) {
            if (i != 0) {
                errorMsg << ", " << EncoderTypeUtil::toString(o);
            }
            errorMsg << EncoderTypeUtil::toString(o);
            i = 1;
        }
        errorMsg << "})";
        NEON_ERROR(errorMsg.str());
    }
    mSet = true;
}

auto EncoderTypeUtil::Cli::getStringOptions() const -> std::string
{
    std::stringstream s;
    auto              options = EncoderTypeUtil::getOptions();
    int               i = 0;
    for (auto o : options) {
        if (i != 0) {
            s << ", ";
        }
        s << EncoderTypeUtil::toString(o);
        i = 1;
    }
    std::string msg = s.str();
    return msg;
}

auto EncoderTypeUtil::Cli::getStringOption() const -> std::string
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "TransferSemantic was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return EncoderTypeUtil::toString(mOption);
}

auto EncoderTypeUtil::Cli::getDoc() const -> std::string
{
    std::stringstream s;
    s << getStringOptions();
    s << " default: " << getStringOptions();
    return s.str();
}

auto EncoderTypeUtil::Cli::addToReport(Neon::Report& report) const -> void
{
    report.addMember("EncoderType", EncoderTypeUtil::toString(this->getOption()));
}

auto EncoderTypeUtil::Cli::addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) const -> void
{
    report.addMember("EncoderType", EncoderTypeUtil::toString(this->getOption()), &subBlock);
}

}  // namespace Neon::domain::tool::spaceCurves
