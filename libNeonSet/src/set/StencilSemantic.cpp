#include "Neon/set/StencilSemantic.h"

namespace Neon::set {

auto StencilSemanticUtils::toString(StencilSemantic option) -> std::string
{
    switch (option) {
        case StencilSemantic::streaming: {
            return "streaming";
        }
        case StencilSemantic::standard: {
            return "grid";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto StencilSemanticUtils::fromString(const std::string& occ) -> StencilSemantic
{
    std::array<StencilSemantic, 4> opts{StencilSemantic::standard, StencilSemantic::streaming};
    for (auto a : opts) {
        if (toString(a) == occ) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto StencilSemanticUtils::getOptions() -> std::array<StencilSemantic, nOptions>
{
    std::array<StencilSemantic, nOptions> opts = {StencilSemantic::standard, StencilSemantic::streaming};
    return opts;
}

StencilSemanticUtils::Cli::Cli()
{
    mSet = false;
}

StencilSemanticUtils::Cli::Cli(std::string s)
{
    set(s);
}

StencilSemanticUtils::Cli::Cli(StencilSemantic model)
{
    mOption = model;
}

auto StencilSemanticUtils::Cli::getOption() -> StencilSemantic
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "TransferSemantic was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return mOption;
}

auto StencilSemanticUtils::Cli::set(const std::string& opt)
    -> void
{
    try {
        mOption = StencilSemanticUtils::fromString(opt);
    } catch (...) {
        std::stringstream errorMsg;
        errorMsg << "TransferSemantic: " << opt << " is not a valid option (valid options are {";
        auto options = StencilSemanticUtils::getOptions();
        int i = 0;
        for (auto o : options) {
            if(i!=0){
                errorMsg << ", "<< StencilSemanticUtils::toString(o) ;
            }
            errorMsg << StencilSemanticUtils::toString(o);
            i=1;
        }
        errorMsg << "})";
        NEON_ERROR(errorMsg.str());
    }
    mSet = true;
}

auto StencilSemanticUtils::Cli::getStringOptions() -> std::string
{
    std::stringstream s;
    auto              options = StencilSemanticUtils::getOptions();
    int               i = 0;
    for (auto o : options) {
        if (i != 0) {
            s << ", " ;
        }
        s << StencilSemanticUtils::toString(o);
        i = 1;
    }
    std::string msg= s.str();
    return msg;
}
}  // namespace Neon
