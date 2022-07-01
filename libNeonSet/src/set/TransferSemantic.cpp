#include "Neon/set/TransferSemantic.h"

namespace Neon::set {

auto TransferSemanticUtils::toString(TransferSemantic occ) -> std::string
{
    switch (occ) {
        case TransferSemantic::lattice: {
            return "lattice";
        }
        case TransferSemantic::grid: {
            return "grid";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto TransferSemanticUtils::fromString(const std::string& occ) -> TransferSemantic
{
    std::array<TransferSemantic, 4> opts{TransferSemantic::grid, TransferSemantic::lattice};
    for (auto a : opts) {
        if (toString(a) == occ) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto TransferSemanticUtils::getOptions() -> std::array<TransferSemantic, nOptions>
{
    std::array<TransferSemantic, nOptions> opts = {TransferSemantic::grid, TransferSemantic::lattice};
    return opts;
}

TransferSemanticUtils::Cli::Cli()
{
    mSet = false;
}

TransferSemanticUtils::Cli::Cli(std::string s)
{
    set(s);
}

TransferSemanticUtils::Cli::Cli(TransferSemantic model)
{
    mOption = model;
}

auto TransferSemanticUtils::Cli::getOption() -> TransferSemantic
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "TransferSemantic was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return mOption;
}

auto TransferSemanticUtils::Cli::set(const std::string& opt)
    -> void
{
    try {
        mOption = TransferSemanticUtils::fromString(opt);
    } catch (...) {
        std::stringstream errorMsg;
        errorMsg << "TransferSemantic: " << opt << " is not a valid option (valid options are {";
        auto options = TransferSemanticUtils::getOptions();
        int i = 0;
        for (auto o : options) {
            if(i!=0){
                errorMsg << ", "<< TransferSemanticUtils::toString(o) ;
            }
            errorMsg << TransferSemanticUtils::toString(o);
            i=1;
        }
        errorMsg << "})";
        NEON_ERROR(errorMsg.str());
    }
    mSet = true;
}

auto TransferSemanticUtils::Cli::getStringOptions() -> std::string
{
    std::stringstream s;
    auto              options = TransferSemanticUtils::getOptions();
    int               i = 0;
    for (auto o : options) {
        if (i != 0) {
            s << ", " ;
        }
        s << TransferSemanticUtils::toString(o);
        i = 1;
    }
    std::string msg= s.str();
    return msg;
}
}  // namespace Neon
