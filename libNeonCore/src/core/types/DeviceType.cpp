#include "Neon/core/types/DeviceType.h"
#include "Neon/core/core.h"

#include <vector>
namespace Neon {

static const std::vector<const char*> devTypeIdNames_vec = {"CPU", "OMP", "CUDA", "MPI", "NONE"};


auto DeviceTypeUtil::toString(DeviceType type) -> std::string
{
    return devTypeIdNames_vec[static_cast<int>(type)];
}

auto DeviceTypeUtil::toInt(DeviceType dt) -> int32_t
{
    return static_cast<int>(dt);
}

auto DeviceTypeUtil::getOptions() -> std::array<DeviceType, DeviceTypeUtil::nConfig>
{
    return {DeviceType::CPU,
            DeviceType::OMP,
            DeviceType::CUDA,
            DeviceType::MPI};
}

auto DeviceTypeUtil::fromString(const std::string& option) -> DeviceType
{
    auto options = DeviceTypeUtil::getOptions();
    for (auto a : options) {
        if (DeviceTypeUtil::toString(a) == option) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto DeviceTypeUtil::getDevType(int devTypeidx) -> DeviceType
{
    if (devTypeidx == 0)
        return DeviceType::CPU;
    if (devTypeidx == 1)
        return DeviceType::OMP;
    if (devTypeidx == 2)
        return DeviceType::CUDA;
    if (devTypeidx == 3)
        return DeviceType::MPI;
    NEON_THROW_UNSUPPORTED_OPERATION("");
}

auto DeviceTypeUtil::getExecution(Neon::DeviceType devType) -> Neon::Execution
{
    if (DeviceType::CPU == devType)
        return Execution::host;
    if (DeviceType::OMP == devType)
        return Execution::host;
    if (DeviceType::CUDA == devType)
        return Execution::device;
    NEON_THROW_UNSUPPORTED_OPERATION("");}


std::ostream& operator<<(std::ostream& os, Neon::DeviceType const& m)
{
    return os << std::string(Neon::DeviceTypeUtil::toString(m));
}

DeviceTypeUtil::Cli::Cli(std::string str)
{
    set(str);
}

auto DeviceTypeUtil::Cli::getOption() -> DeviceType
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "Executor model was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return mOption;
}

DeviceTypeUtil::Cli::Cli()
{
    mSet = false;
}
DeviceTypeUtil::Cli::Cli(DeviceType model)
{
    mSet = true;
    mOption = model;
}

auto DeviceTypeUtil::Cli::set(const std::string& opt)
    -> void
{
    try {
        mOption = DeviceTypeUtil::fromString(opt);
    } catch (...) {
        std::stringstream errorMsg;
        errorMsg << "Executor: " << opt << " is not a valid option (valid options are {";
        auto options = DeviceTypeUtil::getOptions();
        int  i = 0;
        for (auto o : options) {
            if (i != 0) {
                errorMsg << ", " << DeviceTypeUtil::toString(o);
            }
            errorMsg << DeviceTypeUtil::toString(o);
            i = 1;
        }
        errorMsg << "})";
        NEON_ERROR(errorMsg.str());
    }
    mSet = true;
}
auto DeviceTypeUtil::Cli::getStringOptions() -> std::string
{
    std::stringstream s;
    auto              options = DeviceTypeUtil::getOptions();
    int               i = 0;
    for (auto o : options) {
        if (i != 0) {
            s << ", ";
        }
        s << DeviceTypeUtil::toString(o);
        i = 1;
    }
    std::string msg = s.str();
    return msg;
}

auto DeviceTypeUtil::Cli::addToReport(Neon::core::Report& report, Neon::core::Report::SubBlock& subBlock) -> void
{
    report.addMember("DeviceType", DeviceTypeUtil::toString(this->getOption()), &subBlock);
}

auto DeviceTypeUtil::Cli::addToReport(Neon::core::Report& report) -> void
{
    report.addMember("DeviceType", DeviceTypeUtil::toString(this->getOption()));
}













}  // namespace Neon
