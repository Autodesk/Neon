#pragma once

#include <stdint.h>
#include <array>
#include <iostream>
#include "Neon/core/tools/Report.h"
#include "Neon/core/types/Place.h"
namespace Neon {

/**
 * Enumeration of possible names for a device
 */
enum struct DeviceType
{
    CPU = 0,
    OMP = 1,
    CUDA = 2,
    MPI = 3,
    NONE = 4,
    NUM_USER_OPTIONS = 4 /** We don't count SYSTEM as USER option */
};

struct DeviceTypeUtil
{
    static const int nConfig{static_cast<int>(DeviceType::NUM_USER_OPTIONS)};
    static auto      getOptions() -> std::array<DeviceType, nConfig>;
    static auto      getDevType(int devTypeidx) -> DeviceType;
    static auto      toString(DeviceType dataView) -> std::string;
    static auto      toInt(DeviceType dt) -> int32_t;
    static auto      fromString(const std::string& option) -> DeviceType;
    static auto      getExecution(Neon::DeviceType devType) -> Neon::Place;

    struct Cli
    {
        explicit Cli(std::string);
        explicit Cli(DeviceType model);
        Cli();

        auto getOption() -> DeviceType;
        auto getStringOptions() -> std::string;
        auto set(const std::string& opt) -> void;

        auto addToReport(Neon::core::Report& report, Neon::core::Report::SubBlock& subBlock) -> void;
        auto addToReport(Neon::core::Report& report) -> void;

       private:
        bool       mSet = false;
        DeviceType mOption;
    };
};

std::ostream& operator<<(std::ostream& os, Neon::DeviceType const& m);

}  // namespace Neon
