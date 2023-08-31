#pragma once
#include "Neon/Report.h"
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"

namespace Neon::skeleton {

enum class Occ
{
    standard,
    extended,
    twoWayExtended,
    none,
};

struct OccUtils
{
    static constexpr int nOptions = 4;

    static auto toString(Occ occ) -> std::string;
    static auto fromString(const std::string& occ) -> Occ;
    static auto getOptions() -> std::array<Occ, nOptions>;

    struct Cli
    {
        explicit Cli(std::string);
        explicit Cli(Occ model);
        Cli();

        auto getOption() const -> Occ;
        auto set(const std::string& opt) -> void;
        auto getStringOptions() const -> std::string;
        auto getDoc() const -> std::string;

        auto addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) const -> void;
        auto addToReport(Neon::Report& report) const -> void;

       private:
        bool mSet = false;
        Occ  mOption;
    };
};


}  // namespace Neon::skeleton