#pragma once
#include "Neon/Report.h"
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"


enum class Collision
{
    bgk,
    kbc
};

struct CollisionUtils
{
    static constexpr int nOptions = 2;

    static auto toString(Collision occ) -> std::string;
    static auto fromString(const std::string& occ) -> Collision;
    static auto getOptions() -> std::array<Collision, nOptions>;

    struct Cli
    {
        explicit Cli(std::string);
        explicit Cli(Collision model);
        Cli();

        auto getOption() const -> Collision;
        auto getOptionStr() const -> std::string;

        auto set(const std::string& opt) -> void;
        auto getAllOptionsStr() const -> std::string;
        auto getDoc() const -> std::string;

        auto addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) const -> void;
        auto addToReport(Neon::Report& report) const -> void;

       private:
        bool mSet = false;
        Collision  mOption;
    };
};


