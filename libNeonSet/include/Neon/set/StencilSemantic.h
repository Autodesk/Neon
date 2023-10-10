#pragma once
#include <string>
#include <vector>

#include "Neon/Report.h"
#include "Neon/core/core.h"

namespace Neon::set {

enum struct StencilSemantic
{
    standard = 0 /*<  Transfer for halo update on grid structure    */,
    lattice = 1 /*< Transfer for halo update on lattice structure */
};


struct StencilSemanticUtils
{
    static constexpr int nOptions = 2;

    static auto toString(StencilSemantic opt) -> std::string;
    static auto fromString(const std::string& opt) -> StencilSemantic;
    static auto getOptions() -> std::array<StencilSemantic, nOptions>;

    struct Cli
    {
        explicit Cli(std::string);
        explicit Cli(StencilSemantic model);
        Cli();

        auto getOption() const -> StencilSemantic;
        auto set(const std::string& opt) -> void;
        auto getStringOptions() const -> std::string;
        auto getStringOption() const -> std::string;
        auto getDoc() const -> std::string;

        auto addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) const -> void;
        auto addToReport(Neon::Report& report) const -> void;

       private:
        bool            mSet = false;
        StencilSemantic mOption;
    };
};


}  // namespace Neon::set
