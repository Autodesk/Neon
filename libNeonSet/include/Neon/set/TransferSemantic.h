#pragma once
#include <string>
#include <vector>

#include "Neon/core/core.h"

namespace Neon::set {


enum struct TransferSemantic
{
    grid = 0 /*<    Transfer for halo update on grid structure    */,
    lattice = 1 /*< Transfer for halo update on lattice structure */
};


struct TransferSemanticUtils
{
    static constexpr int nOptions = 2;

    static auto toString(TransferSemantic opt) -> std::string;
    static auto fromString(const std::string& opt) -> TransferSemantic;
    static auto getOptions() -> std::array<TransferSemantic, nOptions>;
    
    struct Cli
    {
        explicit Cli(std::string);
        explicit Cli(TransferSemantic model);
        Cli();

        auto getOption() -> TransferSemantic;
        auto set(const std::string& opt) -> void;
        auto getStringOptions() -> std::string;

       private:
        bool mSet = false;
        TransferSemantic  mOption;
    };
};


}  // namespace Neon::set
