#pragma once
#include <string>
#include <vector>

#include "Neon/core/core.h"

namespace Neon::set {

enum struct TransferMode
{
    put = 0,
    get = 1
};

class TransferModeUtils
{
   public:
    static constexpr int nOptions = 2;
    static auto          toString(TransferMode occ) -> std::string;
    static auto          fromString(const std::string& occ) -> TransferMode;
    static auto          getOptions() -> std::array<TransferMode, nOptions>;

    struct Cli
    {
        explicit Cli(std::string);
        explicit Cli(TransferMode model);
        Cli();

        auto getOption() -> TransferMode;
        auto set(const std::string& opt) -> void;
        auto getStringOptions() -> std::string;

       private:
        bool         mSet = false;
        TransferMode mOption;
    };
};

}  // namespace Neon::set
