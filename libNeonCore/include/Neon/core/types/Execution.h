#pragma once

#include <iostream>
#include <string>
#include <array>
namespace Neon {

enum struct Execution
{
    device = 0 /**< A device accelerated execution */,
    host = 1 /**< Pre/post processing execution done on the host */
};

struct ExecutionUtils
{
    constexpr static int numConfigurations = 2;

    /**
     * Safely convert a ExecutionType to string
     *
     * @param option
     * @return
     */
    static auto toString(Neon::Execution option) -> const char*;

    /**
     * Safely convert an Execution to integer
     * @param option
     * @return
     */
    static auto toInt(Neon::Execution option) -> int;

    static auto getAllOptions()
        -> const std::array<Execution, ExecutionUtils::numConfigurations>&;

   private:
    static constexpr std::array<Execution, ExecutionUtils::numConfigurations> mAllOptions{Execution::device, Execution::host};
};

}  // namespace Neon


/**
 * operator<<
 *
 * @param os
 * @param m
 * @return
 */
std::ostream& operator<<(std::ostream& os, Neon::Execution const& m);