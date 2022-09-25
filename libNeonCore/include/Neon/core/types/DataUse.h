#pragma once

#include <iostream>
#include <string>

namespace Neon {

/**
 * Type of use for a Neon data container.
 * IO_POSTPROCESSING: used for pre and post processing.
 *                    For example to load data into a field or to do some computation on a field before
 *                    IO_POSTPROCESSING operations are always computed on the CPU
 *
 * COMPUTE: used both for pre-processing (IO) and actual computation (COMPUTE).
 *          When deploying on an accelerator, compute is always run on the accelerator.
 *
 * IO_COMPUTE: both for IO_POSTPROCESSING and COMPUTE
 *
 */
enum struct DataUse
{
    IO_COMPUTE = 0,
    COMPUTE = 1,
    IO_POSTPROCESSING = 2,
};

struct DataUseUtils
{
    constexpr static int numConfigurations = 3;

    /**
     * Convert a DataUse to string
     *
     * @param option
     * @return
     */
    static auto toString(Neon::DataUse option) -> const char*;
};

/**
 * operator<<
 *
 * @param os
 * @param m
 * @return
 */
std::ostream& operator<<(std::ostream& os, Neon::DataUse const& m);

}  // namespace Neon


