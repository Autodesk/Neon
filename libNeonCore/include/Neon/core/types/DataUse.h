#pragma once

#include <iostream>
#include <string>

namespace Neon {

/**
 * Type of use for a Neon data container.
 * HOST: used for pre and post processing.
 *                    For example to load data into a field or to do some computation on a field before
 *                    IO_POSTPROCESSING operations are always computed on the CPU
 *
 * DEVICE: used for computation on the device..
 *          When deploying on an accelerator, compute is always run on the accelerator.
 *
 * HOST_DEVICE: both for IO_POSTPROCESSING and COMPUTE
 *
 */
enum struct DataUse
{
    HOST_DEVICE = 0,
    DEVICE = 1,
    HOST = 2,
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


