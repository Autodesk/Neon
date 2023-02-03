#pragma once

#include <array>
#include <iostream>
#include <string>
#include <vector>

#include "Neon/core/types/DataUse.h"

namespace Neon {

enum struct Place
{
    device = 0 /**< A device accelerated execution */,
    host = 1 /**< Pre/post processing execution done on the host */
};

struct PlaceUtils
{
    constexpr static int numConfigurations = 2;

    /**
     * Safely convert a ExecutionType to string
     *
     * @param option
     * @return
     */
    static auto toString(Neon::Place option) -> const char*;

    /**
     * Safely convert an ExecutionPlace to integer
     * @param option
     * @return
     */
    static auto toInt(Neon::Place option) -> int;

    static auto getAllOptions()
        -> const std::array<Place, PlaceUtils::numConfigurations>&;

    static auto getCompatibleOptions(Neon::DataUse dataUse)
        -> std::vector<Place>;

    static auto checkCompatibility(Neon::DataUse   dataUse,
                                    Neon::Place execution)
        -> bool;

   private:
    static constexpr std::array<Place, PlaceUtils::numConfigurations> mAllOptions{Place::device, Place::host};
};

/**
 * operator<<
 *
 * @param os
 * @param m
 * @return
 */
std::ostream& operator<<(std::ostream& os, Neon::Place const& m);
}  // namespace Neon
