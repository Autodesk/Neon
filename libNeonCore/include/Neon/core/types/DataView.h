#pragma once
#if !defined(NEON_WARP_COMPILATION)
#include <array>
#endif
#include <string>

namespace Neon {

enum struct DataView
{
    STANDARD = 0,
    INTERNAL = 1,
    BOUNDARY = 2,    
};

/**
 * Set of utilities for DataView options.
 */
struct DataViewUtil
{
    /**
     * Number of configurations for the enum
     */
    static const int nConfig{static_cast<int>(3)};

    /**
     * Convert enum value to string
     *
     * @param dataView
     * @return
     */
    static auto toString(DataView dataView) -> std::string;

#if !defined(NEON_WARP_COMPILATION)
    /**
     * Returns all valid configuration for DataView
     * @return
     */
    static auto validOptions() -> std::array<Neon::DataView, DataViewUtil::nConfig>;
#endif

    static auto fromInt(int val) -> DataView;

    static auto toInt(DataView dataView) -> int;

};

#if !defined(NEON_WARP_COMPILATION)
/**
 * operator<<
 *
 * @param os
 * @param m
 * @return
 */
std::ostream& operator<<(std::ostream& os, Neon::DataView const& m);
#endif
}  // namespace Neon
