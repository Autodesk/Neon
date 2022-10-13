#include "Neon/domain/internal/experimantal/staggeredGrid/node/NodeToVoxelMask.h"


namespace Neon::domain::internal::experimental::staggeredGrid::details {


auto NodeToVoxelMask::reset() -> void
{
    activeNodesOfVoxel = 0;
}

auto NodeToVoxelMask::setAsValid(int8_t a, int8_t b, int8_t c) -> void
{
    const size_t jump = (6) * (a + 1) +
                        (3) * (b + 1) +
                        (3) * (c + 1) / 2;
    const uint32_t nodeFlags = activeNodesOfVoxel;
    const uint32_t mask = 1 << jump;
    const uint32_t masked = nodeFlags | mask;
    activeNodesOfVoxel = masked;
}

auto NodeToVoxelMask::isNeighbourVoxelValid(int8_t a, int8_t b, int8_t c) const -> bool
{
    const size_t jump = (6) * (a + 1) +
                        (3) * (b + 1) +
                        (3) * (c + 1) / 2;
    const uint32_t nodeFlags = activeNodesOfVoxel;
    const uint32_t mask = 1 << jump;
    const uint32_t masked = nodeFlags & mask;

    return masked != 0;
}

NodeToVoxelMask::NodeToVoxelMask(uint8_t val)
{
    activeNodesOfVoxel = val;
}


}  // namespace Neon::domain::internal::experimental::staggeredGrid::details
