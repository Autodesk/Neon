#pragma once
#include "CellType.h"
#include "D3Q19.h"
#include "Neon/Neon.h"
#include "Neon/set/Containter.h"

namespace pull {
template <typename Precision_,
          typename Lattice,
          typename Grid>
struct ContainerFactory
{
};
}  // namespace pull

namespace push {
template <typename Precision_,
          typename Lattice,
          typename Grid>
struct ContainerFactory
{
};
}  // namespace push

namespace common {
template <typename Precision_,
          typename Lattice,
          typename Grid>
struct ContainerFactory
{
};
}  // namespace common
#include "ContainersD3Q19.h"
#include "ContainersD3Q27.h"