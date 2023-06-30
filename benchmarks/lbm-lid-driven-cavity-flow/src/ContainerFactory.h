#include "CellType.h"
#include "D3Q19.h"
#include "Neon/Neon.h"
#include "Neon/set/Containter.h"

template <typename Precision_,
          typename Lattice,
          typename Grid>
struct LbmContainers
{
};

#include "ContainersD3Q19.h"