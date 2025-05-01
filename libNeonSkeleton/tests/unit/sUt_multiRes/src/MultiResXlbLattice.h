#pragma once
#include "Neon/domain/Grids.h"
namespace xlb {
auto getD3Q19(bool filterCenterOut)
{
    std::vector<Neon::index_3d> points = std::vector<Neon::index_3d>({{-1,- 1, 0},
                                                                      {-1, 0, -1},
                                                                      {-1, 0, 0},
                                                                      {-1, 0, 1},
                                                                      {-1, 1, 0},
                                                                      {0, -1, -1},
                                                                      {0, -1, 0},
                                                                      {0, -1, 1},
                                                                      {0, 0, -1},
                                                                      {0, 0, 0},
                                                                      {0, 0, 1},
                                                                      {0, 1, -1},
                                                                      {0, 1, 0},
                                                                      {0, 1, 1},
                                                                      {1, -1, 0},
                                                                      {1, 0, -1},
                                                                      {1, 0, 0},
                                                                      {1, 0, 1},
                                                                      {1, 1, 0}});
    Neon::domain::Stencil       res(points, filterCenterOut);
    return res;
}
}  // namespace xlb