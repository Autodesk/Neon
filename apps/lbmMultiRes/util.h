#pragma once
#include "Neon/Neon.h"
#include "lattice.h"

constexpr NEON_CUDA_HOST_DEVICE Neon::int8_3d getDir(const int8_t q)
{
    return Neon::int8_3d(latticeVelocity[q][0], latticeVelocity[q][1], latticeVelocity[q][2]);
}

template <typename T>
constexpr NEON_CUDA_HOST_DEVICE inline Neon::int8_3d uncleOffset(const T& cell, const Neon::int8_3d& q)
{
    //given a local index within a cell and a population direction (q)
    //find the uncle's (the parent neighbor) offset from which the desired population (q) should be read
    //this offset is wrt the cell containing the localID (i.e., the parent of localID)
    auto off = [](const int8_t i, const int8_t j) {
        //0, -1 --> -1
        //1, -1 --> 0
        //0, 0 --> 0
        //0, 1 --> 0
        //1, 1 --> 1
        const int8_t s = i + j;
        return (s <= 0) ? s : s - 1;
    };
    Neon::int8_3d offset(off(cell.x % Neon::domain::details::mGrid::kUserBlockSizeX, q.x),
                         off(cell.y % Neon::domain::details::mGrid::kUserBlockSizeY, q.y),
                         off(cell.z % Neon::domain::details::mGrid::kUserBlockSizeZ, q.z));
    return offset;
}

template <typename T>
NEON_CUDA_HOST_DEVICE T computeOmega(T omega0, int level, int numLevels)
{
    int ilevel = numLevels - level - 1;
    // scalbln(1.0, x) = 2^x
    return 2 * omega0 / (scalbln(1.0, ilevel + 1) + (1. - scalbln(1.0, ilevel)) * omega0);
}

template <typename T, int Q>
NEON_CUDA_HOST_DEVICE Neon::Vec_3d<T> velocity(const T* fin,
                                               const T  rho)
{
    Neon::Vec_3d<T> vel(0, 0, 0);
    for (int i = 0; i < Q; ++i) {
        const T f = fin[i];
        for (int d = 0; d < 3; ++d) {
            vel.v[d] += f * latticeVelocity[i][d];
        }
    }
    for (int d = 0; d < 3; ++d) {
        vel.v[d] /= rho;
    }
    return vel;
}


inline float sdfCube(Neon::index_3d id, Neon::index_3d dim, Neon::float_3d b = {1.0, 1.0, 1.0})
{
    auto mapToCube = [&](Neon::index_3d id) {
        //map p to an axis-aligned cube from -1 to 1
        Neon::float_3d half_dim = dim.newType<float>() * 0.5;
        Neon::float_3d ret = (id.newType<float>() - half_dim) / half_dim;
        return ret;
    };
    Neon::float_3d p = mapToCube(id);

    Neon::float_3d d(std::abs(p.x) - b.x, std::abs(p.y) - b.y, std::abs(p.z) - b.z);

    Neon::float_3d d_max(std::max(d.x, 0.f), std::max(d.y, 0.f), std::max(d.z, 0.f));
    float          len = std::sqrt(d_max.x * d_max.x + d_max.y * d_max.y + d_max.z * d_max.z);
    float          val = std::min(std::max(d.x, std::max(d.y, d.z)), 0.f) + len;
    return val;
}