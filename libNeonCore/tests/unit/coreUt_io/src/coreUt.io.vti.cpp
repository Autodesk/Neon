
#include <Neon/core/types/vec.h>

#include <cstring>
#include <iostream>

#include "Neon/core/core.h"
#include "Neon/core/tools/io/ioToVti.h"
#include "Neon/core/tools/io/exportVTI.h"
#include "Neon/core/tools/io/ioToVTK.h"
#include "gtest/gtest.h"

template <typename T_ta>
void exportLinear(T_ta startingVal, Neon::index_3d dim)
{
    using namespace Neon;
    size_t nEl = dim.rMulTyped<size_t>();
    T_ta*  mem = (T_ta*)malloc(nEl * sizeof(T_ta));

    for (int z = 0; z < dim.z; z++) {
        for (int y = 0; y < dim.y; y++) {
            for (int x = 0; x < dim.x; x++) {
                const size_3d pitch3d(x, y, z);
                const size_t  pitch = pitch3d.mPitch(dim);

                mem[pitch] = startingVal + pitch;
                //std::cout<<mem[pitch]<<" ";
            }
        }
    }
    const Vec_3d<T_ta> spacingData(1);
    const Vec_3d<T_ta> origin(0.0);
    std::string        filename("coreUt_vti_test.vti");

    Neon::exportVti<Neon::vti_e::NODE, T_ta, T_ta>(mem, 1, "coreUt_vti_test_nodes.vti", dim, spacingData, origin);
    Neon::exportVti<Neon::vti_e::VOXEL, T_ta, T_ta>(mem, 1, "coreUt_vti_test_vox.vti", dim, spacingData, origin);

    free(mem);
}

TEST(CoreUt_io, ExportLinear)
{
    exportLinear<double>(1.0, {10, 10, 10});
}

TEST(CoreUt_io, ImplicitExport)
{
    Neon::index_3d voxDim(5, 5, 5);
    Neon::index_3d nodDim = voxDim + 1;
    auto           nodeLinear = [&](Neon::index_3d idx, int /*vIdx*/) -> double {
        return idx.x + idx.y + idx.z;
    };
    auto voxLinear = [&](Neon::index_3d idx, int /*vIdx*/) -> double {
        return -idx.x + idx.y - idx.z;
    };
   
    std::vector<Neon::VtiInputData_t<double, int>> inps;
    Neon::ioToVTI({{nodeLinear, 1, "nodeLinear", true, Neon::IoFileType::ASCII},
                   {voxLinear, 1, "voxLinear", false, Neon::IoFileType::ASCII}},
                  "coreUt_vti_test_implicit.vti", nodDim, voxDim, 1.0, 0.0);
}

TEST(CoreUt_io, implicitExportTuple)
{
    Neon::index_3d voxDim(5, 5, 5);
    Neon::index_3d nodDim = voxDim + 1;
    auto           velocityNorm = [&](const Neon::Integer_3d<int>& idx, int /*vIdx*/) -> double {
        return idx.x + idx.y + idx.z;
    };
    auto density = [&](const Neon::index_3d& idx, int /*vIdx*/) -> double {
        return -idx.x + idx.y - idx.z;
    };

    Neon::ioToVTI({{velocityNorm, 1, "velocityNorm", true, Neon::IoFileType::ASCII}},
                  "coreUt_vti_test_implicit_tuple_ASCII.vti",
                  nodDim, voxDim,
                  1.0,
                  0.0);

    Neon::ioToVTI({{velocityNorm, 1, "velocityNorm", true, Neon::IoFileType::BINARY},
                   {density, 1, "density", false, Neon::IoFileType::BINARY}},
                  "coreUt_vti_test_implicit_tuple_BINARY.vti",
                  nodDim, voxDim,
                  1.0,
                  0.0);
}

TEST(CoreUt_io, implicitLegacyExportTuple)
{
    Neon::index_3d voxDim(5, 5, 5);
    Neon::index_3d nodDim = voxDim + 1;

    auto velocityNorm = [&](const Neon::Integer_3d<int>& idx, int /*vIdx*/) -> double {
        return idx.x;
    };

    auto density = [&](const Neon::index_3d& idx, int /*vIdx*/) -> double {
        return -idx.x + idx.y - idx.z;
    };
    {
        Neon::IoToVTK ioToVTK("coreUt_vti_test_legacy_implicit_tuple_ASCII",
                              nodDim,
                              1.0,
                              0.0, Neon::IoFileType::ASCII);

        ioToVTK.addField(velocityNorm, 1, "velocityNorm", Neon::ioToVTKns::node);
        ioToVTK.addField(density, 1, "density", Neon::ioToVTKns::voxel);
        ioToVTK.flush();
    }
    {
        Neon::IoToVTK ioToVTK("coreUt_vti_test_legacy_implicit_tuple_BINARY",
                              nodDim,
                              2.0,
                              0.0, Neon::IoFileType::BINARY);

        ioToVTK.addField(velocityNorm, 1, "velocityNorm", Neon::ioToVTKns::node);
        ioToVTK.addField(density, 1, "density", Neon::ioToVTKns::voxel);
        ioToVTK.flush();
    }
//    Neon::ioToVTK({{velocityNorm, 1, "velocityNorm", Neon::ioToVTKns::node},
//                   {density, 1, "density", Neon::ioToVTKns::voxel}},
//                  "coreUt_vti_test_legacy_implicit_tuple_BINARY",
//                  nodDim,
//                  1.0,
//                  0.0,
//                  Neon::IoFileType::BINARY);
}