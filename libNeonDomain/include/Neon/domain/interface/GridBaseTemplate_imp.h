#pragma once

#include "Neon/core/tools/io/ioToVTK.h"
#include "Neon/core/tools/io/ioToNanoVDB.h"
#include "Neon/core/tools/io/ioToHDF5.h"

namespace Neon::domain::interface {

template <typename GridT, typename CellT>
auto GridBaseTemplate<GridT, CellT>::ioDomainToVtk(const std::string& fileName, Neon::IoFileType ioFileType) const -> void
{
    IoToVTK<int, double> io(fileName,
                            this->getDimension() + 1,
                            Vec_3d<double>(1, 1, 1),
                            Vec_3d<double>(0, 0, 0),
                            ioFileType);

    io.addField([&](const Neon::index_3d& idx, int) {
        bool isActiveVox = isInsideDomain(idx);
        return isActiveVox;
    },
                1, "Domain", ioToVTKns::VtiDataType_e::voxel);

    io.addField([&](const Neon::index_3d& idx, int) {
        const auto& cellProperties = this->getProperties(idx);
        if (!cellProperties.isInside()) {
            return -1;
        }
        auto setIdx = cellProperties.getSetIdx();
        return setIdx.idx();
    },
                1, "Partition", ioToVTKns::VtiDataType_e::voxel);

    io.flushAndClear();
    return;
}

template <typename GridT, typename CellT>
auto GridBaseTemplate<GridT, CellT>::ioDomainToNanoVDB(const std::string& fileName) const -> void
{
    ioToNanoVDB<int, float> io1(fileName + "_domain",
                            this->getDimension(),
                            [&](const Neon::index_3d& idx, int) {
                                bool isActiveVox = isInsideDomain(idx);
                                return isActiveVox;
                            },
                            1,
                            1.0,
                            Neon::Integer_3d<int>(0, 0, 0));

    ioToNanoVDB<int, float> io2(fileName + "_partition",
                            this->getDimension(),
                            [&](const Neon::index_3d& idx, int) {
                                const auto& cellProperties = this->getProperties(idx);
                                if (!cellProperties.isInside()) {
                                    return -1;
                                }
                                auto setIdx = cellProperties.getSetIdx();
                                return setIdx.idx();
                            },
                            1,
                            1.0,
                            Neon::Integer_3d<int>(0, 0, 0));

    io1.flush();
    io2.flush();
    return;
}

template <typename GridT, typename CellT>
auto GridBaseTemplate<GridT, CellT>::ioDomainToHDF5(const std::string& fileName) const -> void
{
    ioToHDF5<int, float> io1(fileName + "_domain",
                            this->getDimension(),
                            [&](const Neon::index_3d& idx, int) {
                                bool isActiveVox = isInsideDomain(idx);
                                return isActiveVox;
                            },
                            1,
                            1.0,
                            Neon::Integer_3d<int>(0, 0, 0));

    ioToHDF5<int, float> io2(fileName + "_partition",
                            this->getDimension(),
                            [&](const Neon::index_3d& idx, int) {
                                const auto& cellProperties = this->getProperties(idx);
                                if (!cellProperties.isInside()) {
                                    return -1;
                                }
                                auto setIdx = cellProperties.getSetIdx();
                                return setIdx.idx();
                            },
                            1,
                            1.0,
                            Neon::Integer_3d<int>(0, 0, 0));

    io1.flush();
    io2.flush();
    return;
}
}  // namespace Neon::domain::interface
