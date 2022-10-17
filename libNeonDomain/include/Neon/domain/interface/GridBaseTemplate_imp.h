#pragma once

#include "Neon/core/tools/io/ioToVTK.h"

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
        bool isActieVox = isInsideDomain(idx);
        return isActieVox;
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
}  // namespace Neon::domain::interface
