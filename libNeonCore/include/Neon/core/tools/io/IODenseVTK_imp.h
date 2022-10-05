#pragma once

namespace Neon {

template <typename IntType, class RealType>
IODenseVTK<IntType, RealType>::IODenseVTK(const std::string&    filename,
                                          const Vec_3d<double>& spacingData,
                                          const Vec_3d<double>& origin,
                                          IoFileType            vtiIOe)
    : IoToVTK<IntType, RealType>(filename,
                                 Neon::Integer_3d<IntType>(0, 0, 0),
                                 spacingData,
                                 origin,
                                 vtiIOe),
      m_nodeSpace(0)
{
}

template <typename IntType, class RealType>
template <typename ExportType_ta>
auto IODenseVTK<IntType, RealType>::addField(IODense<ExportType_ta, IntType> dense,
                                             const std::string&              fname,
                                             bool                            isNodeSpace) -> void
{
    if (m_nodeSpace == 0) {
        if (isNodeSpace) {
            m_nodeSpace = dense.getDimension();
        } else {
            m_nodeSpace = dense.getDimension() + 1;
        }
    }

    bool isValidConfiguration = false;

    if (dense.getDimension() == m_nodeSpace && isNodeSpace) {
        isValidConfiguration = true;
    }

    if (dense.getDimension() + 1 == m_nodeSpace && !isNodeSpace) {
        isValidConfiguration = true;
    }

    if (!isValidConfiguration) {
        NeonException exception("IODenseVTK");
        exception << "Incompatible size detected " << dense.getDimension() << " vs " << m_nodeSpace;
        NEON_THROW(exception);
    }

    IoToVTK<IntType, RealType>::addField(
        [&](index_3d idx, int card, int) -> RealType {
            return dense(idx, card);
        },
        dense.getCardinality(),
        fname,
        isNodeSpace ? ioToVTKns::VtiDataType_e::node : ioToVTKns::VtiDataType_e::voxel);
}

}  // namespace Neon
