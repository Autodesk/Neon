#pragma once

namespace Neon::domain {

template <class RealType, typename IntType>
IOGridVTK<RealType, IntType>::IOGridVTK(const Neon::domain::interface::GridBase& grid,
                                        const std::string&                       filename,
                                        bool                                     isNodeSpace,
                                        Neon::IoFileType                         vtiIOe)
    : IoToVTK<IntType, RealType>(filename,
                                 isNodeSpace ? grid.getDimension() : grid.getDimension() + 1,
                                 grid.getSpacing(),
                                 grid.getOrigin(),
                                 vtiIOe),
      mVtiDataTypeE(isNodeSpace ? ioToVTKns::VtiDataType_e::node : ioToVTKns::VtiDataType_e::voxel),
      mDimension(grid.getDimension())
{
}

template <class RealType, typename IntType>
template <typename Field>
auto IOGridVTK<RealType, IntType>::addField(const Field&       field,
                                            const std::string& name) -> void
{
    ioToVTKns::VtiDataType_e vtiDataTypeE = mVtiDataTypeE;
    bool                     isValidConfiguration = false;
    if (field.getBaseGridTool().getDimension() == mDimension) {
        vtiDataTypeE = mVtiDataTypeE;
        isValidConfiguration = true;
    }
    if (field.getBaseGridTool().getDimension() + 1 == mDimension && mVtiDataTypeE == ioToVTKns::VtiDataType_e::voxel) {
        vtiDataTypeE = ioToVTKns::VtiDataType_e::node;
        isValidConfiguration = true;
    }
    if (field.getBaseGridTool().getDimension() - 1 == mDimension && mVtiDataTypeE == ioToVTKns::VtiDataType_e::node) {
        vtiDataTypeE = ioToVTKns::VtiDataType_e::voxel;
        isValidConfiguration = true;
    }

    if (!isValidConfiguration) {
        NeonException exception("IOGridVTK");
        exception << "Incompatible size detected " << field.getBaseGridTool().getDimension() << " vs " << mDimension;
        NEON_THROW(exception);
    }

    IoToVTK<IntType, RealType>::addField([&](Neon::Integer_3d<IntType> idx, int card) -> RealType {
        return field(idx, card);
    },
                                         field.getCardinality(), name, vtiDataTypeE);
}

}  // namespace Neon::domain
