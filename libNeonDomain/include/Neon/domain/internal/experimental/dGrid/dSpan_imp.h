#pragma once

namespace Neon::domain::internal::exp::dGrid {

NEON_CUDA_HOST_DEVICE inline auto
dSpan::setAndValidate(Voxel&        cell,
                      const size_t& x,
                      const size_t& y,
                      const size_t& z)
    const
    -> bool
{
    bool res = false;
    cell.set().x = int(x);
    cell.set().y = int(y);
    cell.set().z = int(z);

    switch (mDataView) {
        case Neon::DataView::STANDARD: {
            if (cell.get() < mDim) {
                res = true;
            }
            cell.set().z += mZHaloRadius;
            return res;
        }
        case Neon::DataView::INTERNAL: {
            if (cell.get().x < (mDim.x) &&
                cell.get().y < (mDim.y) &&
                cell.get().z < (mDim.z - 2 * mZBoundaryRadius)) {
                res = true;
            }
            cell.set().z += mZHaloRadius + mZBoundaryRadius;

            return res;
        }
        case Neon::DataView::BOUNDARY: {
            if (cell.get().x < (mDim.x) &&
                cell.get().y < (mDim.y) &&
                cell.get().z < (mZBoundaryRadius * 2)) {
                res = true;
            }
            cell.set().z += cell.get().z < mZBoundaryRadius
                                ? 0
                                : (mDim.z - 1) + (-1 * mZBoundaryRadius /* we remove zBoundaryRadius as the first zBoundaryRadius will manage the lower slices */);
            cell.set().z += mZHaloRadius;

            return res;
        }
        default: {
        }
    }
    return false;
}

}  // namespace Neon::domain::internal::exp::dGrid