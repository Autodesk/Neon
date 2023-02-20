#pragma once

namespace Neon::domain::internal::exp::dGrid {


NEON_CUDA_HOST_DEVICE inline auto
dSpan::setAndValidate(Cell&         cell,
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

    switch (m_dataView) {
        case Neon::DataView::STANDARD: {
            if (cell.get() < m_dim) {
                res = true;
            }
            cell.set().z += m_zHaloRadius;
            return res;
        }
        case Neon::DataView::INTERNAL: {
            if (cell.get().x < (m_dim.x) &&
                cell.get().y < (m_dim.y) &&
                cell.get().z < (m_dim.z - 2 * m_zBoundaryRadius)) {
                res = true;
            }
            cell.set().z += m_zHaloRadius + m_zBoundaryRadius;

            return res;
        }
        case Neon::DataView::BOUNDARY: {
            if (cell.get().x < (m_dim.x) &&
                cell.get().y < (m_dim.y) &&
                cell.get().z < (m_zBoundaryRadius * 2)) {
                res = true;
            }
            cell.set().z += cell.get().z < m_zBoundaryRadius ? 0 : (m_dim.z - 1) + (-1 * m_zBoundaryRadius /* we remove zBoundaryRadius as the first zBoundaryRadius will manage the lower slices */);
            cell.set().z += m_zHaloRadius;

            return res;
        }
        default: {
        }
    }
    return false;
}

}  // namespace Neon::domain::internal::exp::dGrid