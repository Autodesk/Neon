#pragma once

#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/domain/bGrid.h"
#include "Neon/solver/linear/MatVec.h"

namespace Neon {
namespace solver {

/**
 * LaplacianMatVec represents a finite-difference Laplacian operator
 * @tparam Grid Type of the grid where this operation will be executed
 * @tparam Real Real value type (double or float)
 */
template <typename Grid_, typename Real>
class LaplacianMatVec : public MatVec<Grid_, Real>
{
    // Step size h in finite-difference stencil
    Real m_h;

   public:
    using self_t = LaplacianMatVec<Grid_, Real>;
    using Grid = Grid_;
    using Field = typename Grid::template Field<Real>;
    using bdField = typename Grid::template Field<int8_t>;

    LaplacianMatVec(Real h)
        : MatVec<Grid, Real>(), m_h(h)
    {
    }

    /**
     * Get the finite-difference step-size
     * @return Step size h
     */
    Real stepSize() const
    {
        return m_h;
    }

    /**
     * Set the finite-difference step-size
     * @param[in] Step size h
     */
    void setStepSize(Real h)
    {
        m_h = h;
    }

    /**
     * Applies the finite-difference Laplacian operator L
     * @param[in] backend Backend specifying the device type and device set to use
     * @param[in] input Real valued input field x
     * @param[in] bd int8_t valued field marking Dirichlet boundary with 1 and 0 otherwise
     * @param[inout] output Real valued field holding the output L * x
     */
    virtual Neon::set::Container matVec(const Field& input, const bdField& bd, Field& output) override;
};

// Extern template instantiations
extern template class LaplacianMatVec<Neon::domain::details::eGrid::eGrid, double>;
extern template class LaplacianMatVec<Neon::domain::details::eGrid::eGrid, float>;
extern template class LaplacianMatVec<Neon::dGrid, double>;
extern template class LaplacianMatVec<Neon::dGrid, float>;
extern template class LaplacianMatVec<Neon::domain::bGrid, double>;
extern template class LaplacianMatVec<Neon::domain::bGrid, float>;

}  //namespace solver
}  // namespace Neon