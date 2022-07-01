#pragma once

#include "Neon/domain/bGrid.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/set/GpuStreamSet.h"
#include "Neon/solver/linear/IterativeLinearSolver.h"

namespace Neon {
namespace solver {

/**
 * Conjugate Gradient solver for symmetric, positive-definite problems of the form Ax = b.
 * Reference: https://en.wikipedia.org/wiki/Conjugate_gradient_method
 * @tparam Grid_ta Grid datastructure
 * @tparam Real_ta Real number type (typically double or float)
 */
template <typename Grid_ta, typename Real_ta>
class CG_t : public IterativeLinearSolver_t<Grid_ta, Real_ta>
{
   public:
    using self_t = CG_t<Grid_ta, Real_ta>;
    using grid_t = Grid_ta;
    using Field = typename grid_t::template Field<Real_ta>;
    using BdField = typename grid_t::template Field<int8_t>;
    using matVec_t = MatVec<Grid_ta, Real_ta>;

   protected:
    Field m_p, m_s, m_r; /**< Extra fields and memory needed for the CG_t*/

   public:
    /**
     * Constructor for the conjugate gradient solver
     * @param[in] grid Grid definition
     */
    CG_t()
        : IterativeLinearSolver_t<Grid_ta, Real_ta>()
    {
    }

    inline auto self() -> self_t& { return *this; }
    inline auto self() const -> const self_t& { return *this; }
    inline auto cself() const -> const self_t& { return *this; }

    /**
     * Return the name of the solver ("CG_t").
     * @return Solver name
     */
    virtual std::string name() const override
    {
        return "CG";
    }

    /**
     * Solve the linear system Ax = b with the Conjugate Gradient method.
     *
     * The solver will return SolverStatus::NumericalIssue if the matrix is not positive-definite and
     * SolverParams::numericalIssueIsFailure flag is set to true.
     *
     * @param[in] A Matrix-vector multiply operation representing the linear operator
     * @param[in,out] x Unknown to solve for
     * @param[in] b RHS of the linear system
     * @param[in] bd Dirichlet boundary conditions in the domain (1: on boundary, 0: interior)
     * @param[in] params Parameters for the solve
     * @param[in,out] result Resulting information from the solve
     * @return Status of the solve
     * \sa SolverParams, SolverResultInfo, SolverStatus
     */
    virtual SolverStatus
    solve(NEON_IN std::shared_ptr<matVec_t> A /*!     Mat vec object                                                            */,
          NEON_IO Field& x /*!                      Unknown to solve for                                                      */,
          NEON_IN Field& b /*!                      b RHS of the linear system                                                */,
          NEON_IN BdField&               bd /*!     Dirichlet boundary conditions in the domain (1: on boundary, 0: interior) */,
          const SolverParams&            params /*! Parameters for the solve                                                  */,
          SolverResultInfo&              result /*! Resulting information from the solve                                      */,
          const Neon::skeleton::Options& opt = Neon::skeleton::Options(Neon::skeleton::Occ::standard, Neon::set::TransferMode::get)) override;

    /*
     * Reset the data structure used by the solver such that it can be
     * reused again.
     */
    virtual void reset() override;

   protected:
    /**
     * One time initializations for the CG_t solver
     */
    virtual void doInit(Field& x) override;

    /**
     * Compute the residual and store in m_r
     * ||r_0||_2 = ||b - Ax_0||_2
     * @return The residual squared norm
     */
    virtual Real_ta h_computeResidual(std::shared_ptr<matVec_t> A, Field& x, Field& b, BdField& bc);
};

extern template class CG_t<Neon::domain::eGrid, double>;
extern template class CG_t<Neon::domain::eGrid, float>;
extern template class CG_t<Neon::domain::bGrid, double>;
extern template class CG_t<Neon::domain::bGrid, float>;
extern template class CG_t<Neon::domain::dGrid, double>;
extern template class CG_t<Neon::domain::dGrid, float>;

}  // namespace solver
}  // namespace Neon
