#pragma once

#include <stdint.h>
#include <limits>
#include "Neon/set/Backend.h"
#include "Neon/set/memory/memSet.h"
#include "Neon/skeleton/Skeleton.h"
#include "Neon/solver/linear/MatVec.h"

namespace Neon {
namespace solver {

/**
 * Enum used to specify the boundary conditions
 * Solvers in Neon assume this convention of using 0 for Dirichlet (Fixed) BC, and 1 for Neumann (Free) BC
 */
enum BoundaryCondition : int8_t
{
    Fixed,  /// Dirichlet (Fixed) boundary condition where solution is specified
    Free,   /// Neumann boundary condition where directional derivative in normal direction is specified / free DoF
};

/**
 * Input parameters to the iterative solver
 */
struct SolverParams
{
    size_t                  maxIterations = 100;             /// Maximum iterations allowed in the solver
    double                  toleranceRel = 1e-5;             /// Relative tolerance for checking residual convergence
    double                  toleranceAbs = 0.0;              /// Absolute tolerance for checking residual convergence
    double                  toleranceDiv = 1e+5;             /// Relative tolerance for checking if residual is diverging
    bool                    needResiduals = false;           /// Whether to store each iteration's residual in SolverResultInfo
    bool                    numericalIssueIsFailure = true;  /// Whether to stop the solver in case of numerical issue (e.g. non-positive definiteness)
    Neon::skeleton::Options skeletonOptions;
};

/**
 * Status of the solver
 */
enum class SolverStatus
{
    Converged,      /**< Converged before reaching maximum iterations */
    IterationLimit, /**< Reached the maximum iteration limit */
    Iterating,      /**< Iteration in progress */
    NumericalIssue, /**< Did not converge due to numerical issue */
    Error           /**< Error in solver */
};

/**
 * Stores the result of a solver
 */
struct SolverResultInfo
{
    std::string         solverName = "Unknown";  /// Solver name
    double              residualStart = 0.0;     /// Residual before solve
    double              residualEnd = 0.0;       /// Residual after solve
    size_t              numIterations = 0;       /// Number of iterations taken to solve
    double              totalTime = 0.0;         /// Total time taken to solve including pre- and post-solve steps
    double              solveTime = 0.0;         /// Time taken only for solve iterations
    std::vector<double> residuals;               /// List of residuals for each iteration
};

/**
 * Interface for all iterative solvers
 * @tparam Grid_ta Template argument representing the grid type
 * @tparam Real_ta Datatype of fields (typically double or float)
 */
template <typename Grid_ta, typename Real_ta>
class IterativeLinearSolver_t
{
   protected:
    bool m_isInit = false;

   public:
    // Aliases to reduce typing
    using self_t = IterativeLinearSolver_t<Grid_ta, Real_ta>;
    using Grid = Grid_ta;
    using Field = typename Grid::template Field<Real_ta>;
    using BdField = typename Grid::template Field<int8_t>;
    using matVec_t = MatVec<Grid_ta, Real_ta>;

    /**
     * Virtual destructor
     */
    virtual ~IterativeLinearSolver_t() = default;

    /**
     * Return the name of the solver.
     * Override this method in the derived solver class and
     * provide a unique name for each solver.
     * @return Solver name
     */
    virtual std::string name() const = 0;

    /**
     * One time initialization for the solver
     * @param[in] x A field representating the unknown to be solved for (used for initializing internal field props like cardinality, halo etc.)
     * Additional functionality for derived solvers can be written in doInit()
     */
    void init(Field& x)
    {
        doInit(x);
        m_isInit = true;
    }

    /**
     * Check whether the solver has been initialized
     * @return Initialized or not
     */
    bool isInit() const
    {
        return m_isInit;
    }

    /**
     * Reset the solver for reusing in a new solve
     */
    virtual void reset()
    {
    }

    /**
     * Solve the linear system using Container API
     * @param[in] params Parameters for the solve
     * @param[in,out] result Resulting information from the solve
     * @return Status of the solve
     * \sa SolverParams, SolverResultInfo, SolverStatus
     */
    virtual SolverStatus solve(std::shared_ptr<matVec_t>      matVecOp,
                               Field&                         x,
                               Field&                         b,
                               BdField&                       bc,
                               const SolverParams&            params,
                               SolverResultInfo&              result,
                               const Neon::skeleton::Options& opt = Neon::skeleton::Options(Neon::skeleton::Occ::standard, Neon::set::TransferMode::get)) = 0;

   protected:
    IterativeLinearSolver_t() = default;

    /**
     * Additional operations to do during init()
     * This method must be overriden in the derived class.
     * The input field can be used as a guide to initialize internal field
     * attributes e.g., the halo.
     * @param[in] x Input field used as reference for initializing internal fields
     */
    virtual void doInit([[maybe_unused]] Field& x)
    {
    }

    /**
     * Convergence check. Convergence is determined by the following equation:
     * rnorm < max(reltol * rnorm_0, abstol)
     * Divergence is determined by checking if:
     * rnorm > divtol * rnorm_0
     * Can be overriden in derived class
     * @param[in] residual Current residual
     * @param[in] residualinit Initial residual
     * @param[in] iteration Current iteration
     * @param[in] params Solver parameters for determining convergence
     * @return Convergence status
     */
    virtual SolverStatus converged(Real_ta residual, Real_ta residualInit, size_t iteration, const SolverParams& params)
    {
        // Early terminate if the initial residual is tiny
        if (residualInit <= std::numeric_limits<Real_ta>::epsilon()) {
            return SolverStatus::Converged;
        }
        // NaN check
        if (std::isnan(residual) || std::isnan(residualInit)) {
            return SolverStatus::Error;
        }
        // Absolute/relative residual convergence check
        if (residual < std::max(residualInit * params.toleranceRel, params.toleranceAbs)) {
            return SolverStatus::Converged;
        }
        // Divergence check
        if (residual > residualInit * params.toleranceDiv) {
            return SolverStatus::Error;
        }
        // Max iteration limit check
        if (iteration == params.maxIterations - 1) {
            return SolverStatus::IterationLimit;
        }
        return SolverStatus::Iterating;
    }

    /**
     * A helper function to extract the Backend_t object from a field
     *
     * @param[in] f Field to extract the backend from
     * @return const-ref to Backend_t
     */
    const Neon::Backend& h_getBackend(const Field& f) const
    {
        return f.grid().backend();
    }

    /**
     * A helper function to extract the Backend_t object from a field
     *
     * @param[in] f Field to extract the backend from
     * @return mutable-ref to Backend_t
     */
    auto h_getBackend(Field& f) const -> const Neon::Backend&
    {
        return f.getGrid().getBackend();
    }
};

}  // Namespace solver
}  // Namespace Neon
