#pragma once

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "Neon/core/core.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/domain/bGrid.h"
#include "Neon/set/DevSet.h"
#include "Neon/solver/linear/IterativeLinearSolver.h"
#include "Neon/solver/linear/krylov/CG.h"
#include "Neon/solver/linear/matvecs/LaplacianMatVec.h"

// Alias for pointer to base solver
template <typename Grid, typename Real>
using SolverPtr = std::shared_ptr<Neon::solver::IterativeLinearSolver_t<Grid, Real>>;

/**
 * Set up a Poisson problem \Delta u = rhs, s.t. u(:, :, z_min) = bdZmin, u(:, :, z_max) = bdZmax,
 * to interpolate boundary values into the interior of the domain
 * @tparam Grid Type of the grid
 * @tparam Cardinality Cardinality of the fields
 * @param[in] grid Grid
 * @param[in] u Unknowns to solve for
 * @param[in] rhs RHS of the equation
 * @param[in] bd Dirichlet boundary mask (0: boundary, 1: interior)
 * @param[in] bdZmin Value of boundary at z == 0
 * @param[in] bdZmax Value of boundary at z == domainSize - 1
 */
template <typename Grid, typename Real, int Cardinality>
void setupPoissonProblem(const Grid&                               grid,
                         typename Grid::template Field<Real, 0>&   u,
                         typename Grid::template Field<Real, 0>&   rhs,
                         typename Grid::template Field<int8_t, 0>& bd,
                         std::array<Real, Cardinality>             bdZmin,
                         std::array<Real, Cardinality>             bdZmax)
{
    using namespace Neon;
    using namespace Neon::domain::details::eGrid;

    const Neon::index_3d dims = grid.getDimension();
    // u: Unknown to solve for
    u.forEachActiveCell([dims, bdZmin, bdZmax](const Neon::index_3d& idx,
                                           const int& card, Real& val) {
        if (idx.z == 0) {
            val = bdZmin[card];
        } else if (idx.z == dims.z - 1) {
            val = bdZmax[card];
        } else {
            val = 0.0;
        }
    });
    // bd: Dirichlet boundary
    bd.forEachActiveCell([dims](const Neon::index_3d& idx, const int& /*card*/, int8_t& val) {
        val = (idx.z == 0 || idx.z == dims.z - 1) ? Neon::solver::BoundaryCondition::Fixed : Neon::solver::BoundaryCondition::Free;
    });
    // rhs: Right hand side of Poisson equation
    rhs.forEachActiveCell([](const Neon::index_3d& /*idx*/, const int& /*card*/, Real& val) {
        val = 0.0;
    });

    // Move data to GPU if using CUDA backend
    if (grid.getBackend().devType() == DeviceType::CUDA) {
        u.updateDeviceData(0);
        rhs.updateDeviceData(0);
        bd.updateDeviceData(0);
    }
}

/**
 * Create a solver
 * @tparam Grid Type of the grid
 * @tparam Real Real number type (float or double)
 * @param[in] name Name of the solver
 * @param[in] grid Grid where the solver is run
 */
template <typename Grid, typename Real>
SolverPtr<Grid, Real> createSolver(std::string name)
{
    if (name == "CG") {
        return std::make_shared<Neon::solver::CG_t<Grid, Real>>();
    }
    throw std::runtime_error("Unknown solver name. Expected one of: 'CG', ");
}

/**
 * Setup the problem domain
 * @tparam Grid Type of the grid
 * @param[in] domainSize Size of the grid
 * @param[in] deviceSet Devices across which the grid will span
 */
template <typename Grid>
Grid createGrid(const Neon::Backend& /*backend*/, int /*domainSize*/)
{
    throw std::invalid_argument("Unsupported grid type. Expected Grid to be one of (eGrid_t, ...)");
}

// Specialization for eGrid_t
template <>
inline Neon::domain::details::eGrid::eGrid createGrid<Neon::domain::details::eGrid::eGrid>(const Neon::Backend& backend, int domainSize);

// Specialization for dGrid_t
template <>
inline Neon::dGrid createGrid<Neon::dGrid>(const Neon::Backend& backend, int domainSize);

// Specialization for bGrid_t
template <>
inline Neon::domain::bGrid createGrid<Neon::domain::bGrid>(const Neon::Backend& backend, int domainSize);

/**
 * Solve the poisson problem
 * @tparam Grid Type of the grid
 * @tparam Real Type of data in the fields
 * @tparam Cardinality Cardinality of elements in the fields
 * @param[in] domainSize Size of the grid
 * @param[in] backend Backend to use
 * @param[in] deviceSet Devices across which the grid will span
 * @param[in] solverName Name of the solver
 * @param[in] domainSize Size of the grid domain
 * @param[in] bdZmin Dirichlet boundary value at z = 0 of the grid
 * @param[in] bdZmax Dirichlet boundary value at z = domainSize - 1 of the grid
 * @param[in] maxIterations Maximum iterations for solver
 * @param[in] tolerance Tolerance for convergence check
 */
template <typename Grid, typename Real, int Cardinality>
auto testPoissonContainers(const Neon::Backend&           backend,
                           const std::string&             solverName,
                           int                            domainSize,
                           std::array<Real, Cardinality>  bdZmin,
                           std::array<Real, Cardinality>  bdZmax,
                           size_t                         maxIterations,
                           Real                           tolerance,
                           Neon::skeleton::Occ occE,
                           Neon::set::TransferMode transferE)
    -> std::pair<Neon::solver::SolverResultInfo, Neon::solver::SolverStatus>
{
    using namespace Neon;
    using namespace Neon::set;
    using namespace Neon::solver;

    // Setup problem
    Grid grid = createGrid<Grid>(backend, domainSize);

    auto u = grid.template newField<Real>("u", Cardinality, Real(0), DataUse::HOST_DEVICE);
    auto rhs = grid.template newField<Real>("rhs", Cardinality, Real(0), DataUse::HOST_DEVICE);
    auto bd = grid.template newField<int8_t>("bd", Cardinality, int8_t(0), DataUse::HOST_DEVICE);

    setupPoissonProblem<Grid, Real, Cardinality>(grid, u, rhs, bd, bdZmin, bdZmax);

    // Laplacian matvec operation
    auto L = std::make_shared<Neon::solver::LaplacianMatVec<Grid, Real>>(Real(1.0));

    // Create solver and solve problem
    auto solverBase = createSolver<Grid, Real>(solverName);
    auto solver = std::dynamic_pointer_cast<CG_t<Grid, Real>>(solverBase);
    solver->init(u);
    SolverParams params;
    params.maxIterations = maxIterations;
    params.toleranceAbs = tolerance;
    params.toleranceRel = 0.0;
    SolverResultInfo result;
    NEON_INFO(std::string("Backend") + backend.toString());
    const Neon::skeleton::Options   skeletonOpt(occE, transferE);
    SolverStatus                    status = solver->solve(L, u, rhs, bd, params, result, skeletonOpt);
    switch (status) {
        case SolverStatus::Converged:
            printf("SolverStatus is Converged\n");
            break;
        case SolverStatus::IterationLimit:
            printf("SolverStatus is IterationLimit\n");
            break;
        case SolverStatus::Error:
            printf("SolverStatus is Error\n");
            break;
        case SolverStatus::Iterating:
            printf("SolverStatus is Iterating\n");
            break;
        case SolverStatus::NumericalIssue:
            printf("SolverStatus is NumericalIssue\n");
            break;
    }
    printf("Start residual: %.11f\nEnd residual: %.11f\nIterations: %zd\nSolve time: %.11f ms\nTotal time: %.11f ms\n",
           result.residualStart, result.residualEnd, result.numIterations, result.solveTime, result.totalTime);
    return {result, status};
}

#define EXTERN_TEMPLATE_INST(GRID, REAL, CARD)                                             \
    extern template std::pair<Neon::solver::SolverResultInfo, Neon::solver::SolverStatus>  \
    testPoissonContainers<GRID, REAL, CARD>(const Neon::Backend& backend,                  \
                                            const std::string&   solverName,               \
                                            int domainSize, std::array<REAL, CARD> bdZmin, \
                                            std::array<REAL, CARD> bdZmax,                 \
                                            size_t maxIterations, REAL tolerance,          \
                                            Neon::skeleton::Occ occE, Neon::set::TransferMode transferE);

EXTERN_TEMPLATE_INST(Neon::dGrid, double, 1)
EXTERN_TEMPLATE_INST(Neon::dGrid, double, 3)
EXTERN_TEMPLATE_INST(Neon::domain::bGrid, double, 1)
EXTERN_TEMPLATE_INST(Neon::domain::bGrid, double, 3)
EXTERN_TEMPLATE_INST(Neon::domain::eGrid, double, 1)
EXTERN_TEMPLATE_INST(Neon::domain::eGrid, double, 3)

EXTERN_TEMPLATE_INST(Neon::dGrid, float, 1)
EXTERN_TEMPLATE_INST(Neon::dGrid, float, 3)
EXTERN_TEMPLATE_INST(Neon::domain::bGrid, float, 1)
EXTERN_TEMPLATE_INST(Neon::domain::bGrid, float, 3)
EXTERN_TEMPLATE_INST(Neon::domain::eGrid, float, 1)
EXTERN_TEMPLATE_INST(Neon::domain::eGrid, float, 3)

#undef EXTERN_TEMPLATE_INST