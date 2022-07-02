#include "Neon/solver/linear/krylov/CG.h"
#include "Neon/core/tools/metaprogramming/debugHelp.h"

#include "Neon/domain/dGrid.h"
#include "Neon/domain/bGrid.h"
#include "Neon/domain/interface/common.h"
#include "Neon/domain/eGrid.h"
#include "Neon/skeleton/Skeleton.h"
#include "Neon/solver/linear/krylov/CGContainers.h"

namespace Neon {
namespace solver {

template <typename Grid_ta, typename Real_ta>
void CG_t<Grid_ta, Real_ta>::doInit(Field& x)
{
    // Get the cardinality of x for creating internal fields
    const int cardinality = x.getPartition(Neon::DeviceType::CPU, 0, Neon::DataView::STANDARD).cardinality();

    m_p = x.getGrid().template newField<Real_ta>("p", cardinality, Real_ta(0.), Neon::DataUse::COMPUTE);
    m_s = x.getGrid().template newField<Real_ta>("s", cardinality, Real_ta(0.), Neon::DataUse::COMPUTE);
    m_r = x.getGrid().template newField<Real_ta>("r", cardinality, Real_ta(0.), Neon::DataUse::COMPUTE);
}

template <typename Grid_ta, typename Real_ta>
Real_ta CG_t<Grid_ta, Real_ta>::h_computeResidual(std::shared_ptr<matVec_t> A, Field& x, Field& b, BdField& bd)
{
    // Copy x into r so that Dirichlet BC values are in r. These will cancel out when Ax is subtracted from r.
    // then copy b into r in the free DOFs
    //
    // r := (bnd == 1) ? b : x
    // s := Ax
    // r := r - Ax  = r - s
    // rr = <r,r>
    // p := r

    auto& bk = this->h_getBackend(m_r);

    Neon::skeleton::Skeleton   skeleton(bk);
    auto                       delta_init = m_r.getGrid().template newPatternScalar<Real_ta>();

    skeleton.sequence({initR<Grid_ta, Real_ta>(m_r, x, b, bd),
                       A->matVec(x, bd, m_s),
                       AXPY<Grid_ta, Real_ta>(m_r, m_s),
                       m_r.getGrid().dot("init_rTr", m_r, m_r, delta_init),
                       copy<Grid_ta, Real_ta>(m_p, m_r)},
                      "CG::computeInitResidual");
    skeleton.run();
    bk.sync();

    return delta_init();
}

template <typename Grid_ta, typename Real_ta>
SolverStatus CG_t<Grid_ta, Real_ta>::solve(std::shared_ptr<matVec_t>        A,
                                           Field&                           x,
                                           Field&                           b,
                                           BdField&                         bd,
                                           const SolverParams&              params,
                                           SolverResultInfo&                result,
                                           const Neon::skeleton::Options& opt)
{
    // Make sure one time initializations have been done by the user by calling init()
    if (!this->isInit()) {
        NeonException exc("CG_t::solve");
        exc << "Attempting to call solve() before calling init()";
        NEON_THROW(exc);
    }
    Neon::Timer_ms timerSolution;
    Neon::Timer_ms timerTotal;

    // Preparations before the solve loop
    result.solverName = this->name();
    timerTotal.start();

    auto& bk = this->h_getBackend(x);

    // Compute initial residual
    bk.sync(Neon::Backend::mainStreamIdx);
    const Real_ta delta_init = h_computeResidual(A, x, b, bd);
    const Real_ta delta_init_sq = std::sqrt(delta_init);
    result.residualStart = delta_init_sq;

    // Store all residuals if requested
    if (params.needResiduals) {
        result.residuals.reserve(params.maxIterations);
        result.residuals.push_back(delta_init_sq);
    }


    // Solve loop
    size_t       iter = 0;
    SolverStatus status = SolverStatus::Error;

    Neon::skeleton::Skeleton cgIter(bk);

    auto delta_new = m_r.getGrid().template newPatternScalar<Real_ta>();
    auto delta_old = m_r.getGrid().template newPatternScalar<Real_ta>();
    auto pAp = m_r.getGrid().template newPatternScalar<Real_ta>();

    delta_new() = delta_init;

    // beta := delta_new/delta_old (computed on the fly inside updateP container)
    // p := r + beta*s (updateP container)
    // s := Ap (matVec container)
    // pAp := <p,s> (dot container)
    // alpha := delta_new/pAp (computed on the fly inside updateXandR container)
    // x := x + alpha*p (updateXandR container)
    // r := r - alpha*S (updateXandR container)
    // delta_old := delta_new (done inside updateXandR container)
    // delta_new := <r,r> (dot container)
    cgIter.sequence({updateP<Grid_ta, Real_ta>(m_p, m_r, delta_new(), delta_old()),
                     A->matVec(m_p, bd, m_s),
                     m_p.getGrid().dot("pAp", m_p, m_s, pAp),
                     updateXandR<Grid_ta, Real_ta>(x, m_r, m_p, m_s, delta_new(), pAp(), delta_old()),
                     m_r.getGrid().dot("rTr", m_r, m_r, delta_new)},
                    result.solverName, opt);
    delta_old() = 0;

    // Save the multi-GPU graph
    cgIter.ioToDot(result.solverName +
                       "_" + Neon::skeleton::OccUtils::toString(opt.occ()) +
                       "_" + Neon::set::TransferModeUtils::toString(opt.transferMode()),
                   "");

    bk.syncAll();
    timerSolution.start();


    for (iter = 0; iter < params.maxIterations; ++iter) {
        // Stop if converged/diverged/reached maximum iteration
        status = this->converged(delta_new(), delta_init_sq, iter, params);
        if (status == SolverStatus::Converged || status == SolverStatus::Error || status == SolverStatus::IterationLimit) {
            break;
        }

        cgIter.run();

        result.residualEnd = std::sqrt(delta_new());

        // Store residual norms if requested
        if (params.needResiduals) {
            result.residuals.push_back(result.residualEnd);
        }
    }


    // Post-processing after the solve loop
    timerSolution.stop();
    bk.sync();
    result.numIterations = iter;
    timerTotal.stop();
    result.solveTime = timerSolution.time();
    result.totalTime = timerTotal.time();
    return status;
}


template <typename Grid_ta, typename Real_ta>
void CG_t<Grid_ta, Real_ta>::reset()
{
    auto& bk = this->h_getBackend(m_r);

    Neon::skeleton::Skeleton skeleton(bk);

    skeleton.sequence({set<Grid_ta, Real_ta>(m_r, Real_ta(0.0)),
                       set<Grid_ta, Real_ta>(m_p, Real_ta(0.0)),
                       set<Grid_ta, Real_ta>(m_s, Real_ta(0.0))},
                      "CG::Reset");
    skeleton.run();
    bk.sync();
}


template class CG_t<Neon::domain::eGrid, double>;
template class CG_t<Neon::domain::eGrid, float>;
template class CG_t<Neon::domain::dGrid, double>;
template class CG_t<Neon::domain::dGrid, float>;
template class CG_t<Neon::domain::bGrid, double>;
template class CG_t<Neon::domain::bGrid, float>;

}  // namespace solver
}  // namespace Neon
