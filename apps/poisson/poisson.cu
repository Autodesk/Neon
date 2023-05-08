#include <array>

#include "Neon/Neon.h"
#include "Neon/domain/dGrid.h"
#include "Neon/solver/linear/krylov/CG.h"
#include "Neon/solver/linear/matvecs/LaplacianMatVec.h"

template <typename Grid, typename T>
class MxV : public Neon::solver::MatVec<Grid, T>
{
    T m_h;

   public:
    using FieldT = typename Grid::template Field<T>;
    using bdFieldT = typename Grid::template Field<int8_t>;

    MxV(T h)
        : Neon::solver::MatVec<Grid, T>(), m_h(h) {}

    virtual Neon::set::Container matVec(const FieldT&   input,
                                        const bdFieldT& boundary,
                                        FieldT&         output) override
    {
        auto container =
            input.getGrid().getContainer("MxV", [&](Neon::set::Loader& L) {
                auto&   inp = L.load(input, Neon::Compute::STENCIL);
                auto&   bnd = L.load(boundary);
                auto&   out = L.load(output);
                auto    step = m_h;
                const T invh2 = T(1.0) / (step * step);

                return [=] NEON_CUDA_HOST_DEVICE(
                           const typename FieldT::Cell& idx) mutable {
                    constexpr T defaultVal = 0;

                    const int cardinality = inp.cardinality();

                    for (int c = 0; c < cardinality; ++c) {
                        const T center = inp(idx, c);

                        if (bnd(idx, c) == 0) {
                            out(idx, c) = center;
                        } else {
                            T   sum(0.0);
                            int numNeighb = 0;

                            for (int8_t nghIdx = 0; nghIdx < 6; ++nghIdx) {
                                auto neighbor = inp.nghVal(idx, nghIdx, c, defaultVal);
                                if (neighbor.isValid) {
                                    ++numNeighb;
                                    sum += neighbor.value;
                                }
                            }

                            out(idx, c) = (-sum + static_cast<T>(numNeighb) * center) * invh2;
                        }
                    }
                };
            });
        return container;
    }
};


template <typename Grid, typename T, int Cardinality>
void testPoisson(const Neon::Backend&             backend,
                 const Neon::index_3d             domainSize,
                 const std::array<T, Cardinality> bdZmin,
                 const std::array<T, Cardinality> bdZmax,
                 const size_t                     maxIterations,
                 const T                          tolerance)
{
    // create a grid
    Grid grid(
        backend, domainSize, [](Neon::index_3d idx) { return true; },
        Neon::domain::Stencil::s7_Laplace_t());


    // u: Unknown to solve for
    auto u = grid.template newField<T>("u", Cardinality, 0);
    u.forEachActiveCell([&](const Neon::index_3d& idx, const int& card, T& val) {
        if (idx.z == 0) {
            val = bdZmin[card];
        } else if (idx.z == domainSize.z - 1) {
            val = bdZmax[card];
        } else {
            val = 0.0;
        }
    });

    u.ioToVtk("poisson_init", "u");


    // rhs: Right hand side of Poisson equation
    auto bd = grid.template newField<int8_t>("bd", Cardinality, 0);
    bd.forEachActiveCell([=](const Neon::index_3d& idx, const int&, int8_t& val) {
        val = (idx.z == 0 || idx.z == domainSize.z - 1)
                  ? Neon::solver::BoundaryCondition::Fixed
                  : Neon::solver::BoundaryCondition::Free;
    });

    // bd: Dirichlet boundary
    auto rhs = grid.template newField<T>("rhs", Cardinality, 0);
    rhs.forEachActiveCell(
        [=](const Neon::index_3d& idx, const int& card, double& val) { val = 0.0; });


    // Move data to GPU
    u.updateDeviceData(0);
    rhs.updateDeviceData(0);
    bd.updateDeviceData(0);


    // Laplacian matvec operation
    auto L = std::make_shared<MxV<Grid, T>>(1.0);
    //auto L = std::make_shared<Neon::solver::LaplacianMatVec<Grid, T>>(1.0);

    //  Create solver and solve problem
    Neon::solver::CG_t<Grid, T> solver;

    solver.init(u);

    Neon::solver::SolverParams params;
    params.maxIterations = maxIterations;
    params.toleranceAbs = tolerance;
    params.toleranceRel = 0.0;
    //params.needResiduals = true;

    Neon::solver::SolverResultInfo result;
    Neon::solver::SolverStatus     status = solver.solve(L, u, rhs, bd, params, result);
    printf(
        "Backend: %s\nStart residual: %.4f\nEnd residual: %.4f\nIterations: "
        "%zd\nSolve time: %.4f ms\nTotal time: %.4f ms\n",
        backend.toString().c_str(), result.residualStart, result.residualEnd,
        result.numIterations, result.solveTime, result.totalTime);
    if (status == Neon::solver::SolverStatus::Error || result.residualEnd >= tolerance) {
        printf("Solver Failed!!\n ");
    }

    // Plotting
    printf("Updating and exporting to VTI...\n");
    u.updateHostData(0);
    u.ioToVtk("poisson", "u");
}


int main(int argc, char** agrv)
{
    // Backend
    Neon::init();
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {

        auto             runtime = Neon::Runtime::stream;
        std::vector<int> gpu_ids{0};
        Neon::Backend    backend(gpu_ids, runtime);

        std::array<double, 1> bdZMin{-20.0};
        std::array<double, 1> bdZMax{20.0};

        Neon::index_3d domain_size(128, 128, 128);
        size_t         max_iterations = 1000;
        double         tolerance = 1e-10;

        testPoisson<Neon::dGrid, double, 1>(backend, domain_size, bdZMin, bdZMax, max_iterations, tolerance);
    }

    return 0;
}