#include "Neon/core/core.h"
#include "Neon/domain/interface/LaunchConfig.h"
#include "Neon/solver/linear/matvecs/LaplacianMatVec.h"

namespace Neon {
namespace solver {

template <typename Grid, typename Real>
inline Neon::set::Container LaplacianMatVec<Grid, Real>::matVec(const Field&   input,
                                                                      const bdField& boundary,
                                                                      Field&         output)
{
    Real stepSize = m_h;

    auto cont = input.getGrid().getContainer("Laplacian", [&, stepSize](Neon::set::Loader& L) {
        auto& inp = L.load(input, Neon::Compute::STENCIL);
        auto& bnd = L.load(boundary);
        auto& out = L.load(output);

        // Precompute 1/h^2
        const Real invh2 = Real(1.0) / (stepSize * stepSize);

        return [=] NEON_CUDA_HOST_DEVICE(const typename Grid::template Field<Real>::Cell& cell) mutable {
            const int cardinality = inp.cardinality();

            // Iterate through each element's cardinality
            for (int c = 0; c < cardinality; ++c) {
                const Real center = inp(cell, c);
                if (bnd(cell, c) == 0) {
                    out(cell, c) = center;
                } else {
                    Real       sum(0.0);
                    int           numNeighb = 0;
                    const Real defaultVal{0};

                    auto checkNeighbor = [&sum, &numNeighb](Neon::domain::NghInfo<Real>& neighbor) {
                        if (neighbor.isValid) {
                            ++numNeighb;
                            sum += neighbor.value;
                        }
                    };
                    // Laplacian stencil operates on 6 neighbors (assuming 3D)
                    if constexpr (std::is_same<Grid, Neon::domain::internal::eGrid::eGrid>::value) {
                        for (int8_t nghIdx = 0; nghIdx < 6; ++nghIdx) {
                            auto neighbor = inp.nghVal(cell, nghIdx, c, defaultVal);
                            checkNeighbor(neighbor);
                        }
                    } else {
                        typename Grid::template Field<Real, 0>::Partition::nghIdx_t ngh(0, 0, 0);

                        //+x
                        ngh.x = 1;
                        ngh.y = 0;
                        ngh.z = 0;
                        auto neighbor = inp.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor);

                        //-x
                        ngh.x = -1;
                        ngh.y = 0;
                        ngh.z = 0;
                        neighbor = inp.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor);

                        //+y
                        ngh.x = 0;
                        ngh.y = 1;
                        ngh.z = 0;
                        neighbor = inp.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor);

                        //-y
                        ngh.x = 0;
                        ngh.y = -1;
                        ngh.z = 0;
                        neighbor = inp.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor);

                        //+z
                        ngh.x = 0;
                        ngh.y = 0;
                        ngh.z = 1;
                        neighbor = inp.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor);

                        //-z
                        ngh.x = 0;
                        ngh.y = 0;
                        ngh.z = -1;
                        neighbor = inp.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor);
                    }
                    out(cell, c) = (-sum + static_cast<Real>(numNeighb) * center) * invh2;
                }
            }
        };
    });
    return cont;
}

// Template instantiations
template class LaplacianMatVec<Neon::domain::eGrid, double>;
template class LaplacianMatVec<Neon::domain::eGrid, float>;
template class LaplacianMatVec<Neon::domain::dGrid, double>;
template class LaplacianMatVec<Neon::domain::dGrid, float>;
template class LaplacianMatVec<Neon::domain::bGrid, double>;
template class LaplacianMatVec<Neon::domain::bGrid, float>;

}  // namespace solver
}  // namespace Neon