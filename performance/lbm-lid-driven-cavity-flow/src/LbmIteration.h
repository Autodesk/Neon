#include "CellType.h"
#include "D3Q19.h"
#include "Neon/Neon.h"
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"

template <typename PopulationField,
          typename LbmComputeType>
struct LbmIteration
{
    using LbmStoreType = typename PopulationField::Type;
    using CellTypeField = typename PopulationField::Grid::template Field<CellType, 1>;
    using D3Q19 = D3Q19<LbmStoreType, LbmComputeType>;

    static auto
    iterationUnrolled(Neon::set::TransferSemantic stencilSemantic,
                      const D3Q19&                d3q19,
                      const PopulationField&      fInField /*!   inpout population field */,
                      PopulationField&            fOutField /*!  output Population field */,
                      const CellTypeField&        cellTypeField /*!       Cell type field     */,
                      const LatticeParameters&    lpGlobal /*!         Lattice parameters  */,
                      const LbmComputeType        omega /*! LBM omega parameter */)
        -> Neon::set::Container
    {
        Neon::set::Container container = fInField.grid().getContainer(
            "LBM_iteration",
            [&, omega](Neon::set::Loader& L) -> auto {
                auto&       fIn = L.load(fInField, Neon::Compute::STENCIL, stencilSemantic);
                auto&       fOut = L.load(fOutField);
                const auto& cellInfo = L.load(cellTypeField);

                const auto&               Lp = L.load(lpGlobal);

                return [=] NEON_CUDA_HOST_DEVICE(const PopulationField::Cell& cell) mutable {
                    const CellType cellType = cellInfo(cell);
                    if (cellType.classification == CellType::bulk) {
                        LbmComputeType popIn[D3Q19::q];
                        Krn::loadPop(s19->s, i, flagData.bounceBackNghBitflag, flag, fIn, popIn);

                        const auto [rho, u] = Krn::macroscopic(i, popIn);

                        const Element usqr = 1.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
                        Krn::collideAndStreamBgkUnrolled(s19->s,
                                                         i,
                                                         popIn,
                                                         fIn,
                                                         Lp,
                                                         rho, u,
                                                         usqr, omega,
                                                         flagData.bounceBackNghBitflag,
                                                         flag,
                                                         fOut);
                    }
                };
            });
        return Kontainer;
    }
};