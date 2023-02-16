#include "Config.h"
#include "D3Q19.h"
#include "Neon/domain/dGrid.h"

#include "CellType.h"
#include "Metrics.h"
#include "Repoert.h"

namespace CavityTwoPop {
template <typename FieldFlag,
          typename FieldPop,
          typename FieldPopBCType,
          typename StorageFP,
          typename ComputeFP>
auto problemSetup(Config&                              config,
                  FieldFlag&                           flagField,
                  FieldPop&                            popInField,
                  FieldPop&                            popOutField,
                  FieldPopBCType                       bcTypeForDebugging,
                  D3Q19Template<StorageFP, ComputeFP>& lattice,
                  Report&                              report) -> void
{
    using Lattice = D3Q19Template<StorageFP, ComputeFP>;

    const double radiusDomainLenRatio = 1.0 / 7;
    const double rhoPrescribedInlet = 1.0;
    const double rhoPrescribedOutlet = 1.005;

    const Neon::double_3d center = {config.N / 2.0, config.N / 2.0, config.N / 2.0};
    const double          radius = config.N * radiusDomainLenRatio;
    const auto&           t = lattice.t_vect;

    auto isFluidDomain =
        [&](const Neon::index_3d& idx)
        -> bool {
        if (idx < 0)
            return false;
        if (idx.x >= config.N ||
            idx.y >= config.N ||
            idx.z >= config.N) {
            return false;
        }
        const auto point = idx.newType<double>();
        const auto offset = std::pow(point.x - center.x, 2) +
                            std::pow(point.y - center.y, 2) +
                            std::pow(point.z - center.z, 2);
        if (offset <= radius * radius) {
            // we are in the sphere
            return false;
        }
        return true;
    };

    auto isInsideSphere =
        [&](const Neon::index_3d& idx) -> bool {
        if (idx.x < 0 ||
            idx.y < 0 ||
            idx.z < 0)
            return false;
        if (idx.x >= config.N ||
            idx.y >= config.N ||
            idx.z >= config.N) {
            return false;
        }
        const auto point = idx.newType<double>();
        const auto offset = std::pow(point.x - center.x, 2) +
                            std::pow(point.y - center.y, 2) +
                            std::pow(point.z - center.z, 2);
        if (offset <= radius * radius) {
            // we are in the sphere
            return true;
        }
        return false;
    };

    auto getBoundaryType =
        [&](const Neon::index_3d& idx) -> CellType::Classification {
        if (idx.z == 0 || idx.z == config.N - 1) {
            return CellType::Classification::bounceBack;
        }
        if (idx.y == 0 || idx.y == config.N - 1) {
            return CellType::Classification::bounceBack;
        }
        if (idx.x == 0 || idx.x == config.N - 1) {
            return CellType::Classification::bounceBack;
        }

        auto idEdge = [idx, config](int d1, int d2) {
            if ((idx.v[d1] == 1 && idx.v[d2] == 1) ||
                (idx.v[d1] == 1 && idx.v[d2] == config.N - 2) ||
                (idx.v[d1] == config.N - 2 && idx.v[d2] == 1) ||
                (idx.v[d1] == config.N - 2 && idx.v[d2] == config.N - 2)) {
                return true;
            }
            return false;
        };

        if (idEdge(0, 1)) {
            return CellType::Classification::bulk;
        }
        if (idEdge(0, 2)) {
            return CellType::Classification::bulk;
        }
        if (idEdge(1, 2)) {
            return CellType::Classification::bulk;
        }

        if (idx.x == 1) {
            return CellType::Classification::pressure;
        }
        if (idx.x == config.N - 2) {
            return CellType::Classification::velocity;
        }
        if (isInsideSphere(idx)) {
            return CellType::Classification::undefined;
        }
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                for (int k = -1; k < 2; k++) {
                    Neon::index_3d offset(i, j, k);
                    Neon::index_3d neighbour = idx + offset;
                    bool           isIn = isInsideSphere(neighbour);
                    if (isIn) {
                        return CellType::Classification::bounceBack;
                    }
                }
            }
        }
        return CellType::Classification::bulk;
    };


    // Problem Setup
    // 1. init all lattice to equilibrium


    Neon::index_3d dim(config.N, config.N, config.N);


    flagField.forEachActiveCell([&](const Neon::index_3d& idx,
                                    const int&,
                                    CellType& flagVal) {
        flagVal.classification = CellType::undefined;
        flagVal.wallNghBitflag = 0;
        flagVal.classification = getBoundaryType(idx);

        bcTypeForDebugging.getReference(idx, 0) = static_cast<double>(flagVal.classification);
        bcTypeForDebugging.ioToVtk("bcFlags", "cb", false);
    });

    flagField.hostHaloUpdate();

    // Population initialization
    popInField.forEachActiveCell([&](const Neon::index_3d& idx,
                                     const int&            k,
                                     StorageFP&            val) {
        val = t.at(k);
        if (flagField(idx, 0).classification == CellType::bounceBack) {
            val = 0;
        }
    });

    popOutField.forEachActiveCell([&](const Neon::index_3d& idx,
                                      const int&            k,
                                      StorageFP&            val) {
        val = t.at(k);
        if (flagField(idx, 0).classification == CellType::bounceBack) {
            val = 0;
        }
    });


    popInField.updateCompute(Neon::Backend::mainStreamIdx);
    popOutField.updateCompute(Neon::Backend::mainStreamIdx);
    flagField.updateCompute(Neon::Backend::mainStreamIdx);

    flagField.getBackend().syncAll();
    Neon::set::HuOptions hu(Neon::set::TransferMode::get,
                            false,
                            Neon::Backend::mainStreamIdx,
                            Neon::set::StencilSemantic::standard);

    flagField.haloUpdate(hu);
    flagField.getBackend().syncAll();

    auto computeWallNghMask = LbmContainers<Lattice, FieldPop, ComputeFP>::computeWallNghMask(flagField, flagField);
    computeWallNghMask.run(Neon::Backend::mainStreamIdx);

    Neon::Real_3d<StorageFP> prescrivedVel (rhoPrescribedOutlet,0,0);

    auto computeZouheGhostCells = LbmContainers<Lattice, FieldPop, ComputeFP>::computeZouheGhostCells(
        flagField,
        popInField,
        popOutField,
        rhoPrescribedInlet,
        prescrivedVel);

    computeZouheGhostCells.run(Neon::Backend::mainStreamIdx);

    flagField.getBackend().syncAll();
}

}  // namespace CavityTwoPop
