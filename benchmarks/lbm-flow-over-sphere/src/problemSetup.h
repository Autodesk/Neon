#include "Config.h"
#include "D3Q19.h"
#include "Neon/domain/dGrid.h"

#include "CellType.h"
#include "Metrics.h"
#include "Repoert.h"

struct LatticeStructure
{

   private:
    struct BBox
    {
        Neon::index_3d origin;
        Neon::index_3d max;
        Neon::index_3d len;
    } mBb;
    int    mUnitCellSize;
    double mUnitCellRadious;

    double cylinderSDF(const double P[3], const double center[3], const double K[3], double r)
    {
        // Compute the unit vector parallel to K
        double mag_K = std::sqrt(K[0] * K[0] + K[1] * K[1] + K[2] * K[2]);
        double u[3] = {K[0] / mag_K, K[1] / mag_K, K[2] / mag_K};

        // Project P onto the plane perpendicular to K and centered at C
        double Q[3] = {P[0] - (P[0] - center[0]) * u[0] - (P[1] - center[1]) * u[1] - (P[2] - center[2]) * u[2],
                       P[1] - (P[0] - center[0]) * u[1] - (P[1] - center[1]) * u[1] - (P[2] - center[2]) * u[2],
                       P[2] - (P[0] - center[0]) * u[2] - (P[1] - center[1]) * u[1] - (P[2] - center[2]) * u[2]};

        // Compute the distance from Q to C
        double d = std::sqrt((Q[0] - center[0]) * (Q[0] - center[0]) + (Q[1] - center[1]) * (Q[1] - center[1]) + (Q[2] - center[2]) * (Q[2] - center[2]));

        // Compute the signed distance
        if (d < r) {
            // Inside the cylinder
            double h = std::sqrt((P[0] - Q[0]) * (P[0] - Q[0]) + (P[1] - Q[1]) * (P[1] - Q[1]) + (P[2] - Q[2]) * (P[2] - Q[2]) - r * r);
            return -std::sqrt(h * h + d * d - r * r);
        } else {
            // Outside the cylinder
            return d - r;
        }
    }

    auto isInsideCrossBars(const Neon::double_3d p /** p is a point in the unit cell [0-1]^3 **/,
                           double                radius)
        -> bool
    {
        // Neon::double_3d center(.5);
        //        std::vector<Neon::double_3d> lowerCorners /** uppser corners of a unit cube */ {{1., 0., 0.}, {1., 1., 0.}, {0., 1., 0.}, {0., 0., 0.}};
        //        for (const auto& corner : lowerCorners) {
        //            Neon::double_3d opposite(corner.x == 0 ? 1 : 0,
        //                                     corner.y == 0 ? 1 : 0,
        //                                     corner.z == 0 ? 1 : 0);
        //            auto   direction = corner - opposite;
        //            double d = cylinderSDF(p.v, center.v, direction.v, radius);
        //            if (d <= 0)
        //                return false;
        //        }
        //        return true;

        bool xCylinder = std::pow(p.z - .5, 2) + std::pow(p.y - .5, 2) <= radius * radius;
        bool yCylinder = std::pow(p.z - .5, 2) + std::pow(p.x - .5, 2) <= radius * radius;
        bool zCylinder = std::pow(p.y - .5, 2) + std::pow(p.x - .5, 2) <= radius * radius;

        if (xCylinder ||
            yCylinder ||
            zCylinder) {
            return true;
        }
        return false;
    }

   public:
    LatticeStructure(Neon::index_3d bbOrigin,
                     Neon::index_3d bbLen,
                     int            uCellLen,
                     double         uCellRadisu)
    {
        mBb.origin = bbOrigin;
        mBb.len = bbLen;
        mUnitCellSize = uCellLen;
        mUnitCellRadious = uCellRadisu;
        mBb.max = mBb.origin + mBb.len;
    }

    bool isIn(Neon::index_3d p)
    {
        if (p.x < mBb.origin.x ||
            p.y < mBb.origin.y ||
            p.z < mBb.origin.z) {
            return false;
        }
        if (p.x > mBb.max.x ||
            p.y > mBb.max.y ||
            p.z > mBb.max.z) {
            return false;
        }
        Neon::index_3d reminder(p.x % mUnitCellSize,
                                p.y % mUnitCellSize,
                                p.z % mUnitCellSize);

        Neon::double_3d unitCellPosition = reminder.newType<double>() / (1.0 * mUnitCellSize);
        // if (unitCellPosition.x < .3) {
        //     return true;
        // }
        // return false;
        return isInsideCrossBars(unitCellPosition, mUnitCellRadious);
        //return false;
    }
};

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

    LatticeStructure latticeStructure({40, 6, 6},
                                      {config.N - 80, config.N - 10, config.N - 10}, 20, .25);

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
//        const auto point = idx.newType<double>();
//        const auto offset = std::pow(point.x - center.x, 2) +
//                            std::pow(point.y - center.y, 2) +
//                            std::pow(point.z - center.z, 2);
//        if (offset <= radius * radius) {
//            // we are in the sphere
//            return true;
//        }
//        return false;
        return latticeStructure.isIn(idx);

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

    std::cout << "Init flags..." << std::flush;
    flagField.forEachActiveCell([&](const Neon::index_3d& idx,
                                    const int&,
                                    CellType& flagVal) {
        flagVal.classification = CellType::undefined;
        flagVal.wallNghBitflag = 0;
        flagVal.classification = getBoundaryType(idx);

        bcTypeForDebugging.getReference(idx, 0) = static_cast<double>(flagVal.classification);
    });
    bcTypeForDebugging.ioToVtk("bcFlags", "cb", false);

    std::cout << "... [DONE]\n";

    std::cout << "Init Population..." << std::flush;
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
    std::cout << "... [DONE]\n";

    std::cout << "Update Device ..." << std::flush;
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
    std::cout << "... [DONE]\n";


    std::cout << "Init masks ...";
    auto computeWallNghMask = LbmContainers<Lattice, FieldPop, ComputeFP>::computeWallNghMask(flagField, flagField);
    computeWallNghMask.run(Neon::Backend::mainStreamIdx);
    std::cout << "... [DONE]\n";

    Neon::Real_3d<StorageFP> prescrivedVel(rhoPrescribedOutlet, 0, 0);

    std::cout << "Init zhoue ...";
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
