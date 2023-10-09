#include "Config.h"

#include "D3Q19.h"
#include "D3Q27.h"

#include "Neon/domain/bGrid.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/details/dGridSoA/dGridSoA.h"
#include "Neon/domain/eGrid.h"

#include "./Lbm.h"
#include "CellType.h"
#include "Metrics.h"
#include "Repoert.h"
namespace CavityTwoPop {

int backendWasReported = false;
// #include <fenv.h>
#include "/usr/include/fenv.h"
namespace details {
template <lbm::Method method_,
          Collision   CollisionType,
          typename Lattice_,
          typename Grid,
          typename Storage_,
          typename Compute_>
auto run(Config&                             config,
         Report&                             report,
         [[maybe_unused]] std::stringstream& code) -> void
{
    using Storage = Storage_;
    using Compute = Compute_;
    using Precision = Precision<Storage, Compute>;
    using Lattice = Lattice_;  // D3Q27<Precision>;

    code << "_" << config.deviceType << "_";
    for (auto const& id : config.devices) {
        code << id;
    }
    code << "_SS" << config.stencilSemanticCli.getStringOption();
    code << "_SF" << config.spaceCurveCli.getStringOption();
    code << "_TM" << config.transferModeCli.getStringOption();
    code << "_Occ" << config.occCli.getStringOption();
    code << "__";
    // using PopulationField = typename Grid::template Field<Storage, Lattice::Q>;

    // using PopField = typename Grid::template Field<typename Precision::Storage, Lattice::Q>;
    // using CellTypeField = typename Grid::template Field<CellType, 1>;

    // using Idx = typename PopField::Idx;
    // using RhoField = typename Grid::template Field<typename Precision::Storage, 1>;
    // using UField = typename Grid::template Field<typename Precision::Storage, 3>;

    Neon::double_3d ulid(1., 0., 0.);
    // Neon Grid and Fields initialization
    Neon::index_3d domainDim(config.N, config.N, config.N);

    Lbm<Grid, method_, CollisionType, Precision, Lattice> lbm(config,
                                                              report,
                                                              [](Neon::index_3d const&) { return true; });
    auto                                                  ulb = config.ulb;
    lbm.setBC([=] NEON_CUDA_HOST_DEVICE(Neon::index_3d const& globalIdx,
                                        NEON_OUT Storage      p[Lattice::Q],
                                        NEON_OUT CellType::Classification& cellClass) {
        typename Lattice::Precision::Storage popVal = 0;

        if (globalIdx.x == 0 || globalIdx.x == domainDim.x - 1 ||
            globalIdx.y == 0 || globalIdx.y == domainDim.y - 1 ||
            globalIdx.z == 0 || globalIdx.z == domainDim.z - 1) {
            cellClass = CellType::bounceBack;

            if (globalIdx.y == domainDim.y - 1) {
                cellClass = CellType::movingWall;
            }

            Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                using M = typename Lattice::template RegisterMapper<q>;
                if (globalIdx.y == domainDim.y - 1) {
                    popVal = -6. * Lattice::Registers::template getT<M::fwdRegQ>() * ulb *
                             (Lattice::Registers::template getVelocityComponent<M::fwdRegQ, 0>() * ulid.v[0] +
                              Lattice::Registers::template getVelocityComponent<M::fwdRegQ, 1>() * ulid.v[1] +
                              Lattice::Registers::template getVelocityComponent<M::fwdRegQ, 2>() * ulid.v[2]);
                } else {
                    popVal = 0;
                }
                p[q] = popVal;
            });
        } else {
            cellClass = CellType::bulk;
            Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                using M = typename Lattice::template RegisterMapper<q>;
                p[q] = Lattice::Registers::template getT<M::fwdRegQ>();
            });
        }
    });
    lbm.iterate();
}


template <Collision CollisionType, typename Lattice, typename Grid, typename Storage, typename Compute>
auto runFilterMethod(Config&            config,
                     Report&            report,
                     std::stringstream& testCode) -> void
{
    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);  // Enable all floating point exceptions but FE_INEXACT
    if (config.streamingMethod == "push") {
        if (config.devices.size() != 1) {
            NEON_THROW_UNSUPPORTED_OPERATION("We only support PUSH in a single device configuration for now.")
        }
        testCode << "_push";
        return run<lbm::Method::push, CollisionType, Lattice, Grid, Storage, Compute>(config, report, testCode);
    }
    if (config.streamingMethod == "pull") {
        testCode << "_pull";
        return run<lbm::Method::pull, CollisionType, Lattice, Grid, Storage, Compute>(config, report, testCode);
    }
    if (config.streamingMethod == "aa") {
        if (config.devices.size() != 1) {
            NEON_THROW_UNSUPPORTED_OPERATION("We only support AA in a single device configuration for now.")
        }
        testCode << "_aa";
        return run<lbm::Method::aa, CollisionType, Lattice, Grid, Storage, Compute>(config, report, testCode);
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename Lattice, typename Grid, typename Storage, typename Compute>
auto runFilterCollision(Config&            config,
                        Report&            report,
                        std::stringstream& testCode) -> void
{
    if (config.collisionCli.getOption() == Collision::bgk) {
        testCode << "_bgk";
        return runFilterMethod<Collision::bgk, Lattice, Grid, Storage, Compute>(config, report, testCode);
    }
    if (config.collisionCli.getOption() == Collision::kbc) {
        if (config.lattice != "d3q27" && config.lattice != "D3Q27") {
            Neon::NeonException e("runFilterCollision");
            e << "LBM kbc collision model only supports d3q27 lattice";
            NEON_THROW(e);
        }
        testCode << "_kbc";
        using L = D3Q27<Precision<Storage, Compute>>;
        if constexpr (std::is_same_v<Lattice, L>) {
            return runFilterMethod<Collision::kbc, Lattice, Grid, Storage, Compute>(config, report, testCode);
        }
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename Grid, typename Storage, typename Compute>
auto runFilterLattice(Config&            config,
                      Report&            report,
                      std::stringstream& testCode) -> void
{
    using P = Precision<Storage, Compute>;

    if (config.lattice == "d3q19" || config.lattice == "D3Q19") {
        testCode << "_D3Q19";
        using L = D3Q19<P>;
        return runFilterCollision<L, Grid, Storage, Compute>(config, report, testCode);
    }
    if (config.lattice == "d3q27" || config.lattice == "D3Q27") {
        testCode << "_D3Q27";
        using L = D3Q27<P>;
        return runFilterCollision<L, Grid, Storage, Compute>(config, report, testCode);
    }
    NEON_DEV_UNDER_CONSTRUCTION("Lattice type not supported. Available options: D3Q19 and D3Q27");
}


template <typename Grid, typename Storage>
auto runFilterComputeType(Config&            config,
                          Report&            report,
                          std::stringstream& testCode)
{
    if (config.computeTypeStr == "double") {
        testCode << "_Sdouble";
        return runFilterLattice<Grid, Storage, double>(config, report, testCode);
    }
    if (config.computeTypeStr == "float") {
        testCode << "_Sfloat";
        return runFilterLattice<Grid, Storage, float>(config, report, testCode);
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename Grid>
auto runFilterStoreType(Config&            config,
                        Report&            report,
                        std::stringstream& testCode)
    -> void
{
    if (config.storeTypeStr == "double") {
        testCode << "_Cdouble";
        return runFilterComputeType<Grid, double>(config, report, testCode);
    }
    if (config.storeTypeStr == "float") {
        testCode << "_Cfloat";
        return runFilterComputeType<Grid, float>(config, report, testCode);
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}
}  // namespace details

#ifdef NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS
constexpr bool skipTest = false;
#else
constexpr bool skipTest = false;
#endif

auto run(Config&            config,
         Report&            report,
         std::stringstream& testCode) -> void
{
    testCode << "___" << config.N << "_";
    testCode << "_numDevs_" << config.devices.size();

    if (config.gridType == "dGrid") {
        testCode << "_dGrid";
        return details::runFilterStoreType<Neon::dGrid>(config, report, testCode);
    }
    //    if (config.gridType == "eGrid") {
    //        if constexpr (!skipTest) {
    //            return details::runFilterStoreType<Neon::eGrid>(config, report);
    //        } else {
    //            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
    //        }
    //    }
    //    if (config.gridType == "bGrid" || config.gridType == "bGrid_8_8_8") {
    //        return details::runFilterStoreType<Neon::bGrid>(config, report);
    //    }
    if (config.gridType == "bGrid_4_4_4") {
        if constexpr (!skipTest) {
            testCode << "_bGrid_4_4_4";
            using Sblock = Neon::domain::details::bGrid::StaticBlock<4, 4, 4>;
            using Grid = Neon::domain::details::bGrid::bGrid<Sblock>;
            return details::runFilterStoreType<Grid>(config, report, testCode);
        } else {
            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
        }
    }
    //    if (config.gridType == "bGrid_8_8_8") {
    //        if constexpr (!skipTest) {
    //            using Sblock = Neon::domain::details::bGrid::StaticBlock<8, 8, 8>;
    //            using Grid = Neon::domain::details::bGrid::bGrid<Sblock>;
    //            return details::runFilterStoreType<Grid>(config, report, testCode);
    //        } else {
    //            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
    //        }
    //    }
    //    if (config.gridType == "bGrid_2_2_2") {
    //        if constexpr (!skipTest) {
    //            using Sblock = Neon::domain::details::bGrid::StaticBlock<2, 2, 2>;
    //            using Grid = Neon::domain::details::bGrid::bGrid<Sblock>;
    //            return details::runFilterStoreType<Grid>(config, report);
    //        } else {
    //            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
    //        }
    //    }
    //    if (config.gridType == "bGrid_32_8_4") {
    //        if constexpr (!skipTest) {
    //            using Sblock = Neon::domain::details::bGrid::StaticBlock<32, 8, 4>;
    //            using Grid = Neon::domain::details::bGrid::bGrid<Sblock>;
    //            return details::runFilterStoreType<Grid>(config, report);
    //        } else {
    //            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
    //        }
    //    }
    //    if (config.gridType == "bGrid_32_8_4") {
    //        if constexpr (!skipTest) {
    //            using Sblock = Neon::domain::details::bGrid::StaticBlock<32, 8, 4>;
    //            using Grid = Neon::domain::details::bGrid::bGrid<Sblock>;
    //            return details::runFilterStoreType<Grid>(config, report);
    //        } else {
    //            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
    //        }
    //    }
    //    if (config.gridType == "bGrid_32_2_8") {
    //        if constexpr (!skipTest) {
    //            using Sblock = Neon::domain::details::bGrid::StaticBlock<32, 2, 8>;
    //            using Grid = Neon::domain::details::bGrid::bGrid<Sblock>;
    //            return details::runFilterStoreType<Grid>(config, report);
    //        } else {
    //            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
    //        }
    //    }
    //    if (config.gridType == "bGrid_32_8_2") {
    //        if constexpr (!skipTest) {
    //            using Sblock = Neon::domain::details::bGrid::StaticBlock<32, 8, 2>;
    //            using Grid = Neon::domain::details::bGrid::bGrid<Sblock>;
    //            return details::runFilterStoreType<Grid>(config, report);
    //        } else {
    //            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
    //        }
    //    }
    //    if (config.gridType == "dGridSoA") {
    //        if constexpr (!skipTest) {
    //            return details::runFilterStoreType<Neon::domain::details::dGridSoA::dGridSoA>(config, report);
    //        } else {
    //            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
    //        }
    //    }
    NEON_THROW_UNSUPPORTED_OPERATION("Unknown grid type: " + config.gridType);
}
}  // namespace CavityTwoPop
