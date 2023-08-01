#include "Config.h"
#include "D3Q19.h"
#include "Neon/domain/bGrid.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/details/dGridSoA/dGridSoA.h"
#include "Neon/domain/eGrid.h"

#include "./Lbm.h"
#include "CellType.h"
#include "LbmSkeleton.h"
#include "Metrics.h"
#include "Repoert.h"
namespace CavityTwoPop {

int backendWasReported = false;

namespace details {
template <lbm::Method method_,
          typename Grid,
          typename Storage_,
          typename Compute_>
auto run(Config& config,
         Report& report) -> void
{
    using Storage = Storage_;
    using Compute = Compute_;
    using Precision = Precision<Storage, Compute>;
    using Lattice = D3Q19<Precision>;
    using PopulationField = typename Grid::template Field<Storage, Lattice::Q>;

    using PopField = typename Grid::template Field<typename Precision::Storage, Lattice::Q>;
    using CellTypeField = typename Grid::template Field<CellType, 1>;

    using Idx = typename PopField::Idx;
    using RhoField = typename Grid::template Field<typename Precision::Storage, 1>;
    using UField = typename Grid::template Field<typename Precision::Storage, 3>;

    Neon::double_3d ulid(1., 0., 0.);
    // Neon Grid and Fields initialization
    Neon::index_3d dim(config.N, config.N, config.N);

    Lbm<Grid, method_, Precision, Lattice> lbm(config,
                                               report,
                                               dim,
                                               [](Neon::index_3d const&) { return true; });
    auto ulb = config.ulb;
    auto domainDim = dim;
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
                    popVal = -6. * Lattice::Registers::template getT<M::fwdRegIdx>() * ulb *
                             (Lattice::Registers::template getDirection<M::fwdRegIdx>().v[0] * ulid.v[0] +
                              Lattice::Registers::template getDirection<M::fwdRegIdx>().v[1] * ulid.v[1] +
                              Lattice::Registers::template getDirection<M::fwdRegIdx>().v[2] * ulid.v[2]);
                } else {
                    popVal = 0;
                }
                p[q] = popVal;
            });
        } else {
            cellClass = CellType::bulk;
            Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                using M = typename Lattice::template RegisterMapper<q>;
                p[q] = Lattice::Registers::template getT<M::fwdRegIdx>();
            });
        }
    });
    lbm.iterate();
}

template <typename Grid, typename Storage, typename Compute>
auto runFilterMethod(Config& config, Report& report) -> void
{
    return run<lbm::Method::push, Grid, Storage, double>(config, report);
}

template <typename Grid, typename Storage>
auto runFilterComputeType(Config& config, Report& report) -> void
{
    if (config.computeType == "double") {
        return runFilterMethod<Grid, Storage, double>(config, report);
    }
    //    if (config.computeType == "float") {
    //        return run<Grid, Storage, float>(config, report);
    //    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename Grid>
auto runFilterStoreType(Config& config,
                        Report& report)
    -> void
{
    if (config.storeType == "double") {
        return runFilterComputeType<Grid, double>(config, report);
    }
    //    if (config.storeType == "float") {
    //        return runFilterComputeType<Grid, float>(config, report);
    //    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}
}  // namespace details

#ifdef NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS
constexpr bool skipTest = false;
#else
constexpr bool skipTest = false;
#endif

auto run(Config& config,
         Report& report) -> void
{
    if (config.gridType == "dGrid") {
        return details::runFilterStoreType<Neon::dGrid>(config, report);
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
    //    if (config.gridType == "bGrid_4_4_4") {
    //        if constexpr (!skipTest) {
    //            using Sblock = Neon::domain::details::bGrid::StaticBlock<4, 4, 4>;
    //            using Grid = Neon::domain::details::bGrid::bGrid<Sblock>;
    //            return details::runFilterStoreType<Grid>(config, report);
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
