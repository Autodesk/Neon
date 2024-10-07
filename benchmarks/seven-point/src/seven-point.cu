#include "config.h"


#include "Neon/domain/Grids.h"
#include "Neon/domain/details/dGridDisg/dGrid.h"
#include "Neon/domain/details/dGridSoA/dGridSoA.h"
#include "Neon/domain/details/dGridDisg/dGrid.h"
#include "./simulation.h"
#include "parameters.h"

#include "metrics.h"
namespace CavityTwoPop {

int backendWasReported = false;
// #include <fenv.h>
// #include "/usr/include/fenv.h"

namespace details {
template <typename Parameters>
auto run(Config&                             config,
         Report&                             report,
         [[maybe_unused]] std::stringstream& code) -> void
{

    code << "_" << config.deviceType << "_";
    for (auto const& id : config.devices) {
        code << id;
    }
    code << "_SF" << config.spaceCurveCli.getStringOption();
    code << "_TM" << config.transferModeCli.getStringOption();
    code << "_Occ" << config.occCli.getStringOption();
    code << "_N" << config.N;
    code << "__";

    // Neon Grid and Fields initialization
    Neon::index_3d domainDim = config.N;

    Simulation<Parameters> sim(config,
                               report);

    sim.iterate();
}


template <typename P_>
auto runFilterComputeType(Config&            config,
                          Report&            report,
                          std::stringstream& testCode)
{
    if (config.computeTypeStr == "double") {
        testCode << "_Sdouble";
        using P = Parameters<P_::spaceDim, P_::fieldCard, typename P_::Grid,  double>;
        return run<P>(config, report, testCode);
    }
    if (config.computeTypeStr == "float") {
        testCode << "_Sfloat";
        using P = Parameters<P_::spaceDim, P_::fieldCard, typename P_::Grid, float>;
        return run<P>(config, report, testCode);
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}
template <typename P_>
auto filterBySpaceDim(Config&            config,
                          Report&            report,
                          std::stringstream& testCode)
{
    if (config.dimSpaceStr == "2D") {
        testCode << "_2D";
        using P = Parameters<2, 2, typename P_::Grid,  typename P_::Type>;
        return runFilterComputeType<P>(config, report, testCode);
    }
    if (config.dimSpaceStr == "3D") {
        testCode << "_3D";
        using P = Parameters<3, 3, typename P_::Grid, typename P_::Type>;
        return runFilterComputeType<P>(config, report, testCode);
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
        using Grid = Neon::dGrid;
        using P = Parameters<0, 0, Grid,  nullptr_t>;
        return details::filterBySpaceDim<P>(config, report, testCode);
    }
    if (config.gridType == "dGridDisg") {
        testCode << "_dGridDisg";
        using Grid = Neon::domain::details::disaggregated::dGrid::dGrid;
        using P = Parameters<0, 0, Grid,  nullptr_t>;
        return details::filterBySpaceDim<P>(config, report, testCode);
    }
    NEON_THROW_UNSUPPORTED_OPERATION("Unknown grid type: " + config.gridType);
}
}  // namespace CavityTwoPop
