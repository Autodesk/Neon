
#include <map>
#include "gtest/gtest.h"
#include "sPt_common.h"
#include "sPt_geometry.h"
#include "sPt_laplacian.h"

#include "Neon/core/core.h"
#include "Neon/core/tools/clipp.h"
#include "Neon/core/tools/io/ioToVti.h"

#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"

#include "Neon/skeleton/Skeleton.h"


using namespace Neon;
using namespace Neon::domain;


/**
 *
 * @tparam T
 * @param length
 * @param cardinality
 * @param backend
 */
template <typename Grid_ta, typename T_ta>
double filterAverage(TestConfigurations& config)
{

    storage_t<Grid_ta, T_ta> storage(config.m_dim, config.m_geo, config.m_nGPUs, 1, config.m_backend);
    storage.initLinearly();

    auto& inF = storage.Xf;
    auto& outF = storage.Yf;
    auto& inD = storage.Xd;
    auto& outD = storage.Yd;

    config.m_backend.syncAll();
    // storage.ioToVti("Before_" + std::to_string(0));

    Neon::skeleton::Skeleton skl(storage.m_backend);
    skl.sequence({sk::axpy(inF.constSelf(), T_ta(1.0), outF),
                  sk::laplacianFilter(outF.constSelf(), inF)},
                 "axpy_laplacian", config.getSklOpt());

    {  // WARMING UP
        skl.run();
        config.m_backend.syncAll();
    }

    {  // TIMING
        {
            for (int i = 0; i < config.m_nIterations; i++) {
                config.m_timer.start();
                skl.run();
                storage.m_backend.sync();
                config.m_timer.stop();
                NEON_INFO("Time per iteration {}", config.m_timer.time());
            }
        }
        skl.ioToDot("test.dot");
    }

    {  // CORRECTNESS TEST
        if (config.m_compare) {
            storage.axpy_f(inD, T_ta(1.0), outD);
            storage.laplacianFilter_f(outD, inD, storage.m_stencil.neighbours());
            for (int i = 0; i < config.m_nIterations; i++) {
                storage.axpy_f(inD, T_ta(1.0), outD);
                storage.laplacianFilter_f(outD, inD, storage.m_stencil.neighbours());
            }
            // storage.ioToVti("Compare_" + std::to_string(0));
            bool isOk = storage.compare(outD, outF);
            if (!isOk) {
                NeonException ex("Test");
                ex << "TEST FAILED";
                NEON_THROW(ex);
            }
        }
    }
    return config.m_timer.time();
}

[[maybe_unused]] void runAllConfig(std::function<void(TestConfigurations&)> f,
                                   TestConfigurations                       config)
{
    std::vector<int> nGpuTest;
    const int        maxnGPUs = Neon::set::DevSet::maxSet().setCardinality();
    if (maxnGPUs < config.m_nGPUs) {
        //     NEON_THROW_UNSUPPORTED_OPERATION("Not enought GPUs");
    }

    for (int i = 0; i < config.m_nGPUs; i++) {
        nGpuTest.push_back(i + 1);
    }

    auto runtime = Neon::Runtime::stream;

    Neon::Report report("Skeleton - AXPY, Laplace");

    auto h_computeTestName = [&]() {
        std::string name = std::string("Skeleton_Filter_") +
                           Neon::skeleton::OccUtils::toString(config.m_optSkelOCC) + "_" +
                           Neon::set::TransferModeUtils::toString(config.m_optSkelTransfer) + "_" +
                           "_x" + std::to_string(config.m_dim.x) +
                           "_y" + std::to_string(config.m_dim.y) +
                           "_z" + std::to_string(config.m_dim.z);
        return name;
    };

    for (const auto& ngpu : nGpuTest) {
        for (auto& geo : {topologies_e::FullDomain}) {

            std::vector<int> ids;
            for (int i = 0; i < ngpu; i++) {
                ids.push_back(i % maxnGPUs);
            }
            if (ids.size() != size_t(config.m_nGPUs))
                continue;
            Neon::Backend backend(ids, runtime);
            config.m_backend = backend;

            std::cout
                << " [Dim] " << config.m_dim
                << " [Topology] " << topologiesToString(geo)
                << " [nGpus] " << config.m_nGPUs
                << " [Backend] " << backend.toString()
                << " [compare] " << (config.m_compare ? "Active" : "Disabled")
                << std::endl;


            Geometry geometry(geo, config.m_dim);
            report.addMember("gridType", "eGrid");
            report.addMember("Backend", backend.toString());
            report.addMember("devSet", backend.devSet().toString());
            config.storeInforInReport(report, maxnGPUs);

            std::vector<double> times;
            for (int r = 0; r < config.m_nRepetitions; r++) {
                f(config);
                times.push_back(config.m_timer.time());
            }
            report.addMember("timeToSolution_ms", times);
        }
    }
    report.write(h_computeTestName(), false);
}