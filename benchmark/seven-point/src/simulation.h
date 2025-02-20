#include "./config.h"
#include "./metrics.h"
#include "Neon/Neon.h"
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"
#include "Neon/skeleton/Skeleton.h"
#include "containers.h"
#include "field.h"
#include "parameters.h"
int backendWasReported = false;

template <typename Parameters>
struct Simulation
{
    using Type = typename Parameters::Type;
    using Field = typename Parameters::Field;
    using Grid = typename Parameters::Grid;
    static constexpr int spaceDim = Parameters::spaceDim;
    static constexpr int fieldCard = Parameters::fieldCard;

    // using CommonContainerFactory = common::ContainerFactory<Precision, Lattice, Grid>;

    Simulation(Config& config,
               Report& report)
    {
        configurations = config;
        reportPtr = &report;


        // Setting the backend
        Neon::Backend bk = [&] {
            if (config.deviceType == "cpu") {
                Neon::Backend bk(config.devices, Neon::Runtime::openmp);
                return bk;
            }
            if (config.deviceType == "gpu") {
                Neon::Backend bk(config.devices, Neon::Runtime::stream);
                return bk;
            }
            Neon::NeonException exce("run");
            exce << config.deviceType << " is not a supported option as device type";
            NEON_THROW(exce);
        }();

        bk.toReport(report.helpGetReport(), nullptr);

        auto [gridInitClockStart, notcare] = metrics::restartClock(bk, true);

        // Setting the grid
        grid = Grid(
            bk, config.N,
            [&](const Neon::index_3d& p) { return true; },
            Neon::domain::Stencil::s7_Laplace_t(),
            1.0, 0.0,
            config.spaceCurveCli.getOption());

        grid.toReport(report.helpGetReport(), false);


        // Allocating cell type field
        fields = Fields<Parameters>(grid);

        metrics::recordGridInitMetrics(bk, *reportPtr, gridInitClockStart);

        helpPrep();
    }


    auto helpPrep() -> void
    {
        auto i0 = ContainerFactory<Parameters>::iteration(fields.get(0), fields.get(1), Type(1.0));
        auto i1 = ContainerFactory<Parameters>::iteration(fields.get(1), fields.get(0), Type(1.0));

        skeleton[0] = Neon::skeleton::Skeleton(grid.getBackend());
        skeleton[1] = Neon::skeleton::Skeleton(grid.getBackend());

        Neon::skeleton::Options options(Neon::skeleton::Occ::standard,
                                        Neon::set::TransferMode::get);

        std::vector<Neon::set::Container> s0, s1;
        s0.push_back(i0);
        s1.push_back(i1);
        skeleton[0].sequence(s0, "odd", options);
        skeleton[1].sequence(s1, "even", options);
    }

    auto iterate() -> void
    {
        // Iteration keep track of all iterations
        // clock_iter keeps tracks of the iteration done after the last clock reset
        std::cout << "Starting main Solver loop." << std::endl;
        int   target_skeleton = 0;
        auto& bk = grid.getBackend();
        auto [start, clock_iter] = metrics::restartClock(bk, true);
        int time_iter = 0;
        // Reset the clock, to be used when a benchmark simulation is executed.
        tie(start, clock_iter) = metrics::restartClock(bk, true);

        for (time_iter = 0; time_iter < configurations.benchMaxIter; ++time_iter) {
            if ((configurations.vti > 1) && ((time_iter % configurations.vti) == 0)) {
                bk.syncAll();
                helpExportVti();
            }

            if (configurations.benchmark && time_iter == configurations.benchIniIter) {
                std::cout << "Warm up completed (" << time_iter << " iterations ).\n"
                          << "Starting benchmark step ("
                          << configurations.benchMaxIter - configurations.benchIniIter << " iterations)."
                          << std::endl;
                tie(start, clock_iter) = metrics::restartClock(bk, false);
            }

            skeleton[target_skeleton].run();

            ++clock_iter;
            target_skeleton = target_skeleton == 0 ? 1 : 0;
        }
        std::cout << "Iterations completed." << std::endl;
        metrics::recordMetrics(bk, configurations, *reportPtr, start, clock_iter);
    }

    auto helpExportVti() -> void
    {
        //        grid.getBackend().syncAll();
        //        auto& pop = pFieldList.at(iterationPhase.getOutputIdx());
        //        bool  done = false;
        //        if constexpr (method == lbm::Method::push) {
        //            auto computeRhoAndU = ContainerFactory::Push::computeRhoAndU(pop, cellFlagField, rho, u);
        //            computeRhoAndU.run(Neon::Backend::mainStreamIdx);
        //            done = true;
        //        }
        //        if constexpr (method == lbm::Method::pull) {
        //            pop.newHaloUpdate(Neon::set::StencilSemantic::standard,
        //                              Neon::set::TransferMode::get,
        //                              Neon::Execution::device)
        //                .run(Neon::Backend::mainStreamIdx);
        //            auto computeRhoAndU = ContainerFactory::Pull::computeRhoAndU(pop, cellFlagField, rho, u);
        //            computeRhoAndU.run(Neon::Backend::mainStreamIdx);
        //            done = true;
        //        }
        //        if constexpr (method == lbm::Method::aa) {
        //            if (iterationPhase.getPhase() == IterationPhase::Phase::even) {
        //                auto computeRhoAndU = ContainerFactory::AA::Even::computeRhoAndU(pop, cellFlagField, rho, u);
        //                computeRhoAndU.run(Neon::Backend::mainStreamIdx);
        //            } else {
        //                auto computeRhoAndU = ContainerFactory::AA::Odd::computeRhoAndU(pop, cellFlagField, rho, u);
        //                computeRhoAndU.run(Neon::Backend::mainStreamIdx);
        //            }
        //            done = true;
        //        }
        //        if (!done) {
        //            NEON_DEV_UNDER_CONSTRUCTION("helpExportVti");
        //        }
        //        u.updateHostData(Neon::Backend::mainStreamIdx);
        //        rho.updateHostData(Neon::Backend::mainStreamIdx);
        //        // pop.updateHostData(Neon::Backend::mainStreamIdx);
        //        grid.getBackend().sync(Neon::Backend::mainStreamIdx);
        //
        //        size_t      numDigits = 5;
        //        std::string iterIdStr = std::to_string(iterationPhase.getCounter());
        //        iterIdStr = std::string(numDigits - std::min(numDigits, iterIdStr.length()), '0') + iterIdStr;
        //
        //        // pop.ioToVtk("pop_" + iterIdStr, "pop", false);
        //        u.ioToVtk("u_" + iterIdStr, "u", false, Neon::IoFileType::BINARY);
        //        rho.ioToVtk("rho_" + iterIdStr, "rho", false, Neon::IoFileType::BINARY);
        //        cellFlagField.template ioToVtk<int>("cellFlagField_" + iterIdStr, "flag", false);
    }


    Config                   configurations;
    bool                     prepDone = false;
    Grid                     grid;
    Fields<Parameters>       fields;
    Neon::skeleton::Skeleton skeleton[2];
    Report*                  reportPtr;
};
