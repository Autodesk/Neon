
#include "MapSequence.h"
#include "RunTest.h"

#include <map>

namespace Test {
namespace help {


template <typename Grid,
          typename Type,
          int Cardinality>
auto getTestData(Cli::UserData& userData)
    -> Neon::domain::tool::testing::TestData<Grid, Type, Cardinality>
{
    Neon::MemoryOptions memoryOptions;

    Neon::Backend bk(userData.deviceIds,
                     [&] {
                         if (userData.deviceType.getOption() == Neon::DeviceType::CUDA) {
                             return Neon::Runtime::stream;
                         }
                         return Neon::Runtime::openmp;
                     }());

    Neon::domain::tool::testing::TestData<Grid, Type, Cardinality> testData(
        bk,
        userData.dimensions,
        1,
        memoryOptions,
        userData.targetGeometry.getOption(),
        .8,
        .5,
        Neon::domain::Stencil::s7_Laplace_t(),
        Type(0));

    testData.resetValuesToLinear(1);

    return testData;
}

template <typename Grid,
          typename Type,
          int Cardinality,
          typename TestData>
auto getSequence(Cli::UserData& userData,
                 TestData&      testData)
    -> std::vector<Neon::set::Container>
{
    if (userData.targetApp.getOption() == Cli::Apps::map) {
        return Test::sequence::mapAxpy<Grid, Type, Cardinality>(userData, testData);
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename Grid,
          typename Type,
          int Cardinality,
          typename TestData>
auto runGoldenFilter(Cli::UserData& userData,
                     TestData&      testData)
    -> void
{
    if (userData.targetApp.getOption() == Cli::Apps::map) {
        return Test::sequence::mapAxpyGoldenRun<Grid, Type, Cardinality>(userData, testData);
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename Grid,
          typename Type,
          int Cardinality,
          typename TestData>
auto runGolden(Cli::UserData& userData,
               TestData&      testData)
    -> bool
{
    using FieldNames = Neon::domain::tool::testing::FieldNames;

    testData.updateIO();
    for (int i = 0; i < userData.nIterations + userData.warmupIterations; i++) {
        help::runGoldenFilter<Grid, Type, Cardinality>(userData, testData);
    }
    bool isXok = testData.compare(FieldNames::X);
    bool isYok = testData.compare(FieldNames::Y);
    bool isZok = testData.compare(FieldNames::Z);
    bool isWok = testData.compare(FieldNames::W);
    return isXok && isYok && isZok && isWok;
}

auto getSkeletonOption(Cli::UserData& userData)
    -> Neon::skeleton::Options
{
    Neon::skeleton::Options options(userData.occModel.getOption(), Neon::set::TransferMode::get);
    return options;
}

template <typename Grid,
          typename Type,
          int Cardinality>
struct PerformanceMetrics
{
    using Self = PerformanceMetrics<Grid, Type, Cardinality>;

    PerformanceMetrics() = default;

    PerformanceMetrics(Cli::UserData&                                                  userData,
                       Neon::domain::tool::testing::TestData<Grid, Type, Cardinality>& testData,
                       Neon::Timer_us&                                                 timerUs,
                       int                                                             repId)
    {
        elapsedUs = timerUs.time();
        iterationTimeUs = elapsedUs / userData.nIterations;
        nActiveCells = testData.getGrid().getNumActiveCells();
        MCPS = (elapsedUs / 1.0e6) / (nActiveCells / (1.0e6));
        MCPSPD = MCPS / testData.getBackend().devSet().setCardinality();
        repetitionId = repId;
        nGPUs = testData.getBackend().devSet().setCardinality();

        NEON_INFO(
            "Performance Repetition ID {} => [MCPSPD {}], [MCPS {}], [Elapsed Time {} us], [Iteration Time {} us], [Size {}], [Iterations]",
            repetitionId, MCPSPD, MCPS, elapsedUs, iterationTimeUs, userData.dimensions.to_string(),
            userData.nIterations);
    }

    auto log(Neon::Report& report) -> void
    {  // Adding benchmarks metrics
        NEON_INFO(
            "Performance Repetition ID {} => [MCPSPD {}], [MCPS {}], [Iteration Time {} us], [Elapsed Time {} us]",
            repetitionId, MCPSPD, MCPS, iterationTimeUs, elapsedUs);
        auto subdoc = report.getSubdoc();
    }

    static auto toReport(Cli::UserData& userData, std::vector<Self>& perfVec, Neon::Report& report) -> void
    {
        Self average;
        Self stdDev;
        int  nSamples = 0;

        auto addSample = [](const int& nSamplesPLusOne, double& movingAvg, double& movingStd, double x) {
            double nextM = movingAvg + (x - movingAvg) / nSamplesPLusOne;
            movingStd += (x - movingAvg) * (x - nextM);
            movingAvg = nextM;
        };
        auto computeStandardDeviation = [](const int& nSamples, double& movingStd) {
            if (nSamples == 1) {
                movingStd = 0.0;
                return;
            }
            double variace = movingStd / static_cast<double>(nSamples);
            movingStd = std::sqrt(variace);
        };

        for (auto newSample : perfVec) {
            ++nSamples;
            addSample(nSamples, average.elapsedUs, stdDev.elapsedUs, newSample.elapsedUs);
            addSample(nSamples, average.iterationTimeUs, stdDev.iterationTimeUs, newSample.iterationTimeUs);
            addSample(nSamples, average.MCPS, stdDev.MCPS, newSample.MCPS);
            addSample(nSamples, average.MCPSPD, stdDev.MCPSPD, newSample.MCPSPD);
        }

        computeStandardDeviation(nSamples, stdDev.elapsedUs);
        computeStandardDeviation(nSamples, stdDev.iterationTimeUs);
        computeStandardDeviation(nSamples, stdDev.MCPS);
        computeStandardDeviation(nSamples, stdDev.MCPSPD);


        average.nActiveCells = perfVec[0].nActiveCells;
        average.nGPUs = perfVec[0].nGPUs;
        average.repetitionId = -1;


        auto subdocSatistics = report.getSubdoc();

        average.nActiveCells = perfVec[0].nActiveCells;
        average.nGPUs = perfVec[0].nGPUs;
        average.repetitionId = -1;

        report.addMember("Average MCPS", average.MCPS, &subdocSatistics);
        report.addMember("Average MCPSPD", average.MCPSPD, &subdocSatistics);
        report.addMember("Average time per iteration (us)", average.iterationTimeUs, &subdocSatistics);
        report.addMember("Average elapsed time (us)", average.elapsedUs, &subdocSatistics);

        report.addMember("Std Deviation MCPS", stdDev.MCPS, &subdocSatistics);
        report.addMember("Std Deviation MCPSPD", stdDev.MCPSPD, &subdocSatistics);
        report.addMember("Std Deviation time per iteration (us)", stdDev.iterationTimeUs, &subdocSatistics);
        report.addMember("Std Deviation elapsed time (us)", stdDev.elapsedUs, &subdocSatistics);

        report.addMember("Number active cells", average.nActiveCells, &subdocSatistics);
        report.addMember("Number GPUs", average.nGPUs, &subdocSatistics);

        report.addSubdoc(Cli::AppsUtils::toString(userData.targetApp.getOption())+"_Performance_Statistics", subdocSatistics);

        NEON_INFO("Performance statistics => [MCPSPD {}], [MCPS {}],  [Iteration Time {} us], [Elapsed Time {} us]",
                  std::to_string(average.MCPSPD) + "+-" + std::to_string(stdDev.MCPSPD),
                  std::to_string(average.MCPS) + "+-" + std::to_string(stdDev.MCPS),
                  std::to_string(average.iterationTimeUs) + "+-" + std::to_string(stdDev.iterationTimeUs),
                  std::to_string(average.elapsedUs) + "+-" + std::to_string(stdDev.elapsedUs));

        {  // Dumping all data to the report

            std::vector<double> mcpspdVec;
            std::vector<double> iterationTimeUsVec;

            for (auto const& data : perfVec) {
                mcpspdVec.push_back(data.MCPSPD);
                iterationTimeUsVec.push_back(data.iterationTimeUs);
            }

            auto subdoc = report.getSubdoc();
            report.addMember("MCPSPD", mcpspdVec, &subdoc);
            report.addMember("Time per iteration (us)", iterationTimeUsVec, &subdoc);
            report.addSubdoc(Cli::AppsUtils::toString(userData.targetApp.getOption()), subdoc);
        }
    }

    double elapsedUs{0.0};
    double iterationTimeUs{0.0};
    double nActiveCells{0.0};
    double MCPS{0.0};
    double MCPSPD{0.0};
    int    repetitionId{0};
    int    nGPUs{0};
};

template <typename Grid,
          typename Type,
          int Cardinality>
auto testTemplate(Cli::UserData& userData,
                  Neon::Report&  report)
    -> void
{
    using TestData = Neon::domain::tool::testing::TestData<Grid, Type, Cardinality>;
    TestData testData = help::getTestData<Grid, Type, Cardinality>(userData);
    auto     sequence = help::getSequence<Grid, Type, Cardinality>(userData, testData);
    auto     option = help::getSkeletonOption(userData);

    testData.getGrid().toReport(report, true);
    testData.resetValuesToLinear(1);

    Neon::skeleton::Skeleton skeleton(testData.getBackend());
    skeleton.sequence(sequence, userData.testPrefix, option);

    using Metrics = PerformanceMetrics<Grid, Type, Cardinality>;
    std::vector<Metrics> metricVec;

    for (int repetition = 0; repetition < userData.repetitions; repetition++) {
        for (int i = 0; i < userData.warmupIterations; i++) {
            skeleton.run();
        }

        testData.getBackend().syncAll();

        Neon::Timer_us timerUs;
        timerUs.start();
        for (int i = 0; i < userData.nIterations; i++) {
            skeleton.run();
        }
        testData.getBackend().syncAll();
        timerUs.stop();

        Metrics newSample(userData, testData, timerUs, repetition);
        newSample.log(report);
        metricVec.push_back(newSample);

        if (userData.correctness.getOption() == Cli::Correctnesss::on) {
            bool isOk = help::runGolden<Grid, Type, Cardinality>(userData, testData);
            if (!isOk) {
                NEON_ERROR("Wrong data");
            }
        }

        if (repetition + 1 < repetition) {
            testData.resetValuesToLinear(1);
        }
    }
    Metrics::toReport(userData, metricVec, report);
}
}  // namespace help
auto Run(Cli::UserData& userData,
         Neon::Report&  report) -> void
{
    switch (userData.cardinality.getOption()) {
        case Cli::Cardinality::one: {
            switch (userData.runtimeType.getOption()) {
                case Cli::Type::INT64: {
                    if (userData.gridType.getOption() == Cli::GridType::eGrid)
                        return help::testTemplate<Neon::domain::eGrid, int64_t, 1>(userData, report);
                }
                case Cli::Type::DOUBLE: {
                    if (userData.gridType.getOption() == Cli::GridType::eGrid)
                        return help::testTemplate<Neon::domain::eGrid, double, 1>(userData, report);
                }
            }
        }

        case Cli::Cardinality::three: {
            switch (userData.runtimeType.getOption()) {
                case Cli::Type::INT64: {
                    if (userData.gridType.getOption() == Cli::GridType::eGrid)
                        return help::testTemplate<Neon::domain::eGrid, int64_t, 3>(userData, report);
                }
                case Cli::Type::DOUBLE: {
                    if (userData.gridType.getOption() == Cli::GridType::eGrid)
                        return help::testTemplate<Neon::domain::eGrid, double, 3>(userData, report);
                }
            }
        }

        case Cli::Cardinality::nineteen: {
            switch (userData.runtimeType.getOption()) {
                case Cli::Type::INT64: {
                    if (userData.gridType.getOption() == Cli::GridType::eGrid)
                        return help::testTemplate<Neon::domain::eGrid, int64_t, 19>(userData, report);
                }
                case Cli::Type::DOUBLE: {
                    if (userData.gridType.getOption() == Cli::GridType::eGrid)
                        return help::testTemplate<Neon::domain::eGrid, double, 19>(userData, report);
                }
            }
        }
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}
}  // namespace Test