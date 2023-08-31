#include "Config.h"
#include <string>
#include <vector>

auto Config::toString() const -> std::string
{
    std::stringstream s;
    const Config&     c = *this;

    auto vecToSting = [](const std::vector<int>& v) {
        std::stringstream s;
        bool              firstTime = true;
        for (auto e : v) {
            if (firstTime) {
                firstTime = false;
            } else {
                s << " ";
            }
            s << std::to_string(e);
        }
        return s.str();
    };

    s << "Neon Runtime Parameters" << std::endl;
    s << ".......... deviceType " << c.deviceType << std::endl;
    s << ".......... numDevices " << c.devices.size() << std::endl;
    s << "............. devices " << vecToSting(c.devices) << std::endl;
    s << ".......... reportFile " << c.reportFile << std::endl;
    s << "............ gridType " << c.gridType << std::endl;

    s << ".......... spaceCurve " << c.spaceCurveCli.getStringOptions() << std::endl;
    s << "................. occ " << c.occCli.getStringOptions() << std::endl;
    s << "....... transferMode " << c.transferModeCli.getStringOptions() << std::endl;
    s << ".... stencilSemantic " << c.stencilSemanticCli.getStringOptions() << std::endl;

    s << "LBM Implementation" << std::endl;
    s << "............ lattice " << c.lattice << std::endl;
    s << "... streaming method " << c.streamingMethod << std::endl;
    s << "......... computeType " << c.computeTypeStr << std::endl;
    s << "........... storeType " << c.storeTypeStr << std::endl;

    s << "Physics Parameters" << std::endl;
    s << ".................. Re " << c.Re << std::endl;
    s << "................. ulb " << c.ulb << std::endl;
    s << "................... N " << c.N << std::endl;
    s << "................. nu " << mLbmParameters.nu << std::endl;
    s << ".............. omega " << mLbmParameters.omega << std::endl;
    s << "................. dx " << mLbmParameters.dx << std::endl;
    s << "................. dt " << mLbmParameters.dt << std::endl;

    s << "Test Parameters" << std::endl;
    s << "........... benchmark " << c.benchmark << std::endl;
    s << "............... max_t " << c.max_t << std::endl;
    s << "................. vti " << c.vti << std::endl;
    s << "........ benchIniIter " << c.benchIniIter << std::endl;
    s << "........ benchMaxIter " << c.benchMaxIter << std::endl;


    return s.str();
}

auto Config::parseArgs(const int argc, char* argv[])
    -> int
{
    auto& config = *this;

    auto cli =
        (

            clipp::required("--deviceType") & clipp::value("deviceType", config.deviceType) % "Device type (cpu or gpu)",
            clipp::required("--deviceIds") & clipp::integers("ids", config.devices) % "Device ids",

            clipp::option("--grid") & clipp::value("grid", config.gridType) % Config::getOptionList(config.gridTypeOptions, config.gridType),
            clipp::option("--domain-size") & clipp::integer("domain_size", config.N) % "Voxels along each dimension of the cube domain",
            clipp::option("--max-iter") & clipp::integer("max_iter", config.benchMaxIter) % "Maximum solver iterations",
            clipp::option("--report-filename ") & clipp::value("keeper_filename", config.reportFile) % "Output perf keeper filename",

            clipp::option("--computeFP") & clipp::value("computeFP", config.computeTypeStr) % Config::getOptionList(config.gridTypeOptions, config.gridType),
            clipp::option("--storageFP") & clipp::value("storageFP", config.storeTypeStr) % "double, float",

            clipp::option("--occ")([&config](const std::string& s) { config.occCli.set(s); }) % config.occCli.getDoc(),
            clipp::option("--transferMode")([&config](const std::string& s) { config.transferModeCli.set(s); }) % config.transferModeCli.getDoc(),
            clipp::option("--stencilSemantic")([&config](const std::string& s) { config.stencilSemanticCli.set(s); }) % config.stencilSemanticCli.getDoc(),
            clipp::option("--spaceCurve")([&config](const std::string& s) { config.spaceCurveCli.set(s); }) % config.spaceCurveCli.getDoc(),

            clipp::option("--streamingMethod") & clipp::value("streamingMethod", config.streamingMethod) % Config::getOptionList(config.streamingMethodOption, config.streamingMethod),
            clipp::option("--lattice") & clipp::value("lattice", config.lattice) % Config::getOptionList(config.latticeOptions, config.lattice),
            (
                (
                    clipp::option("--benchmark").set(config.benchmark, true) % "Run benchmark mode",
                    clipp::option("--warmup-iter") & clipp::integer("warmup_iter", config.benchIniIter) % "Number of iteration for warm up. max_iter = warmup_iter + timed_iters",
                    clipp::option("--repetitions") & clipp::integer("repetitions", config.repetitions) % "Number of times the benchmark is run."

                    ) |
                (clipp::option("--vti") & clipp::integer("OutputFrequency", config.vti) % "Voxels along each dimension of the cube domain"))

        );


    if (!clipp::parse(argc, argv, cli)) {
        auto fmt = clipp::doc_formatting{}.doc_column(31);
        std::cout << make_man_page(cli, argv[0], fmt) << '\n';
        std::cout << '\n';
        std::cout << '\n';
        std::cout << "Export example" << '\n';
        std::cout << "./lbm --deviceType cpu --deviceIds 0  --grid dGrid  --domain-size 100 --max-iter 2000 --nOCC --huGrid --vti 1" << '\n';
        std::cout << "Benchmark example " << '\n';
        std::cout << "./lbm --deviceType gpu --deviceIds 0 1 2 3 4  --grid dGrid  --domain-size 100 --max-iter 2000 --computeFP double --storageFP double --nOCC --huGrid --benchmark --warmup-iter 10 --repetitions 5" << '\n';

        return -1;
    }

    helpSetLbmParameters();

    return 0;
}

auto Config::helpSetLbmParameters() -> void
{
    mLbmParameters.nu = ulb * static_cast<double>(N - 2) / Re;
    mLbmParameters.omega = 1. / (3. * mLbmParameters.nu + 0.5);
    mLbmParameters.dx = 1. / static_cast<double>(N - 2);
    mLbmParameters.dt = mLbmParameters.dx * ulb;
}
