#include "Config.h"
#include <string>
#include <vector>


auto Config::parseArgs(const int argc, char* argv[])
    -> int
{
    auto& config = *this;

    auto cli =
        (

            clipp::required("--deviceType") & clipp::value("deviceType", config.deviceType) % "Device type (cpu or gpu)",
            clipp::required("--deviceIds") & clipp::integers("ids", config.devices) % "Device ids",

            clipp::option("--grid") & clipp::value("grid", config.gridType) % Config::getOptionList(config.gridTypeOptions, config.gridType),
            clipp::option("--domain-size") & clipp::value("domain_size")([&config](const std::string& s) { config.fromArgStringToDim(s); }) % "Voxels along each dimension of the cube domain",
            clipp::option("--max-iter") & clipp::integer("max_iter", config.benchMaxIter) % "Maximum solver iterations",
            clipp::option("--report-filename ") & clipp::value("keeper_filename", config.reportFile) % "Output perf keeper filename",

            clipp::option("--computeFP") & clipp::value("computeFP", config.computeTypeStr) % Config::getOptionList(config.gridTypeOptions, config.gridType),
            clipp::option("--storageFP") & clipp::value("storageFP", config.storeTypeStr) % "double, float",

            clipp::option("--occ") & clipp::value("occ")([&config](const std::string& s) { config.occCli.set(s); }) % config.occCli.getDoc(),
            clipp::option("--transferMode") & clipp::value("transferMode")([&config](const std::string& s) { config.transferModeCli.set(s); }) % config.transferModeCli.getDoc(),
            clipp::option("--stencilSemantic") & clipp::value("stencilSemantic")([&config](const std::string& s) { config.stencilSemanticCli.set(s); }) % config.stencilSemanticCli.getDoc(),
            clipp::option("--spaceCurve") & clipp::value("spaceCurve")([&config](const std::string& s) { config.spaceCurveCli.set(s); }) % config.spaceCurveCli.getDoc(),
            clipp::option("--collision") & clipp::value("collision")([&config](const std::string& s) { config.collisionCli.set(s); }) % config.collisionCli.getDoc(),

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

        std::cout << " ./lbm --deviceType gpu\\\n"
                     "     --deviceIds 0\\\n"
                     "     --grid dGrid\\\n"
                     "     --domain-size 100\\\n"
                     "     --max-iter 1000\\\n"
                     "     --computeFP float\\\n"
                     "     --storageFP float\\\n"
                     "     --occ none\\\n"
                     "     --transferMode put\\\n"
                     "     --stencilSemantic standard\\\n"
                     "     --spaceCurve sweep\\\n"
                     "     --collision bgk\\\n"
                     "     --streamingMethod pull\\\n"
                     "     --lattice d3q19\\\n"
                     "     --vti 10";

        return -1;
    }

    helpSetLbmParameters();

    std::stringstream s;
    for (int i = 0; i < argc; i++) {
        s << argv[i];
        if (i + 1 != argc) {
            s << " ";
        }
    }
    mArgv = s.str();

    return 0;
}

auto Config::helpSetLbmParameters() -> void
{
    mLbmParameters.nu = ulb * static_cast<double>(N.x - 2) / Re;
    mLbmParameters.omega = 1. / (3. * mLbmParameters.nu + 0.5);
    mLbmParameters.dx = 1. / static_cast<double>(N.x - 2);
    mLbmParameters.dt = mLbmParameters.dx * ulb;
}
