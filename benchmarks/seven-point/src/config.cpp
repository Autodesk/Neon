#include "config.h"
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

    s << "\n==>[Neon Runtime Parameters]" << std::endl;
    s << ".......... deviceType " << c.deviceType << std::endl;
    s << ".......... numDevices " << c.devices.size() << std::endl;
    s << "............. devices " << vecToSting(c.devices) << std::endl;
    s << ".......... reportFile " << c.reportFile << std::endl;
    s << "............ gridType " << c.gridType << std::endl;
    s << "............ dimSpace " << c.dimSpaceStr << std::endl;

    s << ".......... spaceCurve " << c.spaceCurveCli.getStringOption() << std::endl;
    s << "................. occ " << c.occCli.getStringOption() << std::endl;
    s << "........ transferMode " << c.transferModeCli.getStringOption() << std::endl;
    s << "..... stencilSemantic " << c.stencilSemanticCli.getStringOption() << std::endl;

    s << "\n==>[Solver Implementation]" << std::endl;
    s << "......... computeType " << c.computeTypeStr << std::endl;
    s << "................... N " << c.N << std::endl;


    s << "\n==>[Test Parameters]" << std::endl;
    s << "........... benchmark " << c.benchmark << std::endl;
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
            clipp::option("--domain-size") & clipp::value("domain_size")([&config](const std::string& s) { config.fromArgStringToDim(s); }) % "Voxels along each dimension of the cube domain",
            clipp::option("--max-iter") & clipp::integer("max_iter", config.benchMaxIter) % "Maximum solver iterations",
            clipp::option("--report-filename ") & clipp::value("keeper_filename", config.reportFile) % "Output perf keeper filename",

            clipp::option("--computeFP") & clipp::value("computeFP", config.computeTypeStr) % Config::getOptionList(config.computeTypeOptions, config.computeTypeStr),
            clipp::option("--dimSpace") & clipp::value("computeFP", config.dimSpaceStr) % Config::getOptionList(config.dimSpaceOptions, config.dimSpaceStr),

            clipp::option("--occ") & clipp::value("occ")([&config](const std::string& s) { config.occCli.set(s); }) % config.occCli.getDoc(),
            clipp::option("--transferMode") & clipp::value("transferMode")([&config](const std::string& s) { config.transferModeCli.set(s); }) % config.transferModeCli.getDoc(),
            clipp::option("--spaceCurve") & clipp::value("spaceCurve")([&config](const std::string& s) { config.spaceCurveCli.set(s); }) % config.spaceCurveCli.getDoc(),

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
                     "     --occ none\\\n"
                     "     --transferMode put\\\n"
                     "     --spaceCurve sweep\\\n"
                     "     --vti 10";

        return -1;
    }

    if (dimSpaceStr == "2D"){
        config.N.y = 1;
    }


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
