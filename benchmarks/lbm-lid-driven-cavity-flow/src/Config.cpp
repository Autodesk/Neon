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

    s << ".................. Re " << c.Re << std::endl;
    s << "................. ulb " << c.ulb << std::endl;
    s << "................... N " << c.N << std::endl;
    s << "........... benchmark " << c.benchmark << std::endl;
    s << "............... max_t " << c.max_t << std::endl;
    s << "........ outFrequency " << c.outFrequency << std::endl;
    s << "....... dataFrequency " << c.dataFrequency << std::endl;
    s << "................. vti " << c.vti << std::endl;

    s << "........ benchIniIter " << c.benchIniIter << std::endl;
    s << "........ benchMaxIter " << c.benchMaxIter << std::endl;

    s << ".......... deviceType " << c.deviceType << std::endl;
    s << ".......... numDevices " << c.devices.size() << std::endl;
    s << "............. devices " << vecToSting(c.devices) << std::endl;
    s << ".......... reportFile " << c.reportFile << std::endl;
    s << "............ gridType " << c.gridType << std::endl;

    s << "......... computeType " << c.computeType << std::endl;
    s << "........... storeType " << c.storeType << std::endl;

    s << ". ............... occ " << Neon::skeleton::OccUtils::toString(c.occ) << std::endl;
    s << "....... transfer Mode " << Neon::set::TransferModeUtils::toString(c.transferMode) << std::endl;
    s << "... transfer Semantic " << Neon::set::StencilSemanticUtils::toString(c.stencilSemantic) << std::endl;

    s << ". ............... nu " << mLbmParameters.nu << std::endl;
    s << ".............. omega " << mLbmParameters.omega << std::endl;
    s << "................. dx " << mLbmParameters.dx << std::endl;
    s << "................. dt " << mLbmParameters.dt << std::endl;

    return s.str();
}

auto Config::parseArgs(const int argc, char* argv[])
    -> int
{
    auto& config = *this;

    auto cli =
        (
            clipp::required("--deviceType") & clipp::value("deviceType", config.deviceType) % "Device ids to use",
            clipp::required("--deviceIds") & clipp::integers("gpus", config.devices) % "Device ids to use",
            clipp::option("--grid") & clipp::value("grid", config.gridType) % "Could be dGrid, eGrid, bGrid",
            clipp::option("--domain-size") & clipp::integer("domain_size", config.N) % "Voxels along each dimension of the cube domain",
            clipp::option("--warmup-iter") & clipp::integer("warmup_iter", config.benchIniIter) % "Number of iteration for warm up. max_iter = warmup_iter + timed_iters",
            clipp::option("--max-iter") & clipp::integer("max_iter", config.benchMaxIter) % "Maximum solver iterations",
            clipp::option("--repetitions") & clipp::integer("repetitions", config.repetitions) % "Number of times the benchmark is run.",
            clipp::option("--report-filename ") & clipp::value("keeper_filename", config.reportFile) % "Output perf keeper filename",

            clipp::option("--computeFP") & clipp::value("computeFP", config.computeType) % "Could be double or float",
            clipp::option("--storageFP") & clipp::value("storageFP", config.storeType) % "Could be double or float",

            (
                (clipp::option("--sOCC").set(config.occ, Neon::skeleton::Occ::standard) % "Standard OCC") |
                (clipp::option("--nOCC").set(config.occ, Neon::skeleton::Occ::none) % "No OCC (on by default)")),
            (
                (clipp::option("--put").set(config.transferMode, Neon::set::TransferMode::put) % "Set transfer mode to PUT") |
                (clipp::option("--get").set(config.transferMode, Neon::set::TransferMode::get) % "Set transfer mode to GET (on by default)")),
            (
                (clipp::option("--huLattice").set(config.stencilSemantic, Neon::set::StencilSemantic::streaming) % "Halo update with lattice semantic (on by default)") |
                (clipp::option("--huGrid").set(config.stencilSemantic, Neon::set::StencilSemantic::standard) % "Halo update with grid semantic ")),
            (
                (clipp::option("--benchmark").set(config.benchmark, true) % "Run benchmark mode") |
                (clipp::option("--visual").set(config.benchmark, false) % "Run export partial data")),

            (
                clipp::option("--vti").set(config.vti, true) % "Standard OCC")

        );

    if (!clipp::parse(argc, argv, cli)) {
        auto fmt = clipp::doc_formatting{}.doc_column(31);
        std::cout << make_man_page(cli, argv[0], fmt) << '\n';
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
