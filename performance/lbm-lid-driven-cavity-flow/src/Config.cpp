#include "Config.h"
#include <string>
#include <vector>

auto Config::toString() const -> std::string
{
    std::stringstream s;
    const Config&     c = *this;

    auto vecToSting = [](const std::vector<int>& v) {
        std::stringstream s;
        for (auto e : v) {
            s << " " + std::to_string(e);
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

    s << "........ benchIniIter " << c.benchIniIter << std::endl;
    s << "........ benchMaxIter " << c.benchMaxIter << std::endl;

    s << ".......... numDevices " << c.devices.size() << std::endl;
    s << "............. devices " << vecToSting(c.devices) << std::endl;
    s << ".......... reportFile " << c.reportFile << std::endl;
    s << "............ gridType " << c.gridType << std::endl;

    s << ". ............... occ " << Neon::skeleton::OccUtils::toString(c.occ) << std::endl;
    s << "....... transfer Mode " << Neon::set::TransferModeUtils::toString(c.transferMode) << std::endl;
    s << "... transfer Semantic " << Neon::set::TransferSemanticUtils::toString(c.transferSemantic) << std::endl;

    return s.str();
}

auto Config::getClip()
     -> clipp::group&
{
        return mClip;
}

Config::Config()
{
    auto& config = *this;

    auto cli =
        (

            clipp::option("--gpus") & clipp::integers("gpus", config.devices) % "Device ids to use",
            clipp::option("--grid") & clipp::value("grid", config.gridType) % "Could be eGrid or dGrid",
            clipp::option("--domain-size") & clipp::integer("domain_size", config.N) % "Voxels along each dimension of the cube domain",
            clipp::option("--warmup-iter") & clipp::integer("warmup_iter", config.benchIniIter) % "Number of iteration for warm up. max_iter = warmup_iter + timed_iters",
            clipp::option("--max-iter") & clipp::integer("max_iter", config.benchMaxIter) % "Maximum solver iterations",
            clipp::option("--report-filename ") & clipp::value("keeper_filename", config.reportFile) % "Output perf keeper filename",

            (
                (clipp::option("--sOCC").set(config.occ, Neon::skeleton::Occ::standard) % "Standard OCC") |
                (clipp::option("--nOCC").set(config.occ, Neon::skeleton::Occ::none) % "No OCC (on by default)")),
            (
                (clipp::option("--put").set(config.transferMode, Neon::set::TransferMode::put) % "Set transfer mode to PUT") |
                (clipp::option("--get").set(config.transferMode, Neon::set::TransferMode::get) % "Set transfer mode to GET (on by default)")),
            (
                (clipp::option("--huLattice").set(config.transferSemantic, Neon::set::TransferSemantic::lattice) % "Halo update with lattice semantic (on by default)") |
                (clipp::option("--huGrid").set(config.transferSemantic, Neon::set::TransferSemantic::grid) % "Halo update with grid semantic ")),
            (
                (clipp::option("--benchmark").set(config.benchmark, true) % "Run benchmark mode") |
                (clipp::option("--visual").set(config.benchmark, false) % "Run export partial data")),

            (
                clipp::option("--vti").set(config.vti, true) % "Standard OCC")

        );
}
