#include "Config.h"
#include <string>
#include <vector>

auto Config::parseArgs(int argc, char* argv[]) -> int
{
    auto& config = *this;


    auto cli =
        (clipp::required("--n") & clipp::value("N", config.n) % "Dimension of the cubic domain",
         clipp::option("--op") & clipp::value("op")([&config](const std::string& s) { config.op = OpUtils::formString(s); }) % "operation",
         clipp::option("--iterations") & clipp::integer("iterations", config.iterations) % "Iteration for average",
         clipp::option("--repetitions") & clipp::integer("repetitions", config.repetitions) % "Repetition for standard deviation",
         clipp::option("--cardinality") & clipp::integer("cardinality", config.cardinality) % "Field cardinality",
         clipp::option("--blockSize") & clipp::integer("blockSize", config.blockSize) % "Cuda block size",

         clipp::option("--report ") & clipp::value("report", config.reportName) % "Output perf keeper filename");

    if (!clipp::parse(argc, argv, cli)) {
        auto fmt = clipp::doc_formatting{}.doc_column(31);
        std::cout << make_man_page(cli, argv[0], fmt) << '\n';


        return -1;
    }
    return 0;
}

auto Config::toString() const -> std::string
{
    std::stringstream s;
    const Config&     c = *this;

    [[maybe_unused]] auto vecToSting = [](const std::vector<int>& v) {
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
    s << ".............. n " << c.n << std::endl;
    s << "............. op " << OpUtils::toString(c.op) << std::endl;
    s << "..... Iterations " << c.iterations << std::endl;
    s << ".... Repetitions " << c.repetitions << std::endl;
    s << ".... Cardinality " << c.cardinality << std::endl;
    s << "...... BlockSize " << c.blockSize << std::endl;
    s << "......... Report " << c.reportName << std::endl;

    return s.str();
}
