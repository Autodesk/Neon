
#include "Config.h"
#include "Repoert.h"

#include "Neon/Neon.h"
#include "Neon/domain/dGrid.h"
#include "test.h"
int main(int argc, char* argv[])
{
    Neon::init();

    Config config{};
    config.n = 100;
    config.op = Op::axpy;
    config.iterations = 100;
    config.iterations = 5;
    config.cardinality = 1;
    config.blockSize = 256;
    config.reportName = "report";

    

    if (config.parseArgs(argc, argv) != 0) {
        return -1;
    }

    std::cout << "--------------- Parameters ---------------\n";
    std::cout << config.toString();
    std::cout << "-------------------------------------------\n";

    Report            report(config);
    std::stringstream testCode;
    for (int i = 0; i < config.repetitions; i++) {
        testCode = std::stringstream();
        run(config, report, testCode);
    }

    report.save(testCode);

    return 0;
}
