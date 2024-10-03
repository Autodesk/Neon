
#include "report.h"
#include "config.h"
#include "seven-point.h"

#include "Neon/Neon.h"
#include "Neon/core/tools/clipp.h"
#include "Neon/domain/dGrid.h"

int main(int argc, char** argv)
{
    Config config;
    Neon::init();

    config.N = 160;           // Number of nodes in x-direction
    config.benchmark = true;  // Run in benchmark mode ?



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
        CavityTwoPop::run(config, report, testCode);
    }

    report.save(testCode);

    return 0;
}
