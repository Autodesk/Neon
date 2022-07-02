// -x 1 -y 1 -z 3 -deviceIds 0 -devType OMP -gridType eGrid -skeletonRuntime ompAtGraphLevel -occ standard -app map -type INT64 -iterations 1 -warmup 0 -prefix random
#include "Neon/Neon.h"
#include "Neon/Report.h"

#include "CLiApps.h"
#include "CLiType.h"
#include "Neon/core/tools/clipp.h"
#include "Neon/domain/tools/Geometries.h"
#include "UserData.h"
#include "RunTest.h"

#include "Neon/skeleton/Skeleton.h"

#include <map>

int main(int argc, char** argv)
{
    Neon::init();

    Cli::UserData user;

    auto cli = (
        // BEGIN

        // GEOMETRY
        ((clipp::required("-x") & clipp::value("x", user.dimensions.x),
          clipp::required("-y") & clipp::value("y", user.dimensions.y),
          clipp::required("-z") & clipp::value("z", user.dimensions.z)) |
         (clipp::required("-xyz") & clipp::value("xyz").call([&](const std::string& f) {
        int val = std::atoi(f.c_str());
        user.dimensions.x = val;
        user.dimensions.y = val;
             user.dimensions.z = val; })))
            .doc("Background grid size."),

        clipp::option("-geo") & clipp::value("geo").call([&](const std::string& f) { user.targetGeometry.set(f); }).doc(std::string("Options: {") + user.targetGeometry.getStringOptions() + "}"),

        // SYSTEM
        clipp::required("-deviceIds") & clipp::integers("deviceIds", user.deviceIds).doc("Selected devices"),
        clipp::required("-devType") & clipp::value("devType").call([&](const std::string& f) { user.deviceType.set(f); }).doc(std::string("Options: {") + user.deviceType.getStringOptions() + "}"),
        clipp::required("-gridType") & clipp::value("gridType").call([&](const std::string& f) { user.gridType.set(f); }).doc(std::string("Options: {") + user.gridType.getStringOptions() + "}"),


        // SKELETON
        clipp::required("-skeletonRuntime") & clipp::value("skeletonRuntime").call([&](const std::string& f) { user.executorModel.set(f); }).doc(std::string("Options: {") + user.executorModel.getStringOptions() + "}"),
        clipp::required("-occ") & clipp::value("occ").call([&](const std::string& f) { user.occModel.set(f); }).doc(std::string("Options: {") + user.occModel.getStringOptions() + "}"),

        // APPS
        clipp::required("-app") & clipp::value("app").call([&](const std::string& f) { user.targetApp.set(f); }).doc(std::string("Options: {") + user.targetApp.getStringOptions() + "}"),
        clipp::required("-type") & clipp::value("type").call([&](const std::string& f) { user.runtimeType.set(f); }).doc(std::string("Options: {") + user.runtimeType.getStringOptions() + "}"),
        clipp::option("-cardinality") & clipp::value("cardinality").call([&](const std::string& f) { user.cardinality.set(f); }).doc(std::string("Options: {") + user.cardinality.getStringOptions() + "}"),
        clipp::option("-iterations") & clipp::value("iterations", user.nIterations).doc(std::string("Number of iterations, by default ") + std::to_string(Cli::UserData::Defaults::nIterations)),
        clipp::option("-warmup") & clipp::value("warmup", user.warmupIterations).doc(std::string("Number of warmup iterations, by default ") + std::to_string(Cli::UserData::Defaults::warmupIterations)),
        clipp::option("-repetitions") & clipp::value("repetitions", user.repetitions).doc(std::string("Number of warmup iterations, by default ") + std::to_string(Cli::UserData::Defaults::repetitions)),
        clipp::option("-correctness") & clipp::value("correctness").call([&](const std::string& f) { user.correctness.set(f); }).doc(std::string("Options: {") + user.correctness.getStringOptions() + "}"),

        // TEST
        clipp::required("-prefix") & clipp::opt_value("prefix", user.testPrefix).doc("User defined prefix for file output")

        // END
    );

    auto parsingErrors = clipp::parse(argc, argv, cli);
    if (parsingErrors.any_error()) {
        auto fmt = clipp::doc_formatting{}.doc_column(60);
        std::cout << "Invalid input arguments!\n";
        for (const auto& m : parsingErrors.missing()) {
            if (m.param()->label() != std::string("")) {
                std::cout << "- missing " << m.param()->label() << '\n';
            }
        }
        std::cout << '\n'
                  << make_man_page(cli, argv[0], fmt)
                  << '\n';
        exit(EXIT_FAILURE);
    }

    //    testConfigurations.m_optSkelOCC = Neon::skeleton::Options::fromStringToOcc(occ);
    //    testConfigurations.m_optSkelTransfer = Neon::skeleton::Options::fromStringToTransfer(transfer);
    //    testConfigurations.m_dataType = DataTypeStr2Val(dataType);
    //
    //    NEON_INFO(testConfigurations.toString());
    //    return RUN_ALL_TESTS();

    user.log();
    Neon::Report report(std::string("SkeletonSyntheticBench" + user.testPrefix));
    user.toReport(report);
    report.commandLine(argc, argv);
    report.setToken(user.testPrefix);
    Test::Run(user,report);
    report.write(user.testPrefix);

    return 0;
}
