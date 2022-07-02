#pragma once
#include "Neon/Neon.h"
#include "Neon/Report.h"

#include "CLiApps.h"
#include "CLiCardinality.h"
#include "CLiCorrectness.h"
#include "CLiType.h"
#include "GridType.h"

#include "Neon/core/tools/clipp.h"
#include "Neon/domain/tools/Geometries.h"

#include "Neon/skeleton/Skeleton.h"

#include <map>
namespace Cli {

struct UserData
{
    UserData()
    {
        cardinality.set("one");
        correctness.set("on");
    }
    auto log() -> void
    {
        NEON_INFO("--- [GEOMETRY]");
        NEON_INFO("Dimensions: {}", dimensions.to_string());
        NEON_INFO("Geometry:   {}", Neon::domain::tool::GeometryUtils::toString(targetGeometry.getOption()));
        NEON_INFO("--- [SYSTEM]");
        NEON_INFO("DeviceType: {}", Neon::DeviceTypeUtil::toString(deviceType.getOption()));
        NEON_INFO("DeviceIds:  {}", [&] {
            std::stringstream s;
            s << "[";
            bool first = true;
            for (auto id : deviceIds) {
                if (!first)
                    s << " ";
                s << id;
                first = false;
            }
            s << "]";
            return s.str();
        }());
        NEON_INFO("GridType:     {}", GridTypeUtils::toString(gridType.getOption()));
        NEON_INFO("--- [SKELETON]");
        NEON_INFO("Executor:     {}", Neon::skeleton::ExecutorUtils::toString(executorModel.getOption()));
        NEON_INFO("Occ:          {}", Neon::skeleton::OccUtils::toString(occModel.getOption()));
        NEON_INFO("--- [APPLICATION]");
        NEON_INFO("App:          {}", Cli::AppsUtils::toString(targetApp.getOption()));
        NEON_INFO("Type:         {}", Cli::TypeUtils::toString(runtimeType.getOption()));
        NEON_INFO("Cardinality:  {}", Cli::CardinalityUtils::toString(cardinality.getOption()));
        NEON_INFO("Iterations:   {}", std::to_string(nIterations));
        NEON_INFO("Warmup:       {}", std::to_string(warmupIterations));
        NEON_INFO("CheckResults: {}", Cli::CorrectnesssUtils::toString(correctness.getOption()));
        NEON_INFO("--- [TEST]");
        NEON_INFO("Prefix:      {}", testPrefix);
    }

    auto toReport(Neon::Report& report) -> void
    {
        auto subdoc = report.getSubdoc();
        report.addMember("x", dimensions.x, &subdoc);
        report.addMember("y", dimensions.y, &subdoc);
        report.addMember("z", dimensions.z, &subdoc);

        executorModel.addToReport(report, subdoc);
        occModel.addToReport(report, subdoc);
        targetApp.addToReport(report, subdoc);
        report.addMember("Iterations", std::to_string(nIterations), &subdoc);
        report.addMember("Warmup", std::to_string(warmupIterations), &subdoc);
        runtimeType.addToReport(report, subdoc);
        targetGeometry.addToReport(report, subdoc);
        deviceType.addToReport(report, subdoc);
        gridType.addToReport(report, subdoc);
        cardinality.addToReport(report, subdoc);
        correctness.addToReport(report, subdoc);

        report.addMember(
            "DeviceIds", [&] {
                std::stringstream s;
                s << "[";
                bool first = true;
                for (auto id : deviceIds) {
                    if (!first)
                        s << " ";
                    s << id;
                    first = false;
                }
                s << "]";
                return s.str();
            }(),
            &subdoc);
        report.addMember("testPrefix", testPrefix, &subdoc);

        report.addSubdoc("CLI", subdoc);
    }

    Neon::index_3d                     dimensions;
    Neon::skeleton::ExecutorUtils::Cli executorModel;
    Neon::skeleton::OccUtils::Cli      occModel;
    Cli::AppsUtils::Cli                targetApp;
    Cli::TypeUtils::Cli                runtimeType;
    Cli::CardinalityUtils::Cli         cardinality;

    std::string                            testPrefix;
    Neon::domain::tool::GeometryUtils::Cli targetGeometry{Neon::domain::tool::Geometry::FullDomain};
    std::vector<int>                       deviceIds;
    Neon::DeviceTypeUtil::Cli              deviceType;
    Cli::GridTypeUtils::Cli                gridType;
    int                                    nIterations{Defaults::nIterations};
    int                                    warmupIterations{Defaults::warmupIterations};
    int                                    repetitions{Defaults::repetitions};

    Cli::CorrectnesssUtils::Cli correctness;
    struct Defaults
    {
        static constexpr int nIterations = 100;
        static constexpr int warmupIterations = 10;
        static constexpr int repetitions = 1;
    };
};
}  // namespace Cli
