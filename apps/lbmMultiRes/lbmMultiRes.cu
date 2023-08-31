#include "Neon/core/tools/clipp.h"

#define BGK
//#define KBC

#include "Neon/Neon.h"
#include "Neon/Report.h"
#include "Neon/domain/mGrid.h"
#include "Neon/skeleton/Skeleton.h"


Neon::Report report;

#include "flowOverCylinder.h"
#include "lidDrivenCavity.h"


int main(int argc, char** argv)
{
    Neon::init();

    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        report = Neon::Report("Lid Driven Cavity MultiRes LBM");
        report.commandLine(argc, argv);

        std::string deviceType = "gpu";
        std::string problemType = "lid";
        int         freq = 100;
        int         Re = 100;
        int         deviceId = 99;
        int         numIter = 2;
        bool        benchmark = true;
        bool        fineInitStore = false;
        bool        streamFusedExpl = false;
        bool        streamFusedCoal = false;
        bool        streamFuseAll = false;
        bool        collisionFusedStore = false;
        int         problemId = 0;
        std::string dataType = "float";

        auto cli =
            (clipp::option("--deviceType") & clipp::value("deviceType", deviceType) % "Type of device (gpu or cpu)",
             clipp::option("--deviceId") & clipp::integers("deviceId", deviceId) % "Device id",
             clipp::option("--numIter") & clipp::integer("numIter", numIter) % "LBM number of iterations",
             clipp::option("--problemType") & clipp::value("problemType", problemType) % "Problem type ('lid' for lid-driven cavity or 'cylinder' for flow over cylinder)",
             clipp::option("--problemId") & clipp::integer("problemId", problemId) % "Problem ID (0-1 for lid)",
             clipp::option("--dataType") & clipp::value("dataType", dataType) % "Data type (float or double)",
             clipp::option("--re") & clipp::integers("Re", Re) % "Reynolds number",

             ((clipp::option("--benchmark").set(benchmark, true) % "Run benchmark mode") |
              (clipp::option("--visual").set(benchmark, false) % "Run export partial data")),

             clipp::option("--freq") & clipp::integers("freq", freq) % "Output frequency (only works with visual mode)",


             ((clipp::option("--storeFine").set(fineInitStore, true) % "Initiate the Store operation from the fine level") |
              (clipp::option("--storeCoarse").set(fineInitStore, false) % "Initiate the Store operation from the coarse level")
#ifdef BGK
              | (clipp::option("--collisionFusedStore").set(collisionFusedStore, true) % "Fuse Collision with Store operation")
#endif
                  ),

             ((clipp::option("--streamFusedExpl").set(streamFusedExpl, true) % "Fuse Stream with Explosion") |
              (clipp::option("--streamFusedCoal").set(streamFusedCoal, true) % "Fuse Stream with Coalescence") |
              (clipp::option("--streamFuseAll").set(streamFuseAll, true) % "Fuse Stream with Coalescence and Explosion")));


        if (!clipp::parse(argc, argv, cli)) {
            auto fmt = clipp::doc_formatting{}.doc_column(31);
            std::cout << make_man_page(cli, argv[0], fmt) << '\n';
            return -1;
        }

        if (deviceType != "cpu" && deviceType != "gpu") {
            Neon::NeonException exp("app-lbmMultiRes");
            exp << "Unknown input device type " << deviceType;
            NEON_THROW(exp);
        }

        if (problemType != "lid" && problemType != "cylinder") {
            Neon::NeonException exp("app-lbmMultiRes");
            exp << "Unknown input problem type " << problemType;
            NEON_THROW(exp);
        }

        if (dataType != "float" && dataType != "double") {
            Neon::NeonException exp("app-lbmMultiRes");
            exp << "Unknown input data type " << dataType;
            NEON_THROW(exp);
        }


        //Neon grid and backend
        Neon::Runtime runtime = Neon::Runtime::stream;
        if (deviceType == "cpu") {
            runtime = Neon::Runtime::openmp;
        }

        std::vector<int> gpu_ids{deviceId};
        Neon::Backend    backend(gpu_ids, runtime);

#ifdef KBC
        constexpr int Q = 27;
#endif
#ifdef BGK
        constexpr int Q = 19;
#endif
        if (dataType == "float") {
            if (problemType == "lid") {
                lidDrivenCavity<float, Q>(problemId, backend, numIter, Re, fineInitStore, streamFusedExpl, streamFusedCoal, streamFuseAll, collisionFusedStore, benchmark, freq);
            } else if (problemType == "cylinder") {
                flowOverCylinder<float, Q>(problemId, backend, numIter, Re, fineInitStore, streamFusedExpl, streamFusedCoal, streamFuseAll, collisionFusedStore, benchmark, freq);
            }
        } else if (dataType == "double") {
            if (problemType == "lid") {
                lidDrivenCavity<double, Q>(problemId, backend, numIter, Re, fineInitStore, streamFusedExpl, streamFusedCoal, streamFuseAll, collisionFusedStore, benchmark, freq);
            } else if (problemType == "cylinder") {
                flowOverCylinder<double, Q>(problemId, backend, numIter, Re, fineInitStore, streamFusedExpl, streamFusedCoal, streamFuseAll, collisionFusedStore, benchmark, freq);
            }
        }
    }
    return 0;
}