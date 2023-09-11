#include "Neon/core/tools/clipp.h"

//#define BGK
#define KBC

#include "Neon/Neon.h"
#include "Neon/Report.h"
#include "Neon/domain/mGrid.h"
#include "Neon/skeleton/Skeleton.h"


Neon::Report report;

struct Params
{
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
    int         sliceX = -1;
    int         sliceY = -1;
    int         sliceZ = 1;
    bool        vtk = false;
    bool        gui = false;
    int         scale = 2;
    std::string dataType = "float";
};

#include "flowOverShape.h"
#include "lidDrivenCavity.h"


int main(int argc, char** argv)
{
    Neon::init();

    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        report = Neon::Report("Lid Driven Cavity MultiRes LBM");
        report.commandLine(argc, argv);

        Params params;

        auto cli =
            (clipp::option("--deviceType") & clipp::value("deviceType", params.deviceType) % "Type of device (gpu or cpu)",
             clipp::required("--deviceId") & clipp::integers("deviceId", params.deviceId) % "Device id",
             clipp::option("--numIter") & clipp::integer("numIter", params.numIter) % "LBM number of iterations",
             clipp::option("--problemType") & clipp::value("problemType", params.problemType) % "Problem type ('lid' for lid-driven cavity, 'sphere' for flow over sphere, or 'jet' for flow over jet fighter)",
             clipp::option("--dataType") & clipp::value("dataType", params.dataType) % "Data type (float or double)",
             clipp::option("--re") & clipp::integers("Re", params.Re) % "Reynolds number",
             clipp::option("--scale") & clipp::integers("scale", params.scale) % "Scale of the problem for parametrized problems. 0-9 for lid. jet is up to 112. Sphere is 2 (or maybe more)",

             clipp::option("--sliceX") & clipp::integer("sliceX", params.sliceX) % "Slice along X for output images/VTK",
             clipp::option("--sliceY") & clipp::integer("sliceY", params.sliceY) % "Slice along Y for output images/VTK",
             clipp::option("--sliceZ") & clipp::integer("sliceZ", params.sliceZ) % "Slice along Z for output images/VTK",

             ((clipp::option("--benchmark").set(params.benchmark, true) % "Run benchmark mode") |
              (clipp::option("--visual").set(params.benchmark, false) % "Run export partial data")),

             clipp::option("--vtk").set(params.vtk, true) % "Output VTK files. Active only with if 'visual' is true",
             clipp::option("--gui").set(params.gui, true) % "Show Polyscope gui. Active only with if 'visual' is true",

             clipp::option("--freq") & clipp::integers("freq", params.freq) % "Output frequency (only works with visual mode)",

             ((clipp::option("--storeFine").set(params.fineInitStore, true) % "Initiate the Store operation from the fine level") |
              (clipp::option("--storeCoarse").set(params.fineInitStore, false) % "Initiate the Store operation from the coarse level")
#ifdef BGK
              | (clipp::option("--collisionFusedStore").set(params.collisionFusedStore, true) % "Fuse Collision with Store operation")
#endif
                  ),

             ((clipp::option("--streamFusedExpl").set(params.streamFusedExpl, true) % "Fuse Stream with Explosion") |
              (clipp::option("--streamFusedCoal").set(params.streamFusedCoal, true) % "Fuse Stream with Coalescence") |
              (clipp::option("--streamFuseAll").set(params.streamFuseAll, true) % "Fuse Stream with Coalescence and Explosion")));


        if (!clipp::parse(argc, argv, cli)) {
            auto fmt = clipp::doc_formatting{}.doc_column(31);
            std::cout << make_man_page(cli, argv[0], fmt) << '\n';
            return -1;
        }

        if (params.deviceType != "cpu" && params.deviceType != "gpu") {
            Neon::NeonException exp("app-lbmMultiRes");
            exp << "Unknown input device type " << params.deviceType;
            NEON_THROW(exp);
        }

        if (params.problemType != "lid" && params.problemType != "sphere" && params.problemType != "jet") {
            Neon::NeonException exp("app-lbmMultiRes");
            exp << "Unknown input problem type " << params.problemType;
            NEON_THROW(exp);
        }

        if (params.dataType != "float" && params.dataType != "double") {
            Neon::NeonException exp("app-lbmMultiRes");
            exp << "Unknown input data type " << params.dataType;
            NEON_THROW(exp);
        }


        //Neon grid and backend
        Neon::Runtime runtime = Neon::Runtime::stream;
        if (params.deviceType == "cpu") {
            runtime = Neon::Runtime::openmp;
        }

        std::vector<int> gpu_ids{params.deviceId};
        Neon::Backend    backend(gpu_ids, runtime);

#ifdef KBC
        constexpr int Q = 27;
#endif
#ifdef BGK
        constexpr int Q = 19;
#endif
        if (params.dataType == "float") {
            if (params.problemType == "lid") {
                lidDrivenCavity<float, Q>(backend, params);
            }
            if (params.problemType == "sphere") {
                flowOverSphere<float, Q>(backend, params);
            }
            if (params.problemType == "jet") {
                flowOverJet<float, Q>(backend, params);
            }
        } else if (params.dataType == "double") {
            if (params.problemType == "lid") {
                lidDrivenCavity<double, Q>(backend, params);
            }
            if (params.problemType == "sphere") {
                flowOverSphere<double, Q>(backend, params);
            }
            if (params.problemType == "jet") {
                flowOverJet<double, Q>(backend, params);
            }
        }
    }
    return 0;
}