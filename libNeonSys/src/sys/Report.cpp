#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef _WIN32
#include <Winsock2.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <unistd.h>
#endif

#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include "Neon/core/tools/Logger.h"

#include "Neon/sys/global/GpuSysGlobal.h"

#include "Neon/Report.h"


namespace Neon {
Report::Report(const std::string& record_name)
    : Neon::core::Report(record_name)
{
    system();
    device();
}

auto Report::system() -> void
{
    auto subdoc = getSubdoc();

#ifdef _WIN32
    // https://stackoverflow.com/a/11828223/1608232
    char    szPath[128] = "";
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
    gethostname(szPath, sizeof(szPath));
    std::string hostname(szPath);
    addMember("Hostname", hostname, &subdoc);
    WSACleanup();
#else
    char hostname[300];
    gethostname(hostname, 300 - 1);
    hostname[300 - 1] = '\0';
    std::string hostname_str(hostname);
    addMember("Hostname", hostname_str, &subdoc);
#endif


#ifdef _MSC_VER
    addMember(
        "Microsoft Full Compiler Version", int32_t(_MSC_FULL_VER), &subdoc);
    addMember("Microsoft Compiler Version", int32_t(_MSC_VER), &subdoc);
#else

    // https://stackoverflow.com/a/38531037/1608232
    std::string true_cxx =
#ifdef __clang__
        "clang++";
#elif __GNUC__
        "g++";
#else
        "unknown"
#endif
    auto ver_string = [](int a, int b, int c) {
        std::ostringstream ss;
        ss << a << '.' << b << '.' << c;
        return ss.str();
    };

    std::string true_cxx_ver =
#ifdef __clang__
        ver_string(__clang_major__, __clang_minor__, __clang_patchlevel__);
#elif __GNUC__
        ver_string(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
        "-1";
#endif

    addMember("compiler_name", true_cxx, &subdoc);
    addMember("compiler_version", true_cxx_ver, &subdoc);
#endif
    addMember("C++ version", int(__cplusplus), &subdoc);

#if NDEBUG
    std::string build_mode = "Release";
    addMember("Build Mode", build_mode, &subdoc);
#else
    std::string build_mode = "Debug";
    addMember("Build Mode", build_mode, &subdoc);

#endif

    addSubdoc("System", subdoc);
}

auto Report::device() -> void
{
    if (!Neon::sys::globalSpace::gpuSysObjStorage.isInit()) {
        NeonException exception("Report::device()");
        exception << "Neon has not been initialized. Call Neon::init() first!";
        NEON_THROW(exception);
    }
    int32_t num_gpus = Neon::sys::globalSpace::gpuSysObjStorage.numDevs();
    if (num_gpus >= 1) {

        {
            auto subdoc = getSubdoc();

            int ver = 0;
            if (cudaDriverGetVersion(&ver) != cudaSuccess) {
                NeonException exception("Report::device()");
                exception << "Can not retrieve CUDA runtime version!";
                NEON_THROW(exception);
            }
            addMember("Driver Version", ver, &subdoc);

            if (cudaRuntimeGetVersion(&ver) != cudaSuccess) {
                NeonException exception("Report::device()");
                exception << "Can not retrieve CUDA runtime version!";
                NEON_THROW(exception);
            }
            addMember("Runtime Version", ver, &subdoc);

            addMember("CUDA API Version", CUDA_VERSION, &subdoc);

            addSubdoc("CUDA", subdoc);
        }

        addMember("num_gpus", num_gpus);

        for (int d = 0; d < num_gpus; ++d) {

            auto subdoc = getSubdoc();

            const auto& dev = Neon::sys::globalSpace::gpuSysObjStorage.dev({d});

            addMember("ID", d, &subdoc);

            addMember("Name", dev.tools.getDevName(), &subdoc);

            std::string cc =
                std::to_string(dev.tools.majorComputeCapability()) + "." +
                std::to_string(dev.tools.minorComputeCapability());
            addMember("Compute Capability", cc, &subdoc);

            const auto prop = dev.tools.getDeviceProp();

            addMember("Total amount of global memory (MB)",
                      (float)prop.totalGlobalMem / 1048576.0f,
                      &subdoc);
            addMember("Total amount of shared memory per block (Kb)",
                      (float)prop.sharedMemPerBlock / 1024.0f,
                      &subdoc);
            addMember("Multiprocessors", prop.multiProcessorCount, &subdoc);

            addMember(
                "GPU Max Clock rate (GHz)", prop.clockRate * 1e-6f, &subdoc);
            addMember(
                "Memory Clock rate (GHz)", prop.memoryClockRate * 1e-6f, &subdoc);
            addMember("Memory Bus Width (bit)", prop.memoryBusWidth, &subdoc);
            addMember("Peak Memory Bandwidth (GB/s)",
                      2.0 * prop.memoryClockRate *
                          (prop.memoryBusWidth / 8.0) / 1.0E6,
                      &subdoc);

            addSubdoc("GPU_" + std::to_string(d), subdoc);
        }
    }
}


}  // namespace Neon
