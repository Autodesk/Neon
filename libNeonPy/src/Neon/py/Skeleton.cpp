#include <dlfcn.h>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <rapidjson/reader.h>

#include "Neon/Neon.h"
#include "Neon/core/core.h"
#include "Neon/domain/Grids.h"
#include "Neon/domain/interface/GridBase.h"
#include "Neon/py/CudaDriver.h"
#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Loader.h"

#include "Neon/py/macros.h"
#include "Neon/skeleton/Skeleton.h"

extern "C" auto warp_skeleton_new(
    void** out_handle,
    void*  bkPtr) -> int
{
    NEON_PY_PRINT_BEGIN(*out_handle);

    Neon::Backend& bk = *static_cast<Neon::Backend*>(bkPtr);
    std::cout << "warp_skeleton_new "
              << "bk " << bkPtr << " " << bk.toString() << std::endl;
    auto* skeleton = new (std::nothrow) Neon::skeleton::Skeleton(bk);
    if (skeleton == nullptr) {
        return -1;
    }
    (*out_handle) = skeleton;
    NEON_PY_PRINT_END(*out_handle);
    return 0;
}

extern "C" auto warp_skeleton_delete(
    void** out_handle) -> int
{
    try {
        auto* skeleton = static_cast<Neon::skeleton::Skeleton*>(*out_handle);
        delete skeleton;
        (*out_handle) = nullptr;
        return 0;
    } catch (...) {
        return -1;
    }
}

extern "C" void warp_skeleton_sequence(
    void*                  handle,
    const char*            graphName,
    int                    numContainers,
    Neon::set::Container** containers)
{
    // # Load the shared library
    // # On Linux
    //     lib = ctypes.CDLL('./libexample.so')
    //
    //     # On Windows
    //     # lib = ctypes.CDLL('example.dll')
    //
    //     # Define the argument type for the C++ function
    //     lib.print_message.argtypes = [ctypes.c_char_p]
    //
    //     # Convert Python string to bytes (C char array)
    //     message = "Hello from Python".encode('utf-8')
    //
    //     # Call the C++ function
    //     lib.print_message(message)
    NEON_PY_PRINT_BEGIN(handle);

    Neon::skeleton::Skeleton*         skeleton = static_cast<Neon::skeleton::Skeleton*>(handle);
    std::string                       name(graphName);
    std::vector<Neon::set::Container> operations;
    for (int i = 0; i < numContainers; i++) {
        operations.push_back(*containers[i]);
    }
    skeleton->sequence(operations, name, Neon::skeleton::Options());
    NEON_PY_PRINT_END(handle);
}

extern "C" auto warp_skeleton_run(
    void* handle) -> int
{
    try {
        auto* s = static_cast<Neon::skeleton::Skeleton*>(handle);
        s->run();
        return 0;
    } catch (...) {
        return -1;
    }
}

extern "C" auto warp_skeleton_ioToDot(
    void*       handle,
    const char* fname,
    const char* gname,
    int         debug)
    -> int
{
    try {

        auto* s = static_cast<Neon::skeleton::Skeleton*>(handle);
        s->ioToDot(fname, gname, debug == 1);
    } catch (...) {
        return -1;
    }
    return 0;
}
