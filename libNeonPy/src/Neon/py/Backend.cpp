#include "Neon/py/Backend.h"
#include "Neon/Neon.h"
#include "Neon/py/AllocationCounter.h"
#include "Neon/py/macros.h"
#include "Neon/set/Backend.h"


auto backend_new(
    void**     handle,
    int        runtime,
    int        numDevices,
    const int* devIds)
    -> int
{
    NEON_PY_PRINT_BEGIN(*handle);

    Neon::init();

    std::vector<int> vec(devIds, devIds + numDevices);

    auto backendPtr = new (std::nothrow) Neon::Backend(vec, Neon::Runtime(runtime));
    // std::cout << "NeonPy: Backend created" << backendPtr->toString() << std::endl;

    if (backendPtr == nullptr) {
        std::cout << "NeonPy: Initialization error. Unable to allocage backend " << std::endl;
        return -1;
    }
    *handle = reinterpret_cast<void*>(backendPtr);

    AllocationCounter::Allocation();
    std::cout << "backend_new handle " << backendPtr << std::endl;
    NEON_PY_PRINT_END(*handle);

    return 0;
}

auto backend_delete(
    void** handle)
    -> int
{
    // std::cout << "backend_delete - BEGIN" << std::endl;

    using Backend = Neon::Backend;
    Backend* backendPtr = (Backend*)(*handle);
    // std::cout << "backend_delete backendHandle " << backendPtr << std::endl;

    if (backendPtr != nullptr) {
        delete backendPtr;
        AllocationCounter::Deallocation();
    }

    handle = 0;
    // std::cout << "backend_delete - END" << std::endl;
    return 0;
}

auto backend_get_string(void* handle) -> const char*
{
    // std::cout << "get_string - BEGIN" << std::endl;
    // std::cout << "backendHandle " << handle << std::endl;

    using Backend = Neon::Backend;
    Backend* backendPtr = (Backend*)handle;
    if (backendPtr == nullptr) {
        return "Backend handle is invalid";
    }

    return backendPtr->toString().c_str();
    // std::cout << "get_string - END" << std::endl;
}

auto backend_sync(void* handle) -> int
{
    using Backend = Neon::Backend;
    Backend* backendPtr = (Backend*)handle;
    // std::cout << "NeonPy: Backend sync"<< backendPtr<< std::endl;

    if (backendPtr == nullptr) {
        std::cout << "backend_sync: Backend handle is invalid";
        return -1;
    }
    backendPtr->syncAll();
    return 0;
}

auto backend_info_print(void* handle) -> int
{
    using Backend = Neon::Backend;
    Backend* backendPtr = (Backend*)handle;
    if (backendPtr == nullptr) {
        // print some errors
        std::cout << "info_print: Backend handle is invalid";
        return -1;
    }
    std::cout << backendPtr->toString() << std::endl;
    return 0;
}
