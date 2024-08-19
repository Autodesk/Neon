#include "Neon/py/Backend.h"
#include "Neon/Neon.h"
#include "Neon/py/AllocationCounter.h"
#include "Neon/py/macros.h"
#include "Neon/set/Backend.h"


auto dBackend_new(
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
    std::cout << "NeonPy: Backend created" << backendPtr->toString() << std::endl;

    if (backendPtr == nullptr) {
        std::cout << "NeonPy: Initialization error. Unable to allocage backend " << std::endl;
        return -1;
    }
    *handle = reinterpret_cast<void*>(backendPtr);

    AllocationCounter::Allocation();
    std::cout << "dBackend_new handle " << backendPtr << std::endl;
    NEON_PY_PRINT_END(*handle);

    return 0;
}

auto dBackend_delete(
    void** handle)
    -> int
{
    std::cout << "dBackend_delete - BEGIN" << std::endl;

    using Backend = Neon::Backend;
    Backend* backendPtr = (Backend*)(*handle);
    std::cout << "dBackend_delete backendHandle " << backendPtr << std::endl;

    if (backendPtr != nullptr) {
        delete backendPtr;
        AllocationCounter::Deallocation();
    }

    handle = 0;
    std::cout << "dBackend_delete - END" << std::endl;
    return 0;
}

auto dBackend_get_string(uint64_t& handle) -> const char*
{
    std::cout << "get_string - BEGIN" << std::endl;
    std::cout << "backendHandle " << handle << std::endl;

    using Backend = Neon::Backend;
    Backend* backendPtr = (Backend*)handle;
    if (backendPtr == nullptr) {
        return "Backend handle is invalid";
    }

    return backendPtr->toString().c_str();
    std::cout << "get_string - END" << std::endl;
}

auto dBackend_sync(uint64_t& handle) -> int
{
    std::cout << "dBackend_sync - BEGIN" << std::endl;
    std::cout << "backendHandle " << handle << std::endl;

    using Backend = Neon::Backend;
    Backend* backendPtr = (Backend*)handle;
    if (backendPtr == nullptr) {
        return -1;
    }
    backendPtr->syncAll();

    return 0;
    std::cout << "dBackend_sync - END" << std::endl;
}
