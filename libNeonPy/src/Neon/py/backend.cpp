#include "Neon/py/backend.h"
#include "Neon/set/Backend.h"
#include "Neon/py/AllocationCounter.h"
#include "Neon/Neon.h"

void backend_constructor_prologue(uint64_t& handle) {
    std::cout << "dBackend_new - BEGIN" << std::endl;
    std::cout << "dBackend handle" << handle << std::endl;
}

int backend_constructor_epilogue(uint64_t& handle, Neon::Backend* backendPtr) {
    if (backendPtr == nullptr) {
        std::cout << "NeonPy: Initialization error. Unable to allocage backend " << std::endl;
        return -1;
    }
    handle = reinterpret_cast<uint64_t>(backendPtr);
    std::cout << "allocated backend heap location: " << backendPtr << std::endl;
    std::cout << "backend_new - END" << std::endl;
    return 0;
}

auto dBackend_new(
    uint64_t& handle,
    int runtime,
    int numDevices,
    const int* devIds)
    -> int
{
    Neon::init();
    backend_constructor_prologue(handle);

    std::vector<int> vec(devIds, devIds + numDevices);

    auto backendPtr = new (std::nothrow) Neon::Backend(vec, Neon::Runtime(runtime));
    AllocationCounter::Allocation();

    return backend_constructor_epilogue(handle, backendPtr);
}

auto dBackend_delete(
    uint64_t& handle)
    -> int
{
    std::cout << "dBackend_delete - BEGIN" << std::endl;
    std::cout << "backendHandle " << handle << std::endl;

    using Backend = Neon::Backend;
    Backend* backendPtr = (Backend*)handle;

    if (backendPtr != nullptr) {
        delete backendPtr;
        AllocationCounter::Deallocation();
    }
    handle = 0;
    std::cout << "dBackend_delete - END" << std::endl;
    return 0;
}

auto dBackend_get_string(uint64_t& handle) -> const char* {
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

auto dBackend_sync(uint64_t& handle) -> int {
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
