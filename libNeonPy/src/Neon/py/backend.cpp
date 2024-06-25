#include "Neon/py/backend.h"
#include "Neon/set/Backend.h"
#include "Neon/py/AllocationCounter.h"

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

auto dBackend_new1(
    uint64_t& handle)
    -> int
{
    std::cout << "first constructor" << std::endl;
    AllocationCounter::Allocation();

    backend_constructor_prologue(handle);

    auto backendPtr = new (std::nothrow) Neon::Backend();

    return backend_constructor_epilogue(handle, backendPtr);
}

auto dBackend_new2(
    uint64_t& handle,
    int nGpus,
    int runtime)
    -> int
{
    std::cout << "second constructor" << std::endl;
    AllocationCounter::Allocation();
    backend_constructor_prologue(handle);

    auto backendPtr = new (std::nothrow) Neon::Backend(nGpus, Neon::Runtime(runtime));

    return backend_constructor_epilogue(handle, backendPtr);
}

auto dBackend_new3(
    uint64_t& handle,
    const int* devIds,
    int runtime)
    -> int
{
    std::cout << "third constructor" << std::endl;

    backend_constructor_prologue(handle);

    auto backendPtr = new (std::nothrow) Neon::Backend(std::vector<int>(*devIds), Neon::Runtime(runtime));
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

    return backendPtr->toString().c_str();
}
