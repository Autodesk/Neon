#include "Neon/py/backend.h"
#include "Neon/set/Backend.h"

void backend_constructor_prologue(uint64_t& handle) {
    std::cout << "dBackend_new - BEGIN" << std::endl;
    std::cout << "dBackend handle" << handle << std::endl;
}

int backend_constructor_epilogue(uint64_t& handle, Neon::Backend* backendPtr) {
    if (backendPtr == nullptr) {
        std::cout << "NeonPy: Initialization error. Unable to allocage backend " << std::endl;
        return -1;
    }
    handle = (uint64_t)backendPtr;
    std::cout << "allocated backend heap location: " << backendPtr << std::endl;
    std::cout << "grid_new - END" << std::endl;
    return 0;
}

auto dBackend_new_default(
    uint64_t& handle)
    -> int
{
    return dBackend_new(handle, 1, Neon::Runtime::openmp);
}

auto dBackend_new(
    uint64_t& handle)
    -> int
{
    backend_constructor_prologue(handle);

    auto backendPtr = new (std::nothrow) Neon::Backend();

    return backend_constructor_epilogue(handle, backendPtr);
}

auto dBackend_new(
    uint64_t& handle,
    int nGpus,
    Neon::Runtime runtime)
    -> int
{
    backend_constructor_prologue(handle);

    auto backendPtr = new (std::nothrow) Neon::Backend(nGpus, runtime);

    return backend_constructor_epilogue(handle, backendPtr);
}

auto dBackend_new(
    uint64_t& handle,
    const std::vector<int>& devIds,
    Neon::Runtime runtime)
    -> int
{
    backend_constructor_prologue(handle);

    auto backendPtr = new (std::nothrow) Neon::Backend(devIds, runtime);

    return backend_constructor_epilogue(handle, backendPtr);
}

auto dBackend_new(
    uint64_t& handle,
    const Neon::set::DevSet& devSet,
    Neon::Runtime runtime)
    -> int
{
    backend_constructor_prologue(handle);

    auto backendPtr = new (std::nothrow) Neon::Backend(devSet, runtime);

    return backend_constructor_epilogue(handle, backendPtr);
}

auto dBackend_new(
    uint64_t& handle,
    const std::vector<int>& devIds,
    const Neon::set::StreamSet& streamSet)
    -> int
{
    backend_constructor_prologue(handle);

    auto backendPtr = new (std::nothrow) Neon::Backend(devIds, streamSet);

    return backend_constructor_epilogue(handle, backendPtr);
}

auto dBackend_new(
    uint64_t& handle,
    const std::vector<int>& devSet,
    const Neon::set::StreamSet& streamSet)
    -> int
{
    backend_constructor_prologue(handle);

    auto backendPtr = new (std::nothrow) Neon::Backend(devSet, streamSet);

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
    }
    handle = 0;
    std::cout << "dBackend_delete - END" << std::endl;
    return 0;
}