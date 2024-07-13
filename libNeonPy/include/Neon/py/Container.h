#include <dlfcn.h>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <rapidjson/reader.h>

#include "Neon/Neon.h"
#include "Neon/core/core.h"
#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Loader.h"
#include "Neon/domain/interface/GridBase.h"
#include "Neon/domain/Grids.h"
#include "Neon/py/CudaDriver.h"

/**
*
*/
extern "C" void warp_dgrid_container_new(
    uint64_t&       out_handle,
    Neon::Execution execution,
    uint64_t&       handle_cudaDriver,
    uint64_t&       handle_dgrid,
    void**          kernels_standard,
    void**          kernels_internal,
    void**          kernels_boundary,
    Neon::index_3d* blockSize);

/**
*
*/
extern "C" void warp_container_delete(
    uint64_t& handle);

/**
*
*/
extern "C" void warp_container_run(
    uint64_t&      handle,
    int            streamIdx,
    Neon::DataView dataView);