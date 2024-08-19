#include <dlfcn.h>

#include <cuda.h>
#include <cudaTypedefs.h>

#include "Neon/Neon.h"
#include "Neon/core/core.h"
#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Loader.h"

namespace Neon::py {
class CudaDriver
{

    Neon::Backend backend;

    Neon::set::DataSet<CUdevice>  cu_devices;
    Neon::set::DataSet<CUcontext> cu_contexts;

public:
    using kernel = void*;


    template <typename String>
    inline auto check_cuda_res(CUresult const& res, String const& op) -> void
    {
        if (res != CUDA_SUCCESS) {
            const char* errorName;
            const char* errorString;

            cuGetErrorName(res, &errorName);
            cuGetErrorString(res, &errorString);
            Neon::NeonException e("CudaDriverEntryPoint");
            e << op << " failed with \n" << res << " ";
            e << errorName << "\n";
            e << errorString;
            NEON_THROW(e);
        }
        return;
    }

    /*
     * Constructor
     */
    CudaDriver(Neon::Backend* bk_prt);

    ~CudaDriver();

    auto get_bk_prt() -> Neon::Backend*;

    /*
     * Run a multi-gpu kernel in parallel
     */
    void run_kernel(
        Neon::set::DataSet<kernel> const&  kernelSet,
        Neon::set::LaunchParameters const& launch_params,
        Neon::StreamIdx                    streamIdx);


};
}

extern "C" int cuda_driver_new(void** handle, void* bk_handle);

extern "C" int cuda_driver_delete(void** handle);