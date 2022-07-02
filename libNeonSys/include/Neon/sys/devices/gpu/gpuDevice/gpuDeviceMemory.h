#pragma once
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127)  // C4127 conditional expression is constant
#pragma warning(disable : 4146)  // C4146 unary minus operator applied to unsigned type, result
                                 // still unsigned
#pragma warning(disable : 4211)  // C4211 nonstandard extension used: redefined extern to static
#pragma warning(disable : 4244)  // C4244 conversion from '*' to '*', possible loss of data
#pragma warning(disable : 4251)  // C4251 non dll-interface class 'std::exception' used as base
                                 // for dll-interface class
#pragma warning(disable : 4275)  // C4275	non dll-interface class '...' used as base for
                                 // dll-interface class '...'
#pragma warning(disable : 4706)  // C4706 assignment within conditional expression
#pragma warning(disable : 5033)  // C5033 'register' is no longer a supported storage class
#endif
namespace Neon {
namespace sys {


/**
 * Wrapper around cudaMalloc.
 */
template <typename Type_ta>
Type_ta* GpuDevice::memory_t::malloc(int64_t nElements) const
{
    Type_ta*     mem = nullptr;
    const size_t allocationSize = nElements * sizeof(Type_ta);

    gpuDev.tools.setActiveDevContext();
    cudaError_t res = cudaMalloc((void**)&mem, allocationSize);

    if (res != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaMalloc operation: "
            << "\n   target GPU         " << gpuDev.getIdx()
            << "\n   number of el.      " << nElements
            << "\n   memory size        " << allocationSize
            << "\n   size one element   " << sizeof(Type_ta);
        exc << "\n Error: " << cudaGetErrorString(res);

        NEON_THROW(exc);
    }

    return mem;
}


/**
 * cudaFree.
 */
template <typename Type_ta>
void GpuDevice::memory_t::free(Type_ta*& mem) const
{
    gpuDev.tools.setActiveDevContext();
    cudaError_t res = cudaFree(mem);

    if (res != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaFree operation: "
            << "\n   target GPU       " << gpuDev.getIdx()
            << "\n   address          " << mem;
        NEON_THROW(exc);
    }
    mem = nullptr;
}
/**
 * cudaFree.
 */
template <typename Type_ta>
void GpuDevice::memory_t::free(const Type_ta*& mem) const
{

    gpuDev.tools.setActiveDevContext();
    cudaError_t res = cudaFree(mem);

    if (res != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaFree operation: "
            << "\n   target GPU       " << gpuDev.getIdx()
            << "\n   address          " << mem;
        NEON_THROW(exc);
    }
}


/**
 * Wrapper around cudaMemcpyAsync.
 */
template <typename Type_ta, mem_et::enum_e dest_ta, mem_et::enum_e src_ta, run_et::et runMode>
void GpuDevice::memory_t::transfer(const GpuStream& gpuStream, Type_ta* dest, const Type_ta* src, const int64_t nElemensts) const
{
    cudaError_t res;

    /**
     * cpu to cpu transfers are not supported.
     */
    if (src_ta == dest_ta && (src_ta == mem_et::cpu)) {
        NeonException exc;
        exc << "GpuDevicedo not support cpu to cpu transfer.";
        NEON_THROW(exc);
    }
    /**
     * targeting this device.
     */
    gpuDev.tools.setActiveDevContext();

    /**
     * handling a host to device transfer
     */
    if (src_ta == mem_et::cpu && dest_ta == mem_et::gpu) {
        res = cudaMemcpyAsync(
            dest, src, nElemensts * sizeof(Type_ta), cudaMemcpyHostToDevice, gpuStream.stream());
        if (res == cudaSuccess) {
            gpuStream.sync<runMode>();
            return;
        }
    }

    /**
     * handling a device to host transfer
     */
    if (src_ta == mem_et::gpu && dest_ta == mem_et::cpu) {
        res = cudaMemcpyAsync(
            dest, src, nElemensts * sizeof(Type_ta), cudaMemcpyDeviceToHost, gpuStream.stream());
        if (res == cudaSuccess) {
            gpuStream.sync<runMode>();
            return;
        }
    }

    /**
     * handling a device to device transfer
     */
    if (src_ta == mem_et::gpu && dest_ta == mem_et::gpu) {
        res = cudaMemcpyAsync(
            dest, src, nElemensts * sizeof(Type_ta), cudaMemcpyDeviceToDevice, gpuStream.stream());
        if (res == cudaSuccess) {
            gpuStream.sync<runMode>();
            return;
        }
    }

    NeonException exc;
    exc << "CUDA error completing cudaMemcpy operation: "
        << "to " << (void*)dest << " (" << mem_et::name(dest_ta)
        << ") "
        << "from " << (void*)src << " (" << std::string(mem_et::name(src_ta)) << "). ";
    exc << "\n Error: " << cudaGetErrorString(res);
    NEON_THROW(exc);
}

}  // namespace sys
}  // End of namespace Neon


#ifdef _MSC_VER
#pragma warning(pop)
#endif
