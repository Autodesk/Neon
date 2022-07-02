#pragma once

#include "Neon/core/core.h"
#include "Neon/sys/devices/DevInterface.h"
#include "cuda.h"
#include "cuda_runtime.h"

namespace Neon {
namespace sys {

/**
 * Structure that model information relative to CUDA parameters required to run a kernel.
 */
struct GpuLaunchInfo
{
   public:
    /** defines the semantic of the constructor parameter for GpuKernelInfo_t */
    enum mode_e
    {
        cudaGridMode /**< Input of GpuKernelInfo_t constructor is of a cuda grid */,
        domainGridMode /**< Input of GpuKernelInfo_t constructor is of the domain grid,
                        * the actual cuda grid needs to be computed by the constructor. */
    };

   private:
    dim3   m_cudaGrid /**< */;
    dim3   m_cudaBlock /**< */;
    size_t m_shareMemorySize{0} /**< */;

    mode_e   m_originalMode{mode_e::cudaGridMode};
    int64_3d m_originalGridDim{0, 0, 0};
    int64_3d m_originalBlockDim{0, 0, 0};

    bool m_InitDone{false} /**< */;

    template <typename T>
    void m_init(mode_e mode, const T& gridDim, const T& blockDim, size_t shareMemorySize);

   public:
    GpuLaunchInfo() = default;

    GpuLaunchInfo(GpuLaunchInfo::mode_e mode,
                  const index64_3d&     gridDim,
                  const index_3d&       blockDim,
                  size_t                shareMemorySize);

    GpuLaunchInfo(GpuLaunchInfo::mode_e mode,
                  const int64_t&        gridDim,
                  const index_t&        blockDim,
                  size_t                shareMemorySize);

    GpuLaunchInfo(GpuLaunchInfo::mode_e mode,
                  const index_3d&       gridDim,
                  const index_3d&       blockDim,
                  size_t                shareMemorySize);

    GpuLaunchInfo(const index64_3d& gridDim,
                  const index_3d&   blockDim,
                  size_t            shareMemorySize);

    GpuLaunchInfo(const int64_t& gridDim,
                  const index_t& blockDim,
                  size_t         shareMemorySize);

    GpuLaunchInfo(const index_3d& gridDim,
                  const index_3d& blockDim,
                  size_t          shareMemorySize);

    auto set(mode_e          mode,
             const int32_3d& gridDim,
             const int32_3d& blockDim,
             size_t          shareMemorySize)
        -> void;

    auto set(mode_e          mode,
             const int64_t&  gridDim,
             const int32_3d& blockDim,
             size_t          shareMemorySize)
        -> void;

    auto set(mode_e         mode,
             const int64_t& gridDim,
             const int32_t& blockDim,
             size_t         shareMemorySize)
        -> void;

    auto set(mode_e          mode,
             const int64_3d& gridDim,
             const int32_3d& blockDim,
             size_t          shareMemorySize)
        -> void;

    auto setShm(size_t shareMemorySize) -> void;

    void            reset();
    bool            isValid() const;
    const dim3&     cudaGrid() const;
    const dim3&     cudaBlock() const;
    const size_t&   shrMemSize() const;
    const int64_3d& domainGrid() const;

    index64_3d waste() const;
};


}  // namespace sys
}  // End of namespace Neon
