#include "Neon/sys/devices/gpu/GpuKernelInfo.h"

namespace Neon::sys {

template <typename T>
void GpuLaunchInfo::m_init(mode_e mode, const T& gridDim, const T& blockDim, size_t shareMemorySize)
{
    mOriginalMode = mode;
    mOriginalGridDim = int64_3d(gridDim.x, gridDim.y, gridDim.z);
    mOriginalBlockDim = int64_3d(blockDim.x, blockDim.y, blockDim.z);

    index64_3d cudaGrid{0, 0, 0};
    index64_3d cudaBlock(int64_t(blockDim.x), int64_t(blockDim.y), int64_t(blockDim.z));

    switch (mode) {
        case cudaGridMode: { /* do nothing */
            cudaGrid.set(int64_t(gridDim.x), int64_t(gridDim.y), int64_t(gridDim.z));
            break;
        }
        case domainGridMode: { /* compute the actual grid value*/
            cudaGrid = mOriginalGridDim.cudaGridDim(cudaBlock);
            break;
        }
    }

    for (int i = 0; i < cudaGrid.num_axis; i++) {
        if (cudaGrid.v[i] > int64_t(std::numeric_limits<int>::max())) {
            NeonException exp("GpuLaunchInfo_t");
            exp << "Launch configuration allocates too may blocks";
            NEON_THROW(exp);
        }
        if (cudaBlock.v[i] > int64_t(std::numeric_limits<int>::max())) {
            NeonException exp("GpuLaunchInfo_t");
            exp << "Launch configuration allocates too may blocks";
            NEON_THROW(exp);
        }
    }

    this->mThreadGrid = dim3(int(cudaGrid.x), int(cudaGrid.y), int(cudaGrid.z));
    this->mThreadBlock = dim3(int(cudaBlock.x), int(cudaBlock.y), int(cudaBlock.z));
    this->mShareMemorySize = shareMemorySize;

    this->mInitDone = true;
}

template void GpuLaunchInfo::m_init<dim3>(mode_e mode, const dim3& gridDim, const dim3& blockDim, size_t shareMemorySize);
template void GpuLaunchInfo::m_init<index_3d>(mode_e mode, const index_3d& gridDim, const index_3d& blockDim, size_t shareMemorySize);
template void GpuLaunchInfo::m_init<int64_3d>(mode_e mode, const int64_3d& gridDim, const int64_3d& blockDim, size_t shareMemorySize);


GpuLaunchInfo::GpuLaunchInfo(GpuLaunchInfo::mode_e mode, const index_3d& gridDim, const index_3d& blockDim, size_t shareMemorySize)
{
    m_init(mode, gridDim, blockDim, shareMemorySize);
}

void GpuLaunchInfo::set(mode_e mode, const int32_3d& gridDim, const int32_3d& blockDim, size_t shareMemorySize)
{
    m_init(mode, gridDim, blockDim, shareMemorySize);
}

void GpuLaunchInfo::set(mode_e mode, const int64_t& gridDim, const int32_3d& blockDim, size_t shareMemorySize)
{
    index64_3d gridD(gridDim, 1, 1);
    index64_3d blockD(blockDim.x, blockDim.y, blockDim.z);
    m_init(mode, gridD, blockD, shareMemorySize);
}

void GpuLaunchInfo::set(mode_e mode, const int64_t& gridDim, const int32_t& blockDim, size_t shareMemorySize)
{
    index64_3d gridD(gridDim, 1, 1);
    index64_3d blockD(blockDim, 1, 1);
    m_init(mode, gridD, blockD, shareMemorySize);
}

auto GpuLaunchInfo::set(mode_e          mode,
                        const int64_3d& gridDim,
                        const int32_3d& blockDim,
                        size_t          shareMemorySize)
    -> void
{
    m_init(mode, gridDim, blockDim.template newType<int64_t>(), shareMemorySize);
}

auto GpuLaunchInfo::setShm(size_t shareMemorySize) -> void
{
    mShareMemorySize = shareMemorySize;
}


void GpuLaunchInfo::reset()
{
    mInitDone = (false);

    mThreadGrid.x = 0;
    mThreadGrid.y = 0;
    mThreadGrid.z = 0;

    mThreadBlock.x = 0;
    mThreadBlock.y = 0;
    mThreadBlock.z = 0;

    mShareMemorySize = 0;
}

bool GpuLaunchInfo::isValid() const
{
    return mInitDone;
}

const dim3& GpuLaunchInfo::cudaGrid() const
{
    return mThreadGrid;
}

const dim3& GpuLaunchInfo::cudaBlock() const
{
    return mThreadBlock;
}

const size_t& GpuLaunchInfo::shrMemSize() const
{
    return mShareMemorySize;
}

const int64_3d& GpuLaunchInfo::domainGrid() const
{
    return mOriginalGridDim;
}

GpuLaunchInfo::GpuLaunchInfo(GpuLaunchInfo::mode_e mode, const index64_3d& gridDim, const index_3d& blockDim, size_t shareMemorySize)
{
    auto blockDim64 = blockDim.newType<int64_t>();
    m_init(mode, gridDim, blockDim64, shareMemorySize);
}

GpuLaunchInfo::GpuLaunchInfo(GpuLaunchInfo::mode_e mode, const int64_t& gridDim, const index_t& blockDim, size_t shareMemorySize)
{
    int64_3d gridDim64(gridDim, 1, 1);
    int64_3d blockDim64 = index64_3d(blockDim, 1, 1);

    m_init(mode, gridDim64, blockDim64, shareMemorySize);
}

GpuLaunchInfo::GpuLaunchInfo(const index64_3d& gridDim, const index_3d& blockDim, size_t shareMemorySize)
    : GpuLaunchInfo(GpuLaunchInfo::mode_e::domainGridMode, gridDim, blockDim, shareMemorySize)
{
}

GpuLaunchInfo::GpuLaunchInfo(const int64_t& gridDim, const index_t& blockDim, size_t shareMemorySize)
    : GpuLaunchInfo(GpuLaunchInfo::mode_e::domainGridMode, gridDim, blockDim, shareMemorySize)
{
}

GpuLaunchInfo::GpuLaunchInfo(const index_3d& gridDim, const index_3d& blockDim, size_t shareMemorySize)
    : GpuLaunchInfo(GpuLaunchInfo::mode_e::domainGridMode, gridDim, blockDim, shareMemorySize)
{
}

index64_3d GpuLaunchInfo::getWaste() const
{
    auto       cudaG = cudaGrid();
    auto       cudaB = cudaBlock();
    index64_3d G(cudaG.x, cudaG.y, cudaG.z);
    index64_3d B(cudaB.x, cudaB.y, cudaB.z);
    index64_3d waste = B * G - domainGrid();
    return waste;
}

}  // namespace Neon::sys
