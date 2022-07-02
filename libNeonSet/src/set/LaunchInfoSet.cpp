#include "Neon/set/LaunchParameters.h"
#include "Neon/set/DevSet.h"

namespace Neon {
namespace set {


LaunchParameters::LaunchParameters(const Neon::set::DevSet& gpuSet)
    : LaunchParameters(gpuSet.setCardinality())
{
    //
}

LaunchParameters::LaunchParameters(int setCardinality)
{
    m_infoVec = std::vector<Neon::sys::GpuLaunchInfo>(setCardinality);
}

Neon::sys::GpuLaunchInfo& LaunchParameters::operator[](const int32_t idx)
{
    return m_infoVec.at(idx);
}
const Neon::sys::GpuLaunchInfo& LaunchParameters::operator[](const int32_t idx) const
{
    return m_infoVec.at(idx);
}

Neon::sys::GpuLaunchInfo& LaunchParameters::get(int32_t idx)
{
    return m_infoVec.at(idx);
}
const Neon::sys::GpuLaunchInfo& LaunchParameters::get(int32_t idx) const
{
    return m_infoVec.at(idx);
}

int LaunchParameters::cardinality() const
{
    return int(m_infoVec.size());
}

void LaunchParameters::set(Neon::sys::GpuLaunchInfo::mode_e    mode,
                          const Neon::set::DataSet<index_3d>& gridDims,
                          const index_3d&                       blockDim,
                          size_t                                shareMemorySize)
{
    for (int i = 0; i < int(gridDims.size()); i++) {
        m_infoVec.at(i).set(mode,
                         gridDims[i],
                         blockDim,
                         shareMemorySize);
    }
}

void LaunchParameters::set(Neon::sys::GpuLaunchInfo::mode_e mode,
                          const Neon::SetIdx                 setIdx,
                          const index_3d&                    gridDim,
                          const index_3d&                    blockDim,
                          size_t                             shareMemorySize)
{
    m_infoVec.at(setIdx).set(mode,
                          gridDim,
                          blockDim,
                          shareMemorySize);
}

auto LaunchParameters::set(Neon::sys::GpuLaunchInfo::mode_e mode,
                          const Neon::SetIdx                 setIdx,
                          const int64_t&                     gridDim,
                          const index_3d&                    blockDim,
                          size_t                             shareMemorySize) -> void
{
    m_infoVec.at(setIdx).set(mode,
                          gridDim,
                          blockDim,
                          shareMemorySize);
}


auto LaunchParameters::set(Neon::sys::GpuLaunchInfo::mode_e mode,
                          Neon::SetIdx                       setIdx,
                          int64_t                            gridDim,
                          index_t                            blockDim,
                          size_t                             shareMemorySize) -> void
{
    m_infoVec.at(setIdx).set(mode,
                          gridDim,
                          blockDim,
                          shareMemorySize);
}


}  // namespace set
}  // End of namespace Neon
