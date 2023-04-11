#pragma once


#include "Neon/core/core.h"
#include "Neon/set/DataSet.h"

#include "Neon/sys/devices/DevInterface.h"
#include "Neon/sys/devices/gpu/GpuKernelInfo.h"


namespace Neon {
namespace set {

class DevSet;


class LaunchParameters
{
   public:
    friend class DevSet;

    using launchInfo_e = Neon::sys::GpuLaunchInfo;
    using mode_e = launchInfo_e::mode_e;

    std::vector<Neon::sys::GpuLaunchInfo> m_infoVec;

    LaunchParameters() = default;
    LaunchParameters(const Neon::set::DevSet& devSet);
    LaunchParameters(int setCardinality);

    LaunchParameters(const LaunchParameters&) = default;
    LaunchParameters(LaunchParameters&&) = default;
    LaunchParameters& operator=(const LaunchParameters&) = default;
    ~LaunchParameters() = default;

    Neon::sys::GpuLaunchInfo& operator[](int32_t);
    const Neon::sys::GpuLaunchInfo& operator[](int32_t) const;

    Neon::sys::GpuLaunchInfo&       get(int32_t id);
    const Neon::sys::GpuLaunchInfo& get(int32_t id) const;

    int cardinality() const;

    void set(Neon::sys::GpuLaunchInfo::mode_e    mode,
             const Neon::set::DataSet<index_3d>& gridDim,
             const index_3d&                     blockDim,
             size_t                              shareMemorySize);

    void set(Neon::sys::GpuLaunchInfo::mode_e mode,
             const Neon::SetIdx               setIdx,
             const index_3d&                  gridDim,
             const index_3d&                  blockDim,
             size_t                           shareMemorySize);

    auto set(Neon::sys::GpuLaunchInfo::mode_e mode,
             const Neon::SetIdx               setIdx,
             const int64_t&                   gridDim,
             const index_3d&                  blockDim,
             size_t                           shareMemorySize) -> void;

    auto set(Neon::sys::GpuLaunchInfo::mode_e mode,
             Neon::SetIdx                     setIdx,
             int64_t                          gridDim,
             index_t                          blockDim,
             size_t                           shareMemorySize) -> void;

    template <typename LambdaFun>
    auto forEachSeq(LambdaFun const& lambdaFun) -> void
    {
        for (int i = 0; i < m_infoVec.size(); i++) {
            LambdaFun(Neon::SetIdx(i), this->get(i));
        }
    }
};
// New name after refactoring: https://git.autodesk.com/Research/gd-Neon/issues/374
}  // namespace set
}  // End of namespace Neon
