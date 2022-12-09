#include <omp.h>
#include <vector>
#include "Neon/core/core.h"
#include "Neon/set/GpuStreamSet.h"
#include "Neon/sys/devices/gpu/ComputeID.h"
#include "Neon/sys/global/GpuSysGlobal.h"

namespace Neon {
namespace set {


StreamSet::StreamSet(int setCardinality)
{
    m_streamVec = std::vector<Neon::sys::GpuStream>(setCardinality);
}


void StreamSet::release()
{
    for (auto&& cudaStream : m_streamVec) {
        cudaStream.release();
    }
}


auto StreamSet::set(SetIdx id, Neon::sys::GpuStream&& gpuStream) -> void
{
    if (id.idx() > int32_t(m_streamVec.size()) || m_streamVec.empty()) {
        if (m_streamVec.empty()) {
            Neon::NeonException exp("GpuStreamSet");
            exp << "Incompatible stream id " << id << ". This stream set was not initialized";
            NEON_THROW(exp);
        } else {
            Neon::NeonException exp("GpuStreamSet");
            exp << "Incompatible stream id " << id << ". Id range is: 0," << m_streamVec.size();
            NEON_THROW(exp);
        }
    }
    m_streamVec[id.idx()] = gpuStream;
}


auto StreamSet::get(SetIdx id)
    const
    -> const Neon::sys::GpuStream&
{
    validateId(id);
    return m_streamVec[id.idx()];
}


auto StreamSet::get(SetIdx id)
    -> Neon::sys::GpuStream&
{
    validateId(id);
    return m_streamVec[id.idx()];
}


auto StreamSet::enqueueEvent(GpuEventSet& eventSet) const
    -> void
{
    const int ndevs = int(m_streamVec.size());
#pragma omp parallel num_threads(ndevs)
    {
        int tid = omp_get_thread_num();
        m_streamVec.at(tid).enqueueEvent(eventSet.event<Neon::Access::readWrite>(tid));
    }
}

auto StreamSet::enqueueEvent(Neon::SetIdx setIdx,
                             GpuEventSet& eventSet) const
    -> void
{
    m_streamVec.at(setIdx.idx()).enqueueEvent(eventSet.event<Neon::Access::readWrite>(setIdx.idx()));
}

auto StreamSet::waitForEvent(GpuEventSet& eventSet) const
    -> void
{
    const int ndevs = int(m_streamVec.size());
#pragma omp parallel num_threads(ndevs)
    {
        int tid = omp_get_thread_num();
        m_streamVec.at(tid).waitForEvent(eventSet.event<Neon::Access::readWrite>(tid));
    }
}

auto StreamSet::waitForEvent(Neon::SetIdx setIdx, GpuEventSet& eventSet) const
    -> void
{
    m_streamVec.at(setIdx.idx()).waitForEvent(eventSet.event<Neon::Access::readWrite>(setIdx.idx()));
}


auto StreamSet::validateId(SetIdx id)
    const
    -> void
{
    if (id.idx() > int32_t(m_streamVec.size()) || m_streamVec.empty()) {
        if (m_streamVec.empty()) {
            Neon::NeonException exp("GpuStreamSet");
            exp << "Incompatible stream id " << id << ". This stream set was not initialized";
            NEON_THROW(exp);
        } else {
            Neon::NeonException exp("GpuStreamSet");
            exp << "Incompatible stream id " << id << ". Id range is: 0," << m_streamVec.size();
            NEON_THROW(exp);
        }
    }
}
}  // namespace set
}  // namespace Neon
