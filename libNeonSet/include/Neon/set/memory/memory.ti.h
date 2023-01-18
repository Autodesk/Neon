#pragma once
#include "Neon/set/memory/memory.h"

namespace Neon {
namespace set {

template <typename T_ta>
auto Memory::MemSet(const Neon::Backend&                      bk,
                    int                                 cardinality,
                    const Neon::set::DataSet<uint64_t>& nElementVec,
                    Neon::DataUse                       dataUse,
                    Neon::MemSetOptions_t               cpuConfig,
                    Neon::MemSetOptions_t               gpuConfig) -> MemSet_t<T_ta>
{
    auto& devSet = bk.devSet();

    if (bk.runtime() == Neon::Runtime::system) {
        NEON_THROW_UNSUPPORTED_OPTION("");
    }

    /**
     * If running host only, than we set GPU memory to null memory
     */
    if (bk.runtime() == Neon::Runtime::openmp) {
        gpuConfig.allocator(DeviceType::CUDA) = Neon::Allocator::NULL_MEM;
        gpuConfig.allocator(DeviceType::MPI) = Neon::Allocator::NULL_MEM;
    }

    /**
     * if we are running in COMPUTE mode, than the host side is set to null memory
     */
    if (dataUse == Neon::DataUse::COMPUTE && bk.runtime() == Neon::Runtime::stream) {
        cpuConfig.allocator(DeviceType::CPU) = Neon::Allocator::NULL_MEM;
        cpuConfig.allocator(DeviceType::OMP) = Neon::Allocator::NULL_MEM;
    }


    Neon::memLayout_et::order_e m_orderCPU = cpuConfig.order() == Neon::MemoryLayout::arrayOfStructs
                                                 ? Neon::memLayout_et::order_e::arrayOfStructs
                                                 : Neon::memLayout_et::order_e::structOfArrays;

    Neon::memLayout_et::order_e m_orderGPU = gpuConfig.order() == Neon::MemoryLayout::arrayOfStructs
                                                 ? Neon::memLayout_et::order_e::arrayOfStructs
                                                 : Neon::memLayout_et::order_e::structOfArrays;


    Neon::sys::memConf_t cpuSysConfig(Neon::DeviceType::CPU, cpuConfig.allocator(Neon::DeviceType::CPU), m_orderCPU);
    Neon::sys::memConf_t gpuSysConfig(Neon::DeviceType::CUDA, gpuConfig.allocator(Neon::DeviceType::CUDA), m_orderGPU);

    return devSet.template newMemSet<T_ta>(cardinality, cpuSysConfig, gpuSysConfig, nElementVec);
}

template <typename T_ta>
auto Memory::MemSet(const Neon::Backend&        bk,
                    int                   cardinality,
                    const uint64_t&       nElementInEachPartition,
                    Neon::DataUse         dataUse,
                    Neon::MemSetOptions_t cpuConfig,
                    Neon::MemSetOptions_t gpuConfig)
    -> MemSet_t<T_ta>
{
    auto dataSetSize = bk.devSet().template newDataSet<uint64_t>(nElementInEachPartition);
    return Neon::set::Memory::MemSet<T_ta>(bk, cardinality, dataSetSize, dataUse, cpuConfig, gpuConfig);
}

}  // namespace set
}  // namespace Neon
