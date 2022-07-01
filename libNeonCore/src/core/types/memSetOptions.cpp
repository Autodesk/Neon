#include "Neon/core/types/memSetOptions.h"
#include "Neon/core/core.h"
namespace Neon {

MemSetOptions_t::MemSetOptions_t()
{
    m_allocatorByDev[static_cast<int>(Neon::DeviceType::CUDA)] = Neon::Allocator::CUDA_MEM_DEVICE;
    m_allocatorByDev[static_cast<int>(Neon::DeviceType::CPU)] = Neon::Allocator::MALLOC;
    m_allocatorByDev[static_cast<int>(Neon::DeviceType::MPI)] = Neon::Allocator::MALLOC;
    m_allocatorByDev[static_cast<int>(Neon::DeviceType::OMP)] = Neon::Allocator::MALLOC;
}

auto MemSetOptions_t::order() -> MemoryLayout&
{
    return m_memOrder;
}

auto MemSetOptions_t::padding() -> memPadding_e::e&
{
    return m_memPadding;
}

auto MemSetOptions_t::alignment() -> memAlignment_e::e&
{
    return m_memAlignment;
}

auto MemSetOptions_t::allocator(Neon::DeviceType devType) -> Neon::Allocator&
{
    return m_allocatorByDev[static_cast<int>(devType)];
}

auto MemSetOptions_t::order() const -> const MemoryLayout&
{
    return m_memOrder;
}

auto MemSetOptions_t::padding() const -> const memPadding_e::e&
{
    return m_memPadding;
}

auto MemSetOptions_t::alignment() const -> const memAlignment_e::e&
{
    return m_memAlignment;
}

auto MemSetOptions_t::allocator(Neon::DeviceType devType) const -> const Neon::Allocator&
{
    return m_allocatorByDev[static_cast<int>(devType)];
}

auto MemSetOptions_t::toString() const -> std::string
{
    std::string name = std::string(MemoryLayoutUtils::toString(m_memOrder)) + "_" +
                       memPadding_e::toString(m_memPadding) + "_" +
                       memAlignment_e::toString(m_memAlignment);
    return name;
}

}  // namespace Neon
