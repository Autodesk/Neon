#include "Neon/sys/devices/DevInterface.h"

#include <iomanip>  // std::setprecision
#include <sstream>  // std::ostringstream
#include <vector>


namespace Neon {


namespace sys {
//===============================================
//=================  Device_i   =================
//===============================================

DeviceInterface::DeviceInterface(const DeviceType& devType)
    : typeId(devType) {}
DeviceInterface::DeviceInterface(const DeviceType& devType, const DeviceID& devIdx)
    : typeId(devType), idx(devIdx) {}


std::string DeviceInterface::getSizeWithMemoryUnit(int64_t size)
{
    std::ostringstream ss;

    // Byte
    double sizeD = static_cast<double>(size);
    if (sizeD < 1000) {
        ss << std::setprecision(3) << sizeD << " Byte";
        return ss.str();
    }
    // Kbyte
    sizeD /= 1024;
    if (sizeD < 1000) {
        ss << std::setprecision(3) << sizeD << " MByte";
        return ss.str();
    }
    // Mbyte
    sizeD /= 1024;
    if (sizeD < 1000) {
        ss << std::setprecision(3) << sizeD << " GByte";
        return ss.str();
    }
    // Gbyte
    sizeD /= 1024;
    ss << std::setprecision(3) << sizeD << " TByte";
    return ss.str();
}

std::string DeviceInterface::getLoad(double load)
{
    std::ostringstream ss;
    ss << std::setprecision(0) << load * 100 << "%";
    return ss.str();
}


std::string DeviceInterface::virtMemoryStr() const
{
    return getSizeWithMemoryUnit(this->virtMemory());
}

std::string DeviceInterface::physMemoryStr() const
{
    return getSizeWithMemoryUnit(this->physMemory());
}

std::string DeviceInterface::usedVirtMemoryStr() const
{
    return getSizeWithMemoryUnit(this->physMemory());
}

std::string DeviceInterface::usedPhysMemoryStr() const
{
    return getSizeWithMemoryUnit(this->usedPhysMemory());
}

std::string DeviceInterface::freeVirtMemoryStr() const
{
    return getSizeWithMemoryUnit(this->freeVirtMemory());
}

std::string DeviceInterface::freePhysMemoryStr() const
{
    return getSizeWithMemoryUnit(this->freeVirtMemory());
}

std::string DeviceInterface::processUsedPhysMemoryStr() const
{
    return getSizeWithMemoryUnit(this->processUsedPhysMemory());
}

std::string DeviceInterface::info(const std::string& prefix) const
{
    std::ostringstream os;
    os << prefix << "Device => Type: " << this->type() << " Idx: " << idx << "\n";
    os << prefix << "Device load " << this->physMemoryStr() << " \tused " << this->usedPhysMemoryStr() << " \tfree " << this->freePhysMemoryStr() << "\n";
    os << prefix << "System phys. Mem. " << this->physMemoryStr() << " \tused " << this->usedPhysMemoryStr() << " \tfree " << this->freePhysMemoryStr() << "\n";
    os << prefix << "System virt. Mem. " << this->virtMemoryStr() << " \tused " << this->usedVirtMemoryStr() << " \tfree " << this->freeVirtMemoryStr() << "\n";
    os << prefix << "Process used physical memory:  " << processUsedPhysMemoryStr() << "\n";
    return os.str();
}

void DeviceInterface::setType(DeviceType _typeId)
{
    this->typeId = _typeId;
}
const DeviceType& DeviceInterface::getType() const
{
    return this->typeId;
}

void DeviceInterface::setIdx(DeviceID _idx)
{
    this->idx = _idx;
}
const DeviceID& DeviceInterface::getIdx() const
{
    return this->idx;
};

std::string DeviceInterface::type() const
{
    return std::string(Neon::DeviceTypeUtil::toString(typeId));
}

int64_t DeviceInterface::freePhysMemory() const
{
    const auto used = this->usedPhysMemory();
    const auto sys = this->physMemory();
    if (sys == -1 || used == -1) {
        return -1;
    }
    return sys - used;
}

int64_t DeviceInterface::freeVirtMemory() const
{
    const int64_t used = this->usedVirtMemory();
    const int64_t all = this->virtMemory();

    if (used == -1 || all == -1) {
        return -1;
    }

    return all - used;
}

std::ostream& operator<<(std::ostream& os, DeviceID const& m)
{
    return os << m.idx();
}
std::ostream& operator<<(std::ostream& os, DeviceInterface const& m)
{
    return os << m.info();
}
}

}  // End of namespace Neon
