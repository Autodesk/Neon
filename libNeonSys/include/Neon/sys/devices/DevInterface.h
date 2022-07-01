#pragma once

#include "Neon/core/core.h"

#include <string>


namespace Neon {


using run_et = Neon::run_et;
/**
 * Representing different types of devices
 */

namespace sys {
/**
 * Representing logical relative index when more than one device is present for a specific type.
 */
struct DeviceID
{
   private:
    int32_t m_idx{-1};

   public:
    /**
     * Construct a devIdx_t with a specific device ID
     * @param idx
     */
    DeviceID(int32_t idx)
        : m_idx(idx){};

    /**
     * Default constructor.
     */
    DeviceID() = default;

    /**
     * Returns the device ID managed by this object.
     */
    int32_t idx() const { return m_idx; }

    /**
     * Set the ID managed by this object to invalid
     */
    void setInvalid() { m_idx = -1; }

    /**
     * Returns true if the device Id managed by this object seems to be a valid ID.
     * @return
     */
    bool validate() const { return m_idx >= 0; }

    /**
     * Equal operator
     * @param other
     * @return
     */
    bool operator==(const DeviceID& other) const
    {
        return this->m_idx == other.m_idx;
    }

    /**
     * Non equal operator
     * @param other
     * @return
     */
    bool operator!=(const DeviceID& other) const
    {
        return this->m_idx != other.m_idx;
    }
};

/**
 * Interface for any device.
 */
class DeviceInterface
{
   protected:
    DeviceType typeId;
    DeviceID   idx; /**< used to identify devices of the same type. For example all GPU in the system */

    /**
     * Set the device type
     */
    void setType(DeviceType typeId);

   public:
    /**
     * Constructor for a device with specific type and ID
     * @param devType
     * @param devIdx
     */
    DeviceInterface(const DeviceType& devType, const DeviceID& devIdx);
    /**
     * Constructor for a device with specific type.
     * This device is unique and therefore ID is not given and automatically set to zero.
     * @param devType
     */
    DeviceInterface(const DeviceType& devType);

    /**
     * Default constructor.
     */
    DeviceInterface() = default;

    /**
     * Set the device relative index, used for devices of the same type
     */
    void setIdx(DeviceID _idx);

    /**
     * Get the device relative index, used for devices of the same type
     */
    const DeviceID& getIdx() const;

    /**
     * Returns the device type
     */
    const DeviceType& getType() const;

    /**
     * Returns a string associated to the device type
     */
    virtual std::string type() const;

    /**
     * Returns device usage from 0 to one.
     * Returns -1 if the device/OS does not provide such information.
     */
    virtual double usage() const = 0;

    /**
     * Returns the size of available virtual memory
     */
    virtual int64_t virtMemory() const = 0;

    /**
     * Returns the size of available physical memory
     * Returns -1 if the os does not provide such information
     */
    virtual int64_t physMemory() const = 0;

    /**
     * Returns the size of used virtual memory
     * Returns -1 if the os does not provide such information
     */
    virtual int64_t usedVirtMemory() const = 0;

    /**
     * Returns the size of used physical memory
     * Returns -1 if the os does not provide such information
     */
    virtual int64_t usedPhysMemory() const = 0;

    /**
     * Returns the size of free virtual memory
     * Returns -1 if the os does not provide such information
     */
    int64_t freeVirtMemory() const;

    /**
     * Returns the size of free physical memory
     * Returns -1 if the os does not provide such information
     */
    int64_t freePhysMemory() const;

    /**
     * Returns the size of physical memory used by this process.
     * Returns -1 if the os does not provide such information
     */
    virtual int64_t processUsedPhysMemory() const = 0;

    /**
     * Returns a string of available virtual memory
     */
    std::string virtMemoryStr() const;

    /**
     * Returns a string of used virtual memory
     */
    std::string usedVirtMemoryStr() const;

    /**
     * Returns a string of free virtual memory
     */
    std::string freeVirtMemoryStr() const;

    /**
     * Returns a string of available physical memory
     */
    std::string physMemoryStr() const;

    /**
     * Returns a string of used physical memory
     */
    std::string usedPhysMemoryStr() const;

    /**
     * Returns a string of free physical memory
     */
    std::string freePhysMemoryStr() const;

    /**
     * Returns a string of used physical memory
     */
    std::string processUsedPhysMemoryStr() const;

    /**
     * Returns a string providing a full status of the device
     */
    std::string info(const std::string& prefix = std::string("")) const;

   private:
    static std::string getSizeWithMemoryUnit(int64_t size);

    static std::string getLoad(double load);
};

std::ostream& operator<<(std::ostream& os, DeviceID const& m);
std::ostream& operator<<(std::ostream& os, DeviceInterface const& m);

}  // namespace sys
}  // End of namespace Neon
