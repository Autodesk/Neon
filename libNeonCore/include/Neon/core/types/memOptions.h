#pragma once
#include "MemoryLayout.h"
#include <string>

namespace Neon {

struct memPadding_e
{
    enum e
    {
        OFF = 0,
        ON = 1
    };
    static auto toString(int config) -> const char*;
};

struct memAlignment_e
{
    enum e
    {
        SYSTEM = 0,      /** Basic alignment provided by the allocator */
        L1 = 1,          /** Alignment based on L1 cache size */
        L2 = 2,          /** Alignment based on L2 cache size */
        PAGE = 3,        /** Alignment based on memory page size */
    };
    static auto toString(int config) -> const char*;
};

/**
 * Type of memory layout for either for scalar or vector types
 */
struct [[deprecated("This feature is going to be replaced by a new API for Neon 1.0")]] memLayout_et
{
    /**
     * In the coding of the layout we use the following standard:
     * 1. v stands for the vector component of the user type
     * 2. x,y,z represent the coordinates of a point in the 3D space where the scalar or vector field is defined
     * 3. xyzv means x pitch is less than y pitch which is less than z pitch which is less than v pitch...
     */
    enum [[deprecated("This feature is going to be replaced by a new API for Neon 1.0")]] order_e : int32_t{
        structOfArrays = 0,
        arrayOfStructs = 1};

    enum [[deprecated("This feature is going to be replaced by a new API for Neon 1.0")]] padding_e : int32_t{
        ON = 1,  /*  */
        OFF = 0, /* */
    };

    ~memLayout_et() = default;

    memLayout_et(order_e order = structOfArrays, padding_e padding = padding_e::OFF);
    memLayout_et(padding_e padding);

    memLayout_et::order_e   order() const;
    memLayout_et::padding_e padding() const;

    static const char* order2char(order_e order);
    static const char* padding2char(padding_e padding);

    const char* order2char() const;
    const char* padding2char() const;

    static auto convert(Neon::MemoryLayout order)->order_e;
    static auto convert(Neon::memPadding_e::e padding)->padding_e ;

   private:
    order_e   m_order{order_e::structOfArrays};
    padding_e m_padding{padding_e::OFF};
};

}  // namespace Neon
