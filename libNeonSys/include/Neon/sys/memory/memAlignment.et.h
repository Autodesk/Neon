#pragma once


#include <string>

namespace Neon {
namespace sys {

/**
 * Type of memory layout for 3D memory either for scalar or vector types
 */
struct [[deprecated("This feature is going to be replaced by a new API for Neon 1.0")]] memAlignment_et
{
    enum enum_e : int32_t
    {
        cacheLine = 0, /* Aligned to cache line size*/
        page = 1,      /* Aligned to the page size */
        system = 2,    /* Aligned as the target allocator decides */
        user = 3       /* Aligned to the size decided by user */
    };

    ~memAlignment_et() = default;

    memAlignment_et() = default;

    memAlignment_et(enum_e type);

    memAlignment_et::enum_e type() const;

    static const char* name(enum_e type);

    const char* name() const;

    bool operator==(const memAlignment_et& other)
    {
        bool equal = true;
        equal = (m_type == other.m_type) ? equal : false;
        return equal;
    }

   private:
    enum_e m_type{system};
};

}  // namespace sys
}  // namespace Neon
