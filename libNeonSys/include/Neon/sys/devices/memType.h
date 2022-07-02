#pragma once

#include <string>


namespace Neon {
namespace sys {

struct mem_et
{
    enum enum_e : int32_t
    {
        cpu = 0 /* used for cpu buffers */
        ,
        gpu = 1 /* used for cpu buffers */
        ,
        omp = cpu /* used for cpu buffers */
        ,
        host = cpu /* another name for cpu*/
        ,
        device = 2 /* general name for a device memory. It can be GPU or other */
    };

    mem_et(enum_e type);

    mem_et::enum_e type() const;

    static const char* name(enum_e type);

    const char* name() const;

   private:
    enum_e m_type;
};

}  // namespace sys
}  // namespace Neon
