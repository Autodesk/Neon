#pragma once
#include "Neon/set/Backend.h"
#include "Neon/set/dependencyTools/enum.h"
#include "Neon/skeleton/internal//dependencyTools/Alias.h"

namespace Neon::skeleton::internal {


struct Dependency
{
   private:
    ContainerIdx   m_t0 = 0;
    ContainerIdx   m_t1 = 0;
    Dependencies_e m_type = Dependencies_e::NONE;
    DataUId_t      m_uid = 0;

   public:
    /**
     * Empty constructor
     */
    Dependency() = default;

    /**
     * Defines a dependency of type e between kernel A and B.
     * Note, the order is important.
     *
     * Example: a read after write where kernel B reads results from kernel A
     *          Dependency_t(B, RAW, A)
     * @param A
     * @param B
     */
    Dependency(ContainerIdx t1, Dependencies_e type, DataUId_t m_uid, ContainerIdx t0);

    /**
     * true the object has been initialized with a valid dependency
     * @return
     */
    bool isValid();

    /**
     * Convert the dependency into a string
     * @return
     */
    auto toString() -> std::string;

    /**
     * Returns the dependency type
     * @return
     */
    auto type() -> Dependencies_e;

    /**
     * Returns the "before" kernel id
     * @return
     */
    auto t0() -> ContainerIdx;

    /**
     * Returns the "after" kernel id
     * @return
     */
    auto t1() -> ContainerIdx;

    /**
     * Static method to build an empty dependency
     * @return
     */
    static Dependency getEmpty();
};

}