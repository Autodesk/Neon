#pragma once
#include "Neon/set/Backend.h"
#include "Neon/set/container/Graph.h"
#include "Neon/set/dependency/Alias.h"
#include "Neon/set/dependency/DataDependencyType.h"

namespace Neon::skeleton::internal {


struct Dependency
{
   private:
    Neon::set::container::GraphData::Uid               mT0 = 0;
    Neon::set::container::GraphData::Uid               mT1 = 0;
    Neon::internal::dataDependency::DataDependencyType mType = Neon::internal::dataDependency::DataDependencyType::NONE;
    Neon::internal::dataDependency::DataUId            mDataUid = 0;

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
    Dependency(Neon::set::container::GraphData::Uid               t1,
               Neon::internal::dataDependency::DataDependencyType type,
               Neon::internal::dataDependency::DataUId            m_uid,
               Neon::set::container::GraphData::Uid               t0);

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
    auto type() -> Neon::internal::dataDependency::DataDependencyType;

    /**
     * Returns the "before" kernel id
     * @return
     */
    auto t0() -> Neon::set::container::GraphData::Uid;

    /**
     * Returns the "after" kernel id
     * @return
     */
    auto t1() -> Neon::set::container::GraphData::Uid;

    /**
     * Static method to build an empty dependency
     * @return
     */
    static Dependency getEmpty();
};

}  // namespace Neon::skeleton::internal