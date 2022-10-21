#pragma once
#include "Neon/set/Backend.h"
#include "Neon/set/container/Graph.h"
#include "Neon/set/container/graph/GraphInfo.h"
#include "Neon/set/dependency/DataDependencyType.h"

namespace Neon::skeleton::internal {


struct DataDependency
{
   private:
    Neon::set::container::GraphInfo::NodeUid      mT0 = 0;
    Neon::set::container::GraphInfo::NodeUid      mT1 = 0;
    Neon::set::dataDependency::DataDependencyType mType = Neon::set::dataDependency::DataDependencyType::NONE;
    Neon::set::dataDependency::MultiXpuDataUid    mDataUid = 0;

   public:
    /**
     * Empty constructor
     */
    DataDependency() = default;

    /**
     * Defines a dependency of type e between kernel A and B.
     * Note, the order is important.
     *
     * Example: a read after write where kernel B reads results from kernel A
     *          Dependency_t(B, RAW, A)
     * @param A
     * @param B
     */
    DataDependency(Neon::set::container::GraphInfo::NodeUid      t1,
                   Neon::set::dataDependency::DataDependencyType type,
                   Neon::set::dataDependency::MultiXpuDataUid    m_uid,
                   Neon::set::container::GraphInfo::NodeUid      t0);

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
    auto type() -> Neon::set::dataDependency::DataDependencyType;

    /**
     * Returns the "before" kernel id
     * @return
     */
    auto t0() -> Neon::set::container::GraphInfo::NodeUid;

    /**
     * Returns the "after" kernel id
     * @return
     */
    auto t1() -> Neon::set::container::GraphInfo::NodeUid;

    /**
     * Static method to build an empty dependency
     * @return
     */
    static DataDependency getEmpty();
};

}  // namespace Neon::skeleton::internal