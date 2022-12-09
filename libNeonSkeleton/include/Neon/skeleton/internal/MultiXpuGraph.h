#pragma once
#include <list>
#include "Neon/core/types/digraph.h"
#include "Neon/set//Containter.h"
#include "Neon/set/Backend.h"
#include "Neon/set/container/Graph.h"
#include "Neon/skeleton/Options.h"
#include "Neon/skeleton/internal/dependencyTools/UserDataManager.h"

namespace Neon::skeleton::internal {
/**
 * Graph storing dependency between user kernels
 */
struct MultiXpuGraph
{

   public:
    /**
     * Default empty constructor
     */
    MultiXpuGraph();

    void init(Neon::Backend&                           bk,
              const std::vector<Neon::set::Container>& operations,
              std::string                              name,
              Options                                  options);

    /**
     * Export the graph of the user application
     * @param fname
     * @param graphName
     */
    auto io2DotOriginalApp(const std::string& fname,
                           const std::string& graphName,
                           bool               debug = false)
        -> void;

    /**
     * Export both the data dependency graph and the scheduling graph
     * @param fname
     * @param graphName
     */
    auto ioToDot(const std::string& fname,
                 const std::string& graphName,
                 bool               debug = false)
        -> void;

    auto execute(const Neon::skeleton::Options& options)
        -> void;

   private:


    /**
     * Access methods to members
     * @return
     */
    inline auto getContainers()
        -> std::vector<Neon::set::Container>&
    {
        return mStorage->mContainers;
    }

    inline auto getContainers()
        const -> const std::vector<Neon::set::Container>&
    {
        return mStorage->mContainers;
    }

    /**
     * Access methods to members
     * @return
     */
    inline auto getDataRecords()
        -> UserDataManager&
    {
        return mStorage->mDataRecords;
    }

    inline auto getGraph()
        -> Neon::set::container::Graph&
    {
        return mStorage->mGraph;
    }

    /**
     * Access methods to members
     * @return
     */
    inline auto getSetCardinality()
        -> int&
    {
        return mStorage->mSetCardinality;
    }


    enum heloUpdate
    {
        PUT,
        GET
    };

    /**
     * Parser method
     * @param setCardinalty
     * @param operations
     */
    auto parse(int                                       setCardinalty,
               const std::vector<Neon::set::Container>&& operations)
        -> void;

    /**
     *
     * @param options
     */
    auto optimizations(const Neon::skeleton::Options& options)
        -> void;

    /**
     * Helper function to add a new kernel to the graph
     * The dependencies are extracted from the kernel container
     */
    auto helpParseNewContainer(const Neon::set::Container& inContainer)
        -> void;

    /**
     * Parsing of a container of the task list
     */
    auto helpParseContainer(Neon::set::Container& kernelContainerIdx)
        -> std::vector<Neon::set::dataDependency::Token>;

    auto computeScheduling()
        -> void;



   private:
    auto helpAddNewContainerToGraph(const Neon::set::Container& container)
        -> Neon::set::container::GraphInfo::NodeUid;

    auto optimizeStandardOCC(const Neon::skeleton::Options&)
        -> void;

    auto optimizeExtendedOCC(const Neon::skeleton::Options&)
        -> void;

    auto optimizeTwoWayExtendedOCC(const Neon::skeleton::Options&)
        -> void;

    auto communications(const Neon::skeleton::Options& skeletonOptions)
        -> void;

    auto fixingDependenciesWithBeginNode()
        -> void;

    struct Storage
    {
        std::vector<Neon::set::Container> mContainers;
        UserDataManager                   mDataRecords;
        int                               mSetCardinality = 0;
        Neon::set::container::Graph       mGraph;
    };

    std::shared_ptr<Storage> mStorage;
};
}  // namespace Neon::skeleton::internal
