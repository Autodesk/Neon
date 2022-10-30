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
    auto io2Dot(const std::string& fname,
                const std::string& graphName,
                bool               debug = false)
        -> void;


   private:
    struct Storage
    {
        std::vector<Neon::set::Container> m_kContainers;
        UserDataManager                   m_dataRecords;

        size_t m_rootNodeId;
        size_t m_finalNodeId;
        int    m_setCardinality = 0;

        Neon::set::container::Graph mGraph;
    };


    std::shared_ptr<Storage> m_storage;

    /**
     * Access methods to members
     * @return
     */
    inline auto m_kContainers()
        -> std::vector<Neon::set::Container>&
    {
        return m_storage->m_kContainers;
    }

    inline auto m_kContainers()
        const -> const std::vector<Neon::set::Container>&
    {
        return m_storage->m_kContainers;
    }

    /**
     * Access methods to members
     * @return
     */
    inline auto m_dataRecords()
        -> UserDataManager&
    {
        return m_storage->m_dataRecords;
    }

    inline auto mGraph()
        -> Neon::set::container::Graph&
    {
        return m_storage->mGraph;
    }

    /**
     * Access methods to members
     * @return
     */
    inline auto m_setCardinality()
        -> int&
    {
        return m_storage->m_setCardinality;
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
     * @param container
     */
    auto helpParseNewContainer(const Neon::set::Container& inContainer)
        -> void;

    /**
     * Parsing of a container of the task list
     * @param kernelContainerIdx
     * @return
     */
    auto helpParseContainer(Neon::set::Container& kernelContainerIdx)
        -> std::vector<Neon::set::dataDependency::Token>;

    /**
     * helper function to export a dot file
     * @param fname
     * @param graphName
     */
    auto h_io2Dot([[maybe_unused]] const std::string& fname,
                  [[maybe_unused]] const std::string& graphName)
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

    auto addCommunications(const Neon::skeleton::Options&)
        -> void;

    auto checkCoherency()
        -> void;
};
}  // namespace Neon::skeleton::internal
