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
 * @brief Multi-GPU execution graph for managing kernel dependencies and scheduling
 * 
 * The MultiXpuGraph class manages the execution flow of user-defined kernels across
 * multiple GPUs by building and maintaining a dependency graph. It handles:
 * 
 * - Parsing user-provided container operations to extract dependencies
 * - Building an internal graph representation of kernel execution order
 * - Optimizing the execution schedule using various optimization strategies
 * - Managing data dependencies and communications between kernels
 * - Executing the optimized graph with proper synchronization
 * 
 * Key Features:
 * - Automatic dependency extraction from container operations
 * - Multiple optimization strategies (Standard OCC, Extended OCC, Two-Way Extended OCC)
 * - Support for data dependency management through UserDataManager
 * - Graph visualization and debugging capabilities
 * - Configurable execution options and scheduling policies
 * 
 * Usage:
 * 1. Create a MultiXpuGraph instance
 * 2. Initialize with backend, operations, name, and options
 * 3. The graph will automatically parse dependencies and optimize execution
 * 4. Execute the graph with specified options
 * 
 * @see Neon::skeleton::Options for configuration options
 * @see Neon::set::Container for kernel container operations
 * @see UserDataManager for data dependency management
 */
struct MultiXpuGraph
{

   public:
    /**
     * @brief Default constructor creating an empty graph
     * 
     * Creates an uninitialized MultiXpuGraph. The graph must be initialized
     * using the init() method before it can be used.
     */
    MultiXpuGraph();

    /**
     * @brief Initialize the graph with backend, operations, and configuration
     * 
     * This method sets up the execution graph by:
     * - Storing the backend configuration for multi-GPU execution
     * - Parsing the provided container operations to extract dependencies
     * - Building the internal dependency graph
     * - Applying optimizations based on the provided options
     * 
     * @param bk The Neon backend managing GPU resources and execution
     * @param operations Vector of container operations defining the computational kernels
     * @param name Human-readable name for the graph (used in debugging/visualization)
     * @param options Configuration options controlling optimization and execution behavior
     * 
     * @throws std::runtime_error if initialization fails or invalid operations are provided
     */
    void init(Neon::Backend&                           bk,
              const std::vector<Neon::set::Container>& operations,
              std::string                              name,
              Options                                  options);

    /**
     * @brief Export the original user application graph to DOT format
     * 
     * Generates a DOT file representing the user's original application graph
     * before any optimizations are applied. This is useful for debugging and
     * understanding the initial dependency structure.
     * 
     * @param fname Output filename for the DOT file (should have .dot extension)
     * @param graphName Name to be displayed in the generated graph
     * @param debug If true, includes additional debug information in the output
     * 
     * @note The generated DOT file can be visualized using Graphviz tools
     */
    auto io2DotOriginalApp(const std::string& fname,
                           const std::string& graphName,
                           bool               debug = false)
        -> void;

    /**
     * @brief Export the complete optimized graph to DOT format
     * 
     * Generates a comprehensive DOT file showing both the data dependency graph
     * and the final scheduling graph after all optimizations have been applied.
     * This includes communication nodes and optimized execution order.
     * 
     * @param fname Output filename for the DOT file (should have .dot extension)
     * @param graphName Name to be displayed in the generated graph
     * @param debug If true, includes detailed debug information and node attributes
     * 
     * @note This is the most comprehensive graph export, showing the final
     *       execution plan that will be used during graph execution
     */
    auto ioToDot(const std::string& fname,
                 const std::string& graphName,
                 bool               debug = false)
        -> void;

    /**
     * @brief Execute the optimized graph with the specified options
     * 
     * Executes all kernels in the graph according to the computed schedule,
     * managing dependencies, synchronization, and data transfers between GPUs.
     * The execution follows the optimized order determined during initialization.
     * 
     * @param options Runtime execution options that may override initialization settings
     * 
     * @throws std::runtime_error if execution fails or if the graph is not properly initialized
     * 
     * @note This method blocks until all kernels in the graph have completed execution
     */
    auto execute(const Neon::skeleton::Options& options)
        -> void;

   private:

    /**
     * @brief Get mutable reference to the container operations
     * 
     * Provides access to the vector of container operations that define
     * the computational kernels in the graph.
     * 
     * @return Mutable reference to the vector of container operations
     */
    inline auto getContainers()
        -> std::vector<Neon::set::Container>&
    {
        return mStorage->mContainers;
    }

    /**
     * @brief Get const reference to the container operations
     * 
     * Provides read-only access to the vector of container operations
     * that define the computational kernels in the graph.
     * 
     * @return Const reference to the vector of container operations
     */
    inline auto getContainers()
        const -> const std::vector<Neon::set::Container>&
    {
        return mStorage->mContainers;
    }

    /**
     * @brief Get reference to the user data manager
     * 
     * Provides access to the UserDataManager that handles data dependencies
     * and manages data transfers between kernels and devices.
     * 
     * @return Reference to the UserDataManager instance
     */
    inline auto getDataRecords()
        -> UserDataManager&
    {
        return mStorage->mDataRecords;
    }

    /**
     * @brief Get reference to the internal execution graph
     * 
     * Provides access to the underlying graph structure that represents
     * the optimized execution order and dependencies.
     * 
     * @return Reference to the container graph instance
     */
    inline auto getGraph()
        -> Neon::set::container::Graph&
    {
        return mStorage->mGraph;
    }

    /**
     * @brief Get reference to the set cardinality
     * 
     * Provides access to the number of devices/GPUs in the execution set.
     * This determines the parallelization factor for the graph execution.
     * 
     * @return Reference to the set cardinality (number of devices)
     */
    inline auto getSetCardinality()
        -> int&
    {
        return mStorage->mSetCardinality;
    }


    /**
     * @brief Enumeration for halo update operations
     * 
     * Defines the type of halo update operation for boundary data exchange
     * between neighboring domains in multi-GPU computations.
     */
    enum heloUpdate
    {
        PUT,  ///< Put operation: send boundary data to neighbors
        GET   ///< Get operation: receive boundary data from neighbors
    };

    /**
     * @brief Parse container operations and build the dependency graph
     * 
     * Analyzes the provided container operations to extract data dependencies
     * and builds the internal graph representation. This method processes each
     * container to understand its input/output requirements and establishes
     * the execution order constraints.
     * 
     * @param setCardinalty Number of devices/GPUs in the execution set
     * @param operations Vector of container operations to be parsed (moved)
     */
    auto parse(int                                       setCardinalty,
               const std::vector<Neon::set::Container>&& operations)
        -> void;

    /**
     * @brief Apply optimization strategies to the execution graph
     * 
     * Applies various optimization techniques to improve the execution
     * performance based on the provided options. This may include
     * overlapping communication and computation (OCC), reordering operations,
     * and other scheduling optimizations.
     * 
     * @param options Configuration options specifying which optimizations to apply
     */
    auto optimizations(const Neon::skeleton::Options& options)
        -> void;

    /**
     * @brief Add a new container to the execution graph
     * 
     * Helper function that processes a single container operation and adds it
     * to the execution graph. This method extracts dependencies from the
     * container and establishes connections with previously added containers.
     * 
     * @param inContainer The container operation to be added to the graph
     */
    auto helpParseNewContainer(const Neon::set::Container& inContainer)
        -> void;

    /**
     * @brief Parse a container and extract its data dependency tokens
     * 
     * Analyzes a single container operation to extract its data dependencies
     * in the form of tokens. These tokens are used to establish connections
     * between containers in the dependency graph.
     * 
     * @param kernelContainerIdx The container to be parsed for dependencies
     * @return Vector of data dependency tokens representing the container's dependencies
     */
    auto helpParseContainer(Neon::set::Container& kernelContainerIdx)
        -> std::vector<Neon::set::dataDependency::Token>;

    /**
     * @brief Compute the final execution schedule for the graph
     * 
     * Determines the optimal execution order for all containers in the graph
     * based on their dependencies and optimization settings. This method
     * produces the final schedule that will be used during execution.
     */
    auto computeScheduling()
        -> void;



   private:
    /**
     * @brief Add a container to the internal graph structure
     * 
     * Helper method that adds a new container node to the internal graph
     * and returns its unique identifier. This method handles the low-level
     * graph manipulation required to incorporate the container.
     * 
     * @param container The container to be added to the graph
     * @return Unique node identifier for the newly added container
     */
    auto helpAddNewContainerToGraph(const Neon::set::Container& container)
        -> Neon::set::container::GraphInfo::NodeUid;

    /**
     * @brief Apply standard Overlapping Communication and Computation (OCC) optimization
     * 
     * Implements the standard OCC optimization strategy that overlaps
     * communication operations with computation to improve performance.
     * This optimization identifies opportunities to hide communication
     * latency behind computation.
     * 
     * @param options Configuration options for the optimization
     */
    auto optimizeStandardOCC(const Neon::skeleton::Options& options)
        -> void;

    /**
     * @brief Apply extended Overlapping Communication and Computation (OCC) optimization
     * 
     * Implements an extended version of OCC optimization that provides more
     * aggressive overlapping strategies compared to the standard approach.
     * This may include more sophisticated analysis of communication patterns.
     * 
     * @param options Configuration options for the optimization
     */
    auto optimizeExtendedOCC(const Neon::skeleton::Options& options)
        -> void;

    /**
     * @brief Apply two-way extended OCC optimization
     * 
     * Implements a bidirectional extended OCC optimization that considers
     * communication patterns in both directions of the dependency graph.
     * This provides the most sophisticated overlap optimization available.
     * 
     * @param options Configuration options for the optimization
     */
    auto optimizeTwoWayExtendedOCC(const Neon::skeleton::Options& options)
        -> void;

    /**
     * @brief Handle communication setup and management
     * 
     * Manages the communication aspects of the graph execution, including
     * setting up data transfers between devices, synchronization points,
     * and communication scheduling.
     * 
     * @param skeletonOptions Options controlling communication behavior
     */
    auto communications(const Neon::skeleton::Options& skeletonOptions)
        -> void;

    /**
     * @brief Fix dependency issues by adding begin nodes
     * 
     * Ensures proper dependency management by adding special begin nodes
     * to the graph where necessary. This method resolves dependency
     * conflicts and ensures correct execution ordering.
     */
    auto fixingDependenciesWithBeginNode()
        -> void;

    /**
     * @brief Internal storage structure for graph data
     * 
     * Contains all the internal data structures needed for graph management
     * and execution. This struct is shared through a shared_ptr to enable
     * efficient copying and sharing of graph instances.
     */
    struct Storage
    {
        /// Vector of all container operations in the graph
        std::vector<Neon::set::Container> mContainers;
        
        /// Manager for user data dependencies and transfers
        UserDataManager                   mDataRecords;
        
        /// Number of devices/GPUs in the execution set
        int                               mSetCardinality = 0;
        
        /// Internal graph representation for execution scheduling
        Neon::set::container::Graph       mGraph;
        
        /// Human-readable name for the graph (used in debugging)
        std::string                       mName;
    };

    /// Shared pointer to the internal storage structure
    std::shared_ptr<Storage> mStorage;
};
}  // namespace Neon::skeleton::internal
