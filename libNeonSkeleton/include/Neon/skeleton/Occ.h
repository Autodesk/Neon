/**
 * @file Occ.h
 * @brief Overlap-Communication-Computation (OCC) configuration options
 * 
 * This file defines the OCC enumeration and utilities for configuring how
 * computation and communication operations are overlapped in Neon's skeleton
 * execution framework to optimize performance in distributed and parallel
 * computing environments.
 */

#pragma once
#include "Neon/Report.h"
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"

namespace Neon::skeleton {

/**
 * @brief Overlap-Communication-Computation (OCC) configuration enumeration
 * 
 * The OCC enum defines different strategies for overlapping computation and
 * communication operations in parallel and distributed computing scenarios.
 * These strategies can significantly impact performance by reducing idle time
 * and improving resource utilization.
 * 
 * Communication operations typically involve data transfer between:
 * - Different GPU devices
 * - Different compute nodes in a distributed system
 * - Halo exchanges in domain decomposition
 * 
 * Computation operations include:
 * - Kernel execution on GPU devices
 * 
 * @note The optimal OCC strategy depends on:
 *       - Problem size and computational intensity
 *       - Network bandwidth and latency characteristics
 *       - Hardware capabilities (number of GPUs, memory bandwidth)
 *       - Application communication patterns
 */
enum class Occ
{
    /**
     * @brief Standard overlap mode
     * 
     * Provides basic overlap between computation and communication operations.
     * In this mode, the system attempts to hide communication latency by
     * performing computations on interior domain points while boundary data
     * is being exchanged.
     * 
     * Characteristics:
     * - Moderate complexity in implementation
     * - Good performance for most typical scenarios
     * - Balanced resource utilization
     * - Suitable for applications with regular communication patterns
     * 
     * Use when:
     * - You need reliable performance across diverse workloads
     * - Communication costs are moderate compared to computation
     * - System has sufficient resources for basic overlap
     */
    standard=0,

    /**
     * @brief Extended overlap mode
     * 
     * Implements more aggressive overlap strategies with enhanced optimization
     * techniques by splitting operation between internal and boundary for pre-stencil nodes
     * 
     * Characteristics:
     * - More complex synchronization requirements
     * - Better performance for communication-intensive applications
     * 
     * Use when:
     * - Communication overhead is significant
     * - Application can benefit from aggressive optimization
     */
    extended=1,

    /**
     * @brief Two-way extended overlap mode
     * 
     * Implements more aggressive overlap strategies with enhanced optimization
     * techniques by splitting operation between internal and boundary for pre-stencil and post-stencil nodes
     * 
     * Characteristics:
     * - Maximum theoretical performance potential
     * - The most complex synchronization requirements
     * - The most complex implementation and debugging
     *
     * 
     * @warning This mode may not always provide benefits and can sometimes
     *          degrade performance if system resources are insufficient
     */
    twoWayExtended=2,

    /**
     * @brief No overlap mode (sequential execution)
     * 
     * Disables all overlap optimizations, resulting in strictly sequential
     * execution where communication and computation phases are completely
     * separated. This mode is useful for debugging, benchmarking, and
     * scenarios where overlap causes issues.
     * 
     * Characteristics:
     * - Simplest execution model
     * - Easiest to debug and understand
     * - Minimal resource overhead
     * - Predictable execution patterns
     * - Generally lower performance due to idle time
     * 
     * Use when:
     * - Debugging communication or computation issues
     * - Establishing baseline performance measurements
     * - System resources are very limited
     * - Overlap optimizations cause stability problems
     * - Educational purposes to understand communication costs
     */
    none=3,
};

/**
 * @brief Utility functions and classes for OCC configuration management
 * 
 * The OccUtils struct provides static utility functions for converting between
 * different representations of OCC values and managing OCC configurations in
 * command-line interfaces and reporting systems.
 */
struct OccUtils
{
    /// @brief Total number of available OCC options
    static constexpr int nOptions = 4;

    /**
     * @brief Convert OCC enum value to string representation
     * @param occ The OCC enumeration value to convert
     * @return String representation ("standard", "extended", "twoWayExtended", "none")
     */
    static auto toString(Occ occ) -> std::string;

    /**
     * @brief Convert string representation to OCC enum value
     * @param occ String representation of OCC value
     * @return Corresponding OCC enumeration value
     * @throws Exception if the string is not a valid OCC value
     */
    static auto fromString(const std::string& occ) -> Occ;

    /**
     * @brief Convert integer value to OCC enum value
     * @param occ Integer value (0=standard, 1=extended, 2=twoWayExtended, 3=none)
     * @return Corresponding OCC enumeration value
     * @throws Exception if the integer is not in valid range [0-3]
     */
    static auto fromInt(int occ) -> Occ;

    /**
     * @brief Convert OCC enum value to integer value
     * @param occ OCC enumeration value
     * @return Corresponding integer value
     */
    static auto toInt(Occ occ) -> int;

    /**
     * @brief Get array of all available OCC options
     * @return Array containing all OCC enumeration values
     */
    static auto getOptions() -> std::array<Occ, nOptions>;
    
    /**
     * @brief Command-line interface wrapper for OCC configuration
     * 
     * The Cli class provides a convenient interface for handling OCC
     * configuration in command-line applications, including parsing,
     * validation, and reporting capabilities.
     */
    struct Cli
    {
        /**
         * @brief Construct from string representation
         * @param occString String representation of OCC value
         */
        explicit Cli(std::string);

        /**
         * @brief Construct from OCC enumeration value
         * @param model OCC enumeration value
         */
        explicit Cli(Occ model);

        /**
         * @brief Default constructor (creates unset configuration)
         */
        Cli();

        /**
         * @brief Get the currently configured OCC option
         * @return Current OCC enumeration value
         * @throws Exception if no option has been set
         */
        auto getOption() const -> Occ;

        /**
         * @brief Set OCC option from string representation
         * @param opt String representation of OCC value
         * @throws Exception if the string is not a valid OCC value
         */
        auto set(const std::string& opt) -> void;

        /**
         * @brief Get comma-separated string of all available options
         * @return String listing all valid OCC options
         */
        auto getStringOptions() const -> std::string;

        /**
         * @brief Get documentation string with options and default value
         * @return Documentation string suitable for help text
         */
        auto getDoc() const -> std::string;

        /**
         * @brief Get string representation of current option
         * @return String representation of currently set OCC value
         * @throws Exception if no option has been set
         */
        auto getStringOption() const -> std::string;

        /**
         * @brief Add current OCC configuration to report with sub-block
         * @param report Report object to add configuration to
         * @param subBlock Sub-block within the report for organization
         */
        auto addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) const -> void;

        /**
         * @brief Add current OCC configuration to report
         * @param report Report object to add configuration to
         */
        auto addToReport(Neon::Report& report) const -> void;

       private:
        bool mSet = false;  ///< Flag indicating whether an option has been set
        Occ  mOption;       ///< Currently configured OCC option
    };
};


}  // namespace Neon::skeleton
