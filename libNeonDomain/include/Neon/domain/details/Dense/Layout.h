#pragma once
#include <string>
#include <array>

#if !defined(NEON_WARP_COMPILATION)
#include "Neon/Report.h"
#endif
#include "Neon/core/core.h"

namespace Neon::domain::details::Dense {

/**
 * ═══════════════════════════════════════════════════════════════════════════════════════
 *
 *                                   DENSE GRID LAYOUT
 *                         Memory Layout Options for Dense Grid Storage
 *
 * ═══════════════════════════════════════════════════════════════════════════════════════
 *
 * The Layout enum defines different memory organization strategies for Dense Grid data:
 *
 *   ┌─────────────────────────────────────────────────────────────────────────────────┐
 *   │                          LAYOUT STRATEGIES                                      │
 *   ├─────────────────────────────────────────────────────────────────────────────────┤
 *   │                                                                                 │
 *   │  StructOfArrays (SoA) = 0                                                       │
 *   │  ┌─────┐ ┌─────┐ ┌─────┐                                                        │
 *   │  │ x[] │ │ y[] │ │ z[] │  ← Separate arrays for each field component            │
 *   │  └─────┘ └─────┘ └─────┘                                                        │
 *   │  • Better cache locality for operations on single components                    │
 *   │  • Optimal for SIMD operations and vectorization                                │
 *   │  • Memory coalescing friendly on GPU                                            │
 *   │                                                                                 │
 *   │  ArrayOfStructs (AoS) = 1                                                       │
 *   │  ┌───────────────┐                                                              │
 *   │  │ {x,y,z} {x,y,z} │ ← Interleaved components in single array                   │
 *   │  └───────────────┘                                                              │
 *   │  • Better cache locality for operations using all components                    │
 *   │  • More intuitive memory layout                                                 │
 *   │  • Simpler indexing for complete element access                                 │
 *   │                                                                                 │
 *   │  DisgSoA = 2                                                                    │
 *   │  ┌─────────────────────────────────────┐                                        │
 *   │  │    Dissagregated Structure of Arrays│ ← Advanced layout for specialized use  │
 *   │  └─────────────────────────────────────┘                                        │
 *   │  • Hybrid approach for complex memory access patterns                           │
 *   │  • Specialized for distributed computing scenarios                              │
 *   │  • Optimized for specific computational kernels                                 │
 *   │                                                                                 │
 *   └─────────────────────────────────────────────────────────────────────────────────┘
 *
 * PERFORMANCE CONSIDERATIONS:
 *
 * • StructOfArrays: Best for element-wise operations, GPU kernels, vectorization
 * • ArrayOfStructs: Best for algorithms processing complete elements together
 * • DisgSoA: Best for advanced distributed computing and specialized access patterns
 *
 * ═══════════════════════════════════════════════════════════════════════════════════════
 */
enum class Layout {
    StructOfArrays = 0,  ///< Structure of Arrays - separate arrays per component
    ArrayOfStructs = 1,  ///< Array of Structures - interleaved components
    DisgSoA = 2          ///< Disjoint Structure of Arrays - advanced layout
};

/**
 * ═══════════════════════════════════════════════════════════════════════════════════════
 *
 *                                LAYOUT UTILITIES
 *                    Complete Utility Suite for Layout Management & CLI
 *
 * ═══════════════════════════════════════════════════════════════════════════════════════
 *
 * LayoutUtils provides comprehensive utilities for working with Layout enums:
 *
 *   ┌─────────────────────────────────────────────────────────────────────────────────┐
 *   │                           UTILITY FUNCTIONS                                     │
 *   ├─────────────────────────────────────────────────────────────────────────────────┤
 *   │                                                                                 │
 *   │  STRING CONVERSION                                                              │
 *   │  • toString()    → Convert Layout to human-readable string                     │
 *   │  • fromString()  → Parse Layout from string with validation                    │
 *   │                                                                                 │
 *   │  INTEGER CONVERSION                                                             │
 *   │  • toInt()       → Convert Layout to integer representation                    │
 *   │  • fromInt()     → Convert integer back to Layout enum                         │
 *   │                                                                                 │
 *   │  INTROSPECTION                                                                  │
 *   │  • getOptions()  → Get all available Layout values as array                    │
 *   │  • isDisgSoA()   → Check if layout is DisgSoA                                  │
 *   │  • isSoA()       → Check if layout is StructOfArrays                          │
 *   │  • isAoS()       → Check if layout is ArrayOfStructs                          │
 *   │                                                                                 │
 *   └─────────────────────────────────────────────────────────────────────────────────┘
 *
 * CLI INTEGRATION:
 *
 * The nested Cli struct provides complete command-line argument parsing support:
 *
 *   ┌─────────────────────────────────────────────────────────────────────────────────┐
 *   │                              CLI INTERFACE                                      │
 *   ├─────────────────────────────────────────────────────────────────────────────────┤
 *   │                                                                                 │
 *   │  INITIALIZATION                                                                 │
 *   │  • Cli()                    → Default construction (unset state)               │
 *   │  • Cli(std::string)         → Initialize from string parameter                 │
 *   │  • Cli(Layout)              → Initialize from Layout enum                      │
 *   │                                                                                 │
 *   │  CONFIGURATION                                                                  │
 *   │  • set(string)              → Set layout from string with validation          │
 *   │  • getOption()              → Retrieve selected layout (throws if unset)      │
 *   │                                                                                 │
 *   │  HELP & DOCUMENTATION                                                           │
 *   │  • getStringOptions()       → Get comma-separated list of valid options       │
 *   │  • getStringOption()        → Get current selection as string                 │
 *   │  • getDoc()                 → Get complete documentation string               │
 *   │                                                                                 │
 *   │  REPORTING INTEGRATION                                                          │
 *   │  • addToReport(...)         → Integration with Neon's reporting system        │
 *   │                                                                                 │
 *   └─────────────────────────────────────────────────────────────────────────────────┘
 *
 * USAGE EXAMPLES:
 *
 * ```cpp
 * // Basic utilities
 * Layout layout = Layout::StructOfArrays;
 * std::string name = LayoutUtils::toString(layout);        // "StructOfArrays"
 * bool isSoa = LayoutUtils::isSoA(layout);                 // true
 * 
 * // CLI integration
 * LayoutUtils::Cli cli;
 * cli.set("ArrayOfStructs");                               // Parse from command line
 * Layout selected = cli.getOption();                       // Layout::ArrayOfStructs
 * std::string help = cli.getDoc();                         // Full help text
 * 
 * // Error handling
 * try {
 *     cli.set("InvalidLayout");
 * } catch(...) {
 *     // Descriptive error with valid options listed
 * }
 * ```
 *
 * ═══════════════════════════════════════════════════════════════════════════════════════
 */
struct LayoutUtils {
    static constexpr int nOptions = 3;  ///< Total number of available layout options

    // ═══════════════════════════════════════════════════════════════════════════════════
    // STRING CONVERSION UTILITIES
    // ═══════════════════════════════════════════════════════════════════════════════════

    /**
     * @brief Convert Layout enum to human-readable string representation
     * @param layout The Layout enum value to convert
     * @return String representation ("StructOfArrays", "ArrayOfStructs", "DisgSoA")
     * @throws Neon exception if layout value is invalid
     */
    static auto toString(const Layout& layout) -> std::string;

    /**
     * @brief Parse Layout from string representation
     * @param layout String representation of layout
     * @return Corresponding Layout enum value
     * @throws Neon exception if string doesn't match any valid layout
     */
    static auto fromString(const std::string& layout) -> Layout;

    // ═══════════════════════════════════════════════════════════════════════════════════
    // INTEGER CONVERSION UTILITIES
    // ═══════════════════════════════════════════════════════════════════════════════════

    /**
     * @brief Convert Layout to integer representation
     * @param layout Layout enum to convert
     * @return Integer value (0=StructOfArrays, 1=ArrayOfStructs, 2=DisgSoA)
     */
    static auto toInt(const Layout& layout) -> int;

    /**
     * @brief Convert integer to Layout enum
     * @param layout Integer representation (0-2)
     * @return Corresponding Layout enum value
     * @throws Neon exception if integer is out of valid range
     */
    static auto fromInt(const int& layout) -> Layout;

    // ═══════════════════════════════════════════════════════════════════════════════════
    // INTROSPECTION UTILITIES
    // ═══════════════════════════════════════════════════════════════════════════════════

    /**
     * @brief Get array of all available Layout options
     * @return Array containing all Layout enum values
     */
    static auto getOptions() -> std::array<Layout, nOptions>;

    /**
     * @brief Check if layout is DisgSoA (Disjoint Structure of Arrays)
     * @param layout Layout to check
     * @return true if layout is DisgSoA, false otherwise
     */
    static auto isDisgSoA(const Layout& layout) -> bool;

    /**
     * @brief Check if layout is SoA (Structure of Arrays)
     * @param layout Layout to check
     * @return true if layout is StructOfArrays, false otherwise
     */
    static auto isSoA(const Layout& layout) -> bool;

    /**
     * @brief Check if layout is AoS (Array of Structures)
     * @param layout Layout to check
     * @return true if layout is ArrayOfStructs, false otherwise
     */
    static auto isAoS(const Layout& layout) -> bool;

    // ═══════════════════════════════════════════════════════════════════════════════════
    // COMMAND-LINE INTERFACE INTEGRATION
    // ═══════════════════════════════════════════════════════════════════════════════════

    /**
     * @brief Command-line interface utilities for Layout configuration
     * 
     * Provides complete CLI argument parsing, validation, and help generation
     * for Layout options. Supports initialization from strings or Layout enums,
     * comprehensive error reporting, and integration with Neon's reporting system.
     */
    struct Cli
    {
        /**
         * @brief Initialize CLI with string parameter
         * @param layoutStr String representation of layout ("StructOfArrays", etc.)
         * @throws Neon exception if string is invalid
         */
        explicit Cli(std::string layoutStr);

        /**
         * @brief Initialize CLI with Layout enum
         * @param layout Layout enum value to set
         */
        explicit Cli(Layout layout);

        /**
         * @brief Default constructor - creates unset CLI instance
         * Must call set() before getOption()
         */
        Cli();

        /**
         * @brief Get the selected layout option
         * @return Selected Layout enum value
         * @throws Neon exception if layout was not set
         */
        auto getOption() const -> Layout;

        /**
         * @brief Set layout from string with comprehensive validation
         * @param opt String representation of layout
         * @throws Neon exception with helpful error message if invalid
         */
        auto set(const std::string& opt) -> void;

        /**
         * @brief Get comma-separated list of all valid layout options
         * @return String like "StructOfArrays, ArrayOfStructs, DisgSoA"
         */
        auto getStringOptions() const -> std::string;

        /**
         * @brief Get current selection as string
         * @return String representation of selected layout
         * @throws Neon exception if layout was not set
         */
        auto getStringOption() const -> std::string;

        /**
         * @brief Get complete documentation string for CLI help
         * @return Help string with options and default value
         */
        auto getDoc() const -> std::string;

#if !defined(NEON_WARP_COMPILATION)
        /**
         * @brief Add layout information to Neon report with sub-block
         * @param report Neon report to add to
         * @param subBlock Sub-block to add information under
         */
        auto addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) const -> void;

        /**
         * @brief Add layout information to Neon report
         * @param report Neon report to add to
         */
        auto addToReport(Neon::Report& report) const -> void;
#endif

       private:
        bool   mSet = false;    ///< Flag indicating whether layout has been set
        Layout mOption;         ///< Currently selected layout option
    };
};

} // namespace Neon::domain::details::Dense