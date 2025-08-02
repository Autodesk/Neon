/**
 * @file Timers.h
 * @brief High-performance timer utilities with optional NVTX integration for CUDA profiling
 * 
 * This file provides Timer and TimerManager classes for high-precision timing measurements.
 * When NEON_USE_NVTX is enabled, the TimerManager automatically creates corresponding NVTX ranges
 * for seamless integration with NVIDIA profiling tools like Nsight Systems.
 * 
 * ## Key Features:
 * - Template-based timers supporting different time resolutions (ns, us, ms, s)
 * - Automatic NVTX range creation for CUDA profiling
 * - Single-shot and multi-timer management
 * - Compile-time unit string generation
 * - Exception safety and proper resource cleanup
 * 
 * ## Usage Example:
 * ```cpp
 * // Single timer
 * Neon::TimerUS timer;
 * timer.start();
 * // ... work ...
 * auto elapsed = timer.stop();
 * 
 * // Timer manager with NVTX integration
 * Neon::TimerManagerMS manager;
 * manager.start("computation");     // Creates NVTX range "computation"
 * // ... CUDA kernels or CPU work ...
 * auto time = manager.stop("computation");  // Ends NVTX range
 * ```
 * 
 */
#pragma once

#if !defined(NEON_WARP_COMPILATION)

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <iomanip>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Neon/core/types/Exceptions.h"
#include "Neon/core/types/Macros.h"
#include "Neon/core/tools/Logger.h"

#ifdef NEON_USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

/**
 * @namespace Neon
 * @brief Main namespace for the Neon computational framework
 * 
 * Contains all core Neon functionality including timing utilities, domain grids,
 * computational backends, and profiling integration.
 */
namespace Neon {

/**
 * @brief Helper function to obtain a string representation of a duration unit at compile time.
 *
 * This constexpr function maps std::chrono duration types to their corresponding
 * unit string representations. It's used internally by Timer classes to generate
 * human-readable time measurements.
 *
 * @tparam Duration The std::chrono duration type (hours, minutes, seconds, milliseconds, microseconds, nanoseconds)
 * @return std::string_view A compile-time string literal representing the unit (e.g., "ms", "us", "s")
 * 
 * ## Supported Duration Types:
 * - std::chrono::hours → "h"
 * - std::chrono::minutes → "min"  
 * - std::chrono::seconds → "s"
 * - std::chrono::milliseconds → "ms"
 * - std::chrono::microseconds → "us"
 * - std::chrono::nanoseconds → "ns"
 * 
 * @note Unsupported duration types will trigger a compile-time static assertion
 */
template <typename Duration>
constexpr auto UnitStr() noexcept -> std::string_view {
    if constexpr (std::is_same_v<Duration, std::chrono::hours>) return "h";
    else if constexpr (std::is_same_v<Duration, std::chrono::minutes>) return "min";
    else if constexpr (std::is_same_v<Duration, std::chrono::seconds>) return "s";
    else if constexpr (std::is_same_v<Duration, std::chrono::milliseconds>) return "ms";
    else if constexpr (std::is_same_v<Duration, std::chrono::microseconds>) return "us";
    else if constexpr (std::is_same_v<Duration, std::chrono::nanoseconds>) return "ns";
    else static_assert(sizeof(Duration) == 0, "Unsupported Duration type");
}

/**
 * @brief High-precision single-shot timer for measuring elapsed time.
 *
 * A lightweight timer class that measures elapsed time between start() and stop() calls.
 * The timer is template-based to support different time resolutions and clock types.
 * It provides both raw elapsed time values and formatted string representations.
 *
 * ## Features:
 * - High precision timing using template-specified clock and duration types
 * - Automatic duration conversion to double for fractional time measurements
 * - Formatted string output with appropriate unit labels
 * - Separation of sampling (sample()) and measurement (elapsed()) operations
 * - Exception-free design with noexcept guarantees where possible
 *
 * ## Usage:
 * ```cpp
 * Timer<std::chrono::microseconds> timer;
 * timer.start();
 * // ... work to be timed ...
 * auto elapsed_us = timer.stop();  // Returns elapsed time in microseconds
 * std::cout << timer.elapsedStr(); // Outputs "123.45 us"
 * ```
 *
 * @tparam Duration The time unit for elapsed calculations (default: std::chrono::microseconds)
 * @tparam Clock The clock type to use for time measurement (default: std::chrono::steady_clock)
 * 
 * @note std::chrono::steady_clock is recommended for timing as it's monotonic and not affected by system clock adjustments
 */
template <typename Duration = std::chrono::microseconds,
          typename Clock = std::chrono::steady_clock>
class Timer {
public:
    using DurationType = Duration;  ///< The duration type used for time measurements
    using ClockType    = Clock;     ///< The clock type used for time points
    using TimePoint    = typename Clock::time_point;  ///< The time point type from the clock

    /**
     * @brief Default constructor creates a timer ready for use.
     */
    Timer() noexcept = default;

    /**
     * @brief Start the timer by recording the current time.
     * 
     * Records the current time point as the start time for elapsed time calculations.
     * Can be called multiple times to restart the timer.
     */
    auto start() noexcept -> void {
        m_start = Clock::now();
    }

    /**
     * @brief Sample the current time without stopping the timer.
     * 
     * Records the current time as the end time but does not return the elapsed time.
     * Useful for preparing multiple elapsed time queries without additional clock calls.
     */
    auto sample() noexcept -> void {
        m_end = Clock::now();
    }

    /**
     * @brief Stop the timer and return the elapsed time.
     * 
     * Records the current time as the end time and immediately returns the elapsed time
     * since the last start() call.
     * 
     * @return double The elapsed time in the timer's configured duration units
     */
    [[nodiscard]] auto stop() noexcept -> double {
        m_end = Clock::now();
        return elapsed();
    }

    /**
     * @brief Calculate elapsed time between start and end time points.
     * 
     * Computes the time difference between the recorded start and end time points,
     * converting the result to the timer's configured duration type as a double.
     * 
     * @return double The elapsed time in the timer's configured duration units
     * @note If sample() or stop() haven't been called, uses the current time as end point
     */
    [[nodiscard]] auto elapsed() const noexcept -> double {
        const auto diff = m_end - m_start;
        if constexpr (std::is_same_v<Duration, std::chrono::nanoseconds>) {
            return std::chrono::duration<double, std::nano>(diff).count();
        } else if constexpr (std::is_same_v<Duration, std::chrono::microseconds>) {
            return std::chrono::duration<double, std::micro>(diff).count();
        } else if constexpr (std::is_same_v<Duration, std::chrono::milliseconds>) {
            return std::chrono::duration<double, std::milli>(diff).count();
        } else {
            return std::chrono::duration<double, typename Duration::period>(diff).count();
        }
    }

    /**
     * @brief Get a formatted string representation of the elapsed time.
     * 
     * Returns the elapsed time as a formatted string with 2 decimal places
     * and the appropriate unit suffix (e.g., "123.45 us", "67.89 ms").
     * 
     * @return std::string Formatted elapsed time with unit suffix
     * 
     * ## Example outputs:
     * - "1234.56 us" (for microsecond timer)
     * - "78.90 ms" (for millisecond timer)  
     * - "12.34 s" (for second timer)
     */
    [[nodiscard]] auto elapsedStr() const -> std::string {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2)
            << elapsed() << ' ' << UnitStr<Duration>();
        return oss.str();
    }

private:
    TimePoint m_start{};  ///< Start time point recorded by start()
    TimePoint m_end{};    ///< End time point recorded by sample() or stop()
};

// Convenient type aliases for common timer resolutions
using TimerNS  = Timer<std::chrono::nanoseconds>;   ///< Nanosecond precision timer
using TimerUS  = Timer<std::chrono::microseconds>;  ///< Microsecond precision timer (default)
using TimerMS  = Timer<std::chrono::milliseconds>;  ///< Millisecond precision timer
using TimerSec = Timer<std::chrono::seconds>;       ///< Second precision timer

// Extern template declarations to prevent implicit instantiation
// The explicit instantiations are provided in Timers.cpp
extern template class Timer<std::chrono::nanoseconds>;
extern template class Timer<std::chrono::microseconds>;
extern template class Timer<std::chrono::milliseconds>;
extern template class Timer<std::chrono::seconds>;

/**
 * @brief Manager to track multiple named timers with optional NVTX range integration.
 *
 * This class provides a convenient way to manage multiple named timers while
 * automatically creating corresponding NVTX ranges for CUDA profiling when
 * NEON_USE_NVTX is enabled.
 *
 * ## NVTX Integration Features:
 * - Automatically starts NVTX range when timer.start() is called
 * - Automatically ends NVTX range when timer.stop() is called  
 * - Handles nested timers correctly (each timer gets its own range)
 * - Prevents range leaks by cleaning up on destruction
 * - Provides debugging info about active ranges
 *
 * ## Usage with CUDA Profiling:
 * When NEON_USE_NVTX is enabled and you profile with NVIDIA Nsight Systems,
 * you'll see named ranges in the timeline corresponding to your timer names.
 * This makes it much easier to correlate timing measurements with GPU activity.
 *
 * ## Example:
 * ```cpp
 * Neon::TimerManagerUS manager;
 * manager.start("kernel_execution");  // Starts timer + NVTX range
 * // ... CUDA kernel launch ...
 * auto time = manager.stop("kernel_execution");  // Stops timer + NVTX range
 * ```
 *
 * @tparam Duration The time unit for elapsed calculations.
 * @tparam Clock The clock type to use.
 */
template <typename Duration = std::chrono::microseconds,
          typename Clock = std::chrono::steady_clock>
class TimerManager {
public:
    using TimerType  = Timer<Duration, Clock>;  ///< The underlying timer type
    using StringView = std::string_view;        ///< String view type for efficient string parameters

    /**
     * @brief Default constructor creates an empty timer manager.
     */
    TimerManager() = default;
    
    /**
     * @brief Destructor ensures proper cleanup of NVTX ranges.
     * 
     * Automatically closes any remaining active NVTX ranges to prevent
     * profiling data corruption. This provides exception safety and prevents
     * range leaks when TimerManager is destroyed with active timers.
     */
    ~TimerManager() {
#ifdef NEON_USE_NVTX
        // Clean up any remaining NVTX ranges on destruction
        // nvtxRangePop() doesn't need range names - it pops the most recent range
        for (size_t i = 0; i < m_activeNvtxRanges.size(); ++i) {
            nvtxRangePop();
        }
#endif
        m_activeNvtxRanges.clear();
    }

    // Non-copyable to prevent accidental duplication of NVTX ranges, but movable for flexibility
    TimerManager(const TimerManager&) = delete;            ///< Copy constructor deleted
    TimerManager& operator=(const TimerManager&) = delete; ///< Copy assignment deleted
    TimerManager(TimerManager&&) = default;                ///< Move constructor available
    TimerManager& operator=(TimerManager&&) = default;     ///< Move assignment available

    /**
     * @brief Start a named timer and corresponding NVTX range.
     * 
     * Creates or reuses a timer with the given name and starts timing.
     * When NEON_USE_NVTX is enabled, also pushes an NVTX range with the same name
     * for seamless integration with CUDA profiling tools.
     * 
     * @param name Unique identifier for the timer and NVTX range
     * 
     * ## Behavior:
     * - If timer doesn't exist, creates a new one
     * - If timer already exists, restarts it (ends previous NVTX range if active)
     * - With NVTX enabled: pushes new NVTX range with the timer name
     * - Without NVTX: still tracks active timers for API consistency
     * 
     * ## Example:
     * ```cpp
     * manager.start("gpu_kernel");  // Starts timer + NVTX range "gpu_kernel"
     * // ... launch CUDA kernel ...
     * manager.stop("gpu_kernel");   // Stops timer + ends NVTX range
     * ```
     */
    auto start(StringView name) -> void {
        std::string timerName{name};
        auto [it, inserted] = m_timers.try_emplace(timerName, TimerType{});
        it->second.start();
        
#ifdef NEON_USE_NVTX
        // If timer is already running, end the previous NVTX range first
        auto rangeIt = m_activeNvtxRanges.find(timerName);
        if (rangeIt != m_activeNvtxRanges.end()) {
            nvtxRangePop();
        } else {
            // Only insert if not already present
            m_activeNvtxRanges.insert(timerName);
        }
        nvtxRangePush(timerName.c_str());
#else
        // When NVTX is disabled, still track ranges for consistency
        m_activeNvtxRanges.insert(timerName);
#endif
    }

    /**
     * @brief Start a named timer with NEON_INFO logging.
     * 
     * Starts a timer and immediately logs the timer start using NEON_INFO.
     * This combines timer start functionality with informational logging for
     * better visibility in performance analysis and debugging.
     * 
     * @param name Unique identifier for the timer and NVTX range
     * @param category Optional category/component name for the info message (default: "Timer")
     * 
     * ## Example:
     * ```cpp
     * manager.start_with_info("gpu_kernel");                    // Outputs: Timer: Starting gpu_kernel
     * manager.start_with_info("computation", "Performance");   // Outputs: Performance: Starting computation
     * // ... work ...
     * manager.stop("gpu_kernel");
     * ```
     */
    auto start_with_info(StringView name, StringView category = "Timer") -> void {
        start(name);
        NEON_INFO(std::string{category}, "Starting {}", name);
    }

    /**
     * @brief Start a named timer with NEON_TRACE logging.
     * 
     * Starts a timer and immediately logs the timer start using NEON_TRACE.
     * This combines timer start functionality with trace-level logging for
     * detailed performance tracing during development and debugging.
     * 
     * @param name Unique identifier for the timer and NVTX range
     * @param category Optional category/component name for the trace message (default: "Timer")
     * 
     * ## Example:
     * ```cpp
     * manager.start_with_trace("gpu_kernel");                   // Outputs: Timer: Starting gpu_kernel
     * manager.start_with_trace("fine_detail", "GPU_Debug");    // Outputs: GPU_Debug: Starting fine_detail
     * // ... work ...
     * manager.stop("gpu_kernel");
     * ```
     */
    auto start_with_trace(StringView name, StringView category = "Timer") -> void {
        start(name);
        NEON_TRACE(std::string{category}, "Starting {}", name);
    }

    /**
     * @brief Sample the current time for a named timer without stopping it.
     * 
     * Records the current time as the end point for the specified timer
     * without affecting any active NVTX ranges. This allows multiple
     * elapsed time queries without additional clock calls.
     * 
     * @param name Name of the timer to sample
     * @throws NeonException if the timer doesn't exist
     */
    auto sample(StringView name) -> void {
        getTimer(name).sample();
    }

    /**
     * @brief Stop a named timer and end its corresponding NVTX range.
     * 
     * Stops the specified timer, records the final time, and returns the elapsed time.
     * When NEON_USE_NVTX is enabled, also pops the corresponding NVTX range.
     * 
     * @param name Name of the timer to stop
     * @return double Elapsed time in the timer's configured duration units
     * @throws NeonException if the timer doesn't exist
     * 
     * ## Example:
     * ```cpp
     * manager.start("computation");
     * // ... work ...
     * auto elapsed_us = manager.stop("computation");  // Returns microseconds if using TimerManagerUS
     * ```
     */
    [[nodiscard]] auto stop(StringView name) -> double {
        auto result = getTimer(name).stop();
        
        // End NVTX range if this timer has an active range
        auto rangeIt = m_activeNvtxRanges.find(std::string{name});
        if (rangeIt != m_activeNvtxRanges.end()) {
#ifdef NEON_USE_NVTX
            nvtxRangePop();
#endif
            m_activeNvtxRanges.erase(rangeIt);
        }
        
        return result;
    }

    /**
     * @brief Get elapsed time for a named timer without stopping it.
     * 
     * Returns the elapsed time for the specified timer without affecting
     * the timer state or any active NVTX ranges.
     * 
     * @param name Name of the timer to query
     * @return double Elapsed time in the timer's configured duration units
     * @throws NeonException if the timer doesn't exist
     */
    [[nodiscard]] auto elapsed(StringView name) const -> double {
        return getTimer(name).elapsed();
    }

    /**
     * @brief Get formatted elapsed time string for a named timer.
     * 
     * Returns a formatted string representation of the elapsed time
     * with appropriate unit suffix (e.g., "123.45 us").
     * 
     * @param name Name of the timer to query
     * @return std::string Formatted elapsed time with unit suffix
     * @throws NeonException if the timer doesn't exist
     */
    [[nodiscard]] auto elapsedStr(StringView name) const -> std::string {
        return getTimer(name).elapsedStr();
    }

    /**
     * @brief Generate a formatted summary of all timers and their elapsed times.
     *
     * Creates a multi-line string listing all managed timers with their elapsed times.
     * Timer names are aligned in columns for readability, and each line can be prefixed
     * with a custom string (useful for indentation in logs).
     *
     * @param prefix String to prepend to each line (default is empty)
     * @return std::string Multi-line formatted summary of all timers
     * 
     * ## Example output:
     * ```
     * initialization: 123.45 ms
     * computation   : 567.89 ms  
     * cleanup       : 12.34 ms
     * ```
     * 
     * ## With prefix:
     * ```cpp
     * std::cout << manager.toString("  ");  // Indent each line
     * ```
     * Output:
     * ```
     *   initialization: 123.45 ms
     *   computation   : 567.89 ms
     *   cleanup       : 12.34 ms
     * ```
     */
    [[nodiscard]] auto toString(std::string_view prefix = "") const -> std::string {
        // Compute maximum name length
        std::size_t maxName = 0;
        for (auto const& [name, _] : m_timers) {
            maxName = std::max(maxName, name.size());
        }

        static constexpr auto newLine = "\n";
        std::ostringstream oss;
        for (auto const& [name, timer] : m_timers) {
            oss << prefix << name
                << std::string(maxName - name.size(), ' ')
                << ": " << timer.elapsedStr()
                << newLine;
        }
        return oss.str();
    }

    /**
     * @brief Get a list of all timer names currently managed.
     * 
     * Returns a vector containing the names of all timers that have been
     * created in this manager, regardless of their current state.
     * 
     * @return std::vector<std::string> Names of all managed timers
     * 
     * ## Example:
     * ```cpp
     * auto names = manager.list();
     * for (const auto& name : names) {
     *     std::cout << name << ": " << manager.elapsedStr(name) << std::endl;
     * }
     * ```
     */
    [[nodiscard]] auto list() const -> std::vector<std::string> {
        std::vector<std::string> keys;
        keys.reserve(m_timers.size());
        for (auto const& [key, _] : m_timers) keys.push_back(key);
        return keys;
    }

    /**
     * @brief Remove a timer and clean up its NVTX range.
     * 
     * Completely removes the specified timer from the manager and properly
     * closes any active NVTX range associated with it. This is useful for
     * cleaning up timers that are no longer needed.
     * 
     * @param name Name of the timer to remove
     * 
     * ## Behavior:
     * - Ends active NVTX range if one exists for this timer
     * - Removes the timer from internal storage
     * - Does nothing if timer doesn't exist (no exception thrown)
     * 
     * @note After reset(), the timer name can be reused with start()
     */
    auto reset(StringView name) -> void {
        // End NVTX range if this timer has an active range before resetting
        auto rangeIt = m_activeNvtxRanges.find(std::string{name});
        if (rangeIt != m_activeNvtxRanges.end()) {
#ifdef NEON_USE_NVTX
            nvtxRangePop();
#endif
            m_activeNvtxRanges.erase(rangeIt);
        }
        m_timers.erase(std::string{name});
    }

    /**
     * @brief Get list of timers that currently have active NVTX ranges.
     * 
     * When NEON_USE_NVTX is enabled, returns the names of timers that have
     * active NVTX ranges. When NEON_USE_NVTX is disabled, still tracks
     * active timers for consistency but no actual NVTX ranges are created.
     * 
     * @return std::vector<std::string> Names of timers with active ranges
     */
    [[nodiscard]] auto getActiveNvtxRanges() const -> std::vector<std::string> {
        std::vector<std::string> ranges;
        ranges.reserve(m_activeNvtxRanges.size());
        for (const auto& range : m_activeNvtxRanges) {
            ranges.push_back(range);
        }
        return ranges;
    }

    /**
     * @brief Print elapsed time for a named timer using NEON_TRACE.
     * 
     * Outputs the elapsed time for the specified timer using the NEON_TRACE macro
     * for trace-level logging. This is useful for detailed performance tracing
     * during development and debugging.
     * 
     * @param name Name of the timer to trace
     * @param category Optional category/component name for the trace message (default: "Timer")
     * @throws NeonException if the timer doesn't exist
     * 
     * ## Example:
     * ```cpp
     * manager.start("gpu_kernel");
     * // ... work ...
     * manager.trace("gpu_kernel");                    // Outputs: Timer: gpu_kernel elapsed 123.45 us
     * manager.trace("gpu_kernel", "GPU_Performance"); // Outputs: GPU_Performance: gpu_kernel elapsed 123.45 us
     * ```
     */
    auto trace(StringView name, StringView category = "Timer") const -> void {
        NEON_TRACE(std::string{category}, "{} elapsed {}", name, elapsedStr(name));
    }

    /**
     * @brief Print elapsed time for a named timer using NEON_INFO.
     * 
     * Outputs the elapsed time for the specified timer using the NEON_INFO macro
     * for informational logging. This is useful for performance reporting
     * and general timing information.
     * 
     * @param name Name of the timer to log
     * @param category Optional category/component name for the info message (default: "Timer")
     * @throws NeonException if the timer doesn't exist
     * 
     * ## Example:
     * ```cpp
     * manager.start("computation");
     * // ... work ...
     * manager.log("computation");                        // Outputs: Timer: computation elapsed 567.89 ms
     * manager.log("computation", "Performance_Report");  // Outputs: Performance_Report: computation elapsed 567.89 ms
     * ```
     */
    auto log(StringView name, StringView category = "Timer") const -> void {
        NEON_INFO(std::string{category}, "{} elapsed {}", name, elapsedStr(name));
    }

private:
    /**
     * @brief Get a reference to a timer by name (non-const version).
     * 
     * @param name Name of the timer to retrieve
     * @return TimerType& Reference to the timer
     * @throws NeonException if timer doesn't exist
     */
    auto getTimer(StringView name) -> TimerType& {
        auto it = m_timers.find(std::string{name});
        if (it == m_timers.end()) {
            Neon::NeonException e("TimerManager");
            e << "Timer '" << name << "' not found";
            NEON_THROW(e);
        }
        return it->second;
    }

    /**
     * @brief Get a const reference to a timer by name (const version).
     * 
     * @param name Name of the timer to retrieve
     * @return const TimerType& Const reference to the timer
     * @throws NeonException if timer doesn't exist
     */
    auto getTimer(StringView name) const -> const TimerType& {
        auto it = m_timers.find(std::string{name});
        if (it == m_timers.end()) {
            Neon::NeonException e("TimerManager");
            e << "Timer '" << name << "' not found";
            NEON_THROW(e);
        }
        return it->second;
    }

    std::unordered_map<std::string, TimerType> m_timers;        ///< Storage for named timers
    std::unordered_set<std::string> m_activeNvtxRanges;        ///< Track active NVTX ranges for cleanup
};

// Convenient type aliases for common timer manager resolutions
using TimerManagerNS  = TimerManager<std::chrono::nanoseconds>;   ///< Nanosecond precision timer manager
using TimerManagerUS  = TimerManager<std::chrono::microseconds>;  ///< Microsecond precision timer manager (recommended)
using TimerManagerMS  = TimerManager<std::chrono::milliseconds>;  ///< Millisecond precision timer manager  
using TimerManagerSec = TimerManager<std::chrono::seconds>;       ///< Second precision timer manager

// Extern template declarations to prevent implicit instantiation
// The explicit instantiations are provided in Timers.cpp
extern template class TimerManager<std::chrono::nanoseconds>;
extern template class TimerManager<std::chrono::microseconds>;
extern template class TimerManager<std::chrono::milliseconds>;
extern template class TimerManager<std::chrono::seconds>;

} // namespace Neon

#endif // NEON_WARP_COMPILATION
