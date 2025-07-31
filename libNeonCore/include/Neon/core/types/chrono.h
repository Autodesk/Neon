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
#include <vector>

#include "Neon/core/types/Exceptions.h"
#include "Neon/core/types/Macros.h"

namespace Neon {

/**
 * @brief Helper function to obtain a string representation of a duration unit at compile time.
 *
 * @tparam Duration The std::chrono duration type.
 * @return std::string_view A string literal representing the unit (e.g., "ms", "us").
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
 * @brief Single-shot timer to measure elapsed time.
 *
 * @tparam Duration The time unit for elapsed calculations (e.g., std::chrono::microseconds).
 * @tparam Clock The clock type to use (e.g., std::chrono::steady_clock).
 */
template <typename Duration = std::chrono::microseconds,
          typename Clock = std::chrono::steady_clock>
class Timer {
public:
    using DurationType = Duration;
    using ClockType    = Clock;
    using TimePoint    = typename Clock::time_point;

    Timer() noexcept = default;

    auto start() noexcept -> void {
        m_start = Clock::now();
    }

    auto sample() noexcept -> void {
        m_end = Clock::now();
    }

    [[nodiscard]] auto stop() noexcept -> double {
        m_end = Clock::now();
        return elapsed();
    }

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

    [[nodiscard]] auto elapsedStr() const -> std::string {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2)
            << elapsed() << ' ' << UnitStr<Duration>();
        return oss.str();
    }

private:
    TimePoint m_start{};
    TimePoint m_end{};
};

// Aliases for common resolutions
using TimerNS  = Timer<std::chrono::nanoseconds>;
using TimerUS  = Timer<std::chrono::microseconds>;
using TimerMS  = Timer<std::chrono::milliseconds>;
using TimerSec = Timer<std::chrono::seconds>;

/**
 * @brief Manager to track multiple named timers.
 *
 * @tparam Duration The time unit for elapsed calculations.
 * @tparam Clock The clock type to use.
 */
template <typename Duration = std::chrono::microseconds,
          typename Clock = std::chrono::steady_clock>
class TimerManager {
public:
    using TimerType  = Timer<Duration, Clock>;
    using StringView = std::string_view;

    auto start(StringView name) -> void {
        auto [it, inserted] = m_timers.try_emplace(std::string{name}, TimerType{});
        it->second.start();
    }

    auto sample(StringView name) -> void {
        getTimer(name).sample();
    }

    [[nodiscard]] auto stop(StringView name) -> double {
        return getTimer(name).stop();
    }

    [[nodiscard]] auto elapsed(StringView name) const -> double {
        return getTimer(name).elapsed();
    }

    [[nodiscard]] auto elapsedStr(StringView name) const -> std::string {
        return getTimer(name).elapsedStr();
    }

    /**
     * @brief Get a multi-line string listing all timers and their elapsed times, with an optional prefix per line.
     *
     * @param prefix A string to prepend to each line (default is empty).
     * @return std::string Each line in the format "<prefix><name>: <time unit>", aligned in columns.
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

    [[nodiscard]] auto list() const -> std::vector<std::string> {
        std::vector<std::string> keys;
        keys.reserve(m_timers.size());
        for (auto const& [key, _] : m_timers) keys.push_back(key);
        return keys;
    }

    auto reset(StringView name) -> void {
        m_timers.erase(std::string{name});
    }

private:
    auto getTimer(StringView name) -> TimerType& {
        auto it = m_timers.find(std::string{name});
        if (it == m_timers.end()) {
            Neon::NeonException e("TimerManager");
            e << "Timer '" << name << "' not found";
            NEON_THROW(e);
        }
        return it->second;
    }

    auto getTimer(StringView name) const -> const TimerType& {
        auto it = m_timers.find(std::string{name});
        if (it == m_timers.end()) {
            Neon::NeonException e("TimerManager");
            e << "Timer '" << name << "' not found";
            NEON_THROW(e);
        }
        return it->second;
    }

    std::unordered_map<std::string, TimerType> m_timers;
};

using TimerManagerNS  = TimerManager<std::chrono::nanoseconds>;
using TimerManagerUS  = TimerManager<std::chrono::microseconds>;
using TimerManagerMS  = TimerManager<std::chrono::milliseconds>;
using TimerManagerSec = TimerManager<std::chrono::seconds>;

} // namespace Neon

#endif // NEON_WARP_COMPILATION
