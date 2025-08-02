/**
 * @file Timers.cpp
 * @brief Explicit template instantiations for Timer and TimerManager classes
 * 
 * This file provides explicit template instantiations for common Timer and TimerManager
 * template specializations to reduce compilation time and binary size. The extern template
 * declarations in Timers.h prevent implicit instantiation, while this file provides the
 * actual instantiations that will be linked into the final binary.
 * 
 * @author Neon Development Team
 * @date 2024
 * @copyright Copyright (c) 2024 Neon Project
 */

#include "Neon/core/types/Timers.h"

namespace Neon {

// Explicit template instantiations for Timer class with common duration types
template class Timer<std::chrono::nanoseconds>;
template class Timer<std::chrono::microseconds>;
template class Timer<std::chrono::milliseconds>;
template class Timer<std::chrono::seconds>;

// Explicit template instantiations for TimerManager class with common duration types
template class TimerManager<std::chrono::nanoseconds>;
template class TimerManager<std::chrono::microseconds>;
template class TimerManager<std::chrono::milliseconds>;
template class TimerManager<std::chrono::seconds>;

// Note: UnitStr is a constexpr function template and doesn't need explicit instantiation
// since it's evaluated at compile time.

} // namespace Neon