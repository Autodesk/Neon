/**
 * @file chrono.h
 *
 * Utilities to measure time for a set of samples.
 *
 * @author Massimiliano Meneghin <massimiliano.meneghin@autodesk.com>
 *
 * @version 0.1
 */
#pragma once


#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "Neon/core/types/Exceptions.h"
#include "Neon/core/types/Macros.h"


namespace Neon {

// TODO@[Max](Have a look at http://www.cplusplus.com/forum/general/187899/ )
//using clock_type = typename std::conditional< std::chrono::high_resolution_clock::is_steady,
//        std::chrono::high_resolution_clock,
//        std::chrono::steady_clock >::type ;


template <typename TIME_UNIT_ta = std::chrono::microseconds,
          typename CLOCK_TYPE_ta = std::chrono::high_resolution_clock>
class Timer_t
{
   private:
    std::chrono::time_point<CLOCK_TYPE_ta> m_start, m_end;

   public:
    using s = std::chrono::seconds;
    using ms = std::chrono::milliseconds;
    using us = std::chrono::microseconds;
    using ns = std::chrono::nanoseconds;

    Timer_t()
    {
        if (std::is_same<TIME_UNIT_ta, s>::value || std::is_same<TIME_UNIT_ta, ms>::value || std::is_same<TIME_UNIT_ta, us>::value) {

        } else {
            Neon::NeonException e("Timer_t");
            e << ("Unsupported Time Unit ");
            NEON_THROW(e);
        }
    }


   public:
    static const char* unit()
    {
        if (std::is_same<TIME_UNIT_ta, s>::value) {
            return "s";
        } else if (std::is_same<TIME_UNIT_ta, ms>::value) {
            return "ms";
        } else if (std::is_same<TIME_UNIT_ta, us>::value) {
            return "us";
        } else {
            Neon::NeonException e("Timer_t");
            e << ("Unsupported Time Unit");
            NEON_THROW(e);
        }
    }


    inline void start()
    {
        m_start = CLOCK_TYPE_ta::now();
        return;
    }

    inline void sample()
    {
        m_end = CLOCK_TYPE_ta::now();
        return;
    }

    inline void stop()
    {
        m_end = CLOCK_TYPE_ta::now();
        return;
    }

    double time()
    {
        //std::chrono::duration<TIME_UNIT_ta> fp_ms = (m_end - m_start);
        //auto enlapsedTime = std::chrono::duration_cast<TIME_UNIT_ta>(m_end - m_start);
        if (std::is_same<TIME_UNIT_ta, s>::value) {
            std::chrono::duration<double, std::milli> elapsedTime = m_end - m_start;
            return elapsedTime.count() / 1000.0;
        } else if (std::is_same<TIME_UNIT_ta, ms>::value) {
            std::chrono::duration<double, std::milli> elapsedTime = m_end - m_start;
            return elapsedTime.count();
        } else if (std::is_same<TIME_UNIT_ta, us>::value) {
            std::chrono::duration<double, std::micro> elapsedTime = m_end - m_start;
            return elapsedTime.count();
        } else {
            Neon::NeonException e("Timer_t");
            e << ("Unsupported Time Unit");
            NEON_THROW(e);
        }
    }

    std::string timeStr()
    {
        std::ostringstream msg;
        msg << this->time() << " " << std::string() << this->unit();
        return msg.str();
    }
    
};
using Timer_ns = Timer_t<std::chrono::nanoseconds, std::chrono::high_resolution_clock>;
using Timer_us = Timer_t<std::chrono::microseconds, std::chrono::high_resolution_clock>;
using Timer_ms = Timer_t<std::chrono::milliseconds, std::chrono::steady_clock>;
using Timer_sec = Timer_t<std::chrono::seconds, std::chrono::steady_clock>;

}  // namespace Neon
