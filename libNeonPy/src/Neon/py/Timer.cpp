#include "Neon/Neon.h"
#include "Neon/Report.h"
#include "Neon/core/core.h"
#include "Neon/py/macros.h"

/**
 * Initialize a new grid object on the heap.
 * NOTE: some parameters are still not exposed
 */
extern "C" auto timer_ms_new(
    void** handle)
    -> int
{
    Neon::init();
    auto timerPtr = new (std::nothrow) Neon::TimerMS();
    if (timerPtr != nullptr) {
        timerPtr->start();
        *handle = timerPtr;
        return 0;
    }
    return 1;
}

/**
 * Delete a grid object on the heap.
 */
extern "C" auto timer_ms_delete(
    void** handle)
    -> int
{
    NEON_PY_PRINT_BEGIN(*handle);

    auto timer = reinterpret_cast<Neon::TimerMS*>(*handle);

    if (timer != nullptr) {
        delete timer;
    }
    *handle = nullptr;
    NEON_PY_PRINT_END(*handle);
    return 0;
}

extern "C" auto timer_ms_start(void* handle) -> int
{
    auto timer = reinterpret_cast<Neon::TimerMS*>(handle);
    timer->start();
    return 0;
}

extern "C" auto timer_ms_stop(void* handle) -> double
{
    auto timer = reinterpret_cast<Neon::TimerMS*>(handle);
    return timer->stop();
}

extern "C" auto timer_ms_time(void* handle) -> double
{
    auto timer = reinterpret_cast<Neon::TimerMS*>(handle);
    return timer->elapsed();
}