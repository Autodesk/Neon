#include <sstream>

#include "Neon/Neon.h"

int main(int, char**)
{
    // Step 1 -> Neon backend: choosing the hardware for the computation
    Neon::Backend backend = [] {
        Neon::init();
        // auto runtime = Neon::Runtime::openmp;
        auto runtime = Neon::Runtime::stream;
        // We are overbooking XPU 0 three times
        std::vector<int> xpuIds{0, 0, 0};
        Neon::Backend    backend(xpuIds, runtime);
        // Printing some information
        NEON_INFO(backend.toString());
        return backend;
    }();

    backend.forEachXpu

    return 0;
}