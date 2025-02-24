
#include <Kokkos_Core.hpp>

#include <cstdio>
#include <iostream>


int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    Kokkos::DefaultExecutionSpace().print_configuration(std::cout);

    // if (argc < 2) {
    //     fprintf(stderr, "Usage: %s [<kokkos_options>] <size>\n", argv[0]);
    //     Kokkos::finalize();
    //     exit(1);
    // }

    Kokkos::parallel_for(
        15, KOKKOS_LAMBDA(const int i) {
            // Kokko::printf works for all backends in a parallel kernel;
            // std::ostream does not.
            Kokkos::printf("Hello from i = %i\n", i);
        });

    // You must call finalize() after you are done using Kokkos.
    Kokkos::finalize();
}