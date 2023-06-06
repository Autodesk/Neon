cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

add_subdirectory("domain-globalIdx")
add_subdirectory("domain-host-containers")
add_subdirectory("domain-map")
add_subdirectory("domain-neighbour-globalIdx")
add_subdirectory("domain-halos")
add_subdirectory("domain-stencil")
add_subdirectory("domain-bGrid-tray")

add_subdirectory("domainUt_sGrid")
add_subdirectory("domain-unit-test-eGrid")
add_subdirectory("domain-unit-test-gridInterface")
# add_subdirectory("domain-unit-test-cast")
add_subdirectory("domain-unit-test-patterns-containers")

# add_subdirectory("gUt_dataView_patterns")
add_subdirectory("domainUt_swap")
# add_subdirectory("domain-unit-test-staggered-grid")
add_subdirectory("gUt_tools")
add_subdirectory("gUt_vtk")
add_subdirectory("gUt_mGrid")

