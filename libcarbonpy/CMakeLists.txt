add_library(carbonpy SHARED
  src/carbonpy.cpp
  src/dense.cpp
  src/block.cpp
  src/mgrid.cpp
  src/cuda_driver.cpp
  src/warp_kernel.cpp
  src/skeleton.cpp
  src/pod_layout.cpp)

target_include_directories(carbonpy
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(carbonpy PRIVATE carbon CUDA::cudart CUDA::cuda_driver)

set_target_properties(carbonpy PROPERTIES
  OUTPUT_NAME carbonpy
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

target_compile_definitions(carbonpy PRIVATE CARBONPY_EXPORTS)

if(CARBON_BUILD_TESTS)
  set(_carbonpy_plat_dir "linux_${CMAKE_SYSTEM_PROCESSOR}")
  add_custom_command(TARGET carbonpy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
      "${CMAKE_SOURCE_DIR}/py/carbon/lib/${_carbonpy_plat_dir}"
    COMMAND ${CMAKE_COMMAND} -E copy
      "$<TARGET_FILE:carbonpy>"
      "${CMAKE_SOURCE_DIR}/py/carbon/lib/${_carbonpy_plat_dir}/"
    COMMENT "Copy libcarbonpy into py/carbon/lib/${_carbonpy_plat_dir}")
endif()

install(TARGETS carbonpy EXPORT carbonpyTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/ DESTINATION include)
