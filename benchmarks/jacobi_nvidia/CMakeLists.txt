_carbon_benchmark_add(carbon_benchmark_jacobi_nvidia jacobi_nvidia.cu)
target_link_libraries(carbon_benchmark_jacobi_nvidia PRIVATE carbon::carbon
                                                              carbon_benchmark_common)
