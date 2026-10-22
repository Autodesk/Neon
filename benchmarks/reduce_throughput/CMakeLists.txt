_carbon_benchmark_add(carbon_benchmark_reduce_throughput reduce_throughput.cu)
target_link_libraries(carbon_benchmark_reduce_throughput PRIVATE carbon::carbon
                                                                 carbon_benchmark_common)
