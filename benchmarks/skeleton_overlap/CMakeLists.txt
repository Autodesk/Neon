_carbon_benchmark_add(carbon_benchmark_skeleton_overlap skeleton_overlap.cu)
target_link_libraries(carbon_benchmark_skeleton_overlap PRIVATE carbon::carbon
                                                                  carbon_benchmark_common)
