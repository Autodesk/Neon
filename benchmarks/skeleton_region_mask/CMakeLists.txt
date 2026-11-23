_carbon_benchmark_add(carbon_benchmark_skeleton_region_mask skeleton_region_mask.cu)
target_link_libraries(carbon_benchmark_skeleton_region_mask PRIVATE carbon::carbon
                                                                       carbon_benchmark_common)
