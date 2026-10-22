_carbon_benchmark_add(carbon_benchmark_block_sparse block_sparse.cu)
target_link_libraries(carbon_benchmark_block_sparse PRIVATE carbon::carbon
                                                             carbon_benchmark_common)
