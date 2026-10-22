# Halo-exchange scalability benchmark: carbon::Dense vs carbon::DenseDisg.
#
# Single TU; same backend dispatch helper as the LBM benchmark
# (`_carbon_benchmark_add` defined in benchmarks/CMakeLists.txt).

_carbon_benchmark_add(carbon_benchmark_halo_scalability halo_scalability.cu)
target_link_libraries(carbon_benchmark_halo_scalability PRIVATE carbon::carbon)
