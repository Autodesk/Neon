#pragma once

#include <carbon/data_view.h>
#include <carbon/kernel.h>

#include <vector>

#ifdef __cplusplus

namespace carbonpy {

[[nodiscard]] auto
make_warp_dense_kernel(void* dense_container_handle) -> carbon::Kernel;

[[nodiscard]] auto
make_warp_block_kernel(void* block_container_handle) -> carbon::Kernel;

[[nodiscard]] auto
make_warp_mgrid_kernel(void* mgrid_container_handle) -> carbon::Kernel;

namespace detail {

auto
launch_dense_container_on_carbon_stream(void* dense_container_handle,
                                        size_t gpu,
                                        void* native_stream,
                                        carbon::DataView view) -> int;

auto
launch_block_container_on_carbon_stream(void* block_container_handle,
                                        size_t gpu,
                                        void* native_stream,
                                        carbon::DataView view) -> int;

auto
launch_mgrid_container_on_carbon_stream(void* mgrid_container_handle,
                                          size_t gpu,
                                          void* native_stream,
                                          carbon::DataView view) -> int;

[[nodiscard]] auto
dense_container_tokens(void* dense_container_handle)
    -> const std::vector<carbon::AccessToken>&;

[[nodiscard]] auto
block_container_tokens(void* block_container_handle)
    -> const std::vector<carbon::AccessToken>&;

[[nodiscard]] auto
mgrid_container_tokens(void* mgrid_container_handle)
    -> const std::vector<carbon::AccessToken>&;

}  // namespace detail

}  // namespace carbonpy

#endif
