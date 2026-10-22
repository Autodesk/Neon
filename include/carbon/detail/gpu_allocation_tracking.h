#pragma once

#include <carbon/export.h>

#include <cstddef>

namespace carbon::detail {

/** Add @p bytes to the running total for CUDA device @p device_id (detail::Gpu::buffer). */
CARBON_API auto
gpu_track_allocation(int device_id,
                     size_t bytes) -> void;

/** Subtract @p bytes when a detail::Gpu::buffer allocation is released (shared_ptr deleter). */
CARBON_API auto
gpu_track_deallocation(int device_id,
                       size_t bytes) -> void;

/** Current total bytes tracked for @p device_id (only buffers from detail::Gpu::buffer). */
[[nodiscard]] CARBON_API auto
gpu_tracked_bytes(int device_id) -> size_t;

}  // namespace carbon::detail
