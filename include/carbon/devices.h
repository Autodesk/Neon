#pragma once

#include <carbon/detail/gpu.h>
#include <carbon/export.h>

#include <vector>

namespace carbon {

/**
 * Builds one @c detail::Gpu per CUDA device visible to this process (ordinals
 * @c 0 .. @c Gpu::device_count()-1), in ascending order. Empty if @c cudaGetDeviceCount reports
 * zero. Respects CUDA visibility (e.g. @c CUDA_VISIBLE_DEVICES).
 *
 * Typical use is to feed a @c System over every visible GPU on a single node:
 * @code
 * carbon::init();
 * carbon::System sys(carbon::get_all_devices());
 * @endcode
 *
 * The returned element type @c detail::Gpu is implementation-facing; the only intended use of
 * the vector is to pass it (or a slice of it) to @ref System / @ref detail::GpuGroup
 * constructors.
 */
[[nodiscard]] CARBON_API auto
get_all_devices() -> std::vector<detail::Gpu>;

}  // namespace carbon
