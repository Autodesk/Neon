#pragma once

/**
 * @file halo_pack.h
 * @brief Launchers for the packed halo gather/scatter kernels (CUDA / HIP backends only).
 *
 * The kernels live in @c src/halo_pack.cu and are compiled once into libcarbon, so a host-only
 * translation unit that merely includes @c halo_distributed.h can run a packed @c HaloDistributed without a device
 * toolchain, and there is no per-TU template instantiation (no ODR hazard between a @c .cpp and a
 * @c .cu that both construct @c HaloDistributed<T,N>). The launchers are type-erased: a pack/unpack is a pure
 * copy, so only @c sizeof(T) matters.
 *
 * Planning (extents, component lists, messages) and the per-thread index math are backend-neutral
 * and live in @c halo_pack_plan.h. Under the SYCL backend nothing here is declared and @c HaloDistributed
 * silently keeps using the run-based path.
 */

#include <carbon/detail/backend/select.h>
#include <carbon/detail/halo_pack_plan.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace carbon::detail::halo_pack {

/**
 * Boolean environment switch. Unset -> @p fallback; "0", "off", "false", "no" -> false;
 * anything else -> true. Read once per @c HaloDistributed construction (not per exchange).
 */
[[nodiscard]] inline auto
env_flag(const char* name,
         bool fallback) -> bool {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') {
    return fallback;
  }
  return !(std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0 ||
           std::strcmp(v, "false") == 0 || std::strcmp(v, "no") == 0);
}

/** Element widths the type-erased kernels are instantiated for. */
[[nodiscard]] constexpr auto
element_size_supported(std::size_t bytes) noexcept -> bool {
  return bytes == 1U || bytes == 2U || bytes == 4U || bytes == 8U || bytes == 16U;
}

}  // namespace carbon::detail::halo_pack

#if CARBON_BACKEND_IS_CUDALIKE

#include <carbon/detail/backend/cuda_like.h>
#include <carbon/detail/gpu.h>
#include <carbon/export.h>

namespace carbon::detail::halo_pack {

/**
 * Gather @p a 's box of the field (@p field_base, all components addressed through @p a) into the
 * contiguous staging region @p stage_region. Asynchronous on @p stream. Returns the launch error.
 */
[[nodiscard]] CARBON_API auto
launch_gather(const Gpu& gpu,
              gpu_stream_t stream,
              std::size_t elem_bytes,
              const void* field_base,
              void* stage_region,
              const PackKernelArgs& a) -> gpu_error_t;

/** Inverse of @ref launch_gather: scatter the staging region into the field's box. */
[[nodiscard]] CARBON_API auto
launch_scatter(const Gpu& gpu,
               gpu_stream_t stream,
               std::size_t elem_bytes,
               const void* stage_region,
               void* field_base,
               const PackKernelArgs& a) -> gpu_error_t;

}  // namespace carbon::detail::halo_pack

#endif  // CARBON_BACKEND_IS_CUDALIKE
