#pragma once

/**
 * SYCL twin of @c detail/backend/cudalike/dense_kernel.h. Mirrors the CUDA file's @e concept
 * declaration (host-portable, used by the public @c Dense API for static-checking
 * loading-lambda return types). The SYCL dense kernel launcher itself lives next door
 * in @c detail/backend/sycl/dense_launch.h ; this file is the concept-only shim so
 * @c carbon/dense.h 's transitive include stays cheap under SYCL — the public dense
 * API only needs the concept, not the kernel template.
 */

#include <carbon/data_view.h>
#include <carbon/detail/dense/span.h>

#include <concepts>
#include <utility>

namespace carbon::detail::dense {

/**
 * Callable type for the SYCL dense grid launch (phase 2.4): @c void fn(span, idx).
 * Same shape as the CUDA backend's concept so user lambdas authored against the public
 * @c Dense::make_kernel API satisfy both, no per-backend authoring needed.
 */
template <typename F>
concept dense_kernel_fn = requires(F fn, Span span, Idx idx) {
  {fn(span, idx)}->std::same_as<void>;
};

}  // namespace carbon::detail::dense
