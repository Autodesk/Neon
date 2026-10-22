#pragma once

#include <carbon/export.h>
#include <carbon/version.h>

namespace carbon {

/** Returns the library version string (e.g. "0.1"). */
CARBON_API auto
info() -> const char*;

/** Initializes Carbon (currently the default logger). Safe to call more than once. */
CARBON_API auto
init() -> void;

/**
 * Number of CUDA devices visible to this process (0 if none or if the runtime query fails).
 * Same as detail::Gpu::device_count(); respects CUDA visibility (e.g. CUDA_VISIBLE_DEVICES).
 */
[[nodiscard]] CARBON_API auto
device_count() noexcept -> int;

}  // namespace carbon

#include <carbon/boundary_mode.h>
#include <carbon/checkpoint.h>
#include <carbon/data_view.h>
#include <carbon/dense.h>
#include <carbon/devices.h>
#include <carbon/floating3d.h>
#include <carbon/halo.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/mgrid.h>
#include <carbon/neighbour_value.h>
#include <carbon/overlapped_stencil.h>
#include <carbon/qcriterion.h>
#include <carbon/reduce.h>
#include <carbon/reduce_dist.h>
#include <carbon/render.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/dist_transfer.h>
#include <carbon/distributed.h>
#endif

// The umbrella's direct includes are all from <carbon/...> (no <carbon/detail/...>): new public
// surface always lands in a top-level header so we can see it here. The `detail` subtree is
// still reachable through the public types above (e.g. @c System::gpu_group() returns a
// @c detail::GpuGroup&, and @c carbon/system.h transitively pulls in
// @c <carbon/detail/gpu_group.h> because @c System owns a @c GpuGroup by value), but
// adding a new helper under @c detail/ does not silently expand what @c #include
// @c <carbon/carbon.h> brings in.
