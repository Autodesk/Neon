#pragma once

/**
 * Backend dispatcher for the stage-2 intra-process combine helpers used by the
 * multi-GPU reduce path. Pulls in either the CUDA or SYCL twin under
 * @c detail/backend/<name>/reduce_combine.h based on @c CARBON_BACKEND_IS_* . Both
 * twins expose the same two free functions in @c carbon::detail::dense :
 *
 * - @c launch_combine_partials<T, Op>(stream, src, dst, n, op) — launches a one-thread
 *   fold of the staging buffer into the per-view final scalar.
 * - @c sync_root_after_nonroot(stream, root) — fan-in cross-stream ordering so the
 *   root stream waits for every other per-GPU stream.
 *
 * The reduce @c KernelImpl headers ( @c reduce_kernel_impl.h ,
 * @c universal_reduce_kernel_impl.h ) include this single shim so they don't have to
 * branch on backend at the call site.
 */

#include <carbon/detail/backend/select.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/reduce_combine.h>
#elif CARBON_BACKEND_IS_SYCL
#include <carbon/detail/backend/sycl/reduce_combine.h>
#else
#error "carbon: no compute backend selected (see carbon/detail/backend/select.h)."
#endif
