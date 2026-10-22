#pragma once

/**
 * Compile-time backend selector for carbon.
 *
 * Exactly one of @c CARBON_BACKEND_CUDA / @c CARBON_BACKEND_SYCL / @c CARBON_BACKEND_HIP is
 * defined to @c 1 by the top-level CMake (driven by the @c CARBON_BACKEND cache variable).
 * This header normalises those raw definitions into self-documenting helper macros that the
 * rest of @c detail/ uses to fork backend-specific headers and call sites:
 *
 *  - @c CARBON_BACKEND_IS_CUDA — non-zero iff the active backend is the CUDA backend.
 *  - @c CARBON_BACKEND_IS_SYCL — non-zero iff the active backend is the SYCL backend.
 *  - @c CARBON_BACKEND_IS_HIP  — non-zero iff the active backend is the HIP (ROCm) backend.
 *  - @c CARBON_BACKEND_NAME    — string literal naming the active backend, useful in error
 *                                messages and runtime diagnostics.
 *
 * Two further helpers fold the CUDA and HIP backends together, since HIP is a near-1:1
 * source map of the CUDA runtime + cub (hipcub) and the two share almost all of the
 * @c detail/ device-launch plumbing:
 *
 *  - @c CARBON_BACKEND_IS_CUDALIKE — non-zero iff the active backend is CUDA @e or HIP.
 *  - @c CARBON_CUDALIKE_DEVICE_TU  — non-zero iff the current translation unit is being
 *                                    compiled by a cuda-like device compiler (nvcc defines
 *                                    @c __CUDACC__; AdaptiveCpp / amdclang HIP defines
 *                                    @c __HIPCC__). Replaces bare @c defined(__CUDACC__)
 *                                    tests inside the shared (non-backend-dir) headers so a
 *                                    single arm serves both cuda-like backends.
 *
 * Always include this header (rather than testing @c CARBON_BACKEND_CUDA / @c
 * CARBON_BACKEND_SYCL / @c CARBON_BACKEND_HIP directly) so callers stay forward-compatible
 * if a further backend is ever added.
 *
 * @par Why compile-time
 * Backend choice affects every per-device handle, every kernel-launch syntax, and every
 * memory-copy primitive. A runtime selector would multiply the surface area of the public
 * @c Stream / @c Event / @c Buffer types and force every TU to compile both branches. By
 * fixing the backend at carbon-build time we keep the user-facing types as concrete value
 * types (no @c std::variant, no virtual dispatch) and avoid any toolchain that cannot
 * compile both CUDA and SYCL in the same TU.
 *
 * @par What if both definitions are set
 * The CMake glue refuses to set both at once, but if a downstream build manages to define
 * both we error out at compile time so the mistake surfaces immediately.
 */

#if !defined(CARBON_BACKEND_CUDA) && !defined(CARBON_BACKEND_SYCL) && !defined(CARBON_BACKEND_HIP)
#error \
    "carbon: no compute backend selected. The carbon CMake target must define exactly one of " \
    "CARBON_BACKEND_CUDA, CARBON_BACKEND_SYCL or CARBON_BACKEND_HIP. If you build carbon " \
    "yourself, configure with -DCARBON_BACKEND=Cuda. If you consume carbon via " \
    "find_package(carbon) the macro is propagated automatically from carbon's PUBLIC compile " \
    "definitions."
#endif

#if (defined(CARBON_BACKEND_CUDA) + defined(CARBON_BACKEND_SYCL) + defined(CARBON_BACKEND_HIP)) > 1
#error \
    "carbon: more than one of CARBON_BACKEND_CUDA / CARBON_BACKEND_SYCL / CARBON_BACKEND_HIP is " \
    "defined. Exactly one backend must be selected; pick one in CMake " \
    "(-DCARBON_BACKEND=Cuda|Sycl|Hip)."
#endif

#if defined(CARBON_BACKEND_CUDA) && CARBON_BACKEND_CUDA
#define CARBON_BACKEND_IS_CUDA 1
#define CARBON_BACKEND_IS_SYCL 0
#define CARBON_BACKEND_IS_HIP 0
#define CARBON_BACKEND_NAME "cuda"
#elif defined(CARBON_BACKEND_SYCL) && CARBON_BACKEND_SYCL
#define CARBON_BACKEND_IS_CUDA 0
#define CARBON_BACKEND_IS_SYCL 1
#define CARBON_BACKEND_IS_HIP 0
#define CARBON_BACKEND_NAME "sycl"
#elif defined(CARBON_BACKEND_HIP) && CARBON_BACKEND_HIP
#define CARBON_BACKEND_IS_CUDA 0
#define CARBON_BACKEND_IS_SYCL 0
#define CARBON_BACKEND_IS_HIP 1
#define CARBON_BACKEND_NAME "hip"
#else
#error "carbon: backend macro is defined but evaluates to zero. Use -DCARBON_BACKEND=Cuda|Sycl|Hip."
#endif

// CUDA and HIP share the bulk of carbon's device-launch plumbing (cuda* <-> hip* runtime,
// cub <-> hipcub, identical <<<>>> / __global__ / dim3 surface). The shared (non-backend-dir)
// headers fork on this folded macro rather than spelling out both backends at every call site.
#if CARBON_BACKEND_IS_CUDA || CARBON_BACKEND_IS_HIP
#define CARBON_BACKEND_IS_CUDALIKE 1
#else
#define CARBON_BACKEND_IS_CUDALIKE 0
#endif

// Non-zero only when the current TU is compiled by a cuda-like *device* compiler: nvcc for
// the CUDA backend (defines __CUDACC__), AdaptiveCpp / amdclang for the HIP backend (define
// __HIPCC__). Host-only .cpp TUs (g++/clang++) leave both undefined, so device-only bodies
// (__global__ kernels, cub/hipcub usage) stay out of the host compile.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define CARBON_CUDALIKE_DEVICE_TU 1
#else
#define CARBON_CUDALIKE_DEVICE_TU 0
#endif
