#pragma once

/**
 * Cross-rank stage of the carbon reduce pipeline. Sibling of @ref carbon::reduce that
 * adds the @c ncclAllReduce / @c rcclAllReduce hop on top of the rank-local stage-2
 * combine, plus the trait pair that maps a carbon @c (Op, T) to the matching NCCL
 * operator and dtype.
 *
 * This header is unconditional in the public surface (it always compiles), but the
 * functional bits — the @c dist_allreduce_in_place free function — only have a
 * non-throwing implementation when carbon is built with @c CARBON_ENABLE_MPI=ON
 * (which sets @c CARBON_HAVE_MPI_NCCL ). Without MPI, the function still exists and
 * throws @c std::runtime_error if called with a @c System that has a @c Distributed
 * handle (which itself can only exist under @c CARBON_HAVE_MPI_NCCL ).
 *
 * Trait policy (carbon's "design call 2" answer): only @c Op / @c T pairs that have
 * @ref op_kind and @ref data_kind specialisations are accepted on the cross-rank path.
 * If a user instantiates a reduce kernel against a custom @c Op with no NCCL mapping
 * and the @c System is distributed, the per-launch dispatch will throw via the
 * @c if-constexpr -gated branch (single-process Systems still work — the throw lives
 * inside @c if (sys.distributed() != nullptr) ).
 *
 * Why not host-side @c MPI_Allgather as a fallback? The fallback path would force a
 * per-launch device→host copy for the partial scalar plus a host-side @c MPI_Allgather
 * + Op fold, which materialises a host fence inside what is otherwise a fully async
 * pipeline. The pre-step-5 design discussion converged on @b reject as the safer
 * default; reachable in the future by adding a @c reduce_host_dist.cpp fallback that
 * implements the same surface.
 */

#include <carbon/export.h>
#include <carbon/reduce.h>
#include <carbon/system.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace carbon::reduce {

/**
 * Carbon's enumeration of the reduction operators NCCL/RCCL can express directly.
 * Each entry maps to exactly one @c ncclRedOp_t value inside @c src/dist_reduce.cpp .
 *
 * The set is intentionally kept tight: any user @c Op that wants the cross-rank fast
 * path needs an @ref op_kind specialisation, which is the only place we promise the
 * mapping is well-defined. NCCL also exposes @c ncclAvg and @c ncclSumPostDiv ; carbon
 * does not surface them (Avg = Sum + scalar divide on the result; users who want it can
 * post-process @c host_value(view) ).
 */
enum class ReduceOpKind : int {
  Sum = 0,
  Min = 1,
  Max = 2,
  Prod = 3,
};

/**
 * Carbon's enumeration of the element types NCCL/RCCL accept for a collective. Each
 * entry maps to exactly one @c ncclDataType_t inside @c src/dist_reduce.cpp .
 *
 * @c Float16 / @c BFloat16 are present in the enum so the trait can specialise on
 * @c __half / @c __nv_bfloat16 (or AdaptiveCpp's @c sycl::half ) when those types are
 * actually used; the @c .cpp side guards them on the same NCCL macros NCCL itself uses
 * ( @c NCCL_FLOAT16 / @c NCCL_BFLOAT16 ) so a build against a stripped NCCL fall-back
 * still compiles.
 */
enum class ReduceDataKind : int {
  Int8 = 0,
  Uint8 = 1,
  Int32 = 2,
  Uint32 = 3,
  Int64 = 4,
  Uint64 = 5,
  Float32 = 6,
  Float64 = 7,
  Float16 = 8,
  BFloat16 = 9,
};

// Primary trait templates — intentionally undefined. Specialise to opt a
// (Op, T) pair into the cross-rank path. Anything that hits the primary template
// fails the @ref has_op_kind / @ref has_data_kind concept and gets rejected by the
// reduce KernelImpl when @c sys.distributed() != nullptr at launch time.
template <typename Op>
struct op_kind;

template <typename T>
struct data_kind;

/** True if @p Op has an @ref op_kind specialisation (i.e. an NCCL mapping). */
template <typename Op>
concept has_op_kind = requires {
  {op_kind<Op>::value}->std::convertible_to<ReduceOpKind>;
};

/** True if @p T has a @ref data_kind specialisation (i.e. an NCCL dtype). */
template <typename T>
concept has_data_kind = requires {
  {data_kind<T>::value}->std::convertible_to<ReduceDataKind>;
};

// reduce::Sum<T> / Min<T> / Max<T> / Prod<T> map to the matching NCCL operator on
// the rank-local final scalar. User-defined operators that want the cross-rank fast
// path supply their own @ref op_kind specialisation; anything without one is rejected
// at launch time when the System is distributed.
template <typename T>
struct op_kind<Sum<T>> {
  static constexpr ReduceOpKind value = ReduceOpKind::Sum;
};
template <typename T>
struct op_kind<Min<T>> {
  static constexpr ReduceOpKind value = ReduceOpKind::Min;
};
template <typename T>
struct op_kind<Max<T>> {
  static constexpr ReduceOpKind value = ReduceOpKind::Max;
};
template <typename T>
struct op_kind<Prod<T>> {
  static constexpr ReduceOpKind value = ReduceOpKind::Prod;
};

// Built-in arithmetic specialisations matched to NCCL's standard set. Anything wider
// than 64-bit (e.g. __int128) or non-arithmetic falls through and is rejected.
template <>
struct data_kind<std::int8_t> {
  static constexpr ReduceDataKind value = ReduceDataKind::Int8;
};
template <>
struct data_kind<std::uint8_t> {
  static constexpr ReduceDataKind value = ReduceDataKind::Uint8;
};
template <>
struct data_kind<std::int32_t> {
  static constexpr ReduceDataKind value = ReduceDataKind::Int32;
};
template <>
struct data_kind<std::uint32_t> {
  static constexpr ReduceDataKind value = ReduceDataKind::Uint32;
};
template <>
struct data_kind<int64_t> {
  static constexpr ReduceDataKind value = ReduceDataKind::Int64;
};
template <>
struct data_kind<std::uint64_t> {
  static constexpr ReduceDataKind value = ReduceDataKind::Uint64;
};
template <>
struct data_kind<float> {
  static constexpr ReduceDataKind value = ReduceDataKind::Float32;
};
template <>
struct data_kind<double> {
  static constexpr ReduceDataKind value = ReduceDataKind::Float64;
};

/**
 * @brief In-place cross-rank @c ncclAllReduce on a contiguous @c count -element buffer
 *        sitting on the rank-local root GPU.
 *
 * Routes to @c ncclAllReduce ( @c rcclAllReduce on AMD via @c CARBON_USE_RCCL ) on the
 * @c System 's NCCL communicator, using the rank's per-GPU stream (taken from
 * @c stream[0] under carbon's one-rank-per-GPU convention). The collective is enqueued
 * on the device-side stream and the function returns immediately; pair with
 * @c ReductionResult::sync(stream) before reading @c host_value(view) .
 *
 * @param sys         @c System with a non-null @c Distributed / @c Communicator.
 *                    Throws @c std::runtime_error otherwise.
 * @param device_ptr  Pointer to @c count contiguous @p dtype elements on the rank-local
 *                    GPU. The buffer is both source and destination of the allreduce.
 * @param count       Element count (scalars; not bytes).
 * @param op          NCCL operator selected via @ref op_kind .
 * @param dtype       NCCL dtype selected via @ref data_kind .
 * @param stream      Per-System stream group; only @c stream[0] is used on the
 *                    distributed path (one rank → one local GPU).
 *
 * @throws std::runtime_error on missing @c Communicator, on a stream-group with the
 *         wrong size, or on any NCCL/RCCL failure. Defined in @c src/dist_reduce.cpp .
 *         Without @c CARBON_HAVE_MPI_NCCL the function exists but unconditionally
 *         throws if called with a @c System that somehow has a non-null
 *         @c Distributed (which is only constructible under that define).
 */
CARBON_API auto
dist_allreduce_in_place(const System& sys,
                        void* device_ptr,
                        size_t count,
                        ReduceOpKind op,
                        ReduceDataKind dtype,
                        const Stream& stream) -> void;

/**
 * @brief In-place cross-rank @c MPI_Allreduce on a host-side buffer.
 *
 * Twin of @ref dist_allreduce_in_place for the OpenMP / host reduce path: after the
 * per-slab combine has written the rank-local final scalar to
 * @c ReductionResult::host_data(view) , this hop folds in the contributions from every
 * other rank via @c MPI_Allreduce(MPI_IN_PLACE, ...) on the @c System 's MPI
 * communicator. NCCL is not involved (host buffer + CPU collective).
 *
 * Synchronous: returns when every rank has the combined value in @p host_ptr .
 *
 * @param sys         @c System with a non-null @c Distributed (MPI is required —
 *                    NCCL is not). Throws @c std::runtime_error otherwise.
 * @param host_ptr    Pointer to @c count contiguous @p dtype elements on the host.
 *                    Both source and destination of the allreduce.
 * @param count       Element count (scalars; not bytes).
 * @param op          MPI operator selected via @ref op_kind .
 * @param dtype       MPI dtype selected via @ref data_kind .
 *
 * @throws std::runtime_error on missing @c Distributed handle, on any @c MPI_Allreduce
 *         error, or on an unmappable @p op / @p dtype . Defined in
 *         @c src/dist_reduce.cpp ; the non-MPI build keeps the same defensive stub
 *         shape used by @ref dist_allreduce_in_place .
 */
CARBON_API auto
dist_host_allreduce_in_place(const System& sys,
                             void* host_ptr,
                             size_t count,
                             ReduceOpKind op,
                             ReduceDataKind dtype) -> void;

}  // namespace carbon::reduce
