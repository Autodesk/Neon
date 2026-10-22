#pragma once

/**
 * Access tokens — the tiny @c (field_uid, mode) records the @ref Skeleton uses to
 * build a dependency graph from a heterogeneous @c std::vector<Kernel>.
 *
 * Mirrors Neon's @c Container::tokens / @c MultiXpuGraph::Storage::token resolution:
 * each @c Kernel reports the set of fields it touches and how (read, write, both),
 * and the skeleton turns conflicting @c (field_uid, mode) pairs across the kernel
 * sequence into RAW / WAR / WAW edges in the DAG.
 *
 * @par Field identity
 * @c field_uid is the @c get_uid() of the field's underlying @c BufferGroup<T>
 * (carbon's @c SharedVectorBase numeric id — the address of the shared internal
 * vector, cast to @c uint64_t ). Stable across copies of the same field, distinct
 * across distinct fields, even when the buffer is empty (default-constructed
 * BufferGroup still has a unique address).
 *
 * @par Why a small POD
 * Skeleton DAG construction is O(N²) over kernel count in the worst case (every
 * pair of kernels gets tested for token conflicts). Keeping @c AccessToken to two
 * scalars (a @c uint64_t and a one-byte enum) means the per-kernel token vector
 * stays cache-friendly for the conflict scan; future variants (e.g. a stencil-aware
 * @c StencilToken with a radius / direction byte) can subclass or sit alongside
 * without invalidating this layout.
 */

#include <cstdint>
#include <memory>
#include <string>

namespace carbon {

namespace detail {
/**
 * @brief Type-erased halo-kernel factory snapshotted on stencil-read tokens.
 *
 * Defined fully in @c <carbon/kernel.h> (where @c Kernel itself is complete);
 * here we only forward-declare it so @ref AccessToken can hold a
 * @c std::shared_ptr<HaloFactoryBase> without dragging the kernel headers into
 * the include graph of every TU that mentions a token.
 *
 * Producers — @ref KernelLoader::read_stencil — instantiate the templated
 * @c TypedHaloFactory<FieldT> wrapper and stash it on the token. Consumers —
 * @c Skeleton::compile during the auto-halo insertion pass — call
 * @c HaloFactoryBase::make to materialise a fresh halo @c Kernel for the
 * field at compile time. Trait shape: any @c FieldT with a
 * @c make_halo_kernel() member that returns a @c carbon::Kernel plugs in (today
 * @c Dense::Field<T, Components> ; future grid families satisfy the same
 * requirement).
 */
class HaloFactoryBase;
class ScalarReadbackFactoryBase;
}  // namespace detail

// ScalarReadbackFactoryBase::make is defined in <carbon/kernel.h>.

/**
 * Which logical resource an @ref AccessToken names for skeleton conflict analysis.
 * Field tokens use @c ResourceKind::Field (default); reduction scalars use
 * @c ScalarDevice / @c ScalarHost with @c field_uid set to the scalar's stable id.
 */
enum class ResourceKind : unsigned char {
  Field = 0,
  ScalarDevice = 1,
  ScalarHost = 2,
};

/**
 * How a kernel touches a field. Conservative default for the legacy
 * @c loader(field) call shape (see @c KernelLoader::operator() ) is @c ReadWrite ,
 * since carbon cannot tell from the call alone whether the user's per-cell body
 * reads, writes, or both. Refining a kernel's tokens to @c Read / @c Write through
 * the explicit @c loader.read / @c loader.write methods lets the skeleton expose
 * concurrency the conservative default would hide (two kernels both reading the
 * same field do not need to be serialised).
 */
enum class AccessMode : unsigned char {
  Read = 0,
  Write = 1,
  ReadWrite = 2,
};

/**
 * @brief Compute pattern of a field access — Map (own cell only) or Stencil
 *        (own cell @e plus a configured neighbourhood).
 *
 * Mirrors @c Neon::Pattern (MAP / STENCIL / REDUCE — carbon folds Reduce into
 * the kernel @c kind ()  so this enum carries only the per-cell dependency
 * shape). Recorded on @ref AccessToken alongside @ref AccessMode and consumed
 * by @c Skeleton::compile to:
 *
 *   1. Decide whether a halo update is required before the kernel — only
 *      @c Stencil reads need the halo cells filled (@c Map reads only touch
 *      the cell at @c idx ).
 *   2. Drive the OCC (overlap compute / communication) graph rewrites —
 *      @c Standard splits the stencil node, @c Extended propagates the split
 *      to upstream @c Map nodes, @c TwoWayExtended also to downstream @c Map
 *      nodes (see @c carbon::OverlapMode in @c <carbon/skeleton.h> ).
 *
 * Default for the existing @c loader(f) / @c loader.read(f) / @c loader.write(f)
 * call shapes is @c Map — preserves source compatibility for kernels written
 * before stencil semantics were tracked, at the cost of silently missing
 * halo deps if a pre-existing kernel reads neighbour cells. Use
 * @ref KernelLoader::read_stencil to opt in.
 */
enum class AccessPattern : unsigned char {
  Map = 0,
  Stencil = 1,
};

/**
 * One field access record. Produced by @c KernelLoader 's @c read / @c write /
 * @c read_write / @c read_stencil / @c operator() calls during loading-lambda
 * token discovery, and consumed by @c Skeleton::compile to build the
 * dependency DAG and (when @c Options::overlap_mode != None) to apply the OCC
 * graph rewrite around stencil nodes.
 */
struct AccessToken {
  /** Stable numeric id of the touched field's underlying @c BufferGroup<T>. */
  std::uint64_t field_uid{};
  /** Field buffer vs reduction scalar device/host slot (see @ref ResourceKind). */
  ResourceKind resource{ResourceKind::Field};
  /** How the kernel touches the field (R / W / RW). */
  AccessMode mode{AccessMode::ReadWrite};
  /**
   * @brief Per-cell dependency shape — @c Map (own cell only) or @c Stencil
   *        (own cell + neighbourhood).
   *
   * Set to @c Stencil only by @c KernelLoader::read_stencil ; every other
   * loader entry point records @c Map . Conflict resolution is unchanged
   * (mode-only via @c carbon::conflicts ) — the pattern is consumed
   * separately by the auto-halo insertion pass and the OCC rewrites.
   */
  AccessPattern pattern{AccessPattern::Map};
  /**
   * @brief Diagnostic snapshot of the touched field's @c name() at token-record
   *        time, copied verbatim from @c Field::name . Empty when the field had
   *        no name. Independent of @c field_uid — two fields with the same
   *        name still produce distinct uids, and renaming a field after a token
   *        was recorded does @b not update the snapshot stored here. Consumed
   *        by @c Skeleton::export_dot to label DAG edges and populate node
   *        tooltips; ignored by the conflict scan.
   */
  std::string field_name{};
  /**
   * @brief Type-erased halo-kernel factory captured at @c read_stencil time.
   *
   * Empty (null) for @c Map tokens, halo-kernel tokens (which are halos
   * themselves and don't need to spawn one), and the legacy @c read /
   * @c write / @c read_write / @c operator() call shapes. Populated by
   * @ref KernelLoader::read_stencil with a @c std::shared_ptr to the
   * concrete @c TypedHaloFactory<FieldT> wrapper around the field handle —
   * cheap to copy (refcount bump), survives the loader instance, and lets
   * @c Skeleton::compile materialise a fresh halo @c Kernel for the field
   * during the auto-halo insertion pass without round-tripping through the
   * user's loader.
   *
   * Held by shared pointer so the schedule can (a) defer halo construction
   * until @c compile is actually called and (b) deduplicate halos across
   * sibling stencil reads of the same field by keying the cache off
   * @c field_uid rather than rebuilding through the factory each time.
   */
  std::shared_ptr<detail::HaloFactoryBase> halo_factory{};
  /**
   * @brief Type-erased scalar-readback factory on device-scalar @c Write tokens from
   *        deferred reductions. Populated by @c ReduceKernelImpl when
   *        @c ReduceOptions::host_copy == Deferred ; consumed by
   *        @c Skeleton::compile_auto_insert_scalar_readbacks_ .
   */
  std::shared_ptr<detail::ScalarReadbackFactoryBase> scalar_readback_factory{};
};

/**
 * Conflict predicate: true iff scheduling @p later strictly after @p earlier is
 * required for correctness. Two reads on the same field never conflict; any pair
 * involving a write or read-write does.
 *
 * Used by @c Skeleton::compile to add an edge from an earlier kernel to a later
 * kernel when their token sets share at least one conflicting access on the same
 * @c field_uid . Symmetric in the @c mode arguments — order is purely the
 * earlier-vs-later naming convention.
 */
[[nodiscard]] constexpr auto
conflicts(AccessMode earlier,
          AccessMode later) noexcept -> bool {
  if (earlier == AccessMode::Read && later == AccessMode::Read) {
    return false;
  }
  return true;
}

}  // namespace carbon
