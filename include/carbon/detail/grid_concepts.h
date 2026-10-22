#pragma once

#include <carbon/data_view.h>
#include <carbon/integer3d.h>

#include <concepts>
#include <cstdint>

namespace carbon::detail {

/**
 * @file detail/grid_concepts.h
 *
 * Anchors the cross-grid duck-typed contract that @c KernelImpl s rely on. The
 * three grid families (@c Dense , @c DenseDisg , @c Block ) all expose their own
 * @c Span / @c Idx / @c Partition types and the kernels reach into them via
 * template instantiation. These concepts capture what @b every grid family must
 * provide so a fourth family's gap surfaces as a clean concept error instead of
 * a deep template instantiation stack.
 *
 * The concepts are intentionally minimal: only the surface @c KernelImpl s
 * touch is required. Optional members (e.g. @c BoundaryMode , @c connectivity ,
 * @c origins , brick-specific bitmask APIs) stay grid-specific and are not
 * captured here.
 *
 * Canonical-naming pass (P2.α from
 * @c docs/audit_2026_05_followup.md ) landed both shape unifications as
 * additive overloads (no breaking change to any existing call site):
 *
 * - @c Span::set_and_validate now exposes a value-returning overload on every
 *   grid family ( @c Dense / @c DenseDisg : @c (view, ix, iy, iz) ->
 *   ValidatedIdx ; @c Block : @c (block_idx, ix, iy, iz) -> ValidatedIdx ).
 *   The @c ValidatedIdx return carrier is the shared
 *   @c carbon::detail::Validated<Idx> template (see
 *   @c detail/validated_idx.h ). The historical @c bool / out-param
 *   @c Idx& signature on @c block::Span stays for back-compat.
 * - Every grid family now exposes @c Span::per_card_elements() (no-arg,
 *   per-component buffer pitch) @b and @c Span::linear_storage_elements(int)
 *   (total elements for that many cardinality planes). @c Dense additionally
 *   keeps the historical no-arg @c linear_storage_elements() as a source-
 *   compat alias for @c per_card_elements() . New cross-grid code should
 *   prefer @c per_card_elements() / @c linear_storage_elements(components) .
 *
 * Concept-anchoring story (P1.β closure, third-pass audit):
 *
 * - The cross-grid surface that @c KernelImpl s touch splits cleanly along
 *   the iteration model. @c Dense and @c DenseDisg speak the @b dense-style
 *   per-cell iteration contract — @c iter_extent(view) , @c logic_extent() ,
 *   @c xy_slab_elements() , @c global_location(idx) — captured by
 *   @ref DenseLikeSpanConcept . @c Block speaks the @b brick-iteration
 *   contract — @c iter_block_count(view) , @c view_block_idx(view, idx) ,
 *   @c block_count() , @c n_internal() , @c set_and_validate(block_idx,
 *   ix, iy, iz) , @c linear_offset(idx, components, component_id) ,
 *   @c per_card_elements() , @c linear_storage_elements(int) ,
 *   @c global_location(idx) — captured by @ref BlockLikeSpanConcept .
 * - @ref PartitionConcept (read / write / neighbour-with-value+valid) is
 *   shared by every family — @c Dense , @c DenseDisg , @c Block all expose
 *   the same shape, and the static_assert anchor in
 *   @c DeviceKernelImpl asserts it on both branches.
 * - Both concepts are anchored in @c DeviceKernelImpl 's ctor:
 *   @c if constexpr (grid_has_dense_iteration_v<Grid>) selects
 *   @ref DenseLikeSpanConcept , the @c else branch selects
 *   @ref BlockLikeSpanConcept ; @ref PartitionConcept fires on both
 *   branches with @c Grid::template Partition<int, 1> as the
 *   representative @c (T, Components) . This means a fourth grid family
 *   that ships a mis-shaped @c Span @b or @c Partition lands on a clean
 *   concept diagnostic at kernel-impl ctor time instead of a deep template
 *   instantiation stack inside the launch path.
 *
 * @c SpanConcept is retained as a deprecated alias for
 * @ref DenseLikeSpanConcept so any out-of-tree caller that referenced the
 * historical name keeps compiling; new code should use
 * @ref DenseLikeSpanConcept / @ref BlockLikeSpanConcept directly.
 */

/**
 * Minimal contract for a grid's @c Span type under the @b dense-style
 * per-cell iteration model (@c Dense / @c DenseDisg). Every @c Span must:
 *
 * - expose a default ctor and copy semantics (kernels capture spans by value);
 * - report an iteration extent for a @c DataView (so kernels can size their
 *   launch grid);
 * - report a global voxel location for an @c Idx (so kernels can recover global
 *   coordinates without a side-channel);
 * - expose a logic extent (the owned-cell sub-extent that lives without halos
 *   or ghosts);
 * - expose an xy-slab pitch.
 *
 * The @c IdxT template parameter lets users plug in the matching @c Idx family
 * (e.g. @c dense::Idx , @c dense_disg::Idx ); the @c Span knows how to talk to
 * its own @c Idx and the concept threads the binding through.
 *
 * @c Block deliberately does @b not satisfy this concept — its @c Span speaks
 * the brick-iteration contract captured by @ref BlockLikeSpanConcept instead.
 */
template <typename S, typename IdxT>
concept DenseLikeSpanConcept = requires(const S& s, const IdxT& idx, ::carbon::DataView view) {
  {s.iter_extent(view)}->std::convertible_to<::carbon::int64_3d>;
  {s.logic_extent()}->std::convertible_to<::carbon::int64_3d>;
  {s.xy_slab_elements()}->std::convertible_to<int64_t>;
  {s.global_location(idx)}->std::convertible_to<::carbon::int64_3d>;
};

/**
 * Deprecated alias for @ref DenseLikeSpanConcept retained for source
 * compatibility with the original concept name. New code should reference
 * @ref DenseLikeSpanConcept (and @ref BlockLikeSpanConcept for the brick-
 * iteration shape) directly.
 */
template <typename S, typename IdxT>
concept SpanConcept = DenseLikeSpanConcept<S, IdxT>;

/**
 * Minimal contract for a grid's @c Span type under the @b brick-iteration
 * model (@c Block ). Every @c Span must:
 *
 * - expose a default ctor and copy semantics;
 * - report a @c block_count() (per-device retained-brick count) and an
 *   @c n_internal() (count of bricks whose stencil-radius-1 read stays
 *   inside the slab — Internal / Boundary partitioning of @c iter_block_count );
 * - report an @c iter_block_count(view) (size of the brick-axis launch grid
 *   for @c DataView::Standard / @c Internal / @c Boundary );
 * - translate a launch-grid brick index back to storage via
 *   @c view_block_idx(view, iter_idx) ;
 * - expose a value-returning @c set_and_validate(block_idx, ix, iy, iz) that
 *   builds the matching @c Idx and reports validity (the canonical
 *   cross-family shape used by host iteration paths);
 * - expose @c linear_offset(idx, components, component_id) (the SoA pitch
 *   formula used by @c Partition::read / @c write );
 * - expose @c per_card_elements() (per-component buffer pitch) and
 *   @c linear_storage_elements(int) (total elements for @c components
 *   cardinality planes), the canonical sizing accessors;
 * - expose @c global_location(idx) (kernel-side global voxel coordinates,
 *   shared with the dense-style contract).
 *
 * @c Dense and @c DenseDisg deliberately do @b not satisfy this concept — they
 * speak the per-cell iteration contract captured by @ref DenseLikeSpanConcept .
 *
 * The @c IdxT template parameter binds the matching @c block::Idx ; the concept
 * threads the binding through @c set_and_validate 's return type and through
 * @c linear_offset / @c global_location .
 */
template <typename S, typename IdxT>
concept BlockLikeSpanConcept =
    requires(const S& s, const IdxT& idx, ::carbon::DataView view, std::uint32_t b, int axis) {
  {s.block_count()}->std::convertible_to<std::uint32_t>;
  {s.n_internal()}->std::convertible_to<std::uint32_t>;
  {s.iter_block_count(view)}->std::convertible_to<std::uint32_t>;
  {s.view_block_idx(view, b)}->std::convertible_to<std::uint32_t>;
  // value-returning set_and_validate overload (carrying ValidatedIdx).
  {s.set_and_validate(b, axis, axis, axis).valid}->std::convertible_to<bool>;
  {s.linear_offset(idx, /*components=*/1, /*component_id=*/0)}->std::convertible_to<int64_t>;
  {s.per_card_elements()}->std::convertible_to<int64_t>;
  {s.linear_storage_elements(/*components=*/1)}->std::convertible_to<int64_t>;
  {s.global_location(idx)}->std::convertible_to<::carbon::int64_3d>;
};

/**
 * Minimal contract for a grid's @c Partition type, expressed in terms of the
 * matching @c Span and @c Idx . Every @c Partition must:
 *
 * - read the @p component_id -th component of @p idx through @c read ;
 * - write a value into the same address through @c write (kernels treat
 *   @c Partition as @c const and only mutate through pointer-via-write);
 * - resolve a neighbour through @c neighbour and return a struct carrying at
 *   minimum a @c value and a @c valid flag (the concept can't constrain the
 *   exact return type because each family uses its own name —
 *   @c NeighbourValue<T> for every grid family (Block also exposes @c NghData<T>
 *   as a deprecated alias for backwards source compatibility) — but
 *   it can check the duck-typed shape).
 *
 * @c T is the field's element type; @c neighbour 's return type carries a
 * compatible @c value member.
 *
 * Same shape on @c Dense , @c DenseDisg , and @c Block — the static_assert
 * anchor in @c DeviceKernelImpl asserts this concept on both the dense-iter
 * and block-iter branches.
 */
template <typename P, typename S, typename IdxT, typename T>
concept PartitionConcept =
    requires(const P& p, const S& s, const IdxT& idx, ::carbon::int8_3d offset, T value) {
  {p.read(s, idx, 0)}->std::convertible_to<T>;
  p.write(s, idx, 0, value);
  // neighbour returns a struct with .value (convertible to T) and .valid (bool).
  {p.neighbour(s, idx, offset, 0).valid}->std::convertible_to<bool>;
  {p.neighbour(s, idx, offset, 0).value}->std::convertible_to<T>;
};

/**
 * Minimal @b cross-level contract a multi-resolution (mGrid) @c Partition must add
 * on top of @ref PartitionConcept (which already covers same-level @c read / @c write /
 * @c neighbour ). Mirrors the parent/child surface of Neon's @c mPartition
 * (@c Neon/.../mGrid/mPartition.h ) re-expressed in carbon's idiom — value lookups
 * return @c ::carbon::NeighbourValue<T> ( @c value + @c valid ) like every other
 * carbon neighbour query rather than Neon's @c NghData + reference accessors.
 *
 * A conforming mGrid @c Partition (call it @c MultiResPartition<T, C> ) must:
 *
 * - report topology — @c has_parent(span, idx) and @c has_children(span, idx) — so a
 *   kernel can branch before walking across levels;
 * - resolve the cross-level @c Idx — @c get_parent(span, idx) (the coarser-level cell
 *   covering @p idx ) and @c get_child(span, idx, child_offset) (the finer-level cell at
 *   the user-block @p child_offset under @p idx ), both returning a same-family @c Idx
 *   that indexes the @b adjacent level's storage;
 * - read across levels — @c parent(span, idx, component_id) returns the parent cell's
 *   value, @c child(span, idx, child_offset, component_id) the addressed child's, each as
 *   a @c NeighbourValue<T> whose @c valid tracks "that level's cell exists and is active".
 *
 * Per-level scale ( @c level , @c spacing , refinement factor) is @b not on this concept:
 * carbon parks it on @c block::Span ( @c Span::level() / @c Span::spacing() , added in the
 * pre-mGrid PR3 pass) since it is topology shared by every @c Field over a level, not a
 * property of one field's values. Neon's "uncle" surface ( @c getUncle / @c uncleVal ) is
 * an extension left out of the minimal contract; an mGrid that needs same-generation
 * cross-branch reads adds it without widening this concept.
 *
 * @note Additive and currently @b unsatisfied — no carbon grid family ships a multi-
 *       resolution @c Partition yet. It is the documentary anchor the mGrid port will
 *       @c static_assert its @c Partition against (the way @ref PartitionConcept anchors
 *       the single-level families in @c DeviceKernelImpl ), so the cross-level surface is
 *       pinned now, before any mGrid code exists, and a mis-shaped @c mPartition lands on
 *       a clean concept diagnostic.
 */
template <typename P, typename S, typename IdxT, typename T>
concept MultiResPartitionConcept = requires(const P& p,
                                            const S& s,
                                            const IdxT& idx,
                                            ::carbon::int8_3d child_offset,
                                            int component_id) {
  {p.has_parent(s, idx)}->std::convertible_to<bool>;
  {p.has_children(s, idx)}->std::convertible_to<bool>;
  {p.get_parent(s, idx)}->std::convertible_to<IdxT>;
  {p.get_child(s, idx, child_offset)}->std::convertible_to<IdxT>;
  {p.parent(s, idx, component_id).valid}->std::convertible_to<bool>;
  {p.parent(s, idx, component_id).value}->std::convertible_to<T>;
  {p.child(s, idx, child_offset, component_id).valid}->std::convertible_to<bool>;
  {p.child(s, idx, child_offset, component_id).value}->std::convertible_to<T>;
};

/**
 * Compile-time assertion that a @b dense-style grid family meets the shared
 * concepts. Use this from a @c KernelImpl 's body to anchor the contract
 * against a specific @c (Span, Idx, Partition, T) tuple:
 *
 * @code
 * using Span = typename Grid::Span;
 * using Idx = typename Grid::Idx;
 * using Partition = typename Grid::template Partition<T, Components>;
 * static_assert(detail::dense_like_span_partition_contract<Span, Idx, Partition, T>(),
 *               "Dense-style grid must satisfy DenseLikeSpanConcept / PartitionConcept");
 * @endcode
 *
 * Failing this assertion points at the first member of @c Span / @c Partition
 * that does not match the canonical dense-style shape — same diagnostic shape
 * as a missing @c requires constraint, but anchored at the kernel impl rather
 * than buried deep in the launch path.
 */
template <typename S,
          typename IdxT,
          typename P,
          typename T>
[[nodiscard]] consteval auto
dense_like_span_partition_contract() noexcept -> bool {
  return DenseLikeSpanConcept<S, IdxT> && PartitionConcept<P, S, IdxT, T>;
}

/**
 * Twin of @ref dense_like_span_partition_contract for the @b brick-iteration
 * model (Block). Asserts @ref BlockLikeSpanConcept @b and @ref PartitionConcept
 * (the partition shape is shared across grid families). Used by the
 * @c if constexpr (!grid_has_dense_iteration_v<Grid>) branch in
 * @c DeviceKernelImpl 's anchor.
 */
template <typename S,
          typename IdxT,
          typename P,
          typename T>
[[nodiscard]] consteval auto
block_like_span_partition_contract() noexcept -> bool {
  return BlockLikeSpanConcept<S, IdxT> && PartitionConcept<P, S, IdxT, T>;
}

/**
 * Deprecated alias for @ref dense_like_span_partition_contract retained for
 * source compatibility. New code should call the dense-like or block-like
 * variant directly.
 */
template <typename S,
          typename IdxT,
          typename P,
          typename T>
[[nodiscard]] consteval auto
span_partition_contract() noexcept -> bool {
  return dense_like_span_partition_contract<S, IdxT, P, T>();
}

}  // namespace carbon::detail
