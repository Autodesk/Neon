#pragma once

/**
 * @file detail/validated_idx.h
 *
 * Generic "result of @c Span::set_and_validate " carrier shared across every
 * grid family. Resolves the @c set_and_validate shape half of P2.α from
 * @c docs/audit_2026_05_followup.md : Dense / DenseDisg returned a struct
 * carrying @c {Idx idx, bool valid} by value while Block took an out-param
 * @c Idx& and returned @c bool ; the value-returning shape is the canonical
 * one (no risk of using an uninitialised @c Idx ), and this template lets
 * Block's value-returning overload share storage with the dense forms.
 *
 * Plain POD on purpose so the value travels cleanly across @c __host__ /
 * @c __device__ boundaries without needing @c --expt-relaxed-constexpr (which
 * a @c std::pair<Idx, bool> would require). The default constructor leaves
 * the @c Idx in its family's "invalid" sentinel state (Dense / DenseDisg :
 * default-constructed @c int_3d at the origin ; Block : @c kInvalidBlockIdx )
 * and @c valid set to @c false , so a default-constructed @c Validated<Idx>
 * never accidentally aliases a real cell.
 */

namespace carbon::detail {

template <typename IdxT>
struct Validated {
  IdxT idx{};
  bool valid{false};
};

}  // namespace carbon::detail
