#pragma once

#include <carbon/detail/allocation.h>
#include <carbon/detail/shared_vector_base.h>

#include <utility>
#include <vector>

namespace carbon {
namespace detail {

class GpuGroup;

/**
 * Contiguous sequence of detail::Allocation<T> values in a shared std::vector.
 *
 * Inherits the shallow-copy / move-keeps-valid semantics from @ref SharedVectorBase. Indexed
 * access, @c size(), @c empty(), @c get_uid() and the @c bool conversion all come from the base.
 *
 * @tparam T Element type of each Allocation in the collection.
 */
template <typename T>
class BufferGroup : public SharedVectorBase<BufferGroup<T>, Allocation<T>> {
  using Base = SharedVectorBase<BufferGroup<T>, Allocation<T>>;
  friend class GpuGroup;

public:
  /** Default-construct an empty group (non-null vector with zero elements). */
  BufferGroup() = default;

  /**
   * Build a group that adopts @p buffers as its shared backing vector. Lets non-GpuGroup callers
   * (e.g. tests, manual host-side wiring) bundle a list of per-GPU Allocations into a BufferGroup.
   * The returned group shares ownership through the same shared_ptr scheme as the GpuGroup-built
   * groups; subsequent shallow copies / moves keep the same vector alive.
   */
  [[nodiscard]] static auto
  from_vector(std::vector<Allocation<T>> buffers) -> BufferGroup {
    return BufferGroup(std::make_shared<std::vector<Allocation<T>>>(std::move(buffers)));
  }

private:
  /** Adopting constructor used by @ref GpuGroup and @ref from_vector. */
  using Base::Base;
};

/** Compile-time checks that BufferGroup models DataUid. */
static_assert(DataUid<BufferGroup<float>>);
static_assert(DataUid<BufferGroup<int>>);

}  // namespace detail
}  // namespace carbon
