#pragma once

#include <carbon/detail/data.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace carbon::detail {

/**
 * CRTP base for shallow-copyable, multi-GPU collections of @p T (e.g. BufferGroup<Allocation<T>>,
 * StreamGroup, EventGroup). Centralises the @c std::shared_ptr<std::vector<T>> ownership
 * scheme that all these handle types share:
 *
 * - Default construction yields a non-null shared vector containing no elements (so @c bool
 *   conversion is @c true and @c size() / @c empty() answer @c 0 / @c true).
 * - Copy is shallow: the underlying vector is shared.
 * - Move leaves the source in a valid empty state (a fresh empty vector) so it can be reused.
 *
 * Derived classes typically add a per-type semantic name (e.g. @c BufferGroup::from_vector,
 * @c StreamGroup::stream_at, ...) and a friend declaration for whichever factory needs to call
 * the protected adopting constructor.
 *
 * @tparam Derived  The concrete handle type (used only for CRTP plumbing).
 * @tparam T        Element type stored in the underlying vector.
 */
template <typename Derived, typename T>
class SharedVectorBase {
public:
  /** Underlying element type (mirrors @c std::vector<T>::value_type). */
  using value_type = T;

  /** Default-construct an empty (but non-null) shared vector. */
  SharedVectorBase()
      : data_(std::make_shared<std::vector<T>>()) {}

  /** Shallow copy: shares @p other's vector. */
  SharedVectorBase(const SharedVectorBase&) = default;
  /** Shallow assign: drops the previous shared reference, takes @p other's. */
  SharedVectorBase&
  operator=(const SharedVectorBase&) = default;

  /** Move: leaves @p other with a fresh empty vector so it stays valid. */
  SharedVectorBase(SharedVectorBase&& other) noexcept
      : data_(std::exchange(other.data_,
                            std::make_shared<std::vector<T>>())) {}

  /** Move-assign: leaves @p other with a fresh empty vector so it stays valid. */
  SharedVectorBase&
  operator=(SharedVectorBase&& other) noexcept {
    if (this != &other) {
      data_ = std::exchange(other.data_, std::make_shared<std::vector<T>>());
    }
    return *this;
  }

  /** Mutable access to the underlying vector. */
  [[nodiscard]] auto
  items() noexcept -> std::vector<T>& {
    return *data_;
  }

  /** Const access to the underlying vector. */
  [[nodiscard]] auto
  items() const noexcept -> const std::vector<T>& {
    return *data_;
  }

  /** Number of elements currently in the collection. */
  [[nodiscard]] auto
  size() const noexcept -> size_t {
    return data_->size();
  }

  /** True if the collection has no elements. */
  [[nodiscard]] auto
  empty() const noexcept -> bool {
    return data_->empty();
  }

  /** Element access by index; undefined if @p i is out of range. */
  [[nodiscard]] auto
  operator[](size_t i) noexcept -> T& {
    return (*data_)[i];
  }

  /** Const element access by index; undefined if @p i is out of range. */
  [[nodiscard]] auto
  operator[](size_t i) const noexcept -> const T& {
    return (*data_)[i];
  }

  /**
   * Numeric id: @c std::uint64_t cast of the shared_ptr's raw pointer (the vector object). Zero
   * when the shared pointer is null (which only happens during a moved-from spell that
   * @ref SharedVectorBase guards against). Satisfies the @ref DataUid concept; it is the
   * vector's identity, not the addresses of contained elements.
   */
  [[nodiscard]] auto
  get_uid() const noexcept -> std::uint64_t {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data_.get()));
  }

  /** True if the underlying shared pointer is non-null (always true after move/default ctors). */
  [[nodiscard]] explicit
  operator bool() const noexcept {
    return static_cast<bool>(data_);
  }

protected:
  /**
   * Adopt @p data as the backing store. If @p data is null, falls back to a fresh empty vector to
   * keep the invariant that @c data_ is never null after construction.
   */
  explicit SharedVectorBase(std::shared_ptr<std::vector<T>> data)
      : data_(data ? std::move(data) : std::make_shared<std::vector<T>>()) {}

  /** Shared backing vector; never null. */
  std::shared_ptr<std::vector<T>> data_{};
};

}  // namespace carbon::detail
