#pragma once

#include <carbon/detail/data.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>

namespace carbon {

class Cpu;

namespace detail {

class Gpu;

/**
 * Kind of memory backing an Allocation: host (CPU) or CUDA device (GPU).
 */
enum class AllocationLocation {
  Cpu,   ///< Host-allocated memory (e.g. new[], Cpu::buffer).
  Cuda,  ///< CUDA device memory (e.g. cudaMalloc, detail::Gpu::buffer).
};

/**
 * Owning view of CPU or GPU memory holding @p count elements of type @p T.
 *
 * The underlying allocation is an opaque void pointer (bytes); data() exposes it as T*. Lifetime
 * uses std::shared_ptr<void>: copies share one allocation; when the last Allocation (or shared_ptr
 * owner) is destroyed, the deleter frees the block — reference-counted automatic reclamation
 * (GC-style for that heap object).
 *
 * Two flavours exist:
 * - @b Owning  — created by @ref Cpu::buffer / @ref Gpu::buffer. The refcount's deleter frees
 *                the allocation when it hits zero. @ref owns_storage returns @c true.
 * - @b Unmanaged — created by @ref unmanaged (this class). Wraps an externally-owned raw pointer
 *                with a no-op deleter, so the refcount hitting zero does not free anything. Use
 *                this to describe sub-ranges of other allocations (e.g. halo views for
 *                @ref Transfer) or to adapt foreign allocations without taking ownership.
 *                @ref owns_storage returns @c false.
 *
 * Construction of owning allocations is through @ref Cpu::buffer or @ref Gpu::buffer (friend
 * factories). Unmanaged allocations are constructed directly via the @ref unmanaged static
 * factory. Both paths validate that @c count * sizeof(T) does not overflow @c size_t.
 *
 * @tparam T Element type; size_bytes() is count * sizeof(T). Must be suitable for the allocator
 *         (POD-like usage is typical for GPU allocations).
 */
template <typename T>
class Allocation {
  friend class carbon::Cpu;
  friend class Gpu;

  // Other Allocation<U> instantiations need access to the aliasing constructor so that
  // as_byte_view can build an Allocation<std::byte> that shares this allocation's shared_ptr
  // control block.
  template <typename U>
  friend class Allocation;

public:
  /** Default-constructed allocation: no storage, empty size, invalid ids. */
  Allocation() = default;

  /** Shallow copy: shares the same refcounted allocation as @p other. */
  Allocation(const Allocation&) = default;
  /** Shallow assign: shares @p other's allocation; previous shared ownership is released. */
  Allocation&
  operator=(const Allocation&) = default;
  /** Move: transfers shared ownership; @p other is left valid (empty). */
  Allocation(Allocation&&) noexcept = default;
  /** Move-assign: transfers ownership from @p other. */
  Allocation&
  operator=(Allocation&&) noexcept = default;

  /**
   * Build an unmanaged allocation that wraps @p storage without taking ownership: no
   * @c cudaFree / @c delete[] is called when the last handle is released. The caller is
   * responsible for keeping the underlying allocation alive for as long as any Allocation built
   * from this factory (or an @ref as_byte_view of it) is in use.
   *
   * @param location         @c Cpu or @c Cuda memory class of @p storage.
   * @param storage          Element pointer to wrap (may be null if @p count is 0).
   * @param count            Element count exposed through @ref size (bytes = @p count * sizeof(T)).
   * @param gpu_id           CUDA device ordinal (if @p location == Cuda), or -1 for host memory.
   * @param relative_gpu_id  Optional GPU-group index, or -1.
   *
   * @throws std::overflow_error if @c count * sizeof(T) would overflow @c size_t.
   */
  [[nodiscard]] static auto
  unmanaged(AllocationLocation location,
            T* storage,
            size_t count,
            int gpu_id,
            int relative_gpu_id) -> Allocation {
    check_count_(count);
    std::function<void(void*)> noop = [](void*) {
    };
    std::shared_ptr<void> handle(static_cast<void*>(storage), noop);
    return Allocation(location, count, gpu_id, relative_gpu_id, std::move(handle), std::move(noop),
                      false);
  }

  /** Returns whether this allocation refers to CPU or GPU memory. */
  [[nodiscard]] auto
  location() const noexcept -> AllocationLocation {
    return location_;
  }

  /** Number of elements of type T in this allocation (not bytes). */
  [[nodiscard]] auto
  size() const noexcept -> size_t {
    return count_;
  }

  /** Span of the allocation in bytes: size() * sizeof(T). */
  [[nodiscard]] auto
  size_bytes() const noexcept -> size_t {
    return count_ * sizeof(T);
  }

  /**
   * CUDA device ordinal when location() == Cuda (as reported by the runtime).
   * Returns -1 for CPU allocations.
   */
  [[nodiscard]] auto
  gpu_id() const noexcept -> int {
    return gpu_id_;
  }

  /**
   * Optional logical index among GPUs in a multi-GPU setup (e.g. index in
   * detail::GpuGroup::devices()). Returns -1 for CPU allocations or when not set by the allocator.
   */
  [[nodiscard]] auto
  relative_gpu_id() const noexcept -> int {
    return relative_gpu_id_;
  }

  /**
   * True if this handle owns its storage (the refcount's deleter frees the block). False for
   * allocations built via @ref unmanaged, which just wrap an externally-owned pointer.
   */
  [[nodiscard]] auto
  owns_storage() const noexcept -> bool {
    return owns_storage_;
  }

  /**
   * Pointer to the first element; may be null if empty.
   *
   * Allocation is a shallow handle (refcounted shared_ptr semantics), so the pointer is
   * intentionally exposed as @c T* even on a const handle — same rule as
   * @c std::shared_ptr<T>::get(): const-on-handle does not imply const-on-pointee. Callers that
   * want a read-only view should propagate @c const through their own types.
   */
  [[nodiscard]] auto
  data() const noexcept -> T* {
    return reinterpret_cast<T*>(storage_.get());
  }

  /**
   * Byte-typed view of this allocation. The returned @c Allocation<std::byte> shares this
   * allocation's shared_ptr control block (so the original storage stays alive as long as the
   * view does, and is freed exactly once when all handles drop — same rule as for
   * @ref Allocation copies). Metadata (@c location, @c gpu_id, @c relative_gpu_id,
   * @c owns_storage) is preserved; element count on the view equals this allocation's
   * @c size_bytes().
   */
  [[nodiscard]] auto
  as_byte_view() const -> Allocation<std::byte> {
    return Allocation<std::byte>(location_, size_bytes(), gpu_id_, relative_gpu_id_, storage_,
                                 free_fn_, owns_storage_);
  }

  /**
   * Stable numeric id for this allocation: the raw void* address as std::uint64_t (0 when empty).
   * Satisfies the DataUid concept.
   */
  [[nodiscard]] auto
  get_uid() const noexcept -> std::uint64_t {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(storage_.get()));
  }

  /** True if this allocation holds a non-null shared block. */
  [[nodiscard]] explicit
  operator bool() const noexcept {
    return static_cast<bool>(storage_);
  }

private:
  /**
   * Validates that @p count elements of type @p T do not overflow @c size_t bytes; throws
   * @c std::overflow_error otherwise. Empty (@p count == 0) is always accepted. Inlined and
   * dead-code-eliminated when @c sizeof(T) == 1.
   */
  static auto
  check_count_(size_t count) -> void {
    if constexpr (sizeof(T) > 1U) {
      if (count > (static_cast<size_t>(-1) / sizeof(T))) {
        throw std::overflow_error("Allocation: count * sizeof(T) overflows size_t");
      }
    } else {
      (void)count;
    }
  }

  /**
   * Builds refcounted storage from a raw pointer and deleter (used by Cpu / Gpu owning paths).
   *
   * @param location        Cpu or Cuda.
   * @param count           Element count (size()).
   * @param gpu_id          CUDA device id, or -1 for CPU.
   * @param relative_gpu_id Optional index in a detail::GpuGroup, or -1.
   * @param storage         Raw void* from cudaMalloc, new[], etc.
   * @param free_fn         Deleter for @p storage; used for shared_ptr and free_function().
   */
  Allocation(AllocationLocation location,
             size_t count,
             int gpu_id,
             int relative_gpu_id,
             void* storage,
             std::function<void(void*)> free_fn);

  /**
   * Internal aliasing constructor that adopts an already-constructed shared_ptr. Used by
   * @ref unmanaged and by @ref as_byte_view cross-T rebind (via template friend).
   */
  Allocation(AllocationLocation location,
             size_t count,
             int gpu_id,
             int relative_gpu_id,
             std::shared_ptr<void> storage,
             std::function<void(void*)> free_fn,
             bool owns_storage) noexcept
      : storage_(std::move(storage)),
        free_fn_(std::move(free_fn)),
        count_(count),
        location_(location),
        gpu_id_(gpu_id),
        relative_gpu_id_(relative_gpu_id),
        owns_storage_(owns_storage) {}

  /** Refcounted opaque allocation; deleter runs when refcount hits zero. */
  std::shared_ptr<void> storage_{};
  /** Copy of the deleter retained for the aliasing constructor; actual release is via storage_. */
  std::function<void(void*)> free_fn_{};
  /** Element count. */
  size_t count_{};
  /** Cpu vs Cuda device memory. */
  AllocationLocation location_{AllocationLocation::Cpu};
  /** CUDA device ordinal, or -1. */
  int gpu_id_{-1};
  /** Logical GPU index, or -1. */
  int relative_gpu_id_{-1};
  /** True if the refcount's deleter frees the storage; false for unmanaged wrappers. */
  bool owns_storage_{true};
};

template <typename T>
Allocation<T>::Allocation(AllocationLocation location,
                          size_t count,
                          int gpu_id,
                          int relative_gpu_id,
                          void* storage,
                          std::function<void(void*)> free_fn)
    : storage_(storage,
               free_fn),
      free_fn_(free_fn),
      count_(count),
      location_(location),
      gpu_id_(gpu_id),
      relative_gpu_id_(relative_gpu_id),
      owns_storage_(true) {
  check_count_(count);
}

}  // namespace detail

/** Compile-time checks that Allocation models DataUid. */
static_assert(DataUid<detail::Allocation<float>>);
static_assert(DataUid<detail::Allocation<int>>);
static_assert(DataUid<detail::Allocation<std::byte>>);

}  // namespace carbon
