#pragma once

#include <carbon/detail/allocation.h>
#include <carbon/detail/gpu_group.h>
#include <carbon/detail/stream_group.h>
#include <carbon/export.h>
#include <carbon/logging.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace carbon {

class Distributed;
namespace detail {
class Communicator;
}  // namespace detail

/**
 * User-facing per-@ref System buffer of @c T elements. Conceptually a single handle to "the
 * data on this @c System"; the per-GPU fan-out is a private implementation detail. Backed by
 * @ref detail::BufferGroup<T>, which holds one @ref detail::Allocation<T> per GPU in the
 * system; user code rarely indexes into it directly and should treat it as one buffer.
 *
 * Carbon's public API deliberately uses singular nouns (@c Buffer, @c Stream, @c Event,
 * @c Field) for these multi-GPU aggregates so applications written against the library don't
 * have to special-case single- vs multi-GPU runs at the call site.
 */
template <typename T>
using Buffer = detail::BufferGroup<T>;

/**
 * User-facing per-@ref System CUDA stream. Conceptually one stream; backed internally by one
 * @c detail::CudaStream per GPU (see @ref StreamGroup). Same singular-handle convention as
 * @ref Buffer.
 */
using Stream = StreamGroup;

/**
 * User-facing per-@ref System CUDA event. Conceptually one event; backed internally by one
 * @c detail::Event per GPU (see @ref EventGroup). Same singular-handle convention as
 * @ref Buffer.
 */
using Event = EventGroup;

/**
 * Host-side view of a fixed set of CUDA devices, backed by a detail::GpuGroup.
 *
 * Construct from an existing GpuGroup or from a vector of devices (e.g. carbon::get_all_devices()).
 *
 * Copying a System is a shallow share: all copies refer to the same GpuGroup and the same
 * stored_streams() / stored_events() history. Moving leaves the source in a valid unspecified state.
 *
 * Each call to stream() or event() allocates via the GpuGroup, returns the group, and appends a
 * shallow copy to the shared stored_streams() / stored_events() list.
 */
class CARBON_API System {
public:
  /**
   * Default-construct an empty System (no GPUs, no streams, no events). Equivalent to the state
   * left in a moved-from System; @c gpu_group().empty() is @c true. Useful as a placeholder when
   * holding a System by value (e.g. inside @ref Transfer) without immediately attaching a GpuGroup.
   */
  System();

  explicit System(detail::GpuGroup gpus);

  explicit System(std::vector<detail::Gpu> devices);

#ifdef CARBON_HAVE_MPI_NCCL
  /**
   * Build a one-GPU @c System for a distributed (MPI+NCCL) run. Pins this rank's CUDA device
   * via @c distributed->pin_local_device(), constructs a one-entry @ref detail::GpuGroup over
   * the pinned device, and bootstraps a process-wide NCCL communicator
   * (@c ncclGetUniqueId / @c MPI_Bcast / @c ncclCommInitRank). Declared (and defined) only when
   * Carbon was built with @c CARBON_ENABLE_MPI=ON, so a single-process consumer cannot
   * accidentally reference an unresolved symbol at link time.
   *
   * The shared @c Distributed handle is retained for the lifetime of the @c System (and any of
   * its shallow copies); @c System::distributed() and @c System::communicator() expose the
   * rank / world bookkeeping and the NCCL handle. Held by @c std::shared_ptr<const Distributed>
   * so the same MPI lifetime is shared by every shallow copy of the @c System; the
   * @c Communicator is destroyed before the @c Distributed (member declaration order in
   * @c State).
   */
  explicit System(std::shared_ptr<const Distributed> distributed);
#endif  // CARBON_HAVE_MPI_NCCL

  System(const System&) = default;
  System&
  operator=(const System&) = default;
  System(System&& other) noexcept;
  System&
  operator=(System&& other) noexcept;

  [[nodiscard]] auto
  gpu_group() noexcept -> detail::GpuGroup& {
    return state_->gpu_group;
  }

  [[nodiscard]] auto
  gpu_group() const noexcept -> const detail::GpuGroup& {
    return state_->gpu_group;
  }

  /**
   * One device buffer per GPU; @p counts_per_gpu[i] is the element count on GPU @p i.
   * @p counts_per_gpu.size() must equal gpu_group().size().
   */
  template <typename T>
  [[nodiscard]] auto
  buffer(const std::vector<size_t>& counts_per_gpu) const -> Buffer<T> {
    return state_->gpu_group.buffer<T>(counts_per_gpu);
  }

  /**
   * Empty @c std::vector<T> pre-reserved for @c gpu_group().size() entries, so typical per-GPU
   * build loops don't need a separate @c reserve call:
   * @code
   * auto counts = sys.make_vector<size_t>();
   * for (size_t i = 0; i < sys.gpu_group().size(); ++i) { counts.push_back(...); }
   * @endcode
   */
  template <typename T>
  [[nodiscard]] auto
  make_vector() const -> std::vector<T> {
    std::vector<T> out;
    out.reserve(state_->gpu_group.size());
    return out;
  }

  /**
   * Builds a @c std::vector whose @p i-th entry is @c fn(i) for @p i in @c [0, gpu_group().size()),
   * with element type deduced from @c fn's return. Same shape as @ref make_vector<T>() plus the
   * fill step, so call-site code collapses to one line:
   * @code
   * auto spans = sys.make_vector([&](size_t i) { return build_span(i); });
   * @endcode
   *
   * @p fn is invoked once per GPU in increasing index order. Return type @c void is rejected.
   */
  template <typename F>
  [[nodiscard]] auto
  make_vector(F fn) const -> std::vector<std::decay_t<std::invoke_result_t<F&,
                                                                           size_t>>> {
    using U = std::decay_t<std::invoke_result_t<F&, size_t>>;
    static_assert(!std::is_same_v<U, void>,
                  "System::make_vector(fn): fn must return a non-void value per GPU");
    const size_t n = state_->gpu_group.size();
    std::vector<U> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      out.emplace_back(fn(i));
    }
    return out;
  }

  /**
   * Asynchronously copy @p size_bytes from @p src (starting at @c src.data()) to @p dst
   * (starting at @c dst.data()), choosing the appropriate per-GPU stream out of the
   * @ref Stream group. The dispatch follows the @c (src.location(), dst.location(),
   * gpu_id) pair:
   * - Cpu  → Cpu    : @c std::memcpy on the host (the per-GPU stream is irrelevant).
   * - Cpu  → Device : async H→D copy on @c stream[dst.relative_gpu_id()].
   * - Device → Cpu  : async D→H copy on @c stream[src.relative_gpu_id()].
   * - Device → Device (same @c gpu_id)     : async D→D copy on the destination's stream.
   * - Device → Device (different @c gpu_id): async peer copy on the destination's stream.
   *
   * Zero-byte transfers return immediately. Either buffer's @c size_bytes() must be at
   * least @p size_bytes; otherwise a @c std::out_of_range is thrown via
   * @c carbon::log::throw_with_log (log-then-throw). Backend errors throw
   * @c std::runtime_error.
   *
   * This is the single chokepoint for asynchronous device copies in carbon — every
   * higher-level transfer (@ref Transfer, halo exchange, reduction scalar mirror) routes
   * through here, so the public API does not have to mention any backend-specific stream
   * type. The primitive is byte-typed; the templated wrapper below builds byte views of
   * typed @ref detail::Allocation pairs before dispatching here.
   *
   * @param stream  Per-System @ref Stream. Must have at least @c relative_gpu_id+1 entries
   *                for every device-side allocation it touches. The CPU↔CPU case has no
   *                stream constraint (size 0 is fine).
   */
  auto
  async_copy(const detail::Allocation<std::byte>& src,
             const detail::Allocation<std::byte>& dst,
             size_t size_bytes,
             const Stream& stream) const -> void;

  /**
   * Templated wrapper around the byte-typed @ref async_copy primitive. Both ends must be
   * trivially copyable (byte-wise copy is well-defined). The wrapper builds a byte-typed
   * @ref detail::Allocation::unmanaged view of each side (preserving location, gpu_id, and
   * relative_gpu_id), then forwards to the byte primitive. @p size_bytes is in bytes — there
   * is no per-element @p T scaling — so callers passing typed allocations of differing
   * element types only need the byte total to match on both sides.
   *
   * @throws std::out_of_range  via the byte primitive when @p size_bytes exceeds either
   *         allocation's @c size_bytes().
   * @throws std::runtime_error via the byte primitive on any backend error.
   */
  template <typename T,
            typename U>
  auto
  async_copy(const detail::Allocation<T>& src,
             const detail::Allocation<U>& dst,
             size_t size_bytes,
             const Stream& stream) const -> void {
    static_assert(std::is_trivially_copyable_v<T>,
                  "System::async_copy: source element type must be TriviallyCopyable");
    static_assert(std::is_trivially_copyable_v<U>,
                  "System::async_copy: destination element type must be TriviallyCopyable");
    const auto src_view = detail::Allocation<std::byte>::unmanaged(
        src.location(), reinterpret_cast<std::byte*>(src.data()), src.size_bytes(), src.gpu_id(),
        src.relative_gpu_id());
    const auto dst_view = detail::Allocation<std::byte>::unmanaged(
        dst.location(), reinterpret_cast<std::byte*>(dst.data()), dst.size_bytes(), dst.gpu_id(),
        dst.relative_gpu_id());
    async_copy(src_view, dst_view, size_bytes, stream);
  }

  /**
   * Asynchronous byte-fill primitive: writes the byte @p value into the first @p size_bytes
   * bytes of @p dst. @p value follows @c std::memset / @c cudaMemsetAsync semantics — only
   * the low byte is used. Mirrors the dispatch shape of @ref async_copy : host destinations
   * fall back to @c std::memset (the @c Stream argument is irrelevant in that branch),
   * device destinations route through the per-GPU stream picked up via
   * @c dst.relative_gpu_id() . Backend errors throw @c std::runtime_error ; an out-of-range
   * @c size_bytes vs the allocation's @c size_bytes() throws @c std::out_of_range . Both via
   * @c carbon::log::throw_with_log .
   *
   * The primary use-case is "zero this device buffer before launch" patterns that previously
   * called @c cudaMemsetAsync directly in test bodies / setup code, which kept those tests
   * gated on @c carbon_add_cuda_only_unit_test (audit P2.β / fourth-pass and P2.γ /
   * third-pass). Routing the fill through carbon's backend-portable stream layer lets the
   * same source compile and run under both CUDA and SYCL.
   *
   * @param stream Per-System @ref Stream. Same size constraint as @ref async_copy : at
   *               least @c relative_gpu_id+1 entries when @p dst is a device allocation;
   *               size 0 is fine for CPU destinations.
   */
  auto
  async_memset(const detail::Allocation<std::byte>& dst,
               int value,
               size_t size_bytes,
               const Stream& stream) const -> void;

  /**
   * Templated wrapper around the byte-typed @ref async_memset primitive. The destination
   * element type must be trivially copyable; the wrapper builds a byte-typed
   * @ref detail::Allocation::unmanaged view of @p dst (preserving location, gpu_id, and
   * relative_gpu_id) and forwards. @p size_bytes is in bytes — there is no per-element
   * scaling — so callers can pass a typed buffer and a byte total directly.
   *
   * @throws std::out_of_range  via the byte primitive when @p size_bytes exceeds
   *         @c dst.size_bytes() .
   * @throws std::runtime_error via the byte primitive on any backend error.
   */
  template <typename T>
  auto
  async_memset(const detail::Allocation<T>& dst,
               int value,
               size_t size_bytes,
               const Stream& stream) const -> void {
    static_assert(std::is_trivially_copyable_v<T>,
                  "System::async_memset: destination element type must be TriviallyCopyable");
    const auto dst_view = detail::Allocation<std::byte>::unmanaged(
        dst.location(), reinterpret_cast<std::byte*>(dst.data()), dst.size_bytes(), dst.gpu_id(),
        dst.relative_gpu_id());
    async_memset(dst_view, value, size_bytes, stream);
  }

  /** Allocates a stream per GPU; retains a copy in stored_streams(). */
  [[nodiscard]] auto
  stream() -> Stream;

  /**
   * The System's **main (anchor) stream** — carbon's equivalent of "stream 0": one
   * conceptual stream that every consumer on this System can share.
   *
   * Contrast with @ref stream() , which mints a *fresh* per-GPU stream group on every
   * call. @c default_stream() returns the *same* underlying stream every time, and every
   * shallow copy of this @c System (including the copies held by @c Dense / @c Field )
   * observes that same stream, because the handle lives in the shared @c State .
   *
   * That sharing is what makes it an anchor: work issued by two independent orchestrators
   * (e.g. the two ping-pong @c Skeleton s of an LBM step) lands on one FIFO stream and is
   * therefore ordered, with no host-side drain between them. @c Skeleton uses this as its
   * stream 0 and brackets each run with a start/end fork-join on it.
   *
   * Created lazily on first call, so a default-constructed or moved-from @c System (both
   * device-free placeholders) never touches the driver. On an empty @c System the result is
   * an empty @c Stream , matching @c gpu_group() .
   *
   * This is a real created stream, **not** the CUDA legacy null stream: the null stream's
   * implicit cross-stream synchronisation would defeat the OCC overlap modes, and it cannot
   * be graph-captured.
   *
   * Not thread-safe (same contract as @ref stream() / @ref event() ): the first call
   * mutates the shared state.
   */
  [[nodiscard]] auto
  default_stream() const -> Stream;

  /** Allocates an event per GPU; retains a copy in stored_events(). */
  [[nodiscard]] auto
  event() -> Event;

  /** Shallow copies of every Stream returned from stream() on this System (shared across copies). */
  [[nodiscard]] auto
  stored_streams() const noexcept -> const std::vector<Stream>& {
    return state_->stored_streams;
  }

  /** Shallow copies of every Event returned from event() on this System (shared across copies). */
  [[nodiscard]] auto
  stored_events() const noexcept -> const std::vector<Event>& {
    return state_->stored_events;
  }

  /**
   * Distributed (MPI) handle this @c System was built with, or @c nullptr when this is a
   * single-process @c System. Available only when Carbon was built with
   * @c CARBON_ENABLE_MPI=ON; without that, always returns @c nullptr.
   */
  [[nodiscard]] auto
  distributed() const noexcept -> const Distributed* {
    return state_->distributed.get();
  }

  /**
   * NCCL communicator bound to this rank's GPU, or @c nullptr when this is a single-process
   * @c System. Available only when Carbon was built with @c CARBON_ENABLE_MPI=ON.
   */
  [[nodiscard]] auto
  communicator() const noexcept -> const detail::Communicator* {
    return state_->communicator.get();
  }

private:
  struct State {
    // Declaration order matters for destruction: the NCCL communicator must be destroyed
    // before the Distributed (MPI lifetime) it was built from. With Distributed declared
    // first and Communicator second, ~State runs Communicator's destructor first.
    std::shared_ptr<const Distributed> distributed;
    std::shared_ptr<const detail::Communicator> communicator;
    detail::GpuGroup gpu_group;
    std::vector<Stream> stored_streams;
    std::vector<Event> stored_events;
    // Lazily-populated main/anchor stream handed out by default_stream(). Null until the
    // first call so the device-free placeholders (default ctor / moved-from) never touch
    // the driver — see the comment above the move constructor. Declared last: streams
    // depend on neither Distributed nor Communicator, so destroying it first is safe.
    std::shared_ptr<Stream> default_stream;
  };

  std::shared_ptr<State> state_;
};

inline System::System()
    : state_(std::make_shared<State>(State{nullptr,
                                           nullptr,
                                           detail::GpuGroup{},
                                           {},
                                           {},
                                           {}})) {}

inline System::System(detail::GpuGroup gpus)
    : state_(std::make_shared<State>(State{nullptr,
                                           nullptr,
                                           std::move(gpus),
                                           {},
                                           {},
                                           {}})) {}

inline System::System(std::vector<detail::Gpu> devices)
    : state_(std::make_shared<State>(State{nullptr,
                                           nullptr,
                                           detail::GpuGroup(std::move(devices)),
                                           {},
                                           {},
                                           {}})) {}

inline auto
System::stream() -> Stream {
  Stream s = state_->gpu_group.stream();
  state_->stored_streams.push_back(s);
  return s;
}

// `state_` is a shared_ptr, which propagates const to the pointer and not the pointee, so a
// const System can still populate the lazy slot (same mechanism `distributed() const` uses to
// reach into State).
inline auto
System::default_stream() const -> Stream {
  if (!state_->default_stream) {
    state_->default_stream = std::make_shared<Stream>(state_->gpu_group.stream());
  }
  return *state_->default_stream;
}

inline auto
System::event() -> Event {
  Event e = state_->gpu_group.event();
  state_->stored_events.push_back(e);
  return e;
}

// The moved-from placeholder must stay device-free. `detail::GpuGroup{}` is now
// empty by construction (device enumeration is explicit via
// GpuGroup(get_all_devices())), so this is cheap. This matters because a `Field`
// holds its `Dense` (hence a `System`) by value and `KernelLoader::read_stencil`
// snapshots the field on every launch; if the placeholder re-enumerated devices
// (as an earlier default ctor did) every kernel launch paid a full
// cudaGetDeviceProperties sweep — the dominant host-side stall that starved the
// GPU and made OverlapMode worse than none. See docs/CHANGELOG.md.
inline System::System(System&& other) noexcept
    : state_(std::exchange(other.state_,
                           std::make_shared<State>(State{nullptr,
                                                         nullptr,
                                                         detail::GpuGroup{},
                                                         {},
                                                         {},
                                                         {}}))) {}

inline System&
System::operator=(System&& other) noexcept {
  if (this != &other) {
    state_ = std::exchange(other.state_, std::make_shared<State>(State{
                                             nullptr, nullptr, detail::GpuGroup{}, {}, {}, {}}));
  }
  return *this;
}

}  // namespace carbon
