#pragma once

#include <carbon/detail/allocation.h>
#include <carbon/detail/buffer_group.h>
#include <carbon/detail/stream_group.h>
#include <carbon/export.h>
#include <carbon/logging.h>
#include <carbon/system.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace carbon {

/**
 * Asynchronous multi-GPU buffer-to-buffer transfer descriptor.
 *
 * A @c Transfer ties together two @ref carbon::Buffer collections (one per side, source and
 * destination), the @ref System the underlying allocations belong to, and two parallel pairs
 * of element-indexed sub-ranges. Each entry @c i describes the per-GPU sub-range of the @c i-th
 * buffer that participates in the transfer:
 *
 * - @c system() — the @c System whose @c async_copy primitive executes each per-pair copy.
 *   Captured at construction (shallow copy) so callers don't have to thread it through
 *   @ref operator().
 * - @c src() / @c dst() — the source and destination @ref carbon::Buffer (multi-GPU collections).
 *   Stored by shallow copy so the underlying allocations stay alive for as long as the
 *   @c Transfer exists, even when the views below outlive the constructor's local arguments.
 * - @c src_first_element()[i] — element offset into @c src()[i] of the first element to send
 *   from GPU @c i. Must satisfy @c src_first_element()[i] + src_counts()[i] <= src()[i].size().
 * - @c src_counts()[i] — element count from @c src_first_element()[i].
 * - @c dst_first_element() / @c dst_counts() — the symmetric receiver-side pair.
 *
 * Construction-time invariants (verified, throw on failure):
 * - All six vectors have the same length, equal to @c src().size() and @c dst().size().
 * - For each @c i: @c first_element + count never overshoots its per-GPU buffer's size, and
 *   @c src_counts[i] * sizeof(Src) == dst_counts[i] * sizeof(Dst).
 *
 * Executing the transfer (@ref operator()) loops over per-GPU pairs, builds typed sub-range
 * @ref detail::Allocation views via @ref detail::Allocation::unmanaged for each side, and
 * dispatches to @ref System::async_copy on the captured @c system(). Pairs whose
 * @c src_counts()[i] is zero are skipped. @ref System::async_copy itself picks the per-pair
 * stream out of the @ref Stream group using the per-GPU buffer's @c relative_gpu_id
 * (destination side for H2D / D2D / peer, source side for D2H; ignored for H2H), so this
 * class does not need to mention any backend-specific stream type.
 *
 * @tparam Src Element type of the source @c Buffer.
 * @tparam Dst Element type of the destination @c Buffer (may differ from @c Src as long as byte
 *             counts match per pair).
 */
template <typename Src, typename Dst = Src>
class Transfer {
  // Transfer copies bytes via cudaMemcpy*/std::memcpy, which is well-defined only for trivially
  // copyable element types (no constructors / destructors to skip, no padding hazards beyond what
  // the bit-pattern transfer already preserves). Non-trivial types must be serialised first.
  static_assert(std::is_trivially_copyable_v<Src>,
                "carbon::Transfer requires Src to be TriviallyCopyable");
  static_assert(std::is_trivially_copyable_v<Dst>,
                "carbon::Transfer requires Dst to be TriviallyCopyable");

public:
  /** Multi-GPU source @c Buffer (one per-GPU allocation per entry). */
  using SrcBuffer = ::carbon::Buffer<Src>;
  /** Multi-GPU destination @c Buffer (same cardinality as @c SrcBuffer). */
  using DstBuffer = ::carbon::Buffer<Dst>;

  /**
   * Empty transfer: no system, no source, no destination, no per-GPU views. @ref operator()
   * is a no-op (zero pairs). Useful as a placeholder until a real transfer is assigned in.
   */
  Transfer() = default;

  /**
   * Transfer the entirety of every per-GPU buffer in @p src into the matching entry of @p dst.
   * Equivalent to passing element offsets of 0 and counts of @c src[i].size(). Requires
   * @c src.size() == dst.size() and matching per-pair byte counts; throws
   * @c std::invalid_argument otherwise. @p system is captured (shallow copy) for use by
   * @ref operator().
   */
  Transfer(System system,
           SrcBuffer src,
           DstBuffer dst);

  /**
   * Construct from explicit per-GPU sub-ranges. Each pair @c (src_first_element[i],
   * src_counts[i]) must sit fully within @c src[i] and similarly on the @p dst side; per-pair
   * byte sizes must agree. Throws @c std::invalid_argument on cardinality / byte-size
   * disagreements and @c std::out_of_range on bounds violations. @p src_counts[i] /
   * @p dst_counts[i] may be zero on a per-GPU basis to skip that pair. @p system is captured
   * (shallow copy) for use by @ref operator().
   */
  Transfer(System system,
           SrcBuffer src,
           std::vector<size_t> src_first_element,
           std::vector<size_t> src_counts,
           DstBuffer dst,
           std::vector<size_t> dst_first_element,
           std::vector<size_t> dst_counts);

  /** The @c System this transfer was built against; copies share the same underlying state. */
  [[nodiscard]] auto
  system() const noexcept -> const System& {
    return system_;
  }

  [[nodiscard]] auto
  src() const noexcept -> const SrcBuffer& {
    return src_;
  }

  [[nodiscard]] auto
  dst() const noexcept -> const DstBuffer& {
    return dst_;
  }

  [[nodiscard]] auto
  src_first_element() const noexcept -> const std::vector<size_t>& {
    return src_first_element_;
  }

  [[nodiscard]] auto
  dst_first_element() const noexcept -> const std::vector<size_t>& {
    return dst_first_element_;
  }

  [[nodiscard]] auto
  src_counts() const noexcept -> const std::vector<size_t>& {
    return src_counts_;
  }

  [[nodiscard]] auto
  dst_counts() const noexcept -> const std::vector<size_t>& {
    return dst_counts_;
  }

  /**
   * Issue the per-GPU transfers. For each entry @c i with @c src_counts[i] > 0, builds typed
   * @ref detail::Allocation::unmanaged sub-range views of both sides (location / gpu_id /
   * etc. inherited from the per-GPU base allocation) and forwards them to the captured
   * @c system().async_copy together with @p stream. Pairs with zero element count are
   * silently skipped.
   */
  auto
  operator()(const Stream& stream) const -> void;

private:
  /**
   * Validate the six-way invariant (same cardinality across all vectors; sub-ranges fit; per-pair
   * byte sizes agree). Throws @c std::invalid_argument or @c std::out_of_range on failure with a
   * @c CARBON_LOG_ERROR pointing at the offending entry.
   */
  static auto
  validate_(const SrcBuffer& src,
            const std::vector<size_t>& src_first_element,
            const std::vector<size_t>& src_counts,
            const DstBuffer& dst,
            const std::vector<size_t>& dst_first_element,
            const std::vector<size_t>& dst_counts) -> void;

  System system_{};
  SrcBuffer src_{};
  DstBuffer dst_{};
  std::vector<size_t> src_first_element_{};
  std::vector<size_t> dst_first_element_{};
  std::vector<size_t> src_counts_{};
  std::vector<size_t> dst_counts_{};
};

template <typename Src,
          typename Dst>
inline Transfer<Src,
                Dst>::Transfer(System system,
                               SrcBuffer src,
                               DstBuffer dst) {
  if (src.size() != dst.size()) {
    log::throw_with_log<std::invalid_argument>(
        "Transfer: src/dst Buffer sizes disagree (src={}, dst={})", src.size(), dst.size());
  }
  const size_t n = src.size();
  std::vector<size_t> sf(n, 0U);
  std::vector<size_t> df(n, 0U);
  std::vector<size_t> sc(n);
  std::vector<size_t> dc(n);
  for (size_t i = 0; i < n; ++i) {
    sc[i] = src[i].size();
    dc[i] = dst[i].size();
  }
  validate_(src, sf, sc, dst, df, dc);
  system_ = std::move(system);
  src_ = std::move(src);
  dst_ = std::move(dst);
  src_first_element_ = std::move(sf);
  dst_first_element_ = std::move(df);
  src_counts_ = std::move(sc);
  dst_counts_ = std::move(dc);
}

template <typename Src,
          typename Dst>
inline Transfer<Src,
                Dst>::Transfer(System system,
                               SrcBuffer src,
                               std::vector<size_t> src_first_element,
                               std::vector<size_t> src_counts,
                               DstBuffer dst,
                               std::vector<size_t> dst_first_element,
                               std::vector<size_t> dst_counts) {
  validate_(src, src_first_element, src_counts, dst, dst_first_element, dst_counts);
  system_ = std::move(system);
  src_ = std::move(src);
  dst_ = std::move(dst);
  src_first_element_ = std::move(src_first_element);
  dst_first_element_ = std::move(dst_first_element);
  src_counts_ = std::move(src_counts);
  dst_counts_ = std::move(dst_counts);
}

template <typename Src,
          typename Dst>
inline auto
Transfer<Src,
         Dst>::validate_(const SrcBuffer& src,
                         const std::vector<size_t>& src_first_element,
                         const std::vector<size_t>& src_counts,
                         const DstBuffer& dst,
                         const std::vector<size_t>& dst_first_element,
                         const std::vector<size_t>& dst_counts) -> void {
  const size_t n = src.size();
  if (dst.size() != n) {
    log::throw_with_log<std::invalid_argument>(
        "Transfer: src/dst Buffer sizes disagree (src={}, dst={})", n, dst.size());
  }
  if (src_first_element.size() != n || src_counts.size() != n) {
    log::throw_with_log<std::invalid_argument>(
        "Transfer: src first_element/counts cardinality {}/{} != Buffer size {}",
        src_first_element.size(), src_counts.size(), n);
  }
  if (dst_first_element.size() != n || dst_counts.size() != n) {
    log::throw_with_log<std::invalid_argument>(
        "Transfer: dst first_element/counts cardinality {}/{} != Buffer size {}",
        dst_first_element.size(), dst_counts.size(), n);
  }
  for (size_t i = 0; i < n; ++i) {
    const size_t s_bytes = src_counts[i] * sizeof(Src);
    const size_t d_bytes = dst_counts[i] * sizeof(Dst);
    if (s_bytes != d_bytes) {
      log::throw_with_log<std::invalid_argument>(
          "Transfer: byte sizes disagree for pair {} (src={} bytes, dst={} bytes)", i, s_bytes,
          d_bytes);
    }
    if (src_counts[i] > 0U) {
      const size_t last = src_first_element[i] + src_counts[i];
      if (last < src_first_element[i] || last > src[i].size()) {
        log::throw_with_log<std::out_of_range>(
            "Transfer: src sub-range [{}, {}) out of pair-{} buffer of size {} elements",
            src_first_element[i], last, i, src[i].size());
      }
    }
    if (dst_counts[i] > 0U) {
      const size_t last = dst_first_element[i] + dst_counts[i];
      if (last < dst_first_element[i] || last > dst[i].size()) {
        log::throw_with_log<std::out_of_range>(
            "Transfer: dst sub-range [{}, {}) out of pair-{} buffer of size {} elements",
            dst_first_element[i], last, i, dst[i].size());
      }
    }
  }
}

template <typename Src,
          typename Dst>
inline auto
Transfer<Src,
         Dst>::operator()(const Stream& stream) const -> void {
  const size_t n = src_.size();
  for (size_t i = 0; i < n; ++i) {
    const size_t bytes = src_counts_[i] * sizeof(Src);
    if (bytes == 0U) {
      continue;
    }
    const auto& sb = src_[i];
    const auto& db = dst_[i];
    // Build typed sub-range views starting at the per-pair element offset; the System wrapper
    // converts both ends to byte views before dispatching to the byte primitive, which also
    // picks the per-pair stream out of @p stream using the relative_gpu_id rule.
    const auto src_view =
        detail::Allocation<Src>::unmanaged(sb.location(), sb.data() + src_first_element_[i],
                                           src_counts_[i], sb.gpu_id(), sb.relative_gpu_id());
    const auto dst_view =
        detail::Allocation<Dst>::unmanaged(db.location(), db.data() + dst_first_element_[i],
                                           dst_counts_[i], db.gpu_id(), db.relative_gpu_id());
    system_.async_copy(src_view, dst_view, bytes, stream);
  }
}

}  // namespace carbon
