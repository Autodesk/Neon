#pragma once

#include <carbon/detail/buffer_group.h>
#include <carbon/detail/stream_group.h>
#include <carbon/distributed.h>
#include <carbon/export.h>
#include <carbon/logging.h>
#include <carbon/system.h>

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace carbon {

/**
 * Direction of a single peer-to-peer NCCL transfer entry.
 *
 * @c Send issues @c ncclSend on this rank's communicator (the local buffer is the source);
 * @c Recv issues @c ncclRecv (the local buffer is the destination).
 */
enum class DistOp : int {
  Send = 0,
  Recv = 1,
};

/**
 * Asynchronous cross-rank GPU transfer descriptor for the MPI+NCCL build.
 *
 * Sibling of @ref Transfer for the multi-rank case. Where @c Transfer holds a pair of
 * @ref Buffer collections (both endpoints owned by this process), @c DistTransfer<T> holds a
 * single @ref Buffer (the rank's own GPU buffer; one entry under Carbon's one-rank-per-GPU
 * convention) plus a list of one-sided send/receive entries against peer ranks. On
 * @ref operator(), every entry is issued inside a single @c ncclGroupStart / @c ncclGroupEnd
 * block so NCCL can fuse them into one device-side schedule, mirroring Neon-nccl's
 * @c NcclTransferContainer.
 *
 * Per-entry vectors (parallel, all length @c entry_count()):
 * - @c peer_ranks[i] : NCCL/MPI rank of the remote endpoint for entry @c i.
 * - @c ops[i]        : @c DistOp::Send or @c DistOp::Recv (which side is local).
 * - @c first_elements[i] : element offset into the local @ref Buffer for entry @c i.
 * - @c counts[i]     : element count from @c first_elements[i]. Zero entries are skipped.
 *
 * Construction-time invariants (verified, throw on failure):
 * - All four vectors have the same length.
 * - For each @c i, @c first_elements[i] + counts[i] never overshoots
 *   @c buffer()[0].size().
 * - Each @c peer_ranks[i] is in [0, @c System::distributed().world_size()).
 *
 * Cross-rank pairing (caller's responsibility):
 * - Every @c Send posted by rank A to rank B must be matched by exactly one @c Recv posted by
 *   rank B from rank A with the same byte count, and the relative ordering of sends/recvs to
 *   the same peer must agree across ranks. NCCL deadlocks silently on mismatch.
 *
 * The implementation lives in a separate @c .inl that includes @c <nccl.h>; that header is
 * included only from translation units that already use NCCL types (typically @c .cu test or
 * benchmark code), so user TUs that merely include @c <carbon/carbon.h> never see NCCL
 * declarations.
 *
 * @tparam T Element type of the local buffer. Must be trivially copyable; Carbon transfers
 *           bytes via @c ncclChar.
 */
template <typename T>
class DistTransfer {
  static_assert(std::is_trivially_copyable_v<T>,
                "carbon::DistTransfer requires T to be TriviallyCopyable");

public:
  using BufferT = ::carbon::Buffer<T>;

  /** Empty descriptor; @ref operator() is a no-op. */
  DistTransfer() = default;

  /**
   * Construct from explicit parallel vectors. Throws @c std::invalid_argument when the
   * @c System has no @ref Distributed handle, the buffer cardinality is not 1, or the
   * cardinalities of the four vectors disagree; throws @c std::out_of_range on bounds /
   * peer-rank violations.
   */
  DistTransfer(System system,
               BufferT buffer,
               std::vector<int> peer_ranks,
               std::vector<DistOp> ops,
               std::vector<size_t> first_elements,
               std::vector<size_t> counts);

  [[nodiscard]] auto
  system() const noexcept -> const System& {
    return system_;
  }

  [[nodiscard]] auto
  buffer() const noexcept -> const BufferT& {
    return buffer_;
  }

  [[nodiscard]] auto
  peer_ranks() const noexcept -> const std::vector<int>& {
    return peer_ranks_;
  }

  [[nodiscard]] auto
  ops() const noexcept -> const std::vector<DistOp>& {
    return ops_;
  }

  [[nodiscard]] auto
  first_elements() const noexcept -> const std::vector<size_t>& {
    return first_elements_;
  }

  [[nodiscard]] auto
  counts() const noexcept -> const std::vector<size_t>& {
    return counts_;
  }

  [[nodiscard]] auto
  entry_count() const noexcept -> size_t {
    return counts_.size();
  }

  /**
   * Issue every non-zero entry on the given stream as a single NCCL group. Uses the
   * per-rank handle @c stream[0] internally. Throws @c std::runtime_error on any NCCL
   * error. Defined in @c carbon/dist_transfer.inl, which pulls in @c <nccl.h>.
   */
  auto
  operator()(const Stream& stream) const -> void;

private:
  static auto
  validate_(const System& system,
            const BufferT& buffer,
            const std::vector<int>& peer_ranks,
            const std::vector<DistOp>& ops,
            const std::vector<size_t>& first_elements,
            const std::vector<size_t>& counts) -> void;

  System system_{};
  BufferT buffer_{};
  std::vector<int> peer_ranks_{};
  std::vector<DistOp> ops_{};
  std::vector<size_t> first_elements_{};
  std::vector<size_t> counts_{};
};

template <typename T>
inline DistTransfer<T>::DistTransfer(System system,
                                     BufferT buffer,
                                     std::vector<int> peer_ranks,
                                     std::vector<DistOp> ops,
                                     std::vector<size_t> first_elements,
                                     std::vector<size_t> counts) {
  validate_(system, buffer, peer_ranks, ops, first_elements, counts);
  system_ = std::move(system);
  buffer_ = std::move(buffer);
  peer_ranks_ = std::move(peer_ranks);
  ops_ = std::move(ops);
  first_elements_ = std::move(first_elements);
  counts_ = std::move(counts);
}

template <typename T>
inline auto
DistTransfer<T>::validate_(const System& system,
                           const BufferT& buffer,
                           const std::vector<int>& peer_ranks,
                           const std::vector<DistOp>& ops,
                           const std::vector<size_t>& first_elements,
                           const std::vector<size_t>& counts) -> void {
  if (system.distributed() == nullptr) {
    log::throw_with_log<std::invalid_argument>(
        "DistTransfer: System has no Distributed handle (build with CARBON_ENABLE_MPI=ON and "
        "construct System(std::shared_ptr<const Distributed>))");
  }
  if (buffer.size() != 1U) {
    log::throw_with_log<std::invalid_argument>(
        "DistTransfer: Buffer must hold exactly one per-rank allocation (got {})", buffer.size());
  }
  const size_t n = counts.size();
  if (peer_ranks.size() != n || ops.size() != n || first_elements.size() != n) {
    log::throw_with_log<std::invalid_argument>(
        "DistTransfer: parallel vectors disagree (peer_ranks={}, ops={}, first_elements={}, "
        "counts={})",
        peer_ranks.size(), ops.size(), first_elements.size(), counts.size());
  }
  // Local invariants only; cross-rank pairing is the caller's responsibility (see class doxygen).
  const int world_size = system.distributed()->world_size();
  const size_t buf_size = buffer[0].size();
  for (size_t i = 0; i < n; ++i) {
    if (peer_ranks[i] < 0 || peer_ranks[i] >= world_size) {
      log::throw_with_log<std::out_of_range>("DistTransfer: entry {} peer_rank {} not in [0, {})",
                                             i, peer_ranks[i], world_size);
    }
    if (counts[i] > 0U) {
      const size_t last = first_elements[i] + counts[i];
      if (last < first_elements[i] || last > buf_size) {
        log::throw_with_log<std::out_of_range>(
            "DistTransfer: entry {} sub-range [{}, {}) out of buffer of size {} elements", i,
            first_elements[i], last, buf_size);
      }
    }
  }
}

}  // namespace carbon

#include <carbon/dist_transfer.inl>
