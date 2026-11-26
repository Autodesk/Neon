#pragma once

#include <cstddef>

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
 * One cross-rank message as issued on the wire: direction, remote rank and payload. Halo
 * descriptors report their exchange as a list of these ( @c dist_messages() ), in issue order;
 * @c Skeleton::halo_messages collects them per halo kernel. Plain data, so ranks can exchange
 * lists byte-wise.
 */
struct DistMessage {
  DistOp op{DistOp::Send};
  int peer_rank{};
  /** Buffer length in elements of the field's value type. */
  size_t elements{};
  size_t bytes{};
  /** Index of the @c ncclGroupStart / @c ncclGroupEnd block within one exchange. */
  size_t group{};
};

}  // namespace carbon
