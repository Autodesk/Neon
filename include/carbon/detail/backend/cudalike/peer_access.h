#pragma once

/**
 * @file peer_access.h
 *
 * Peer-access (P2P) enablement policy, shared by the CUDA and HIP backends.
 *
 * @par Why this is not a one-liner
 * The obvious implementation — ask @c gpu_device_can_access_peer and enable every pair it
 * approves — is unsafe. A platform can report peer capability it does not have, accept the
 * enable call, and then service a cross-device copy by transferring @b nothing while still
 * returning success. The destination keeps its previous contents, so a halo exchange
 * delivers stale cells and a multi-GPU reduction quietly drops a device's partials with no
 * error raised anywhere. That was observed on an Ada-generation pair over a PCIe host
 * bridge on an AMD/IOMMU host, where both the runtime API and @c nvidia-smi report P2P
 * read/write as available (see docs/CHANGELOG.md 2026-07).
 *
 * The response is to treat every capability answer as a claim and gate on two independent
 * checks before keeping an enablement:
 *
 * 1. @ref peer_link_is_trusted — is this an NVLink-class link? Only consulted under
 *    @c P2pPolicy::Auto.
 * 2. @ref peer_dma_delivers — does a copy issued exactly the way @c System::async_copy
 *    issues one actually move the bytes?
 *
 * Anything that fails is rolled back to the host-staged fallback, which is always correct.
 * Peer access is a performance path here and nothing about correctness depends on it.
 *
 * @par Direction bookkeeping
 * @c gpu_device_enable_peer_access enables the @b current context to reach a named peer, so
 * the enablement @c (ctx -> peer) is what backs copies running in @c ctx 's context. A
 * device-to-device copy executes on the @b destination's stream, so it is the enablement
 * @c (dst -> src) that a copy @c src -> dst depends on. Each copy direction is therefore
 * probed separately and a failure revokes the specific enablement behind it.
 */

#include <carbon/detail/backend/cuda_like.h>
#include <carbon/detail/backend/select.h>
#include <carbon/logging.h>

#if CARBON_BACKEND_IS_CUDALIKE

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace carbon {
namespace detail {

/**
 * Peer-access selection policy, from the @c CARBON_P2P environment variable.
 *
 * - @c Off  ("0" / "off" / "no" / "false"): never enable peer access. Every cross-device
 *   copy takes the host-staged fallback, which is always correct.
 * - @c Auto (default, or "auto"): enable a direction only if its link reports NVLink-class
 *   capability @b and its peer DMA passes @ref peer_dma_delivers.
 * - @c On   ("1" / "on" / "yes" / "true" / "force"): enable every direction the runtime
 *   claims is capable, subject only to @ref peer_dma_delivers. For a PCIe-P2P platform
 *   known to be good.
 */
enum class P2pPolicy : uint8_t {
  Off,
  Auto,
  On
};

[[nodiscard]] inline auto
p2p_policy_from_env() -> P2pPolicy {
  const char* raw = std::getenv("CARBON_P2P");
  if (raw == nullptr) {
    return P2pPolicy::Auto;
  }
  std::string value(raw);
  for (char& c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (value == "0" || value == "off" || value == "no" || value == "false") {
    return P2pPolicy::Off;
  }
  if (value == "1" || value == "on" || value == "yes" || value == "true" || value == "force") {
    return P2pPolicy::On;
  }
  if (value != "auto") {
    CARBON_LOG_WARN("GpuGroup: unrecognised CARBON_P2P value \"{}\"; falling back to auto", raw);
  }
  return P2pPolicy::Auto;
}

/**
 * Does a cross-device copy issued the way @c System::async_copy issues it — direction
 * inferred from the pointers, on the @b destination device's stream — actually deliver the
 * bytes?
 *
 * The destination is pre-seeded with a sentinel distinct from the pattern, so a copy that
 * moves nothing is distinguishable from one that moves the right bytes. Uses raw
 * allocations rather than carbon's tracked ones: this runs during @c GpuGroup construction,
 * before any buffer plumbing exists, and both scratch blocks are released before returning.
 */
[[nodiscard]] inline auto
probe_peer_dma(int src_id,
               int dst_id) -> bool {
  constexpr int kCount = 16;
  constexpr size_t kBytes = static_cast<size_t>(kCount) * sizeof(int);

  std::array<int, kCount> pattern{};
  for (int i = 0; i < kCount; ++i) {
    pattern[static_cast<size_t>(i)] = 0x5A5A0000 + i;
  }
  std::array<int, kCount> readback{};
  readback.fill(0);

  void* src_dev = nullptr;
  void* dst_dev = nullptr;
  gpu_stream_t stream{};
  bool have_stream = false;
  bool delivered = false;

  if (gpu_set_device(src_id) == gpu_success && gpu_malloc(&src_dev, kBytes) == gpu_success &&
      gpu_memcpy_default(src_dev, pattern.data(), kBytes) == gpu_success &&
      gpu_device_synchronize() == gpu_success && gpu_set_device(dst_id) == gpu_success &&
      gpu_malloc(&dst_dev, kBytes) == gpu_success &&
      gpu_memset(dst_dev, 0xFF, kBytes) == gpu_success &&
      gpu_stream_create(&stream) == gpu_success) {
    have_stream = true;
    if (gpu_device_synchronize() == gpu_success &&
        gpu_memcpy_async_default(dst_dev, src_dev, kBytes, stream) == gpu_success &&
        gpu_stream_synchronize(stream) == gpu_success &&
        gpu_memcpy_default(readback.data(), dst_dev, kBytes) == gpu_success) {
      delivered = (readback == pattern);
    }
  }

  if (have_stream) {
    (void)gpu_stream_destroy(stream);
  }
  if (dst_dev != nullptr) {
    (void)gpu_set_device(dst_id);
    (void)gpu_free(dst_dev);
  }
  if (src_dev != nullptr) {
    (void)gpu_set_device(src_id);
    (void)gpu_free(src_dev);
  }
  (void)gpu_get_last_error();
  return delivered;
}

/**
 * @ref probe_peer_dma memoised per direction for the lifetime of the process.
 *
 * A @c GpuGroup is built once per @c carbon::System and a program may construct several, so
 * without this the probe's allocate / copy / sync cost would be repaid on every
 * construction, once per ordered pair and therefore quadratic in the device count. The
 * property being probed is a platform characteristic, so caching it is safe.
 */
[[nodiscard]] inline auto
peer_dma_delivers(int src_id,
                  int dst_id) -> bool {
  static std::mutex mutex;
  static std::map<std::pair<int, int>, bool> cache;

  const std::pair<int, int> key{src_id, dst_id};
  const std::lock_guard<std::mutex> lock(mutex);
  const auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }
  const bool delivered = probe_peer_dma(src_id, dst_id);
  cache.emplace(key, delivered);
  return delivered;
}

/**
 * Is @p src_id -> @p dst_id an NVLink-class link, i.e. one whose peer DMA carbon is willing
 * to trust under @c P2pPolicy::Auto ?
 *
 * Native-atomic support is the discriminator reachable through the runtime API without
 * linking NVML: NVLink-attached peers support native atomics, PCIe-attached peers do not.
 * It is a proxy rather than a direct "is this NVLink" query, and it is deliberately
 * conservative — a healthy PCIe-P2P platform is demoted to the host-staged path (correct,
 * slower) and can opt back in with @c CARBON_P2P=on.
 *
 * @par Why the gate and not just the probe
 * Because the probe is not a reliable detector on its own. Bypassing this gate with
 * @c CARBON_P2P=on shows the affected platform to be measurably non-deterministic: one run
 * caught a single direction and revoked it while the other direction passed the probe and
 * then delivered wrong data to the very next test, and three consecutive runs afterwards
 * passed both directions cleanly. A construction-time probe samples an idle machine once, so
 * it cannot speak for a link that fails intermittently or only under load. Declining
 * PCIe-class links outright is what actually keeps the default correct; the probe is a
 * backstop for the grossest cases, not the primary defence.
 *
 * @par ROCm
 * The heuristic is derived from NVIDIA behaviour and AMD's reporting for this attribute is
 * unverified here. If ROCm reports 0 for an xGMI-attached pair, the effect under @c Auto is
 * that the HIP backend keeps the host-staged path it already used before this policy
 * existed — never worse than the previous behaviour, but leaving bandwidth unclaimed on a
 * fabric that deserves it. @c CARBON_P2P=on is the escape hatch until someone measures it
 * on an MI-series machine.
 */
[[nodiscard]] inline auto
peer_link_is_trusted(int src_id,
                     int dst_id) -> bool {
  int native_atomic = 0;
  if (gpu_device_native_atomic_supported(&native_atomic, src_id, dst_id) != gpu_success) {
    (void)gpu_get_last_error();
    return false;
  }
  return native_atomic != 0;
}

/**
 * Apply the peer-access policy across @p device_ids.
 *
 * Enables the directions the policy admits, validates each copy direction those
 * enablements are meant to accelerate, and revokes any that does not deliver. Leaves the
 * current device set to @p device_ids front on return. Safe to call with 0 or 1 devices
 * (no pairs, so nothing happens).
 *
 * Without peer access a cross-device copy is staged through a host bounce buffer, which
 * serialises the issuing host thread — that stall both slows the multi-GPU halo exchange
 * and defeats the Skeleton's OverlapMode, since compute cannot run concurrently with a
 * host-blocking halo. That cost is why carbon wants peer access at all; correctness never
 * depends on getting it.
 */
inline auto
configure_peer_access(const std::vector<int>& device_ids) -> void {
  const int n = static_cast<int>(device_ids.size());
  if (n < 2) {
    return;
  }

  const P2pPolicy policy = p2p_policy_from_env();
  std::vector<std::pair<int, int>> enabled;  // (context device, peer device)

  if (policy != P2pPolicy::Off) {
    for (int i = 0; i < n; ++i) {
      const int ctx_id = device_ids[static_cast<size_t>(i)];
      if (gpu_set_device(ctx_id) != gpu_success) {
        continue;
      }
      for (int j = 0; j < n; ++j) {
        if (i == j) {
          continue;
        }
        const int peer_id = device_ids[static_cast<size_t>(j)];
        int can_access = 0;
        if (gpu_device_can_access_peer(&can_access, ctx_id, peer_id) != gpu_success ||
            can_access == 0) {
          continue;
        }
        if (policy == P2pPolicy::Auto && !peer_link_is_trusted(ctx_id, peer_id)) {
          CARBON_LOG_INFO(
              "GpuGroup: peer access {} -> {} left disabled: the link does not report "
              "NVLink-class capability, and PCIe-class peer DMA is not trusted by default "
              "(cross-device copies use the host-staged fallback, which is correct but "
              "slower). Set CARBON_P2P=on to force it.",
              ctx_id, peer_id);
          continue;
        }
        const gpu_error_t perr = gpu_device_enable_peer_access(peer_id);
        if (perr == gpu_success || perr == gpu_error_peer_access_already_enabled) {
          (void)gpu_get_last_error();  // clear the sticky error so it does not leak forward
          enabled.emplace_back(ctx_id, peer_id);
        } else {
          (void)gpu_get_last_error();
          CARBON_LOG_WARN("GpuGroup: enabling peer access {} -> {} failed: {}", ctx_id, peer_id,
                          gpu_error_string(perr));
        }
      }
    }
  }

  // Entry (ctx, peer) backs copies peer -> ctx, since a device-to-device copy runs on the
  // destination's stream and therefore in the destination's context.
  size_t revoked = 0;
  for (const auto& [ctx_id, peer_id] : enabled) {
    if (peer_dma_delivers(peer_id, ctx_id)) {
      continue;
    }
    CARBON_LOG_WARN(
        "GpuGroup: peer DMA {} -> {} reported success but delivered no data; revoking peer "
        "access for this direction so cross-device copies fall back to the host-staged path. "
        "The runtime claims this pair is P2P-capable, so this is a platform defect rather than "
        "a configuration error — carbon stays correct here, just slower.",
        peer_id, ctx_id);
    if (gpu_set_device(ctx_id) == gpu_success) {
      (void)gpu_device_disable_peer_access(peer_id);
      (void)gpu_get_last_error();
    }
    ++revoked;
  }

  if (policy == P2pPolicy::Off) {
    CARBON_LOG_INFO(
        "GpuGroup: peer access disabled by CARBON_P2P; cross-device copies are "
        "host-staged");
  } else if (!enabled.empty()) {
    CARBON_LOG_INFO("GpuGroup: peer access active on {} of {} validated direction(s)",
                    enabled.size() - revoked, enabled.size());
  }

  (void)gpu_set_device(device_ids.front());
}

}  // namespace detail
}  // namespace carbon

#endif  // CARBON_BACKEND_IS_CUDALIKE
