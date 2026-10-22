#pragma once

/**
 * @file halo_build.h
 * @brief Shared phase helpers for building halo-exchange descriptors.
 *
 * @c carbon::Halo (Dense) and @c carbon::HaloDisg (DenseDisg) ship long
 * procedural ctors that share the same five-phase shape:
 *
 * 1. Validate @c StencilSemantic and @c BoundaryMode (and per-family
 *    lattice radius constraints).
 * 2. Read off geometry (@c n_gpus , @c halo_z , @c global_n ,
 *    @c local_offset , @c pitch ) and short-circuit on empty / no-halo
 *    layouts.
 * 3. Build the global adjacency list: @c (g, g+1) for every adjacent
 *    slab plus a wrap pair under @c PeriodicZ .
 * 4. Walk the adjacency list, classify each pair as @b both-local
 *    (intra-process), @b one-local (cross-rank), or @b both-foreign
 *    (skipped), and accumulate the per-direction transfer entries.
 * 5. Construct the @c Transfer<T> (from intra entries) and, under
 *    @c CARBON_HAVE_MPI_NCCL , the @c DistTransfer<T> (from cross-rank
 *    entries).
 *
 * Phases 1 (the universal semantic / boundary-mode part), 3, and 5 are
 * structurally identical between the families. Phase 4's per-pair body
 * differs because the lattice activity model differs (per-component
 * @c c_q.z sign for Dense, run-length-fused @c CardinalityRun ranges for
 * Disg) and because the disg layout's region-start offset arithmetic
 * differs from Dense's per-component planes. This header factors the
 * universal pieces out so each family ctor body can collapse to a thin
 * orchestrator (the same shape @c Skeleton::compile and
 * @c Skeleton::apply_occ_ took after the third-pass P1.α split).
 *
 * The per-family per-pair body stays inside its ctor — making it
 * generic over the iteration model would either re-introduce template
 * machinery the audit's "decision-density" complaint was about or
 * force the offset arithmetic through type-erased runs. The audit's
 * resolution shape (audit P1.α, fourth pass) explicitly calls out
 * "extract universal phases" as the right factoring.
 *
 * Resolution narrative: see @c docs/audit_2026_05_fourth_pass.md
 * P1.α and the closure entry in @c docs/SESSIONS.md .
 */

#include <carbon/boundary_mode.h>
#include <carbon/detail/allocation.h>
#include <carbon/detail/buffer_group.h>
#include <carbon/logging.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/dist_transfer.h>
#endif

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <algorithm>
#include <numeric>
#include <tuple>

namespace carbon::detail::halo_build {

/**
 * Phase 1 (semantic): reject unrecognised @c StencilSemantic values.
 *
 * Both halo families accept @c Standard and @c Lattice ; any other
 * enum value indicates a memory-corruption-style misuse and gets a
 * pointed diagnostic. @p family_tag is prepended to the log message
 * so the user sees @c "Halo:" or @c "HaloDisg:" depending on the
 * caller — same shape as the original inline checks.
 *
 * @throws std::invalid_argument when @p semantic is neither
 *         @c Standard nor @c Lattice .
 */
inline auto
validate_stencil_semantic(StencilSemantic semantic,
                          const char* family_tag) -> void {
  if (semantic != StencilSemantic::Standard && semantic != StencilSemantic::Lattice) {
    log::throw_with_log<std::invalid_argument>("{}: unrecognised StencilSemantic (got {})",
                                               family_tag, static_cast<int>(semantic));
  }
}

/**
 * Phase 1 (boundary): reject unrecognised @c BoundaryMode values and
 * report whether the layout is @c PeriodicZ (the wrap-pair flag the
 * adjacency builder reads).
 *
 * @return @c true iff @p boundary is @c BoundaryMode::PeriodicZ .
 *
 * @throws std::invalid_argument when @p boundary is neither @c Open
 *         nor @c PeriodicZ .
 */
[[nodiscard]] inline auto
validate_boundary_mode_is_periodic_z(BoundaryMode boundary,
                                     const char* family_tag) -> bool {
  if (boundary != BoundaryMode::Open && boundary != BoundaryMode::PeriodicZ) {
    log::throw_with_log<std::invalid_argument>("{}: unrecognised BoundaryMode (got {})", family_tag,
                                               static_cast<int>(boundary));
  }
  return boundary == BoundaryMode::PeriodicZ;
}

/**
 * Phase 3: build the global adjacency list of slab pairs.
 *
 * Every adjacent slab pair @c (g, g + 1) for @c g in
 * @c [0, global_n - 1) contributes one entry. Under @c PeriodicZ
 * (@p periodic_z @c == @c true ) an additional wrap pair
 * @c (global_n - 1, 0) is appended; on a single global slab that
 * collapses to @c (0, 0) — a self-pair that the per-pair body
 * dispatches as a one-buffer copy with non-overlapping offsets.
 *
 * @c PeriodicZ on @c global_n @c == @c 0 emits no wrap pair (the
 * layout is degenerate; the empty-layout short-circuit fires before
 * adjacency construction in any real call site, but this helper is
 * defensive).
 */
[[nodiscard]] inline auto
adjacency_pairs(size_t global_n,
                bool periodic_z) -> std::vector<std::pair<size_t,
                                                          size_t>> {
  std::vector<std::pair<size_t, size_t>> adj;
  adj.reserve(global_n + (periodic_z ? 1U : 0U));
  for (size_t g = 0; g + 1U < global_n; ++g) {
    adj.emplace_back(g, g + 1U);
  }
  if (periodic_z && global_n > 0U) {
    adj.emplace_back(global_n - 1U, 0U);
  }
  return adj;
}

/**
 * Collapse adjacent transfer entries whose source and destination sub-ranges are
 * byte-contiguous within the same allocation pair. Independent @c cudaMemcpyPeerAsync
 * copies are order-insensitive, so entries may be sorted within each allocation group
 * before merging. Non-contiguous entries (e.g. Dense SoA component planes) are left
 * untouched.
 */
template <typename T>
auto
coalesce_contiguous_transfers(std::vector<Allocation<T>>& src_allocs,
                              std::vector<Allocation<T>>& dst_allocs,
                              std::vector<size_t>& src_first,
                              std::vector<size_t>& src_counts,
                              std::vector<size_t>& dst_first,
                              std::vector<size_t>& dst_counts) -> void {
  const size_t n = src_allocs.size();
  if (n <= 1U) {
    return;
  }
  if (dst_allocs.size() != n || src_first.size() != n || src_counts.size() != n ||
      dst_first.size() != n || dst_counts.size() != n) {
    log::throw_with_log<std::invalid_argument>(
        "coalesce_contiguous_transfers: vector cardinalities disagree (n={})", n);
  }

  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), 0U);
  const auto sort_key = [&](size_t i) {
    return std::tuple(src_allocs[i].data(), src_allocs[i].gpu_id(), dst_allocs[i].data(),
                      dst_allocs[i].gpu_id(), src_first[i]);
  };
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return sort_key(a) < sort_key(b);
  });

  std::vector<Allocation<T>> out_src;
  std::vector<Allocation<T>> out_dst;
  std::vector<size_t> out_sf;
  std::vector<size_t> out_sc;
  std::vector<size_t> out_df;
  std::vector<size_t> out_dc;
  out_src.reserve(n);
  out_dst.reserve(n);
  out_sf.reserve(n);
  out_sc.reserve(n);
  out_df.reserve(n);
  out_dc.reserve(n);

  for (size_t oi = 0; oi < n; ++oi) {
    const size_t i = order[oi];
    if (!out_src.empty() && out_src.back().data() == src_allocs[i].data() &&
        out_src.back().gpu_id() == src_allocs[i].gpu_id() &&
        out_dst.back().data() == dst_allocs[i].data() &&
        out_dst.back().gpu_id() == dst_allocs[i].gpu_id() &&
        out_sf.back() + out_sc.back() == src_first[i] &&
        out_df.back() + out_dc.back() == dst_first[i]) {
      out_sc.back() += src_counts[i];
      out_dc.back() += dst_counts[i];
      continue;
    }
    out_src.push_back(src_allocs[i]);
    out_dst.push_back(dst_allocs[i]);
    out_sf.push_back(src_first[i]);
    out_sc.push_back(src_counts[i]);
    out_df.push_back(dst_first[i]);
    out_dc.push_back(dst_counts[i]);
  }

  src_allocs = std::move(out_src);
  dst_allocs = std::move(out_dst);
  src_first = std::move(out_sf);
  src_counts = std::move(out_sc);
  dst_first = std::move(out_df);
  dst_counts = std::move(out_dc);
}

/**
 * Collapse adjacent host-mirror @c std::memcpy entries whose source and
 * destination sub-ranges are contiguous within the same @c (src_gpu, dst_gpu)
 * pair. Same merge rule as @ref coalesce_contiguous_transfers .
 */
template <typename Entry>
auto
coalesce_contiguous_host_intra(std::vector<Entry>& entries) -> void {
  const size_t n = entries.size();
  if (n <= 1U) {
    return;
  }

  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), 0U);
  const auto sort_key = [&](size_t i) {
    return std::tuple(entries[i].src_gpu, entries[i].dst_gpu, entries[i].src_offset);
  };
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return sort_key(a) < sort_key(b);
  });

  std::vector<Entry> out;
  out.reserve(n);
  for (size_t oi = 0; oi < n; ++oi) {
    const Entry& e = entries[order[oi]];
    if (!out.empty() && out.back().src_gpu == e.src_gpu && out.back().dst_gpu == e.dst_gpu &&
        out.back().src_offset + out.back().count == e.src_offset &&
        out.back().dst_offset + out.back().count == e.dst_offset) {
      out.back().count += e.count;
      continue;
    }
    out.push_back(e);
  }
  entries = std::move(out);
}

/**
 * Phase 5 (intra accumulator): collect per-pair intra-process transfer
 * entries and finalise into a @c Transfer<T> .
 *
 * Both halo ctors used to keep six parallel @c std::vector s
 * (allocs / first / counts on each side); this struct is the named
 * pair of accumulator + builder so the ctor body sees one
 * @c push(...) call per direction instead of six lockstep
 * @c push_back lines.
 */
template <typename T>
struct IntraTransferBuilder {
  std::vector<Allocation<T>> src_allocs;
  std::vector<Allocation<T>> dst_allocs;
  std::vector<size_t> src_first;
  std::vector<size_t> src_counts;
  std::vector<size_t> dst_first;
  std::vector<size_t> dst_counts;

  /** Append one transfer entry. Storage layout matches between gpu and cpu mirrors. */
  auto
  push(Allocation<T> src,
       Allocation<T> dst,
       size_t src_off,
       size_t dst_off,
       size_t count) -> void {
    src_allocs.push_back(std::move(src));
    dst_allocs.push_back(std::move(dst));
    src_first.push_back(src_off);
    src_counts.push_back(count);
    dst_first.push_back(dst_off);
    dst_counts.push_back(count);
  }

  [[nodiscard]] auto
  empty() const noexcept -> bool {
    return src_allocs.empty();
  }

  /**
   * Construct a @c Transfer<T> from the accumulated entries. Returns a
   * default-constructed (empty) @c Transfer<T> when no entries were
   * pushed (so the caller can unconditionally assign to its data
   * member without a guard).
   */
  [[nodiscard]] auto
  finalise(const System& sys) -> Transfer<T> {
    if (empty()) {
      return Transfer<T>{};
    }
    coalesce_contiguous_transfers(src_allocs, dst_allocs, src_first, src_counts, dst_first,
                                  dst_counts);
    auto src_group = BufferGroup<T>::from_vector(std::move(src_allocs));
    auto dst_group = BufferGroup<T>::from_vector(std::move(dst_allocs));
    return Transfer<T>(sys, std::move(src_group), std::move(src_first), std::move(src_counts),
                       std::move(dst_group), std::move(dst_first), std::move(dst_counts));
  }
};

#ifdef CARBON_HAVE_MPI_NCCL
/**
 * Phase 5 (dist accumulator): collect per-pair cross-rank transfer
 * entries and finalise into a @c DistTransfer<T> .
 *
 * Both halo ctors used to keep four parallel @c std::vector s
 * (peer / op / first / count); this struct collapses the per-direction
 * push to one call. The Send-before-Recv ordering convention the NCCL
 * per-peer FIFO needs is preserved by the order of @c push() calls in
 * each ctor's per-pair body — this builder doesn't enforce it (and
 * shouldn't: the disg layout interleaves runs_up Sends/Recvs with
 * runs_down Recvs/Sends, and the ordering invariant is a property of
 * how the caller drives the builder).
 */
template <typename T>
struct DistTransferBuilder {
  std::vector<int> peer_ranks;
  std::vector<DistOp> ops;
  std::vector<size_t> first_elements;
  std::vector<size_t> counts;

  /** Append one cross-rank transfer entry. */
  auto
  push(int peer,
       DistOp op,
       size_t first,
       size_t count) -> void {
    peer_ranks.push_back(peer);
    ops.push_back(op);
    first_elements.push_back(first);
    counts.push_back(count);
  }

  [[nodiscard]] auto
  empty() const noexcept -> bool {
    return peer_ranks.empty();
  }

  /**
   * Construct a @c DistTransfer<T> from the accumulated entries.
   * Returns default-constructed (empty) when no entries were pushed.
   * @p gpu_buffers is the field's per-GPU buffer group (shared with
   * the intra path); @p sys must carry a @c Distributed handle (the
   * caller validates this earlier in the cross-rank branch).
   */
  [[nodiscard]] auto
  finalise(const System& sys,
           const BufferGroup<T>& gpu_buffers) -> DistTransfer<T> {
    if (empty()) {
      return DistTransfer<T>{};
    }
    return DistTransfer<T>(sys, gpu_buffers, std::move(peer_ranks), std::move(ops),
                           std::move(first_elements), std::move(counts));
  }
};
#endif

}  // namespace carbon::detail::halo_build
