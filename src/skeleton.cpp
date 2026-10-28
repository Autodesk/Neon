/**
 * @file src/skeleton.cpp
 *
 * Out-of-line definitions for @c carbon::Skeleton 's heavy member functions.
 * @c skeleton.h carries the public class declaration plus the small inline
 * accessors / constructors / @c add / @c reset / @c wait ; the bodies here
 * cover the schedule lowering pipeline:
 *
 *   - @c compile     — token-driven DAG inference, transitive reduction,
 *                      stream assignment, cross-stream event allocation,
 *                      OCC dispatch, anchor fork-join event allocation.
 *   - @c run         — replay the static schedule for one iteration.
 *   - @c compile_step_graph_occ_ — Standard / Extended / TwoWayExtended
 *                      step-graph rewrite around (halo, stencil) pairs.
 *   - @c compile_allocate_anchor_events_ — start / join events for the
 *                                          per-run bracket on the anchor.
 *   - @c export_dot  — GraphViz DOT renderer for the inferred DAG.
 *   - @c export_launch_order — text listing of one @c run iteration's ops.
 *
 * Splitting these out trims @c <carbon/skeleton.h> by ~700 lines of body
 * and removes ~thousand lines of parse work from every consumer TU.
 */

#include <carbon/access_token.h>
#include <carbon/data_view.h>
#include <carbon/detail/scalar_readback_factory.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/skeleton.h>
#include <carbon/system.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::array<std::string_view, 8> kStreamColors = {
    "#cfe2f3", "#fce5cd", "#d9ead3", "#f4cccc", "#ead1dc", "#fff2cc", "#d0e0e3", "#e6e6e6"};

auto
html_escape(std::string_view s) -> std::string {
  std::string out;
  out.reserve(s.size());
  for (const char c : s) {
    switch (c) {
      case '&':
        out += "&amp;";
        break;
      case '<':
        out += "&lt;";
        break;
      case '>':
        out += "&gt;";
        break;
      case '"':
        out += "&quot;";
        break;
      default:
        out += c;
    }
  }
  return out;
}

// Shared look of the HTML-label exports (launch matrix, kernel list).
struct Palette {
  std::string_view fill;
  std::string_view border;
};
constexpr std::string_view kHeaderBg = "#2f3b4c";
constexpr std::string_view kHeaderSubtitle = "#c9d1dc";
constexpr std::array<std::string_view, 2> kRowBg = {"#ffffff", "#f2f4f7"};
constexpr std::string_view kMuted = "#5f6b7a";
constexpr std::string_view kSyncColor = "#c0392b";
constexpr std::string_view kAnchorColor = "#2e6da4";
constexpr Palette kHaloPal{"#fde3c4", "#d97b0f"};
constexpr Palette kInternalPal{"#d6e6f7", "#2e6da4"};
constexpr Palette kBoundaryPal{"#e4dcf3", "#6a4fb0"};
constexpr Palette kStandardPal{"#dcefd6", "#4c9a3a"};
constexpr Palette kReductionPal{"#d4efed", "#1f8a85"};
constexpr Palette kCopyPal{"#eceff3", "#7b8794"};

auto
launch_palette(carbon::KernelKind kind,
               carbon::DataView view) -> Palette {
  if (kind == carbon::KernelKind::Halo) {
    return kHaloPal;
  }
  if (kind == carbon::KernelKind::ScalarReadback) {
    return kCopyPal;
  }
  if (view == carbon::DataView::Internal) {
    return kInternalPal;
  }
  if (view == carbon::DataView::Boundary) {
    return kBoundaryPal;
  }
  return kStandardPal;
}

auto
kind_palette(carbon::KernelKind kind) -> Palette {
  switch (kind) {
    case carbon::KernelKind::Halo:
      return kHaloPal;
    case carbon::KernelKind::Reduction:
    case carbon::KernelKind::ReductionHost:
    case carbon::KernelKind::UniversalReduction:
      return kReductionPal;
    case carbon::KernelKind::Anchor:
    case carbon::KernelKind::ScalarReadback:
      return kCopyPal;
    default:
      return kStandardPal;
  }
}

auto
font(int point_size,
     std::string_view color,
     const std::string& text) -> std::string {
  std::string out = "<FONT POINT-SIZE=\"" + std::to_string(point_size) + "\"";
  if (!color.empty()) {
    out += " COLOR=\"" + std::string{color} + "\"";
  }
  return out + ">" + text + "</FONT>";
}

// Fixed-size bordered box with centred @p body. FIXEDSIZE on both the table and the
// cell is required: otherwise GraphViz stretches nested tables to the parent cell and
// pins the text to the top-left corner.
auto
html_box(Palette p,
         std::string_view style,
         int width,
         int height,
         const std::string& body,
         std::string_view port = {}) -> std::string {
  std::ostringstream b;
  b << "<TABLE";
  if (!port.empty()) {
    b << " PORT=\"" << port << "\"";
  }
  b << " FIXEDSIZE=\"TRUE\" WIDTH=\"" << width << "\" HEIGHT=\"" << height
    << "\" BORDER=\"1\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\" STYLE=\"" << style
    << "\" COLOR=\"" << p.border << "\" BGCOLOR=\"" << p.fill
    << "\"><TR><TD FIXEDSIZE=\"TRUE\" WIDTH=\"" << (width - 2) << "\" HEIGHT=\"" << (height - 2)
    << "\">" << body << "</TD></TR></TABLE>";
  return b.str();
}

// Legend entry: a small swatch followed by its caption.
auto
legend_swatch(Palette p,
              std::string_view style,
              std::string_view text) -> std::string {
  return "<TD BGCOLOR=\"" + std::string{p.fill} + "\" COLOR=\"" + std::string{p.border} +
         "\" BORDER=\"1\" STYLE=\"" + std::string{style} +
         "\" WIDTH=\"16\" HEIGHT=\"10\"></TD><TD ALIGN=\"LEFT\">" + font(9, {}, std::string{text}) +
         "</TD>";
}

auto
plural(size_t n,
       std::string_view word) -> std::string {
  return std::to_string(n) + " " + std::string{word} + (n == 1U ? "" : "s");
}

[[nodiscard]] auto
overlap_mode_label(carbon::OverlapMode mode) -> std::string_view {
  switch (mode) {
    case carbon::OverlapMode::None:
      return "none";
    case carbon::OverlapMode::Standard:
      return "standard";
    case carbon::OverlapMode::Extended:
      return "extended";
    case carbon::OverlapMode::TwoWayExtended:
      return "two-way-extended";
  }
  return "?";
}

// Compact Options line for DOT titles (launch matrix / kernel list).
[[nodiscard]] auto
skeleton_options_summary(const carbon::Skeleton::Options& opts) -> std::string {
  std::string out = "overlap=" + std::string{overlap_mode_label(opts.overlap_mode)};
  if (opts.halo_stream_priority == carbon::StreamPriority::High) {
    out += " &#183; halo_priority=high";
  }
  if (opts.max_streams != 0U) {
    out += " &#183; max_streams=" + std::to_string(opts.max_streams);
  }
  if (opts.use_cuda_graph) {
    out += " &#183; cuda_graph=on";
  }
  if (opts.strict_halo) {
    out += " &#183; strict_halo";
  }
  return out;
}

}  // namespace

namespace carbon::detail {

struct StepGraphNode {
  size_t kernel_idx{0U};
  carbon::DataView view{carbon::DataView::Standard};
  carbon::reduce::ReduceLaunchMode reduce_mode{carbon::reduce::ReduceLaunchMode::Full};
  bool active{true};
  std::vector<size_t> data_succs{};
  std::vector<size_t> hint_succs{};
};

}  // namespace carbon::detail

namespace {

using carbon::detail::StepGraphNode;

auto
step_graph_add_succ(std::vector<size_t>& succs,
                    size_t to) -> void {
  if (std::find(succs.begin(), succs.end(), to) == succs.end()) {
    succs.push_back(to);
  }
}

auto
step_graph_erase_succ(std::vector<size_t>& succs,
                      size_t to) -> bool {
  auto it = std::find(succs.begin(), succs.end(), to);
  if (it == succs.end()) {
    return false;
  }
  succs.erase(it);
  return true;
}

auto
step_graph_hint_preds(const std::vector<StepGraphNode>& nodes,
                      size_t nid) -> std::vector<size_t> {
  std::vector<size_t> preds;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (!nodes[i].active) {
      continue;
    }
    for (size_t s : nodes[i].hint_succs) {
      if (s == nid) {
        preds.push_back(i);
      }
    }
  }
  return preds;
}

auto
step_graph_data_preds(const std::vector<StepGraphNode>& nodes,
                      size_t nid) -> std::vector<size_t> {
  std::vector<size_t> preds;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (!nodes[i].active) {
      continue;
    }
    for (size_t s : nodes[i].data_succs) {
      if (s == nid) {
        preds.push_back(i);
      }
    }
  }
  return preds;
}

auto
step_graph_replace_succ(std::vector<size_t>& succs,
                        size_t from,
                        size_t to_a,
                        size_t to_b,
                        bool both_halves) -> void {
  auto it = std::find(succs.begin(), succs.end(), from);
  if (it == succs.end()) {
    return;
  }
  if (both_halves) {
    *it = to_a;
    succs.push_back(to_b);
  } else {
    *it = to_b;
  }
}

}  // namespace

namespace carbon {

auto
Skeleton::compile() -> void {
  if (compiled_) {
    return;
  }

  // Phase pre-0: contract validation (Stencil-pattern WRITEs are illegal).
  compile_validate_no_stencil_writes_();

  // Phase 0: auto-halo insertion. Builds effective_kernels_ +
  // effective_to_user_ from kernels_ , inserting an auto-halo ahead of
  // every Stencil-read of a dirty field.
  compile_auto_insert_halos_();
  compile_auto_insert_scalar_readbacks_();

  copy_stream_idx_ = kNoStream_;
  has_detached_stream_ = false;

  const size_t n = effective_kernels_.size();
  edges_.assign(n, {});
  min_edges_.assign(n, {});
  schedule_.assign(n, Step{});
  record_event_per_kernel_.assign(n, kNoEvent_);
  kernel_stream_.assign(n, 0U);

  // Recompile path (add() after a prior compile()) rebuilds the event pool from
  // scratch: every phase below only ever push_back()es into events_, so without
  // this clear a second compile would strand the previous compile's CUDA events
  // (a leak) while the freshly-assigned indices point only at the new tail.
  // streams_ is intentionally *not* cleared — extra streams are reused in place.
  events_.clear();

  // Phase 1 + 1b: token-driven conflict scan -> edges_ , followed by the
  // transitive reduction -> min_edges_ .
  compile_build_conflict_dag_();

  // direct_preds_of[j] is the inverse adjacency of min_edges_ — i.e. the
  // *irreducible* predecessors of j. Computed once here and reused by the
  // stream-assignment pass and (when non-None) the OCC rewrite, so a
  // transitive edge i -> j whose ordering is already covered by some
  // intermediate chain does not contribute to stream-assignment
  // preferences and does not allocate a redundant cross-stream event.
  std::vector<std::vector<size_t>> direct_preds_of(n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j : min_edges_[i]) {
      direct_preds_of[j].push_back(i);
    }
  }

  compile_step_graph_and_lower_(direct_preds_of);

  // Phase 5: anchor fork-join events. Allocated AFTER the OCC rewrite so the
  // stream count is final (the rewrite may have moved halos to fresh halo
  // streams), and after every other phase that appends to events_ so the
  // indices stashed here stay valid.
  compile_allocate_anchor_events_();

  compile_prune_syncs_();

  compiled_ = true;

  // A recompile rebuilds the schedule (and its streams / events / captured
  // kernel arguments), so any previously captured graph is stale — drop it and
  // re-arm the capture attempt for the new schedule.
  graph_.reset();
  graph_disabled_ = false;
  captured_view_ = DataView::Standard;

  if (options_.log_dot_on_compile) {
    // export_dot() lazy-compiles, but compiled_ is already true above
    // so it falls through to the rendering path.
    CARBON_LOG_INFO("Skeleton::compile: DOT export\n{}", export_dot());
  }
}

auto
Skeleton::compile_validate_no_stencil_writes_() -> void {
  // carbon, mirroring Neon's stencil semantics, disallows Stencil-pattern
  // WRITES (a kernel's writes must be local — Stencil reads are how
  // neighbour pulls are spelled). Catching this here gives a single,
  // explicit diagnostic instead of a confusing downstream auto-halo or
  // scheduling failure if a future KernelLoader API ever stamps a
  // Stencil pattern on a write token.
  for (size_t k_idx = 0; k_idx < kernels_.size(); ++k_idx) {
    const Kernel& k = kernels_[k_idx];
    for (const AccessToken& t : k.impl()->tokens()) {
      const bool is_write = (t.mode == AccessMode::Write) || (t.mode == AccessMode::ReadWrite);
      if (is_write && t.pattern == AccessPattern::Stencil) {
        log::throw_with_log<std::logic_error>(
            "Skeleton::compile: kernel '{}' (index={}) declares a Stencil-pattern WRITE on "
            "field '{}' (uid={}). carbon (matching Neon) requires writes to be local "
            "(AccessPattern::Map); use loader.write(field) for the write side and "
            "loader.read_stencil(field) only for neighbour reads.",
            std::string{k.name()}, k_idx,
            t.field_name.empty() ? std::string{"<anonymous>"} : t.field_name, t.field_uid);
      }
    }
  }
}

auto
Skeleton::compile_auto_insert_halos_() -> void {
  effective_kernels_.clear();
  effective_to_user_.clear();

  // field_uid -> "Dirty" if a write happened since the last halo
  // (manual or auto). Stencil-read on a dirty field triggers an insert.
  //
  // Every field appearing in any token starts dirty: a freshly uploaded
  // multi-GPU field has UNINITIALIZED halo regions on device (the host
  // mirror's halo cells are never written by Field::fill_owned_host , and
  // a single Transfer<T>(cpu, gpu) blasts the whole buffer including
  // those uninitialized halo bytes onto the device — the GPU halo is
  // garbage until the first halo exchange). Without this, the first
  // stencil read inside the skeleton finds the field "clean" (no
  // in-skeleton write yet) and skips the auto-halo, so the kernel reads
  // garbage across slab boundaries — see the LBM cavity benchmark for a
  // load-bearing example. Pre-existing manual halos in the pipeline still
  // dedup the redundant initial halo: when compile() walks the manual
  // halo first, it cleans the field before the subsequent stencil read
  // sees it (see test_skeleton subtest B' for the dedup contract). For
  // single-slab BoundaryMode::Open layouts the inserted Halo is a no-op
  // (pair_count==0), so the conservative pre-dirty costs nothing there.
  // Under BoundaryMode::PeriodicZ a single slab still gets the self-wrap
  // pair (0, 0): the halo copies the slab's first/last owned z rows into
  // its upper/lower halo rows, and periodic stencil reads depend on it.
  std::vector<std::uint64_t> dirty_fields;
  auto is_dirty = [&dirty_fields](std::uint64_t uid) {
    return std::find(dirty_fields.begin(), dirty_fields.end(), uid) != dirty_fields.end();
  };
  auto mark_clean = [&dirty_fields](std::uint64_t uid) {
    auto it = std::find(dirty_fields.begin(), dirty_fields.end(), uid);
    if (it != dirty_fields.end()) {
      dirty_fields.erase(it);
    }
  };
  auto mark_dirty = [&dirty_fields](std::uint64_t uid) {
    if (std::find(dirty_fields.begin(), dirty_fields.end(), uid) == dirty_fields.end()) {
      dirty_fields.push_back(uid);
    }
  };
  for (const Kernel& k : kernels_) {
    for (const AccessToken& t : k.impl()->tokens()) {
      mark_dirty(t.field_uid);
    }
  }

  for (size_t k_idx = 0; k_idx < kernels_.size(); ++k_idx) {
    const Kernel& k = kernels_[k_idx];
    const auto& toks = k.impl()->tokens();

    // First insert any required halos for this kernel. Track which
    // field_uids we've already inserted a halo for in this kernel so a
    // single kernel reading the same field with multiple Stencil tokens
    // (rare but possible) doesn't generate duplicate halos.
    std::vector<std::uint64_t> halo_inserted_for_this_kernel;
    for (const AccessToken& t : toks) {
      if (t.pattern != AccessPattern::Stencil) {
        continue;
      }
      if (!is_dirty(t.field_uid)) {
        continue;
      }
      if (std::find(halo_inserted_for_this_kernel.begin(), halo_inserted_for_this_kernel.end(),
                    t.field_uid) != halo_inserted_for_this_kernel.end()) {
        continue;
      }
      if (options_.strict_halo) {
        log::throw_with_log<std::logic_error>(
            "Skeleton::compile: strict_halo=true and stencil-read of field '{}' (uid={}) by "
            "kernel '{}' (index={}) finds no preceding halo since the last write. Insert "
            "Field::make_halo_kernel(...) before this kernel or set strict_halo=false to "
            "auto-insert.",
            t.field_name.empty() ? std::string{"<anonymous>"} : t.field_name, t.field_uid,
            std::string{k.name()}, k_idx);
      }
      if (!t.halo_factory) {
        // KernelLoader::read_stencil should always populate the factory.
        // A null factory here means the field's Field type doesn't satisfy
        // the trait — caught at a much better location than later.
        log::throw_with_log<std::logic_error>(
            "Skeleton::compile: stencil-read token for field '{}' (uid={}) carries no halo "
            "factory. The field type must implement make_halo_kernel() returning a "
            "carbon::Kernel — Dense::Field<T, Components> does; user grid families need to "
            "as well.",
            t.field_name.empty() ? std::string{"<anonymous>"} : t.field_name, t.field_uid);
      }
      Kernel auto_halo = t.halo_factory->make();
      effective_kernels_.push_back(std::move(auto_halo));
      effective_to_user_.push_back(kNoKernel_);  // marker: synthetic
      halo_inserted_for_this_kernel.push_back(t.field_uid);
      mark_clean(t.field_uid);
    }

    // Append the user kernel.
    effective_kernels_.push_back(k);
    effective_to_user_.push_back(k_idx);

    // Update the dirty-set based on this kernel's tokens. A Halo kernel
    // (manual or auto) cleans every field it touches; every other
    // kernel's Write / ReadWrite token marks the field dirty. Pure Read
    // tokens leave the state alone.
    const bool is_halo = (k.kind() == KernelKind::Halo);
    for (const AccessToken& t : toks) {
      if (is_halo) {
        mark_clean(t.field_uid);
      } else if (t.mode == AccessMode::Write || t.mode == AccessMode::ReadWrite) {
        mark_dirty(t.field_uid);
      }
    }
  }
}

auto
Skeleton::compile_auto_insert_scalar_readbacks_() -> void {
  std::vector<Kernel> next_kernels;
  std::vector<size_t> next_map;
  next_kernels.reserve(effective_kernels_.size() + 4U);
  next_map.reserve(effective_kernels_.size() + 4U);

  for (size_t k_idx = 0; k_idx < effective_kernels_.size(); ++k_idx) {
    next_kernels.push_back(effective_kernels_[k_idx]);
    next_map.push_back(effective_to_user_[k_idx]);

    std::shared_ptr<detail::ScalarReadbackFactoryBase> factory;
    for (const AccessToken& t : effective_kernels_[k_idx].impl()->tokens()) {
      if (t.scalar_readback_factory) {
        factory = t.scalar_readback_factory;
        break;
      }
    }
    if (!factory) {
      continue;
    }
    if (k_idx + 1U < effective_kernels_.size()) {
      if (effective_kernels_[k_idx + 1U].kind() == KernelKind::ScalarReadback) {
        continue;
      }
    }
    Kernel readback = factory->make();
    next_kernels.push_back(std::move(readback));
    next_map.push_back(kNoKernel_);
  }

  effective_kernels_ = std::move(next_kernels);
  effective_to_user_ = std::move(next_map);
}

auto
Skeleton::compile_build_conflict_dag_() -> void {
  const size_t n = effective_kernels_.size();
  // Phase 1: token-driven conflict scan -> edges_ (directed, i -> j with i < j).
  for (size_t i = 0; i < n; ++i) {
    const auto& tok_i = effective_kernels_[i].impl()->tokens();
    for (size_t j = i + 1; j < n; ++j) {
      const auto& tok_j = effective_kernels_[j].impl()->tokens();
      if (pair_conflicts_(tok_i, tok_j)) {
        edges_[i].push_back(j);
      }
    }
  }

  // Phase 1b: transitive reduction. The conflict scan is intentionally
  // exhaustive (every (i, j) pair with overlapping fields is reported),
  // which is the right semantics for diagnostics but over-eager for
  // scheduling — a path i -> ... -> j already enforces the i -> j
  // ordering at run time. min_edges_ is the unique minimum DAG covering
  // the same partial order as edges_; downstream phases (and the
  // deferred tighter inter-iter barrier) consume it instead of the full
  // graph.
  min_edges_ = compute_min_edges_(edges_);
}

auto
Skeleton::compile_step_graph_and_lower_(const std::vector<std::vector<size_t>>& direct_preds_of)
    -> void {
  const size_t n = effective_kernels_.size();
  std::vector<StepGraphNode> nodes(n);
  for (size_t i = 0; i < n; ++i) {
    nodes[i].kernel_idx = i;
  }
  for (size_t i = 0; i < n; ++i) {
    nodes[i].data_succs = min_edges_[i];
  }

  if (options_.overlap_mode != OverlapMode::None) {
    compile_step_graph_occ_(direct_preds_of, nodes);
  }

  std::vector<size_t> active_ids;
  active_ids.reserve(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].active) {
      active_ids.push_back(i);
    }
  }

  std::vector<size_t> indeg(nodes.size(), 0U);
  for (size_t id : active_ids) {
    for (size_t s : nodes[id].data_succs) {
      if (nodes[s].active) {
        ++indeg[s];
      }
    }
    for (size_t s : nodes[id].hint_succs) {
      if (nodes[s].active) {
        ++indeg[s];
      }
    }
  }

  std::vector<size_t> topo;
  topo.reserve(active_ids.size());
  std::vector<size_t> ready;
  for (size_t id : active_ids) {
    if (indeg[id] == 0U) {
      ready.push_back(id);
    }
  }
  std::sort(ready.begin(), ready.end());
  while (!ready.empty()) {
    const size_t id = ready.front();
    ready.erase(ready.begin());
    topo.push_back(id);
    auto relax = [&](const std::vector<size_t>& succs) {
      for (size_t s : succs) {
        if (!nodes[s].active) {
          continue;
        }
        if (--indeg[s] == 0U) {
          ready.push_back(s);
        }
      }
    };
    relax(nodes[id].data_succs);
    relax(nodes[id].hint_succs);
    std::sort(ready.begin(), ready.end());
  }
  if (topo.size() != active_ids.size()) {
    log::throw_with_log<std::logic_error>(
        "Skeleton::compile_step_graph_and_lower_: step graph cycle detected");
  }

  std::vector<size_t> node_stream(nodes.size(), kNoStream_);
  std::vector<size_t> stream_tail(streams_.size(), kNoKernel_);
  std::vector<size_t> node_record_event(nodes.size(), kNoEvent_);
  std::vector<std::vector<size_t>> node_waits(nodes.size());

  auto alloc_stream = [&](bool prefer_high_priority) -> size_t {
    Stream st = (prefer_high_priority && options_.halo_stream_priority == StreamPriority::High)
                    ? sys_.stream(StreamPriority::High)
                    : sys_.stream();
    streams_.push_back(std::move(st));
    stream_tail.push_back(kNoKernel_);
    return streams_.size() - 1U;
  };

  auto pick_stream = [&](size_t nid, const std::vector<size_t>& data_preds) -> size_t {
    const KernelKind k_kind = effective_kernels_[nodes[nid].kernel_idx].kind();
    if (k_kind == KernelKind::ScalarReadback) {
      if (copy_stream_idx_ == kNoStream_) {
        copy_stream_idx_ = alloc_stream(false);
        has_detached_stream_ = true;
      }
      return copy_stream_idx_;
    }

    size_t chosen = kNoStream_;
    for (size_t p : data_preds) {
      if (node_stream[p] == kNoStream_) {
        continue;
      }
      if (effective_kernels_[nodes[p].kernel_idx].kind() == KernelKind::Halo &&
          nodes[nid].view == DataView::Boundary) {
        chosen = node_stream[p];
        break;
      }
    }
    if (chosen == kNoStream_) {
      for (size_t s = 0; s < stream_tail.size(); ++s) {
        const size_t tail = stream_tail[s];
        if (tail == kNoKernel_) {
          continue;
        }
        for (size_t p : data_preds) {
          if (p == tail) {
            chosen = s;
            break;
          }
        }
        if (chosen != kNoStream_) {
          break;
        }
      }
    }
    if (chosen == kNoStream_) {
      for (size_t s = 0; s < stream_tail.size(); ++s) {
        if (stream_tail[s] == kNoKernel_) {
          chosen = s;
          break;
        }
      }
    }
    if (chosen == kNoStream_) {
      if (options_.max_streams != 0U && streams_.size() >= options_.max_streams) {
        chosen = 0U;
      } else {
        chosen = alloc_stream(false);
      }
    }
    const std::vector<size_t> hint_preds = step_graph_hint_preds(nodes, nid);
    if (k_kind == KernelKind::Halo && !hint_preds.empty() &&
        options_.overlap_mode != OverlapMode::None) {
      if (options_.max_streams != 0U && streams_.size() >= options_.max_streams) {
        chosen = 0U;
      } else if (chosen == 0U && stream_tail.size() > 1U) {
        for (size_t s = 1U; s < stream_tail.size(); ++s) {
          if (stream_tail[s] == kNoKernel_) {
            chosen = s;
            break;
          }
        }
      }
      if (chosen == 0U && (options_.max_streams == 0U || streams_.size() < options_.max_streams)) {
        chosen = alloc_stream(true);
      }
    }
    return chosen;
  };

  for (size_t nid : topo) {
    const std::vector<size_t> data_preds = step_graph_data_preds(nodes, nid);
    const size_t chosen = pick_stream(nid, data_preds);
    node_stream[nid] = chosen;
    stream_tail[chosen] = nid;

    for (size_t p : data_preds) {
      if (node_stream[p] != chosen) {
        if (node_record_event[p] == kNoEvent_) {
          events_.push_back(sys_.event());
          node_record_event[p] = events_.size() - 1U;
        }
        node_waits[nid].push_back(node_record_event[p]);
      }
    }
  }

  for (size_t nid : topo) {
    bool needs_record = false;
    for (size_t s : nodes[nid].data_succs) {
      if (!nodes[s].active) {
        continue;
      }
      if (node_stream[nid] != node_stream[s]) {
        needs_record = true;
        break;
      }
    }
    if (needs_record && node_record_event[nid] == kNoEvent_) {
      events_.push_back(sys_.event());
      node_record_event[nid] = events_.size() - 1U;
    }
  }

  schedule_.clear();
  schedule_.reserve(topo.size());
  kernel_stream_.assign(n, 0U);
  record_event_per_kernel_.assign(n, kNoEvent_);
  // A split kernel reports the stream of its first-issued (Internal) half.
  std::vector<bool> kernel_stream_set(n, false);

  for (size_t nid : topo) {
    Step step{};
    step.kernel_idx = nodes[nid].kernel_idx;
    step.view = nodes[nid].view;
    step.reduce_launch_mode = nodes[nid].reduce_mode;
    step.stream_idx = node_stream[nid];
    step.wait_events = std::move(node_waits[nid]);
    step.record_event_idx = node_record_event[nid];
    schedule_.push_back(step);
    if (!kernel_stream_set[step.kernel_idx]) {
      kernel_stream_[step.kernel_idx] = step.stream_idx;
      kernel_stream_set[step.kernel_idx] = true;
    }
    if (step.record_event_idx != kNoEvent_) {
      record_event_per_kernel_[step.kernel_idx] = step.record_event_idx;
    }
  }
}

auto
Skeleton::compile_step_graph_occ_(const std::vector<std::vector<size_t>>& direct_preds_of,
                                  std::vector<StepGraphNode>& nodes) -> void {
  const std::vector<OccPair> candidates = occ_collect_candidates_(direct_preds_of);
  if (candidates.empty()) {
    return;
  }
  const size_t n = effective_kernels_.size();
  const bool extended = options_.overlap_mode == OverlapMode::Extended ||
                        options_.overlap_mode == OverlapMode::TwoWayExtended;
  const bool two_way = options_.overlap_mode == OverlapMode::TwoWayExtended;

  std::vector<std::vector<size_t>> direct_succs_of;
  if (two_way) {
    direct_succs_of.assign(n, {});
    for (size_t k = 0; k < n; ++k) {
      for (size_t pred : direct_preds_of[k]) {
        direct_succs_of[pred].push_back(k);
      }
    }
  }

  std::vector<size_t> internal_node(n, kNoKernel_);
  std::vector<size_t> boundary_node(n, kNoKernel_);

  // Replaces kernel k's node by an Internal and a Boundary node that both inherit all its edges.
  auto split = [&](size_t k, reduce::ReduceLaunchMode mode) {
    const size_t ki = nodes.size();
    nodes.push_back(StepGraphNode{.kernel_idx = k,
                                  .view = DataView::Internal,
                                  .reduce_mode = mode,
                                  .active = true,
                                  .data_succs = nodes[k].data_succs,
                                  .hint_succs = nodes[k].hint_succs});
    const size_t kb = nodes.size();
    nodes.push_back(StepGraphNode{.kernel_idx = k,
                                  .view = DataView::Boundary,
                                  .reduce_mode = mode,
                                  .active = true,
                                  .data_succs = nodes[k].data_succs,
                                  .hint_succs = nodes[k].hint_succs});
    for (size_t p = 0; p < ki; ++p) {
      if (!nodes[p].active) {
        continue;
      }
      step_graph_replace_succ(nodes[p].data_succs, k, ki, kb, /*both_halves=*/true);
      step_graph_replace_succ(nodes[p].hint_succs, k, ki, kb, /*both_halves=*/true);
    }
    nodes[k].active = false;
    internal_node[k] = ki;
    boundary_node[k] = kb;
  };

  // True when consumer writes nothing producer touches: the only hazard between them is
  // consumer reading producer's output cell-by-cell, so matching halves order each other.
  auto consumes_only = [&](size_t producer, size_t consumer) {
    for (const AccessToken& tc : effective_kernels_[consumer].impl()->tokens()) {
      if (tc.mode != AccessMode::Write && tc.mode != AccessMode::ReadWrite) {
        continue;
      }
      for (const AccessToken& tp : effective_kernels_[producer].impl()->tokens()) {
        if (tp.field_uid == tc.field_uid) {
          return false;
        }
      }
    }
    return true;
  };

  for (const OccPair& pair : candidates) {
    const size_t h = pair.halo;
    const size_t s = pair.stencil;
    if (!nodes[s].active) {
      continue;
    }

    const bool is_reduction = effective_kernels_[s].kind() == KernelKind::Reduction;
    // min_edges_ is transitively reduced, so orderings that reach s only through h
    // (e.g. the previous stencil's writes) would be lost once h no longer precedes s_I.
    const std::vector<size_t> preds_h = step_graph_data_preds(nodes, h);
    split(s, is_reduction ? reduce::ReduceLaunchMode::Partial : reduce::ReduceLaunchMode::Full);
    const size_t si = internal_node[s];
    const size_t sb = boundary_node[s];
    step_graph_erase_succ(nodes[h].data_succs, si);
    if (is_reduction) {
      const size_t sm = nodes.size();
      nodes.push_back(StepGraphNode{.kernel_idx = s,
                                    .view = DataView::Standard,
                                    .reduce_mode = reduce::ReduceLaunchMode::Merge,
                                    .active = true,
                                    .data_succs = nodes[si].data_succs,
                                    .hint_succs = {}});
      nodes[si].data_succs = {sm};
      nodes[sb].data_succs = {sm};
    }
    for (size_t p : preds_h) {
      step_graph_add_succ(nodes[p].data_succs, si);
    }
    nodes[si].hint_succs.push_back(h);

    if (extended) {
      // Compute predecessors reachable through Compute kernels that write a stencil-read field.
      // The walk passes through h itself: min_edges_ usually reaches s from a producer only
      // via h (producer -> h -> s), so stopping at h would never find any producer.
      std::vector<bool> visited(n, false);
      visited[s] = true;
      visited[h] = true;
      std::vector<size_t> stack{s, h};
      std::vector<size_t> map_back;
      while (!stack.empty()) {
        const size_t k = stack.back();
        stack.pop_back();
        for (size_t p : direct_preds_of[k]) {
          if (visited[p]) {
            continue;
          }
          visited[p] = true;
          const Kernel& kp = effective_kernels_[p];
          if (kp.kind() != KernelKind::Compute) {
            continue;
          }
          for (const AccessToken& tp : kp.impl()->tokens()) {
            if ((tp.mode == AccessMode::Write || tp.mode == AccessMode::ReadWrite) &&
                std::find(pair.stencil_fields.begin(), pair.stencil_fields.end(), tp.field_uid) !=
                    pair.stencil_fields.end()) {
              map_back.push_back(p);
              break;
            }
          }
          stack.push_back(p);
        }
      }

      for (size_t p : map_back) {
        if (internal_node[p] == kNoKernel_) {
          if (!nodes[p].active) {
            continue;
          }
          split(p, reduce::ReduceLaunchMode::Full);
          step_graph_add_succ(nodes[boundary_node[p]].data_succs, internal_node[p]);
        }
        const size_t pi = internal_node[p];
        // The halo only sends p's boundary band, so it overlaps p_I instead of waiting on it.
        if (step_graph_erase_succ(nodes[pi].data_succs, h)) {
          for (size_t x : nodes[h].data_succs) {
            if (x != pi) {
              step_graph_add_succ(nodes[pi].data_succs, x);
            }
          }
          step_graph_add_succ(nodes[boundary_node[p]].data_succs, h);
        }
      }
    }

    if (two_way) {
      // Map Compute / Map Reduction successors reading what s (or an earlier Map Compute) wrote.
      // Neon Two-way Extended OCC splits both; stencil-pattern reductions are their own OCC pairs.
      std::vector<std::uint64_t> tainted_fields;
      for (const AccessToken& ts : effective_kernels_[s].impl()->tokens()) {
        if (ts.mode == AccessMode::Write || ts.mode == AccessMode::ReadWrite) {
          tainted_fields.push_back(ts.field_uid);
        }
      }
      auto reads_tainted = [&tainted_fields](const std::vector<AccessToken>& toks) {
        return std::any_of(toks.begin(), toks.end(), [&tainted_fields](const AccessToken& t) {
          return (t.mode == AccessMode::Read || t.mode == AccessMode::ReadWrite) &&
                 std::find(tainted_fields.begin(), tainted_fields.end(), t.field_uid) !=
                     tainted_fields.end();
        });
      };
      auto has_stencil = [](const std::vector<AccessToken>& toks) {
        return std::any_of(toks.begin(), toks.end(), [](const AccessToken& t) {
          return t.pattern == AccessPattern::Stencil;
        });
      };

      std::vector<bool> visited(n, false);
      visited[s] = true;
      std::vector<size_t> queue{s};
      std::vector<size_t> map_forward;
      std::vector<size_t> reduce_forward;
      for (size_t qi = 0; qi < queue.size(); ++qi) {
        for (size_t c : direct_succs_of[queue[qi]]) {
          if (visited[c]) {
            continue;
          }
          visited[c] = true;
          const Kernel& kc = effective_kernels_[c];
          const auto& toks = kc.impl()->tokens();
          if (kc.kind() == KernelKind::Reduction) {
            // Map-pattern only; do not enqueue (scalar write ends the Neon chain).
            if (!has_stencil(toks) && reads_tainted(toks)) {
              reduce_forward.push_back(c);
            }
            continue;
          }
          if (kc.kind() != KernelKind::Compute) {
            continue;
          }
          if (has_stencil(toks) || !reads_tainted(toks)) {
            continue;
          }
          map_forward.push_back(c);
          for (const AccessToken& t : toks) {
            if ((t.mode == AccessMode::Write || t.mode == AccessMode::ReadWrite) &&
                std::find(tainted_fields.begin(), tainted_fields.end(), t.field_uid) ==
                    tainted_fields.end()) {
              tainted_fields.push_back(t.field_uid);
            }
          }
          queue.push_back(c);
        }
      }

      auto half_pair = [&](size_t c) {
        const size_t ci = internal_node[c];
        const size_t cb = boundary_node[c];
        for (size_t q : step_graph_data_preds(nodes, ci)) {
          const size_t kq = nodes[q].kernel_idx;
          if (internal_node[kq] == kNoKernel_ || !consumes_only(kq, c)) {
            continue;
          }
          if (q == internal_node[kq]) {
            step_graph_erase_succ(nodes[q].data_succs, cb);
          } else if (q == boundary_node[kq]) {
            step_graph_erase_succ(nodes[q].data_succs, ci);
          }
        }
      };

      for (size_t c : map_forward) {
        if (internal_node[c] != kNoKernel_ || !nodes[c].active) {
          continue;
        }
        split(c, reduce::ReduceLaunchMode::Full);
        half_pair(c);
      }

      for (size_t c : reduce_forward) {
        if (internal_node[c] != kNoKernel_ || !nodes[c].active) {
          continue;
        }
        split(c, reduce::ReduceLaunchMode::Partial);
        const size_t ci = internal_node[c];
        const size_t cb = boundary_node[c];
        const size_t cm = nodes.size();
        nodes.push_back(StepGraphNode{.kernel_idx = c,
                                      .view = DataView::Standard,
                                      .reduce_mode = reduce::ReduceLaunchMode::Merge,
                                      .active = true,
                                      .data_succs = nodes[ci].data_succs,
                                      .hint_succs = {}});
        nodes[ci].data_succs = {cm};
        nodes[cb].data_succs = {cm};
        half_pair(c);
      }
    }
  }
}

auto
Skeleton::run(DataView view) -> void {
  compile();

  // CUDA/HIP graph fast path: capture the schedule once, replay it thereafter.
  // Only engaged for single-stream / single-GPU schedules on a capture-capable
  // backend (see graph_eligible_); everything else drops through to the eager
  // path below with identical semantics.
  if (graph_eligible_()) {
    run_graph_(view);
    return;
  }

  run_eager_(view);
}

auto
Skeleton::issue_schedule_(DataView view) -> void {
  for (const Step& step : schedule_) {
    Stream& s = streams_[step.stream_idx];
    for (size_t evt_idx : step.wait_events) {
      s.wait_for(events_[evt_idx]);
    }
    // Step::view selects the per-launch DataView. For pipelines with
    // OverlapMode::None (the default) this is always DataView::Standard
    // and the per-step view collapses to the user-supplied @p view (so
    // run(DataView::Internal) still works for callers driving the whole
    // pipeline on one side). Under non-None OverlapMode, Step::view is
    // forced to Internal / Boundary by the OCC rewrite and overrides the
    // user's @p view.
    const DataView effective_view = (step.view == DataView::Standard) ? view : step.view;
    Kernel& k = effective_kernels_[step.kernel_idx];
    if (step.reduce_launch_mode != reduce::ReduceLaunchMode::Full) {
      k.launch_reduction_phase(s, effective_view, step.reduce_launch_mode);
    } else {
      k.launch(s, effective_view);
    }
    if (step.record_event_idx != kNoEvent_) {
      events_[step.record_event_idx].record(s);
    }
  }
}

auto
Skeleton::run_eager_(DataView view) -> void {
  // Anchor fork ("start"): every secondary stream waits until the anchor stream
  // (stream 0) reaches this run's start point. Two consequences, both required for
  // correctness once the anchor is the System's shared default stream:
  //   * Cross-skeleton — the start event is recorded into the anchor's FIFO, which
  //     already holds the previous skeleton's join, so this skeleton's secondary
  //     streams cannot begin until the previous skeleton has fully finished. That is
  //     what lets an alternating ping-pong pair run without a host drain between
  //     iterations (the halo of iteration N+1 can no longer peer-copy a population
  //     that iteration N is still writing).
  //   * Cross-iteration — same argument applied to this skeleton's own previous run,
  //     which makes this the whole inter-iteration barrier: combined with the join
  //     below, no stream can start iteration N+1 before every stream finished N.
  if (streams_.size() > 1U && anchor_start_event_ != kNoEvent_) {
    events_[anchor_start_event_].record(streams_[0]);
    for (size_t s = 1U; s < streams_.size(); ++s) {
      if (s == copy_stream_idx_) {
        continue;
      }
      streams_[s].wait_for(events_[anchor_start_event_]);
    }
  }

  issue_schedule_(view);

  // Anchor join ("end"): the anchor stream waits for every secondary stream's tail,
  // so after run() the anchor alone represents whole-skeleton completion. That is what
  // makes the *next* start event (recorded into the same FIFO) a valid whole-pipeline
  // barrier, and it means wait()ing the anchor is sufficient to drain the skeleton.
  if (streams_.size() > 1U) {
    for (size_t s = 1U; s < streams_.size(); ++s) {
      if (s == copy_stream_idx_) {
        continue;
      }
      if (anchor_join_events_[s] != kNoEvent_) {
        events_[anchor_join_events_[s]].record(streams_[s]);
        streams_[0].wait_for(events_[anchor_join_events_[s]]);
      }
    }
  }
}

auto
Skeleton::graph_eligible_() const noexcept -> bool {
  if (!options_.use_cuda_graph || graph_disabled_ || !detail::GraphReplay::supported()) {
    return false;
  }
  if (has_detached_stream_) {
    return false;
  }
  // A single Skeleton stream (OverlapMode::None) means no cross-*Skeleton*-stream
  // event choreography to capture. The GPUs behind it may be one (ThreadLocal
  // capture) or many (global-mode fork/join capture over the per-GPU streams —
  // the halo's cross-device peer copies + event fences fold into the graph as
  // intra-graph dependencies). Multi-*Skeleton*-stream (OCC) capture is a tracked
  // follow-up.
  if (streams_.size() != 1U || streams_[0].size() == 0U || effective_kernels_.empty()) {
    return false;
  }
  // The distributed (cross-rank NCCL) halo path is not capture-safe here; only
  // the intra-process multi-GPU case is captured. Single-process runs have no
  // Distributed handle.
  if (sys_.distributed() != nullptr) {
    return false;
  }
  // Host (OpenMP) kernels run on the CPU and cannot live in a device graph.
  for (const Kernel& k : effective_kernels_) {
    const KernelKind kind = k.kind();
    if (kind == KernelKind::ComputeHost || kind == KernelKind::ReductionHost) {
      return false;
    }
  }
  return true;
}

auto
Skeleton::run_graph_(DataView view) -> void {
#if CARBON_BACKEND_IS_CUDALIKE
  // Origin stream = per-GPU stream 0 of the single Skeleton stream; the graph is
  // captured and replayed from the origin's device. For a multi-GPU layout the
  // remaining per-GPU streams are folded into the same graph via fork/join.
  const size_t n_gpus = streams_[0].size();
  const auto& devices = sys_.gpu_group().devices();
  std::vector<detail::gpu_stream_t> capture_streams(n_gpus);
  std::vector<int> device_ids(n_gpus);
  for (size_t i = 0; i < n_gpus; ++i) {
    capture_streams[i] = streams_[0].native_at(i);
    device_ids[i] = devices[i].device_id();
  }
  const detail::gpu_stream_t stream = capture_streams[0];

  if (graph_.ready() && captured_view_ == view) {
    if (graph_.launch(stream)) {
      return;
    }
    // A replay that fails at runtime (rare) drops us back to eager for good.
    graph_.reset();
    graph_disabled_ = true;
    run_eager_(view);
    return;
  }

  // (Re)capture for this view. The enqueue lambda records — does not execute —
  // the schedule, so on success we launch the graph to actually run this
  // iteration exactly once. A schedule that issues a capture-illegal op (e.g. a
  // non-P2P peer copy that must stage on the host) surfaces as a thrown
  // exception from the enqueue path; GraphReplay::capture drains the in-flight
  // capture and rethrows, and we treat that identically to a soft capture
  // failure — nothing executed yet, so we fall back to an eager run.
  bool captured = false;
  try {
    captured = graph_.capture(capture_streams, device_ids, [&] {
      issue_schedule_(view);
    });
  } catch (const std::exception& e) {
    graph_.reset();
    graph_disabled_ = true;
    captured_view_ = view;
    CARBON_LOG_WARN(
        "Skeleton: CUDA graph capture raised '{}' for this pipeline; falling back to eager launch "
        "(results unaffected).",
        e.what());
    run_eager_(view);
    return;
  }
  captured_view_ = view;
  if (captured && graph_.launch(stream)) {
    return;
  }
  // Capture or first launch failed: nothing executed, so run the iteration
  // eagerly and stop attempting capture for this compiled schedule.
  graph_.reset();
  graph_disabled_ = true;
  CARBON_LOG_WARN(
      "Skeleton: CUDA graph capture/replay unavailable for this pipeline; falling back to eager "
      "launch (results unaffected).");
  run_eager_(view);
#else
  run_eager_(view);
#endif
}

auto
Skeleton::compile_prune_syncs_() -> void {
  if (schedule_.empty()) {
    return;
  }

  const auto rec_stream_for_events = [&]() -> std::vector<size_t> {
    std::vector<size_t> rec_stream(events_.size(), kNoStream_);
    for (size_t si = 0; si < schedule_.size(); ++si) {
      const size_t evt = schedule_[si].record_event_idx;
      if (evt != kNoEvent_ && evt < rec_stream.size()) {
        rec_stream[evt] = schedule_[si].stream_idx;
      }
    }
    if (anchor_start_event_ != kNoEvent_ && anchor_start_event_ < rec_stream.size()) {
      rec_stream[anchor_start_event_] = 0U;
    }
    for (size_t s = 1U; s < anchor_join_events_.size(); ++s) {
      const size_t evt = anchor_join_events_[s];
      if (evt != kNoEvent_ && evt < rec_stream.size()) {
        rec_stream[evt] = s;
      }
    }
    return rec_stream;
  };

  const size_t num_streams = streams_.size();
  const std::vector<size_t> rec_stream = rec_stream_for_events();

  auto simulate_and_prune_waits = [&]() -> void {
    std::vector<std::vector<size_t>> known_seq(num_streams, std::vector<size_t>(num_streams, 0U));
    std::vector<size_t> event_record_op_seq(events_.size(), 0U);
    std::vector<size_t> last_op_seq_on_stream(num_streams, 0U);

    auto redundant_wait = [&](size_t wait_stream, size_t evt) -> bool {
      if (evt >= rec_stream.size() || rec_stream[evt] == kNoStream_) {
        return false;
      }
      const size_t rec_s = rec_stream[evt];
      const size_t rec_seq = event_record_op_seq[evt];
      if (wait_stream == rec_s) {
        return true;
      }
      return known_seq[wait_stream][rec_s] >= rec_seq;
    };

    auto apply_wait = [&](size_t wait_stream, size_t evt) {
      if (evt >= rec_stream.size() || rec_stream[evt] == kNoStream_) {
        return;
      }
      const size_t rec_s = rec_stream[evt];
      const size_t rec_seq = event_record_op_seq[evt];
      if (wait_stream != rec_s && known_seq[wait_stream][rec_s] < rec_seq) {
        known_seq[wait_stream][rec_s] = rec_seq;
      }
    };

    std::vector<std::pair<size_t, size_t>> drop_step_waits;
    bool saw_anchor_start_wait = false;
    bool kept_anchor_start_wait = false;
    const std::vector<LaunchOrderOp> ops = build_launch_order_ops_();

    // Fill record seqs first so forward dominance can compare event strengths.
    for (const LaunchOrderOp& op : ops) {
      if ((op.kind == LaunchOrderOpKind::EventRecord ||
           op.kind == LaunchOrderOpKind::AnchorStartRecord ||
           op.kind == LaunchOrderOpKind::AnchorJoinRecord) &&
          op.event_idx < event_record_op_seq.size()) {
        event_record_op_seq[op.event_idx] = op.seq;
      }
    }

    // Forward dominance: before the next Launch on a stream, keep only the strongest
    // wait per recorder stream. Weaker earlier waits (e.g. anchor start before a later
    // stream-0 data event) add no ordering and are dropped — including AnchorStartWait.
    {
      struct PendingWait {
        size_t evt{kNoEvent_};
        size_t rec_s{kNoStream_};
        size_t rec_seq{0U};
        LaunchOrderOpKind kind{};
        size_t step_idx{kNoKernel_};
      };
      std::vector<std::vector<PendingWait>> pending(num_streams);

      auto flush_pending = [&](size_t stream_idx) {
        auto& group = pending[stream_idx];
        if (group.empty()) {
          return;
        }
        // Per recorder stream: index of the strongest wait (max rec_seq; last wins ties).
        std::vector<size_t> best_idx(num_streams, static_cast<size_t>(-1));
        for (size_t i = 0; i < group.size(); ++i) {
          const PendingWait& w = group[i];
          if (w.rec_s >= num_streams) {
            continue;
          }
          if (best_idx[w.rec_s] == static_cast<size_t>(-1) ||
              w.rec_seq >= group[best_idx[w.rec_s]].rec_seq) {
            best_idx[w.rec_s] = i;
          }
        }
        std::vector<bool> keep(group.size(), false);
        for (size_t rec_s = 0; rec_s < num_streams; ++rec_s) {
          if (best_idx[rec_s] != static_cast<size_t>(-1)) {
            keep[best_idx[rec_s]] = true;
          }
        }
        for (size_t i = 0; i < group.size(); ++i) {
          const PendingWait& w = group[i];
          if (w.kind == LaunchOrderOpKind::AnchorStartWait) {
            saw_anchor_start_wait = true;
            if (keep[i]) {
              kept_anchor_start_wait = true;
            }
          }
          if (keep[i]) {
            continue;
          }
          if (w.kind == LaunchOrderOpKind::EventWait && w.step_idx < schedule_.size()) {
            drop_step_waits.push_back({w.step_idx, w.evt});
          }
        }
        group.clear();
      };

      for (const LaunchOrderOp& op : ops) {
        if (op.kind == LaunchOrderOpKind::AnchorJoinRecord ||
            op.kind == LaunchOrderOpKind::AnchorJoinWait) {
          continue;
        }
        if (op.stream_idx >= num_streams) {
          continue;
        }
        if (op.kind == LaunchOrderOpKind::EventWait ||
            op.kind == LaunchOrderOpKind::AnchorStartWait) {
          if (op.event_idx >= rec_stream.size() || rec_stream[op.event_idx] == kNoStream_) {
            continue;
          }
          pending[op.stream_idx].push_back(PendingWait{op.event_idx, rec_stream[op.event_idx],
                                                       event_record_op_seq[op.event_idx], op.kind,
                                                       op.step_idx});
          continue;
        }
        if (op.kind == LaunchOrderOpKind::Launch) {
          flush_pending(op.stream_idx);
        }
      }
      for (size_t s = 0; s < num_streams; ++s) {
        flush_pending(s);
      }
    }

    // Global fork: clear only when every secondary's AnchorStartWait was dominated.
    if (saw_anchor_start_wait && !kept_anchor_start_wait) {
      anchor_start_event_ = kNoEvent_;
    }

    // Apply forward-dominated step-wait drops before the backward coverage pass so
    // known_seq reflects the pruned schedule.
    for (const auto& [step_idx, evt] : drop_step_waits) {
      Step& step = schedule_[step_idx];
      step.wait_events.erase(std::remove_if(step.wait_events.begin(), step.wait_events.end(),
                                            [evt](size_t e) {
                                              return e == evt;
                                            }),
                             step.wait_events.end());
    }
    drop_step_waits.clear();

    // Rebuild ops after forward drops (anchor start / step waits) for coverage + joins.
    const std::vector<LaunchOrderOp> ops_pruned = build_launch_order_ops_();
    event_record_op_seq.assign(events_.size(), 0U);
    for (const LaunchOrderOp& op : ops_pruned) {
      if (op.kind == LaunchOrderOpKind::AnchorJoinRecord ||
          op.kind == LaunchOrderOpKind::AnchorJoinWait) {
        continue;
      }
      if (op.kind == LaunchOrderOpKind::EventWait ||
          op.kind == LaunchOrderOpKind::AnchorStartWait) {
        if (redundant_wait(op.stream_idx, op.event_idx)) {
          if (op.kind == LaunchOrderOpKind::EventWait && op.step_idx < schedule_.size()) {
            drop_step_waits.push_back({op.step_idx, op.event_idx});
          }
        } else {
          apply_wait(op.stream_idx, op.event_idx);
        }
        continue;
      }
      if (op.kind == LaunchOrderOpKind::Launch) {
        last_op_seq_on_stream[op.stream_idx] = op.seq;
        continue;
      }
      if (op.kind == LaunchOrderOpKind::EventRecord ||
          op.kind == LaunchOrderOpKind::AnchorStartRecord ||
          op.kind == LaunchOrderOpKind::AnchorJoinRecord) {
        if (op.event_idx < event_record_op_seq.size()) {
          event_record_op_seq[op.event_idx] = op.seq;
        }
        last_op_seq_on_stream[op.stream_idx] = op.seq;
      }
    }

    for (const auto& [step_idx, evt] : drop_step_waits) {
      Step& step = schedule_[step_idx];
      step.wait_events.erase(std::remove_if(step.wait_events.begin(), step.wait_events.end(),
                                            [evt](size_t e) {
                                              return e == evt;
                                            }),
                             step.wait_events.end());
    }

    for (size_t s = 1U; s < streams_.size(); ++s) {
      if (s == copy_stream_idx_ || s >= anchor_join_events_.size()) {
        continue;
      }
      const size_t join_evt = anchor_join_events_[s];
      if (join_evt == kNoEvent_) {
        continue;
      }
      const size_t last_on_s = last_op_seq_on_stream[s];
      if (last_on_s == 0U || known_seq[0U][s] >= last_on_s) {
        anchor_join_events_[s] = kNoEvent_;
      }
    }
  };

  simulate_and_prune_waits();

  std::vector<bool> event_used(events_.size(), false);
  for (const Step& step : schedule_) {
    for (size_t evt : step.wait_events) {
      if (evt < event_used.size()) {
        event_used[evt] = true;
      }
    }
  }
  if (anchor_start_event_ != kNoEvent_ && anchor_start_event_ < event_used.size()) {
    event_used[anchor_start_event_] = true;
  }
  for (size_t s = 1U; s < anchor_join_events_.size(); ++s) {
    const size_t evt = anchor_join_events_[s];
    if (evt != kNoEvent_ && evt < event_used.size()) {
      event_used[evt] = true;
    }
  }
  for (Step& step : schedule_) {
    if (step.record_event_idx != kNoEvent_ && step.record_event_idx < event_used.size() &&
        !event_used[step.record_event_idx]) {
      step.record_event_idx = kNoEvent_;
    }
  }
}

auto
Skeleton::compile_allocate_anchor_events_() -> void {
  // compile() clears events_ , so any indices from a previous compile are stale.
  anchor_start_event_ = kNoEvent_;
  anchor_join_events_.assign(streams_.size(), kNoEvent_);
  // A single-stream schedule needs no fork-join: every launch already lands on the
  // anchor stream in FIFO order, which orders both the kernels within an iteration
  // and successive iterations (and successive skeletons sharing the anchor).
  if (streams_.size() <= 1U) {
    return;
  }
  // Copy-only secondaries (deferred scalar readback) are excluded from the anchor
  // bracket so the next skeleton run can overlap D→H on the copy stream.
  bool secondary_needs_anchor = false;
  for (size_t s = 1U; s < streams_.size(); ++s) {
    if (s != copy_stream_idx_) {
      secondary_needs_anchor = true;
      break;
    }
  }
  if (!secondary_needs_anchor) {
    return;
  }
  events_.push_back(sys_.event());
  anchor_start_event_ = events_.size() - 1U;
  for (size_t s = 1U; s < streams_.size(); ++s) {
    if (s == copy_stream_idx_) {
      continue;
    }
    events_.push_back(sys_.event());
    anchor_join_events_[s] = events_.size() - 1U;
  }
}

auto
Skeleton::occ_collect_candidates_(const std::vector<std::vector<size_t>>& direct_preds_of) const
    -> std::vector<OccPair> {
  const size_t n = effective_kernels_.size();
  std::vector<OccPair> candidates;
  for (size_t s = 0; s < n; ++s) {
    const Kernel& ks = effective_kernels_[s];
    if (ks.kind() != KernelKind::Compute && ks.kind() != KernelKind::Reduction) {
      continue;
    }
    std::vector<std::uint64_t> stencil_fields;
    for (const AccessToken& t : ks.impl()->tokens()) {
      if (t.pattern == AccessPattern::Stencil) {
        stencil_fields.push_back(t.field_uid);
      }
    }
    if (stencil_fields.empty()) {
      continue;
    }
    for (size_t h : direct_preds_of[s]) {
      const Kernel& kh = effective_kernels_[h];
      if (kh.kind() != KernelKind::Halo) {
        continue;
      }
      bool match = false;
      for (const AccessToken& th : kh.impl()->tokens()) {
        if (std::find(stencil_fields.begin(), stencil_fields.end(), th.field_uid) !=
            stencil_fields.end()) {
          match = true;
          break;
        }
      }
      if (!match) {
        continue;
      }
      candidates.push_back({h, s, std::move(stencil_fields)});
      break;
    }
  }
  return candidates;
}

auto
Skeleton::build_launch_order_ops_() const -> std::vector<LaunchOrderOp> {
  std::vector<LaunchOrderOp> ops;
  ops.reserve(schedule_.size() * 3U + streams_.size() * 2U + 4U);

  if (streams_.size() > 1U && anchor_start_event_ != kNoEvent_) {
    ops.push_back({0U, LaunchOrderOpKind::AnchorStartRecord, 0U, kNoKernel_, anchor_start_event_});
    for (size_t s = 1U; s < streams_.size(); ++s) {
      if (s == copy_stream_idx_) {
        continue;
      }
      ops.push_back({0U, LaunchOrderOpKind::AnchorStartWait, s, kNoKernel_, anchor_start_event_});
    }
  }

  for (size_t si = 0; si < schedule_.size(); ++si) {
    const Step& step = schedule_[si];
    for (size_t evt : step.wait_events) {
      ops.push_back({0U, LaunchOrderOpKind::EventWait, step.stream_idx, si, evt});
    }
    ops.push_back({0U, LaunchOrderOpKind::Launch, step.stream_idx, si, kNoEvent_});
    if (step.record_event_idx != kNoEvent_) {
      ops.push_back(
          {0U, LaunchOrderOpKind::EventRecord, step.stream_idx, si, step.record_event_idx});
    }
  }

  if (streams_.size() > 1U) {
    for (size_t s = 1U; s < streams_.size(); ++s) {
      if (s == copy_stream_idx_) {
        continue;
      }
      if (s < anchor_join_events_.size() && anchor_join_events_[s] != kNoEvent_) {
        ops.push_back(
            {0U, LaunchOrderOpKind::AnchorJoinRecord, s, kNoKernel_, anchor_join_events_[s]});
        ops.push_back(
            {0U, LaunchOrderOpKind::AnchorJoinWait, 0U, kNoKernel_, anchor_join_events_[s]});
      }
    }
  }

  for (size_t i = 0; i < ops.size(); ++i) {
    ops[i].seq = i + 1U;
  }
  return ops;
}

auto
Skeleton::step_display_label_(size_t step_idx) const -> std::string {
  const Step& step = schedule_[step_idx];
  const Kernel& kernel = effective_kernels_[step.kernel_idx];
  const std::string_view raw_name = kernel.name();
  std::string name = raw_name.empty() ? (std::string{"k"} + std::to_string(step.kernel_idx))
                                      : std::string{raw_name};
  name += " [";
  switch (kernel.kind()) {
    case KernelKind::Compute:
      name += "compute";
      break;
    case KernelKind::Reduction:
      name += "reduction";
      break;
    case KernelKind::Halo:
      name += "halo";
      break;
    case KernelKind::Anchor:
      name += "anchor";
      break;
    case KernelKind::ComputeHost:
      name += "compute_host";
      break;
    case KernelKind::ReductionHost:
      name += "reduction_host";
      break;
    case KernelKind::Universal:
      name += "universal";
      break;
    case KernelKind::UniversalReduction:
      name += "universal_reduction";
      break;
    case KernelKind::ScalarReadback:
      name += "scalar_readback";
      break;
  }
  name += "/";
  name += to_string(step.view);
  name += "]";
  return name;
}

auto
Skeleton::event_annotation_(size_t event_idx) const -> std::string {
  if (event_idx == anchor_start_event_) {
    return "anchor start";
  }
  for (size_t s = 1U; s < streams_.size(); ++s) {
    if (s < anchor_join_events_.size() && anchor_join_events_[s] == event_idx) {
      return "anchor join (stream " + std::to_string(s) + ")";
    }
  }
  for (size_t si = 0; si < schedule_.size(); ++si) {
    if (schedule_[si].record_event_idx == event_idx) {
      return step_display_label_(si);
    }
  }
  return "?";
}

auto
Skeleton::format_launch_order_line_(const LaunchOrderOp& op) const -> std::string {
  const char* kind_name = "?";
  switch (op.kind) {
    case LaunchOrderOpKind::AnchorStartRecord:
    case LaunchOrderOpKind::EventRecord:
    case LaunchOrderOpKind::AnchorJoinRecord:
      kind_name = "record";
      break;
    case LaunchOrderOpKind::AnchorStartWait:
    case LaunchOrderOpKind::EventWait:
    case LaunchOrderOpKind::AnchorJoinWait:
      kind_name = "wait";
      break;
    case LaunchOrderOpKind::Launch:
      kind_name = "launch";
      break;
  }
  std::ostringstream os;
  os << std::setw(3) << op.seq << "  stream " << op.stream_idx << "  " << kind_name << "  ";
  if (op.kind == LaunchOrderOpKind::Launch) {
    os << step_display_label_(op.step_idx);
    return os.str();
  }
  if (op.event_idx != kNoEvent_) {
    os << 'E' << op.event_idx;
    const std::string note = event_annotation_(op.event_idx);
    if (!note.empty()) {
      os << " (" << note << ')';
    }
  }
  return os.str();
}

auto
Skeleton::export_launch_order() -> std::string {
  compile();
  const std::vector<LaunchOrderOp> ops = build_launch_order_ops_();
  std::ostringstream os;
  for (const LaunchOrderOp& op : ops) {
    os << format_launch_order_line_(op) << '\n';
  }
  return os.str();
}

auto
Skeleton::export_launch_order_file(std::string path) -> bool {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    return false;
  }
  out << export_launch_order();
  return out.good();
}

auto
Skeleton::export_launch_matrix_dot(std::string graph_name) -> std::string {
  compile();
  const std::vector<LaunchOrderOp> ops = build_launch_order_ops_();
  const size_t num_streams = streams_.size();

  // Fixed cell geometry (points) so rows and columns line up like a table.
  constexpr int kCellW = 200;
  constexpr int kCellH = 40;
  constexpr int kSeqW = 30;

  std::ostringstream os;
  os << "digraph " << graph_name << " {\n";
  os << "  graph [rankdir=TB, nodesep=0.04, ranksep=0.04, pad=0.3, bgcolor=\"white\", "
        "fontname=\"Helvetica\", labelloc=t, labeljust=l];\n";
  os << "  node [shape=none, margin=0, fontname=\"Helvetica\"];\n";
  os << "  edge [fontname=\"Helvetica\", fontsize=8, penwidth=1.4, arrowsize=0.7];\n";

  // Title + legend as the graph label, so it never disturbs the grid layout.
  auto swatch = [&](Palette p, std::string_view style, std::string_view text) {
    os << legend_swatch(p, style, text);
  };
  os << "  label=<<TABLE BORDER=\"0\" CELLSPACING=\"4\" CELLPADDING=\"1\">"
     << "<TR><TD COLSPAN=\"16\" ALIGN=\"LEFT\"><FONT POINT-SIZE=\"15\"><B>"
     << html_escape(graph_name) << "</B></FONT><FONT POINT-SIZE=\"10\" COLOR=\"" << kMuted
     << "\">   launch matrix &#183; " << num_streams << " stream" << (num_streams == 1U ? "" : "s")
     << " &#183; " << ops.size() << " ops in run() order</FONT></TD></TR>"
     << "<TR><TD COLSPAN=\"16\" ALIGN=\"LEFT\">"
     << font(10, kMuted, skeleton_options_summary(options_)) << "</TD></TR><TR>";
  swatch(kHaloPal, "rounded", "halo");
  swatch(kInternalPal, "rounded", "internal");
  swatch(kBoundaryPal, "rounded", "boundary");
  swatch(kStandardPal, "rounded", "standard / merge");
  swatch({"#ffffff", kSyncColor}, "rounded", "event record");
  swatch({"#ffffff", kSyncColor}, "dashed", "event wait");
  os << "<TD><FONT COLOR=\"" << kSyncColor << "\" POINT-SIZE=\"11\"><B>&#8594;</B></FONT></TD>"
     << "<TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"9\">cross-stream sync</FONT></TD>"
     << "<TD><FONT COLOR=\"" << kAnchorColor << "\" POINT-SIZE=\"11\"><B>&#8594;</B></FONT></TD>"
     << "<TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"9\">anchor fork / join</FONT></TD>"
     << "</TR></TABLE>>;\n";

  auto emit_cell = [&](const std::string& id, std::string_view bg, int width,
                       const std::string& inner) {
    os << "  " << id << " [label=<<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\" "
       << "CELLPADDING=\"0\" BGCOLOR=\"" << bg << "\"><TR><TD FIXEDSIZE=\"TRUE\" WIDTH=\"" << width
       << "\" HEIGHT=\"" << kCellH << "\">" << inner << "</TD></TR></TABLE>>];\n";
  };

  // Header row: launch number column, then one column per stream.
  emit_cell("h_seq", kRowBg[0], kSeqW,
            "<FONT POINT-SIZE=\"9\" COLOR=\"" + std::string{kMuted} + "\"><B>#</B></FONT>");
  for (size_t s = 0; s < num_streams; ++s) {
    std::string role;
    if (s == 0U) {
      role = "anchor";
    } else if (s == copy_stream_idx_) {
      role = "copy (detached)";
    }
    std::string inner =
        "<FONT POINT-SIZE=\"11\" COLOR=\"white\"><B>stream " + std::to_string(s) + "</B></FONT>";
    if (!role.empty()) {
      inner += "<BR/>" + font(8, kHeaderSubtitle, role);
    }
    emit_cell("h_c" + std::to_string(s), kHeaderBg, kCellW, inner);
  }

  auto is_record = [](LaunchOrderOpKind k) {
    return k == LaunchOrderOpKind::AnchorStartRecord || k == LaunchOrderOpKind::EventRecord ||
           k == LaunchOrderOpKind::AnchorJoinRecord;
  };
  auto is_anchor = [](LaunchOrderOpKind k) {
    return k == LaunchOrderOpKind::AnchorStartRecord || k == LaunchOrderOpKind::AnchorStartWait ||
           k == LaunchOrderOpKind::AnchorJoinRecord || k == LaunchOrderOpKind::AnchorJoinWait;
  };
  auto cell_id = [](size_t seq, size_t s) {
    return "r" + std::to_string(seq) + "_c" + std::to_string(s);
  };

  // PORT="op" lets sync arrows attach to the box, not the whole cell.
  auto op_box = [](Palette p, std::string_view style, int width, int height,
                   const std::string& body) {
    return html_box(p, style, width, height, body, "op");
  };
  constexpr int kKernelBoxW = kCellW - 16;
  constexpr int kKernelBoxH = kCellH - 10;
  constexpr int kEventBoxW = 120;
  constexpr int kEventBoxH = 20;
  constexpr int kAnchorBoxH = 26;

  for (const LaunchOrderOp& op : ops) {
    const std::string_view bg = kRowBg[op.seq % 2U];
    emit_cell("r" + std::to_string(op.seq) + "_seq", bg, kSeqW,
              "<FONT POINT-SIZE=\"9\" COLOR=\"" + std::string{kMuted} + "\">" +
                  std::to_string(op.seq) + "</FONT>");
    for (size_t s = 0; s < num_streams; ++s) {
      const std::string id = cell_id(op.seq, s);
      if (s != op.stream_idx) {
        emit_cell(id, bg, kCellW, "");
        continue;
      }
      if (op.kind == LaunchOrderOpKind::Launch) {
        const Step& step = schedule_[op.step_idx];
        // step_display_label_ is "name [kind/view]"; split it into title + subtitle.
        const std::string label = step_display_label_(op.step_idx);
        const size_t bracket = label.rfind(" [");
        std::string name = label.substr(0, bracket);
        std::string tag = bracket == std::string::npos
                              ? ""
                              : label.substr(bracket + 2U, label.size() - bracket - 3U);
        const size_t slash = tag.find('/');
        if (slash != std::string::npos) {
          tag.replace(slash, 1U, " &#183; ");
        }
        const std::string body = "<FONT POINT-SIZE=\"10\"><B>" + html_escape(name) +
                                 "</B></FONT><BR/><FONT POINT-SIZE=\"8\" COLOR=\"" +
                                 std::string{kMuted} + "\">" + tag + "</FONT>";
        emit_cell(id, bg, kCellW,
                  op_box(launch_palette(effective_kernels_[step.kernel_idx].kind(), step.view),
                         "rounded", kKernelBoxW, kKernelBoxH, body));
        continue;
      }
      const std::string_view color = is_anchor(op.kind) ? kAnchorColor : kSyncColor;
      std::string body = "<FONT POINT-SIZE=\"9\" COLOR=\"" + std::string{color} + "\"><B>" +
                         std::string{is_record(op.kind) ? "record" : "wait"} + "</B> E" +
                         std::to_string(op.event_idx) + "</FONT>";
      if (is_anchor(op.kind)) {
        body += "<BR/><FONT POINT-SIZE=\"7\" COLOR=\"" + std::string{kMuted} + "\">" +
                html_escape(event_annotation_(op.event_idx)) + "</FONT>";
      }
      emit_cell(id, bg, kCellW,
                op_box({"#ffffff", color}, is_record(op.kind) ? "rounded" : "dashed", kEventBoxW,
                       is_anchor(op.kind) ? kAnchorBoxH : kEventBoxH, body));
    }
  }

  // Rows: keep each launch on one rank, columns ordered left to right.
  auto emit_row = [&](const std::string& seq_id, auto&& col_id) {
    os << "  { rank=same; " << seq_id;
    for (size_t s = 0; s < num_streams; ++s) {
      os << " -> " << col_id(s);
    }
    os << " [style=invis]; }\n";
  };
  emit_row("h_seq", [](size_t s) {
    return "h_c" + std::to_string(s);
  });
  for (const LaunchOrderOp& op : ops) {
    emit_row("r" + std::to_string(op.seq) + "_seq", [&](size_t s) {
      return cell_id(op.seq, s);
    });
  }

  // Columns: heavy invisible edges stack each column vertically in launch order.
  std::string prev_seq = "h_seq";
  std::vector<std::string> prev_col(num_streams);
  for (size_t s = 0; s < num_streams; ++s) {
    prev_col[s] = "h_c" + std::to_string(s);
  }
  for (const LaunchOrderOp& op : ops) {
    const std::string seq_id = "r" + std::to_string(op.seq) + "_seq";
    os << "  " << prev_seq << " -> " << seq_id << " [style=invis, weight=100];\n";
    prev_seq = seq_id;
    for (size_t s = 0; s < num_streams; ++s) {
      const std::string id = cell_id(op.seq, s);
      os << "  " << prev_col[s] << " -> " << id << " [style=invis, weight=100];\n";
      prev_col[s] = id;
    }
  }

  // Sync arrows: each event record points at every wait on that event.
  std::unordered_map<size_t, const LaunchOrderOp*> record_by_event;
  for (const LaunchOrderOp& op : ops) {
    if (is_record(op.kind) && op.event_idx != kNoEvent_) {
      record_by_event[op.event_idx] = &op;
    }
  }
  for (const LaunchOrderOp& op : ops) {
    if (is_record(op.kind) || op.kind == LaunchOrderOpKind::Launch) {
      continue;
    }
    const auto it = record_by_event.find(op.event_idx);
    if (it == record_by_event.end()) {
      continue;
    }
    const LaunchOrderOp& rec = *it->second;
    const bool anchor = is_anchor(op.kind);
    const std::string_view color = anchor ? kAnchorColor : kSyncColor;
    os << "  " << cell_id(rec.seq, rec.stream_idx) << ":op -> " << cell_id(op.seq, op.stream_idx)
       << ":op [color=\"" << color << "\", fontcolor=\"" << color
       << "\", style=" << (anchor ? "dashed" : "solid") << ", label=\" E" << op.event_idx
       << " \", constraint=false];\n";
  }

  os << "}\n";
  return os.str();
}

auto
Skeleton::export_launch_matrix_dot_file(std::string path,
                                        std::string graph_name) -> bool {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    return false;
  }
  out << export_launch_matrix_dot(std::move(graph_name));
  return out.good();
}

auto
Skeleton::export_kernel_list_dot(std::string graph_name) -> std::string {
  compile();
  const size_t n = effective_kernels_.size();

  std::unordered_map<std::uint64_t, std::string> uid_to_label;
  for (const Kernel& k : effective_kernels_) {
    for (const AccessToken& t : k.impl()->tokens()) {
      if (!t.field_name.empty()) {
        uid_to_label[t.field_uid] = t.field_name;
      }
    }
  }
  for (const auto& [uid, label] : field_label_overrides_) {
    uid_to_label[uid] = label;
  }

  // Resource columns in first-use order; per-kernel access merged over its tokens.
  struct Resource {
    std::uint64_t uid;
    ResourceKind kind;
  };
  struct Access {
    bool read{false};
    bool write{false};
    bool stencil{false};
  };
  std::vector<Resource> resources;
  std::unordered_map<std::uint64_t, size_t> resource_col;
  std::vector<std::unordered_map<size_t, Access>> access(n);
  for (size_t i = 0; i < n; ++i) {
    for (const AccessToken& t : effective_kernels_[i].impl()->tokens()) {
      auto [it, inserted] = resource_col.try_emplace(t.field_uid, resources.size());
      if (inserted) {
        resources.push_back({t.field_uid, t.resource});
      }
      Access& a = access[i][it->second];
      a.read = a.read || t.mode != AccessMode::Write;
      a.write = a.write || t.mode != AccessMode::Read;
      a.stencil = a.stencil || t.pattern == AccessPattern::Stencil;
    }
  }

  std::vector<std::vector<size_t>> preds(n);
  for (size_t i = 0; i < min_edges_.size() && i < n; ++i) {
    for (size_t j : min_edges_[i]) {
      if (j < n) {
        preds[j].push_back(i);
      }
    }
  }
  for (auto& p : preds) {
    std::sort(p.begin(), p.end());
  }

  std::vector<std::vector<LaunchOrderOp>> launches(n);
  for (const LaunchOrderOp& op : build_launch_order_ops_()) {
    if (op.kind == LaunchOrderOpKind::Launch) {
      launches[schedule_[op.step_idx].kernel_idx].push_back(op);
    }
  }

  size_t num_auto = 0;
  for (size_t i = 0; i < n; ++i) {
    num_auto += effective_to_user_[i] == kNoKernel_ ? 1U : 0U;
  }

  constexpr int kIdW = 34;
  constexpr int kKernelW = 184;
  constexpr int kResourceW = 92;
  constexpr int kDepsW = 104;
  constexpr int kLaunchW = 176;
  constexpr int kRowH = 40;
  constexpr Palette kReadChip{"#ffffff", "#7b8794"};
  constexpr Palette kWriteChip{"#2f3b4c", "#2f3b4c"};
  constexpr Palette kReadWriteChip{"#6a7a8f", "#2f3b4c"};
  constexpr std::string_view kStencilColor = "#d97b0f";

  auto kind_name = [](KernelKind k) -> std::string {
    switch (k) {
      case KernelKind::Compute:
        return "compute";
      case KernelKind::Reduction:
        return "reduction";
      case KernelKind::Halo:
        return "halo";
      case KernelKind::Anchor:
        return "anchor";
      case KernelKind::ComputeHost:
        return "compute (host)";
      case KernelKind::ReductionHost:
        return "reduction (host)";
      case KernelKind::Universal:
        return "universal";
      case KernelKind::UniversalReduction:
        return "universal reduction";
      case KernelKind::ScalarReadback:
        return "scalar readback";
    }
    return "?";
  };
  auto resource_label = [&](const Resource& r) -> std::string {
    const auto it = uid_to_label.find(r.uid);
    return it != uid_to_label.end() && !it->second.empty() ? html_escape(it->second)
                                                           : "uid " + std::to_string(r.uid);
  };
  auto resource_kind = [](ResourceKind k) -> std::string {
    switch (k) {
      case ResourceKind::Field:
        return "field";
      case ResourceKind::ScalarDevice:
        return "scalar &#183; device";
      case ResourceKind::ScalarHost:
        return "scalar &#183; host";
    }
    return "?";
  };
  auto access_chip = [&](const Access& a) -> std::string {
    const bool rw = a.read && a.write;
    const Palette p = rw ? kReadWriteChip : (a.write ? kWriteChip : kReadChip);
    const std::string text = rw ? "RW" : (a.write ? "W" : "R");
    std::string body = font(9, a.write ? "white" : kMuted, "<B>" + text + "</B>");
    if (!a.stencil) {
      return html_box(p, "rounded", 34, 18, body);
    }
    body += "  " + font(7, a.write ? "#f8c471" : kStencilColor, "<B>stencil</B>");
    return html_box({p.fill, kStencilColor}, "rounded", 70, 18, body);
  };
  auto kernel_id = [](size_t i) {
    return "K" + std::to_string(i + 1U);
  };
  auto header = [&](int width, const std::string& title, const std::string& subtitle) {
    return "<TD BGCOLOR=\"" + std::string{kHeaderBg} + "\" WIDTH=\"" + std::to_string(width) +
           "\" HEIGHT=\"" + std::to_string(kRowH) + "\">" +
           font(11, "white", "<B>" + title + "</B>") + "<BR/>" +
           font(8, kHeaderSubtitle, subtitle) + "</TD>";
  };
  auto cell = [&](std::string_view bg, int width, const std::string& inner) {
    return "<TD BGCOLOR=\"" + std::string{bg} + "\" WIDTH=\"" + std::to_string(width) +
           "\" HEIGHT=\"" + std::to_string(kRowH) + "\">" + inner + "</TD>";
  };

  std::ostringstream os;
  os << "digraph " << graph_name << " {\n";
  os << "  graph [pad=0.3, bgcolor=\"white\", fontname=\"Helvetica\", labelloc=t, "
        "labeljust=l];\n";
  os << "  node [shape=none, margin=0, fontname=\"Helvetica\"];\n";

  const size_t legend_cols = 24;
  os << "  label=<<TABLE BORDER=\"0\" CELLSPACING=\"4\" CELLPADDING=\"1\">"
     << "<TR><TD COLSPAN=\"" << legend_cols << "\" ALIGN=\"LEFT\">"
     << font(15, {}, "<B>" + html_escape(graph_name) + "</B>")
     << font(10, kMuted,
             "   kernel list &#183; " + plural(n, "kernel") + " (" + std::to_string(num_auto) +
                 " auto-inserted) &#183; " + plural(resources.size(), "resource") +
                 " &#183; add order")
     << "</TD></TR>"
     << "<TR><TD COLSPAN=\"" << legend_cols << "\" ALIGN=\"LEFT\">"
     << font(10, kMuted, skeleton_options_summary(options_)) << "</TD></TR><TR>";
  os << legend_swatch(kStandardPal, "rounded", "compute");
  os << legend_swatch(kReductionPal, "rounded", "reduction");
  os << legend_swatch(kHaloPal, "rounded", "halo");
  os << legend_swatch(kCopyPal, "rounded", "readback / anchor");
  os << legend_swatch({"#ffffff", kMuted}, "dashed", "auto-inserted");
  for (const auto& [a, text] : {std::pair<Access, const char*>{{true, false, false}, "read"},
                                {{false, true, false}, "write"},
                                {{true, true, false}, "read-write"}}) {
    os << "<TD>" << access_chip(a) << "</TD><TD ALIGN=\"LEFT\">" << font(9, {}, text) << "</TD>";
  }
  os << "<TD>" << access_chip({true, false, true}) << "</TD><TD ALIGN=\"LEFT\">"
     << font(9, {}, "stencil read (needs halo)") << "</TD>";
  os << "</TR></TABLE>>;\n";

  os << "  kernels [label=<<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"3\" "
        "CELLPADDING=\"4\">\n";
  os << "    <TR><TD WIDTH=\"" << kIdW << "\" HEIGHT=\"" << kRowH << "\">"
     << font(9, kMuted, "<B>#</B>") << "</TD>" << header(kKernelW, "kernel", "user add order");
  for (const Resource& r : resources) {
    os << header(kResourceW, resource_label(r), resource_kind(r.kind));
  }
  os << header(kDepsW, "depends on", "reduced data deps")
     << header(kLaunchW, "launches", "# in run() &#183; stream &#183; view") << "</TR>\n";

  if (n == 0U) {
    os << "    <TR><TD COLSPAN=\"" << (4U + resources.size()) << "\" HEIGHT=\"" << kRowH << "\">"
       << font(9, kMuted, "<I>no kernels</I>") << "</TD></TR>\n";
  }
  for (size_t i = 0; i < n; ++i) {
    const std::string_view bg = kRowBg[(i + 1U) % 2U];
    const Kernel& k = effective_kernels_[i];
    const bool is_auto = effective_to_user_[i] == kNoKernel_;
    const std::string name =
        k.name().empty() ? "k" + std::to_string(i) : html_escape(std::string{k.name()});

    os << "    <TR>" << cell(bg, kIdW, font(9, kMuted, kernel_id(i)));
    os << cell(
        bg, kKernelW,
        html_box(
            kind_palette(k.kind()), is_auto ? "rounded,dashed" : "rounded", kKernelW - 16, 30,
            font(10, {}, "<B>" + name + "</B>") + "<BR/>" +
                font(8, kMuted, kind_name(k.kind()) + (is_auto ? " &#183; auto-inserted" : ""))));

    for (size_t c = 0; c < resources.size(); ++c) {
      const auto it = access[i].find(c);
      std::string inner;
      if (it != access[i].end()) {
        inner = access_chip(it->second);
      }
      os << cell(bg, kResourceW, inner);
    }

    std::string deps = font(10, kMuted, "&#8212;");
    if (!preds[i].empty()) {
      deps = "<TABLE BORDER=\"0\" CELLSPACING=\"3\" CELLPADDING=\"0\"><TR>";
      for (size_t p : preds[i]) {
        deps += "<TD>" +
                html_box(kind_palette(effective_kernels_[p].kind()), "rounded", 28, 16,
                         font(8, {}, "<B>" + kernel_id(p) + "</B>")) +
                "</TD>";
      }
      deps += "</TR></TABLE>";
    }
    os << cell(bg, kDepsW, deps);

    std::string launch_cell = font(10, kMuted, "&#8212;");
    if (!launches[i].empty()) {
      launch_cell = "<TABLE BORDER=\"0\" CELLSPACING=\"2\" CELLPADDING=\"0\">";
      for (const LaunchOrderOp& op : launches[i]) {
        const Step& step = schedule_[op.step_idx];
        const std::string view = step.reduce_launch_mode == reduce::ReduceLaunchMode::Merge
                                     ? "merge"
                                     : std::string{to_string(step.view)};
        launch_cell += "<TR><TD>" +
                       html_box(launch_palette(k.kind(), step.view), "rounded", kLaunchW - 24, 18,
                                font(8, {},
                                     "<B>#" + std::to_string(op.seq) + "</B> &#183; stream " +
                                         std::to_string(op.stream_idx) + " &#183; " + view)) +
                       "</TD></TR>";
      }
      launch_cell += "</TABLE>";
    }
    os << cell(bg, kLaunchW, launch_cell) << "</TR>\n";
  }
  os << "  </TABLE>>];\n";
  os << "}\n";
  return os.str();
}

auto
Skeleton::export_kernel_list_dot_file(std::string path,
                                      std::string graph_name) -> bool {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    return false;
  }
  out << export_kernel_list_dot(std::move(graph_name));
  return out.good();
}

auto
Skeleton::export_dot(std::string graph_name) -> std::string {
  compile();
  const std::vector<LaunchOrderOp> launch_ops = build_launch_order_ops_();
  std::unordered_map<size_t, size_t> launch_seq_by_step;
  std::unordered_map<size_t, size_t> record_seq_by_event;
  std::unordered_map<size_t, std::unordered_map<size_t, size_t>> wait_seq_by_event_step;
  for (const LaunchOrderOp& op : launch_ops) {
    switch (op.kind) {
      case LaunchOrderOpKind::Launch:
        launch_seq_by_step[op.step_idx] = op.seq;
        break;
      case LaunchOrderOpKind::EventRecord:
      case LaunchOrderOpKind::AnchorStartRecord:
      case LaunchOrderOpKind::AnchorJoinRecord:
        record_seq_by_event[op.event_idx] = op.seq;
        break;
      case LaunchOrderOpKind::EventWait:
      case LaunchOrderOpKind::AnchorStartWait:
      case LaunchOrderOpKind::AnchorJoinWait:
        wait_seq_by_event_step[op.event_idx][op.step_idx] = op.seq;
        break;
    }
  }

  auto fmt_mode = [](AccessMode m) -> const char* {
    switch (m) {
      case AccessMode::Read:
        return "R";
      case AccessMode::Write:
        return "W";
      case AccessMode::ReadWrite:
        return "RW";
    }
    return "?";
  };
  auto fmt_kind = [](KernelKind k) -> const char* {
    switch (k) {
      case KernelKind::Compute:
        return "compute";
      case KernelKind::Reduction:
        return "reduction";
      case KernelKind::Halo:
        return "halo";
      case KernelKind::Anchor:
        return "anchor";
      case KernelKind::ComputeHost:
        return "compute_host";
      case KernelKind::ReductionHost:
        return "reduction_host";
      case KernelKind::Universal:
        return "universal";
      case KernelKind::UniversalReduction:
        return "universal_reduction";
      case KernelKind::ScalarReadback:
        return "scalar_readback";
    }
    return "?";
  };
  std::unordered_map<std::uint64_t, std::string> uid_to_label;
  for (const Kernel& k : effective_kernels_) {
    for (const AccessToken& t : k.impl()->tokens()) {
      if (!t.field_name.empty()) {
        uid_to_label[t.field_uid] = t.field_name;
      }
    }
  }
  for (const auto& [uid, label] : field_label_overrides_) {
    uid_to_label[uid] = label;
  }
  auto field_label = [&uid_to_label](const AccessToken& t) -> std::string {
    auto it = uid_to_label.find(t.field_uid);
    if (it != uid_to_label.end() && !it->second.empty()) {
      return it->second;
    }
    return std::string{"<uid:"} + std::to_string(t.field_uid) + ">";
  };
  auto escape = [](std::string_view s) -> std::string {
    std::string out;
    out.reserve(s.size() + 4U);
    for (char c : s) {
      if (c == '"' || c == '\\') {
        out.push_back('\\');
      }
      out.push_back(c);
    }
    return out;
  };

  std::ostringstream os;
  os << "digraph " << graph_name << " {\n";
  os << "  rankdir=TB;\n";
  os << "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\"];\n";
  os << "  edge [fontname=\"Helvetica\", fontsize=10];\n";

  if (schedule_.empty()) {
    os << "  /* empty schedule */\n";
    os << "}\n";
    return os.str();
  }

  std::vector<std::vector<size_t>> steps_by_stream(streams_.size());
  for (size_t si = 0; si < schedule_.size(); ++si) {
    const size_t stream_idx = schedule_[si].stream_idx;
    if (stream_idx >= steps_by_stream.size()) {
      continue;
    }
    steps_by_stream[stream_idx].push_back(si);
  }

  for (size_t s = 0; s < streams_.size(); ++s) {
    const auto& bucket = steps_by_stream[s];
    if (bucket.empty()) {
      continue;
    }
    const std::string_view color = kStreamColors[s % kStreamColors.size()];
    os << "  subgraph cluster_stream_" << s << " {\n";
    if (s == copy_stream_idx_) {
      os << "    label=\"copy stream (detached, scalar readback)\";\n";
    } else {
      os << "    label=\"stream " << s << "\";\n";
    }
    os << "    style=\"rounded,dashed\";\n";
    os << "    color=\"#888888\";\n";
    os << "    bgcolor=\"" << color << "\";\n";
    for (size_t si : bucket) {
      const Step& step = schedule_[si];
      const Kernel& kernel = effective_kernels_[step.kernel_idx];
      const std::string_view raw_name = kernel.name();
      const std::string label = raw_name.empty()
                                    ? (std::string{"k"} + std::to_string(step.kernel_idx))
                                    : std::string{raw_name};
      const auto launch_it = launch_seq_by_step.find(si);
      const size_t launch_seq = launch_it != launch_seq_by_step.end() ? launch_it->second : 0U;
      os << "    s" << si << " [label=\"";
      if (launch_seq != 0U) {
        os << '#' << launch_seq << ' ';
      }
      os << escape(label) << "\\n[" << fmt_kind(kernel.kind()) << "/" << to_string(step.view)
         << "]\"";
      const auto& tokens = kernel.impl()->tokens();
      if (!tokens.empty()) {
        os << ", tooltip=\"";
        os << escape(label) << " tokens: ";
        for (size_t i = 0; i < tokens.size(); ++i) {
          if (i > 0) {
            os << ", ";
          }
          os << escape(field_label(tokens[i])) << ":" << fmt_mode(tokens[i].mode);
        }
        os << "\"";
      }
      os << "];\n";
    }
    os << "  }\n";
  }

  for (size_t si = 1; si < schedule_.size(); ++si) {
    os << "  s" << (si - 1U) << " -> s" << si << " [label=\"order\", color=\"#666666\"];\n";
  }

  std::vector<size_t> event_recorder(events_.size(), kNoKernel_);
  for (size_t si = 0; si < schedule_.size(); ++si) {
    const size_t evt = schedule_[si].record_event_idx;
    if (evt != kNoEvent_ && evt < event_recorder.size()) {
      event_recorder[evt] = si;
    }
  }
  auto dep_label = [&](size_t pred_step, size_t succ_step) -> std::string {
    const Kernel& pred_k = effective_kernels_[schedule_[pred_step].kernel_idx];
    const Kernel& succ_k = effective_kernels_[schedule_[succ_step].kernel_idx];
    for (const AccessToken& ta : pred_k.impl()->tokens()) {
      for (const AccessToken& tb : succ_k.impl()->tokens()) {
        if (ta.field_uid == tb.field_uid && conflicts(ta.mode, tb.mode)) {
          return std::string{field_label(ta)};
        }
      }
    }
    return "wait";
  };
  for (size_t sj = 0; sj < schedule_.size(); ++sj) {
    for (size_t evt : schedule_[sj].wait_events) {
      if (evt < event_recorder.size() && event_recorder[evt] != kNoKernel_) {
        const size_t si = event_recorder[evt];
        std::string edge_label = dep_label(si, sj);
        const auto rec_it = record_seq_by_event.find(evt);
        const auto wait_outer = wait_seq_by_event_step.find(evt);
        if (rec_it != record_seq_by_event.end() && wait_outer != wait_seq_by_event_step.end()) {
          const auto wait_it = wait_outer->second.find(sj);
          if (wait_it != wait_outer->second.end()) {
            edge_label += " (rec #" + std::to_string(rec_it->second) + " -> wait #" +
                          std::to_string(wait_it->second) + ')';
          }
        }
        os << "  s" << si << " -> s" << sj << " [label=\"" << escape(edge_label)
           << "\", style=dashed, color=\"#aa3333\"];\n";
      }
    }
  }

  if (uses_anchor_fork_join()) {
    std::vector<size_t> first_step_on_stream(streams_.size(), kNoKernel_);
    std::vector<size_t> last_step_on_stream(streams_.size(), kNoKernel_);
    for (size_t si = 0; si < schedule_.size(); ++si) {
      const size_t st = schedule_[si].stream_idx;
      if (st >= first_step_on_stream.size()) {
        continue;
      }
      if (first_step_on_stream[st] == kNoKernel_) {
        first_step_on_stream[st] = si;
      }
      last_step_on_stream[st] = si;
    }

    os << "  anchor_start [shape=point, width=0.01, label=\"\", fixedsize=true];\n";
    os << "  anchor_join [shape=point, width=0.01, label=\"\", fixedsize=true];\n";

    if (anchor_start_event_ != kNoEvent_) {
      const auto rec_it = record_seq_by_event.find(anchor_start_event_);
      const size_t rec_seq = rec_it != record_seq_by_event.end() ? rec_it->second : 0U;
      for (size_t s = 1U; s < streams_.size(); ++s) {
        if (s == copy_stream_idx_) {
          continue;
        }
        size_t wait_seq = 0U;
        for (const LaunchOrderOp& wop : launch_ops) {
          if (wop.kind == LaunchOrderOpKind::AnchorStartWait && wop.stream_idx == s) {
            wait_seq = wop.seq;
            break;
          }
        }
        if (first_step_on_stream[s] != kNoKernel_ && wait_seq != 0U) {
          os << "  anchor_start -> s" << first_step_on_stream[s] << " [label=\"anchor start (rec #"
             << rec_seq << " -> wait #" << wait_seq << ")\", style=dashed, color=\"#336699\"];\n";
        }
      }
    }

    for (size_t s = 1U; s < streams_.size(); ++s) {
      if (s == copy_stream_idx_ || s >= anchor_join_events_.size()) {
        continue;
      }
      const size_t join_evt = anchor_join_events_[s];
      if (join_evt == kNoEvent_) {
        continue;
      }
      const auto rec_it = record_seq_by_event.find(join_evt);
      size_t wait_seq = 0U;
      for (const LaunchOrderOp& wop : launch_ops) {
        if (wop.kind == LaunchOrderOpKind::AnchorJoinWait && wop.event_idx == join_evt) {
          wait_seq = wop.seq;
          break;
        }
      }
      const size_t rec_seq = rec_it != record_seq_by_event.end() ? rec_it->second : 0U;
      if (last_step_on_stream[s] != kNoKernel_ && rec_seq != 0U && wait_seq != 0U) {
        os << "  s" << last_step_on_stream[s] << " -> anchor_join [label=\"anchor join (rec #"
           << rec_seq << " -> wait #" << wait_seq << ")\", style=dashed, color=\"#336699\"];\n";
      }
    }
  }

  os << "}\n";
  return os.str();
}

auto
Skeleton::export_dot_file(std::string path,
                          std::string graph_name) -> bool {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    return false;
  }
  out << export_dot(std::move(graph_name));
  return out.good();
}

auto
Skeleton::export_dot_png(std::string png_path,
                         std::string graph_name) -> bool {
  std::string dot_path = png_path;
  const size_t slash = dot_path.find_last_of("/\\");
  const size_t dot = dot_path.find_last_of('.');
  if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
    dot_path.replace(dot, dot_path.size() - dot, ".dot");
  } else {
    dot_path += ".dot";
  }
  if (!export_dot_file(dot_path, graph_name)) {
    return false;
  }
  const std::string cmd = "dot -Tpng \"" + dot_path + "\" -o \"" + png_path + "\" >/dev/null 2>&1";
  return std::system(cmd.c_str()) == 0;
}

}  // namespace carbon
