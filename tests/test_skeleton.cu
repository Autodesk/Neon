// carbon::Skeleton end-to-end correctness — token-driven dependency analysis +
// multi-stream static schedule replay.
//
// Two complementary subtests:
//
//   A. Sequential pipeline (Jacobi ping-pong, halo + compute interleaved).
//      Verifies the inferred adjacency matches the expected ping-pong DAG,
//      the schedule lands on a single stream (no parallelism available
//      without the Internal/Boundary auto-overlap), and the final field is
//      bit-for-bit equal to a manual reference loop.
//
//   B. Parallel-sibling pipeline (copy_uv + copy_xy + combine_uvxy).
//      copy_uv and copy_xy share no fields and run concurrently; combine
//      reads from both and waits via a cross-stream event. Verifies the
//      inferred edges, the per-kernel stream assignment, and bit-for-bit
//      parity across multiple iterations against a single-stream reference
//      (which exercises the anchor fork-join the multi-stream backend
//      brackets every run with).
//
// Common corner-case coverage: lazy compile via run(); reset() clears all
// state; empty-pipeline run is a no-op; anchor-only DAG has no edges.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/halo.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

namespace {

constexpr int kCycles = 10;  // Each cycle = halo_a + step_ab + halo_b + step_ba (2 applications).
constexpr int kRefIters = 2 * kCycles;
constexpr float kSentinel = -7777.0F;

[[nodiscard]] CARBON_HD auto
encode(int64_t x,
       int64_t y,
       int64_t gz) -> float {
  return static_cast<float>(x + y * 11 + gz * 113);
}

template <typename Fn>
auto
fill_storage_host(const carbon::Dense& dense,
                  carbon::Dense::Field<float,
                                       1>& f,
                  Fn fn) -> void {
  for (size_t g = 0; g < dense.gpu_count(); ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t z_origin = span.z_offset();
    float* h = f.cpu_buffers()[g].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      const int64_t gz = z_origin + z;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          h[off] = fn(x, y, gz);
        }
      }
    }
  }
}

auto
init_with_sentinel_halo(const carbon::Dense& dense,
                        carbon::Dense::Field<float,
                                             1>& f) -> void {
  fill_storage_host(dense, f, [](int64_t, int64_t, int64_t) {
    return kSentinel;
  });
  for (size_t g = 0; g < dense.gpu_count(); ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    float* h = f.cpu_buffers()[g].data();
    for (int64_t z = halo_z; z < halo_z + lim_z; ++z) {
      const int64_t gz = z_origin + z;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          h[off] = encode(x, y, gz);
        }
      }
    }
  }
}

[[nodiscard]] auto
sync_stream(const carbon::Stream& s,
            const char* tag) -> bool {
  try {
    s.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton: sync after {} failed: {}", tag, e.what());
    return false;
  }
  return true;
}

[[nodiscard]] auto
build_copy_kernel(carbon::Dense& dense,
                  carbon::Dense::Field<float,
                                       1>& fin,
                  carbon::Dense::Field<float,
                                       1>& fout,
                  std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&fin, &fout](carbon::KernelLoader& loader) {
        const auto in = loader.read(fin);
        const auto out = loader.write(fout);
        return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          out.write(s, idx, 0, in.read(s, idx, 0));
        };
      },
      std::move(name),
      /*shared_mem=*/0U);
}

[[nodiscard]] auto
build_combine_kernel(carbon::Dense& dense,
                     carbon::Dense::Field<float,
                                          1>& fu,
                     carbon::Dense::Field<float,
                                          1>& fv,
                     carbon::Dense::Field<float,
                                          1>& fx,
                     carbon::Dense::Field<float,
                                          1>& fy,
                     carbon::Dense::Field<float,
                                          1>& fz,
                     std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&fu, &fv, &fx, &fy, &fz](carbon::KernelLoader& loader) {
        const auto u = loader.read(fu);
        const auto v = loader.read(fv);
        const auto x = loader.read(fx);
        const auto y = loader.read(fy);
        const auto z = loader.write(fz);
        return [u, v, x, y, z] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          const float total =
              u.read(s, idx, 0) + v.read(s, idx, 0) + x.read(s, idx, 0) + y.read(s, idx, 0);
          z.write(s, idx, 0, total);
        };
      },
      std::move(name),
      /*shared_mem=*/0U);
}

[[nodiscard]] auto
build_jacobi_step(carbon::Dense& dense,
                  carbon::Dense::Field<float,
                                       1>& fin,
                  carbon::Dense::Field<float,
                                       1>& fout,
                  std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&fin, &fout](carbon::KernelLoader& loader) {
        const auto in = loader.read(fin);
        const auto out = loader.write(fout);
        return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          const carbon::NeighbourValue<float> xp = in.neighbour(s, idx, {1, 0, 0}, 0);
          const carbon::NeighbourValue<float> xm = in.neighbour(s, idx, {-1, 0, 0}, 0);
          const carbon::NeighbourValue<float> yp = in.neighbour(s, idx, {0, 1, 0}, 0);
          const carbon::NeighbourValue<float> ym = in.neighbour(s, idx, {0, -1, 0}, 0);
          const carbon::NeighbourValue<float> zp = in.neighbour(s, idx, {0, 0, 1}, 0);
          const carbon::NeighbourValue<float> zm = in.neighbour(s, idx, {0, 0, -1}, 0);
          const float self = in.read(s, idx, 0);
          const bool all_valid =
              xp.valid && xm.valid && yp.valid && ym.valid && zp.valid && zm.valid;
          if (!all_valid) {
            out.write(s, idx, 0, self);
            return;
          }
          const float sum = xp.value + xm.value + yp.value + ym.value + zp.value + zm.value;
          out.write(s, idx, 0, sum * (1.0F / 6.0F));
        };
      },
      std::move(name),
      /*shared_mem=*/0U);
}

[[nodiscard]] auto
upload_field(const carbon::System& sys,
             carbon::Dense::Field<float,
                                  1>& f,
             const carbon::Stream& s) -> bool {
  try {
    carbon::Transfer<float>(sys, f.cpu_buffers(), f.gpu_buffers())(s);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton: upload failed: {}", e.what());
    return false;
  }
  return true;
}

[[nodiscard]] auto
download_field(const carbon::System& sys,
               carbon::Dense::Field<float,
                                    1>& f,
               const carbon::Stream& s) -> bool {
  try {
    carbon::Transfer<float>(sys, f.gpu_buffers(), f.cpu_buffers())(s);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton: download failed: {}", e.what());
    return false;
  }
  return true;
}

// Fill every owned cell of @p f with @p value via the host mirror; assumes the
// field was built with MemorySide::Host or HostDevice. Used by the parallel-
// sibling subtest which only cares about a single deterministic per-cell value.
auto
fill_owned_constant(const carbon::Dense& dense,
                    carbon::Dense::Field<float,
                                         1>& f,
                    float value) -> void {
  fill_storage_host(dense, f, [](int64_t, int64_t, int64_t) {
    return kSentinel;
  });
  for (size_t g = 0; g < dense.gpu_count(); ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t lim_z = span.logic_extent().z;
    float* h = f.cpu_buffers()[g].data();
    for (int64_t z = halo_z; z < halo_z + lim_z; ++z) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          h[off] = value;
        }
      }
    }
  }
}

[[nodiscard]] auto
compare_fields(const carbon::Dense& dense,
               const carbon::Dense::Field<float,
                                          1>& a,
               const carbon::Dense::Field<float,
                                          1>& b) -> bool {
  for (size_t g = 0; g < dense.gpu_count(); ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t lim_z = span.logic_extent().z;
    const int64_t z_origin = span.z_offset();
    const float* ha = a.cpu_buffers()[g].data();
    const float* hb = b.cpu_buffers()[g].data();
    for (int64_t z = halo_z; z < halo_z + lim_z; ++z) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          if (ha[off] != hb[off]) {
            CARBON_LOG_ERROR(
                "test_skeleton: mismatch at gpu={} (x={}, y={}, gz={}): test={} ref={}", g, x, y,
                z_origin + z, ha[off], hb[off]);
            return false;
          }
        }
      }
    }
  }
  return true;
}

// ---------- Parallel-sibling subtest ----------
//
// Three independent kernels:
//   copy_uv  : v <- u                       (R u, W v)
//   copy_xy  : y <- x                       (R x, W y)
//   combine  : z <- u + v + x + y           (R u, R v, R x, R y, W z)
//
// copy_uv and copy_xy share no fields, so the inferred DAG has no edge
// between them and they run on different streams. combine touches v and y
// (which copy_uv / copy_xy just wrote) so it must wait on both — a
// cross-stream wait gets emitted on the stream combine ends up on (here:
// stream 0, extending copy_uv's chain).
//
// We run kIters iterations through the skeleton on the test path, and the
// same kernels in user-add order on a single reference stream on the
// reference path, then compare z cell-by-cell. The anchor fork-join that
// brackets every run() is the part of the schedule under test here —
// without it, iteration k+1's combine on stream 0 might launch before
// iteration k's copy_xy on stream 1 lands its writes.
[[nodiscard]] auto
run_parallel_sibling_subtest(carbon::System& sys,
                             carbon::Dense& dense) -> bool {
  constexpr int kSiblingIters = 5;

  // Test path: name every field so the DOT export below has labelled edges.
  auto u = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u");
  auto v = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v");
  auto x = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "x");
  auto y = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "y");
  auto z = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "z");
  // Reference path: chainable set_name covers the post-hoc API surface.
  auto u_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice).set_name("u_ref");
  auto v_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice).set_name("v_ref");
  auto x_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice).set_name("x_ref");
  auto y_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice).set_name("y_ref");
  auto z_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice).set_name("z_ref");

  if (u.name() != "u" || v.name() != "v" || x.name() != "x" || y.name() != "y" || z.name() != "z") {
    CARBON_LOG_ERROR("test_skeleton: Field::name() round-trip failed for ctor-named fields");
    return false;
  }
  if (u_ref.name() != "u_ref" || z_ref.name() != "z_ref") {
    CARBON_LOG_ERROR("test_skeleton: Field::set_name() round-trip failed");
    return false;
  }

  // Distinct constants so a copy bug or a wrong-field combine would surface
  // immediately on the host comparison rather than aliasing into a correct-
  // looking sum.
  fill_owned_constant(dense, u, 1.0F);
  fill_owned_constant(dense, v, -123.0F);  // overwritten by copy_uv on every iter
  fill_owned_constant(dense, x, 2.0F);
  fill_owned_constant(dense, y, -456.0F);  // overwritten by copy_xy on every iter
  fill_owned_constant(dense, z, kSentinel);
  fill_owned_constant(dense, u_ref, 1.0F);
  fill_owned_constant(dense, v_ref, -123.0F);
  fill_owned_constant(dense, x_ref, 2.0F);
  fill_owned_constant(dense, y_ref, -456.0F);
  fill_owned_constant(dense, z_ref, kSentinel);

  carbon::Stream s_test = sys.stream();
  carbon::Stream s_ref = sys.stream();
  if (!upload_field(sys, u, s_test) || !upload_field(sys, v, s_test) ||
      !upload_field(sys, x, s_test) || !upload_field(sys, y, s_test) ||
      !upload_field(sys, z, s_test) || !upload_field(sys, u_ref, s_ref) ||
      !upload_field(sys, v_ref, s_ref) || !upload_field(sys, x_ref, s_ref) ||
      !upload_field(sys, y_ref, s_ref) || !upload_field(sys, z_ref, s_ref) ||
      !sync_stream(s_test, "sibling test upload") || !sync_stream(s_ref, "sibling ref upload")) {
    return false;
  }

  // Reference path: same three kernels, single stream, sequential.
  {
    auto k_uv = build_copy_kernel(dense, u_ref, v_ref, "ref_uv");
    auto k_xy = build_copy_kernel(dense, x_ref, y_ref, "ref_xy");
    auto k_combine = build_combine_kernel(dense, u_ref, v_ref, x_ref, y_ref, z_ref, "ref_combine");
    for (int it = 0; it < kSiblingIters; ++it) {
      try {
        k_uv.launch(s_ref, carbon::DataView::Standard);
        k_xy.launch(s_ref, carbon::DataView::Standard);
        k_combine.launch(s_ref, carbon::DataView::Standard);
      } catch (const std::exception& e) {
        CARBON_LOG_ERROR("test_skeleton: sibling ref launch failed at iter {}: {}", it, e.what());
        return false;
      }
    }
    if (!sync_stream(s_ref, "sibling ref iters")) {
      return false;
    }
  }

  // Test path: skeleton with three kernels.
  carbon::Skeleton sk(sys, s_test);
  sk.add(build_copy_kernel(dense, u, v, "uv"));
  sk.add(build_copy_kernel(dense, x, y, "xy"));
  sk.add(build_combine_kernel(dense, u, v, x, y, z, "combine"));

  sk.compile();

  const std::vector<std::vector<size_t>> expected_edges = {{2U}, {2U}, {}};
  if (sk.edges() != expected_edges) {
    CARBON_LOG_ERROR("test_skeleton: sibling edges mismatch — expected {{2}, {{2}}, {{}}}");
    return false;
  }
  // Sibling DAG is already minimal: combine has two independent direct
  // predecessors (0 and 1) with no path between them, so the transitive
  // reduction equals the full conflict graph.
  if (sk.min_edges() != expected_edges) {
    CARBON_LOG_ERROR(
        "test_skeleton: sibling min_edges mismatch — expected {{2}, {{2}}, {{}}} (already "
        "minimal)");
    return false;
  }

  const std::vector<size_t> expected_streams = {0U, 1U, 0U};
  if (sk.kernel_streams() != expected_streams) {
    for (size_t i = 0; i < sk.kernel_streams().size(); ++i) {
      CARBON_LOG_ERROR("test_skeleton: sibling kernel_streams[{}] = {}, expected {}", i,
                       sk.kernel_streams()[i], expected_streams[i]);
    }
    return false;
  }
  if (sk.num_streams() != 2U) {
    CARBON_LOG_ERROR("test_skeleton: sibling num_streams() = {}, expected 2", sk.num_streams());
    return false;
  }

  // ---------- DOT export ----------
  //
  // The exporter is an inspection / diagnostics surface, not a perf path, so we
  // assert on substring presence rather than parse the full GraphViz syntax. We
  // want to see:
  //   - the digraph header with the user-supplied graph name;
  //   - per-stream cluster scopes (subgraph cluster_stream_0 / 1);
  //   - all three kernel labels (uv, xy, combine) with DataView tags;
  //   - schedule execution-order edges and cross-stream wait styling.
  const std::string dot = sk.export_dot("sibling_pipeline");
  auto contains = [&dot](std::string_view needle) -> bool {
    return dot.find(needle) != std::string::npos;
  };

  for (std::string_view needle :
       {"digraph sibling_pipeline {", "subgraph cluster_stream_0", "subgraph cluster_stream_1",
        "label=\"uv", "label=\"xy", "label=\"combine", "label=\"order\"", "s0 -> s1", "s1 -> s2"}) {
    if (!contains(needle)) {
      CARBON_LOG_ERROR("test_skeleton: DOT export missing expected substring [{}]; got:\n{}",
                       needle, dot);
      return false;
    }
  }
  if (dot.find("style=dashed") == std::string::npos || !contains("label=\"wait\"")) {
    CARBON_LOG_ERROR("test_skeleton: DOT export missing cross-stream wait edge styling; got:\n{}",
                     dot);
    return false;
  }

  for (int it = 0; it < kSiblingIters; ++it) {
    try {
      sk.run();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_skeleton: sibling run failed at iter {}: {}", it, e.what());
      return false;
    }
  }
  sk.wait();

  if (!download_field(sys, z, s_test) || !download_field(sys, z_ref, s_ref) ||
      !sync_stream(s_test, "sibling download test") ||
      !sync_stream(s_ref, "sibling download ref")) {
    return false;
  }
  if (!compare_fields(dense, z, z_ref)) {
    return false;
  }
  return true;
}

// ---------- Auto-halo + Options subtest ----------
//
// Three concerns at once:
//
//   A. KernelLoader::read_stencil() records AccessPattern::Stencil and a
//      non-null halo factory on the AccessToken. Verified by inspecting the
//      kernel's tokens directly (no Skeleton involved).
//   B. Skeleton::compile auto-inserts a halo before every Stencil-read of a
//      "dirty" field (one written since the last halo). The auto-halo
//      kernel appears in effective_kernels(), labelled "auto_halo_<field>".
//   C. Skeleton::Options{strict_halo=true} flips the auto-insert into a
//      throw at compile time, so users who want every halo to be explicit
//      can opt in.
//   D. Skeleton::Options{overlap_mode=OverlapMode::Standard} (and the other
//      non-None modes) drives the OCC graph rewrite in
//      Skeleton::apply_occ_ — halo stream isolation + Internal/Boundary
//      split for stencil-read kernels. The parallel-sibling and overlap
//      subtests below cover the full Standard / Extended / TwoWayExtended
//      matrix; this auto-halo subtest is structural-only.
//
// We don't run() the auto-halo'd skeleton end-to-end here because the
// Jacobi-style end-to-end semantics are already covered by the main subtest
// (with manually-added halos); this subtest is a structural check that the
// auto-insertion + Options machinery does the right thing at compile time.
[[nodiscard]] auto
run_auto_halo_subtest(carbon::System& sys,
                      carbon::Dense& dense) -> bool {
  auto u = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_auto");
  auto v = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v_auto");

  // ----- A. read_stencil records pattern + factory -----
  auto k_seed = build_copy_kernel(dense, u, v, "seed");  // W v from R u
  auto k_stencil = build_jacobi_step(dense, v, u, "jac_v_to_u_via_stencil");
  // build_jacobi_step uses loader.read(); rebuild via a thin wrapper that
  // calls loader.read_stencil() so the token reflects the stencil pattern.
  auto k_stencil_explicit = carbon::Kernel::compute(
      dense,
      [&v, &u](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(v);
        const auto out = loader.write(u);
        return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          out.write(s, idx, 0, in.read(s, idx, 0));
        };
      },
      "jac_v_to_u_via_stencil",
      /*shared_mem=*/0U);

  const auto& tokens = k_stencil_explicit.impl()->tokens();
  bool found_stencil_token = false;
  for (const carbon::AccessToken& t : tokens) {
    if (t.field_uid == v.gpu_buffers().get_uid()) {
      if (t.pattern != carbon::AccessPattern::Stencil) {
        CARBON_LOG_ERROR("test_skeleton: read_stencil token for v_auto missing Stencil pattern");
        return false;
      }
      if (!t.halo_factory) {
        CARBON_LOG_ERROR("test_skeleton: read_stencil token for v_auto carries null halo_factory");
        return false;
      }
      if (t.field_name != "v_auto") {
        CARBON_LOG_ERROR("test_skeleton: read_stencil token field_name = '{}', expected 'v_auto'",
                         t.field_name);
        return false;
      }
      found_stencil_token = true;
    }
  }
  if (!found_stencil_token) {
    CARBON_LOG_ERROR("test_skeleton: no Stencil-pattern token recorded for v_auto");
    return false;
  }

  // ----- B. Auto-halo insertion under default Options -----
  {
    carbon::Skeleton sk(sys);  // default options: overlap_mode=None, strict_halo=false
    sk.add(build_copy_kernel(dense, u, v, "seed_uv"));
    sk.add(carbon::Kernel::compute(
        dense,
        [&v, &u](carbon::KernelLoader& loader) {
          const auto in = loader.read_stencil(v);
          const auto out = loader.write(u);
          return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
            out.write(s, idx, 0, in.read(s, idx, 0));
          };
        },
        "stencil_v_to_u",
        /*shared_mem=*/0U));

    try {
      sk.compile();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_skeleton: auto-halo compile threw: {}", e.what());
      return false;
    }

    // Effective kernels: seed_uv (idx 0), auto_halo_v_auto (idx 1), stencil_v_to_u (idx 2).
    if (sk.effective_kernels().size() != 3U) {
      CARBON_LOG_ERROR("test_skeleton: auto-halo expected effective_kernels().size()==3, got {}",
                       sk.effective_kernels().size());
      return false;
    }
    if (sk.effective_kernels()[1].kind() != carbon::KernelKind::Halo) {
      CARBON_LOG_ERROR("test_skeleton: auto-halo expected effective_kernels()[1] to be Halo");
      return false;
    }
    if (sk.effective_to_user()[1] != carbon::Skeleton::kNoKernel) {
      CARBON_LOG_ERROR(
          "test_skeleton: auto-halo expected effective_to_user[1] == kNoKernel (synthetic)");
      return false;
    }
    if (sk.effective_kernels()[1].name() != "auto_halo_v_auto") {
      CARBON_LOG_ERROR("test_skeleton: auto-halo name = '{}', expected 'auto_halo_v_auto'",
                       std::string{sk.effective_kernels()[1].name()});
      return false;
    }
  }

  // ----- B'. No auto-halo when not needed (no preceding write since last halo) -----
  {
    carbon::Skeleton sk(sys);
    // Manually add halo first, then stencil-read — should NOT add a second halo.
    sk.add(v.make_halo_kernel("manual_halo_v"));
    sk.add(carbon::Kernel::compute(
        dense,
        [&v, &u](carbon::KernelLoader& loader) {
          const auto in = loader.read_stencil(v);
          const auto out = loader.write(u);
          return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
            out.write(s, idx, 0, in.read(s, idx, 0));
          };
        },
        "stencil_after_manual_halo",
        /*shared_mem=*/0U));

    sk.compile();
    if (sk.effective_kernels().size() != 2U) {
      CARBON_LOG_ERROR("test_skeleton: auto-halo dedup expected size==2 (no extra insert), got {}",
                       sk.effective_kernels().size());
      return false;
    }
  }

  // ----- B''. Halo dedup across a 3-stencil chain on the same field -----
  // Sequence: write v ; stencil_read v -> write u1 ; stencil_read v -> write
  // u2 ; stencil_read v -> write u3. The dirty bit on v is set by the
  // initial write, cleared by the auto-halo before stencil 1, and stays
  // clean for stencils 2 and 3 (none of them write v). Expected
  // effective_kernels: [seed, halo_v, stencil1, stencil2, stencil3] —
  // size 5, exactly one Halo kernel inserted, and it sits at index 1.
  // Without the dedup the count would be 7 (3 separate halos) and the
  // schedule would over-serialise the three stencils on a single stream.
  {
    auto u1 = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u1_chain");
    auto u2 = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u2_chain");
    auto u3 = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u3_chain");

    auto build_stencil_to = [&dense, &v](carbon::Dense::Field<float, 1>& dst,
                                         std::string name) -> carbon::Kernel {
      return carbon::Kernel::compute(
          dense,
          [&v, &dst](carbon::KernelLoader& loader) {
            const auto in = loader.read_stencil(v);
            const auto out = loader.write(dst);
            return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
              out.write(s, idx, 0, in.read(s, idx, 0));
            };
          },
          std::move(name),
          /*shared_mem=*/0U);
    };

    carbon::Skeleton sk(sys);
    sk.add(build_copy_kernel(dense, u, v, "chain_seed_v"));
    sk.add(build_stencil_to(u1, "chain_stencil_1"));
    sk.add(build_stencil_to(u2, "chain_stencil_2"));
    sk.add(build_stencil_to(u3, "chain_stencil_3"));

    sk.compile();

    if (sk.effective_kernels().size() != 5U) {
      CARBON_LOG_ERROR(
          "test_skeleton: 3-stencil halo-dedup expected effective_kernels().size()==5 (one halo "
          "shared across all three reads), got {} (would have been 7 without dedup)",
          sk.effective_kernels().size());
      return false;
    }
    size_t halo_count = 0;
    for (const auto& ek : sk.effective_kernels()) {
      if (ek.kind() == carbon::KernelKind::Halo) {
        ++halo_count;
      }
    }
    if (halo_count != 1U) {
      CARBON_LOG_ERROR(
          "test_skeleton: 3-stencil halo-dedup expected exactly 1 Halo kernel in schedule, got {}",
          halo_count);
      return false;
    }
    if (sk.effective_kernels()[1].kind() != carbon::KernelKind::Halo) {
      CARBON_LOG_ERROR(
          "test_skeleton: 3-stencil halo-dedup expected the halo at index 1 (right after the "
          "seed write), got kind={} at index 1",
          static_cast<int>(sk.effective_kernels()[1].kind()));
      return false;
    }
  }

  // ----- C. strict_halo throws on missing halo -----
  {
    carbon::Skeleton sk(sys, carbon::Skeleton::Options{carbon::OverlapMode::None,
                                                       /*strict=*/true});
    sk.add(build_copy_kernel(dense, u, v, "seed_uv2"));
    sk.add(carbon::Kernel::compute(
        dense,
        [&v, &u](carbon::KernelLoader& loader) {
          const auto in = loader.read_stencil(v);
          const auto out = loader.write(u);
          return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
            out.write(s, idx, 0, in.read(s, idx, 0));
          };
        },
        "stencil_no_halo",
        /*shared_mem=*/0U));

    bool threw = false;
    try {
      sk.compile();
    } catch (const std::logic_error&) {
      threw = true;
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_skeleton: strict_halo threw wrong exception type: {}", e.what());
      return false;
    }
    if (!threw) {
      CARBON_LOG_ERROR(
          "test_skeleton: strict_halo=true expected to throw on stencil-read without halo");
      return false;
    }
  }

  // ----- D. OverlapMode::Standard splits the stencil and moves halo to a fresh stream -----
  {
    carbon::Skeleton sk(sys,
                        carbon::Skeleton::Options{carbon::OverlapMode::Standard, /*strict=*/false});
    sk.add(build_copy_kernel(dense, u, v, "seed_uv_std"));
    sk.add(carbon::Kernel::compute(
        dense,
        [&v, &u](carbon::KernelLoader& loader) {
          const auto in = loader.read_stencil(v);
          const auto out = loader.write(u);
          return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
            out.write(s, idx, 0, in.read(s, idx, 0));
          };
        },
        "stencil_std",
        /*shared_mem=*/0U));

    try {
      sk.compile();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_skeleton: Standard OCC compile threw: {}", e.what());
      return false;
    }
    // effective: [seed_uv_std, auto_halo_v_auto, stencil_std].
    // Schedule should have 4 entries because stencil_std splits into Internal + Boundary.
    if (sk.effective_kernels().size() != 3U) {
      CARBON_LOG_ERROR("test_skeleton: Standard OCC effective_kernels size = {}, expected 3",
                       sk.effective_kernels().size());
      return false;
    }
    // Halo (effective idx 1) should be on a different stream than the stencil (idx 2).
    const auto& streams = sk.kernel_streams();
    if (streams[1] == streams[2]) {
      CARBON_LOG_ERROR(
          "test_skeleton: Standard OCC halo (stream {}) should differ from stencil (stream {})",
          streams[1], streams[2]);
      return false;
    }
    if (sk.num_streams() < 2U) {
      CARBON_LOG_ERROR("test_skeleton: Standard OCC expected num_streams >= 2, got {}",
                       sk.num_streams());
      return false;
    }
  }

  // ----- E. Standard OCC end-to-end parity vs OverlapMode::None -----
  {
    auto u_none = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_none");
    auto u_std = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_std");
    auto u_next_none = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_next_none");
    auto u_next_std = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_next_std");

    init_with_sentinel_halo(dense, u_none);
    init_with_sentinel_halo(dense, u_std);
    fill_storage_host(dense, u_next_none, [](int64_t, int64_t, int64_t) {
      return kSentinel;
    });
    fill_storage_host(dense, u_next_std, [](int64_t, int64_t, int64_t) {
      return kSentinel;
    });

    carbon::Stream s_none = sys.stream();
    carbon::Stream s_std = sys.stream();
    if (!upload_field(sys, u_none, s_none) || !upload_field(sys, u_std, s_std) ||
        !upload_field(sys, u_next_none, s_none) || !upload_field(sys, u_next_std, s_std) ||
        !sync_stream(s_none, "OCC parity upload none") ||
        !sync_stream(s_std, "OCC parity upload std")) {
      return false;
    }

    auto build_stencil = [&](carbon::Dense::Field<float, 1>& fin,
                             carbon::Dense::Field<float, 1>& fout, std::string name) {
      return carbon::Kernel::compute(
          dense,
          [&fin, &fout](carbon::KernelLoader& loader) {
            const auto in = loader.read_stencil(fin);
            const auto out = loader.write(fout);
            return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
              const carbon::NeighbourValue<float> xp = in.neighbour(s, idx, {1, 0, 0}, 0);
              const carbon::NeighbourValue<float> xm = in.neighbour(s, idx, {-1, 0, 0}, 0);
              const carbon::NeighbourValue<float> yp = in.neighbour(s, idx, {0, 1, 0}, 0);
              const carbon::NeighbourValue<float> ym = in.neighbour(s, idx, {0, -1, 0}, 0);
              const carbon::NeighbourValue<float> zp = in.neighbour(s, idx, {0, 0, 1}, 0);
              const carbon::NeighbourValue<float> zm = in.neighbour(s, idx, {0, 0, -1}, 0);
              const float self = in.read(s, idx, 0);
              const bool all_valid =
                  xp.valid && xm.valid && yp.valid && ym.valid && zp.valid && zm.valid;
              if (!all_valid) {
                out.write(s, idx, 0, self);
                return;
              }
              const float sum = xp.value + xm.value + yp.value + ym.value + zp.value + zm.value;
              out.write(s, idx, 0, sum * (1.0F / 6.0F));
            };
          },
          std::move(name),
          /*shared_mem=*/0U);
    };

    constexpr int kOccIters = 6;
    carbon::Skeleton sk_none(sys, s_none, carbon::Skeleton::Options{carbon::OverlapMode::None});
    sk_none.add(build_stencil(u_none, u_next_none, "none_ab"));
    sk_none.add(build_stencil(u_next_none, u_none, "none_ba"));

    carbon::Skeleton sk_std(sys, s_std, carbon::Skeleton::Options{carbon::OverlapMode::Standard});
    sk_std.add(build_stencil(u_std, u_next_std, "std_ab"));
    sk_std.add(build_stencil(u_next_std, u_std, "std_ba"));

    try {
      for (int it = 0; it < kOccIters; ++it) {
        sk_none.run();
        sk_std.run();
      }
      sk_none.wait();
      sk_std.wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_skeleton: Standard OCC parity run threw: {}", e.what());
      return false;
    }

    if (!download_field(sys, u_none, s_none) || !download_field(sys, u_std, s_std) ||
        !sync_stream(s_none, "OCC parity download none") ||
        !sync_stream(s_std, "OCC parity download std")) {
      return false;
    }
    if (!compare_fields(dense, u_none, u_std)) {
      CARBON_LOG_ERROR("test_skeleton: Standard OCC parity vs None failed");
      return false;
    }
  }

  // ----- F. Extended OCC end-to-end parity vs OverlapMode::None -----
  // Pipeline: copy(seed -> work) ; stencil(work -> out). The copy is a
  // Map predecessor that writes the field stencil-read by the next step,
  // so Extended's Map-back walk should find and split it. Parity must
  // still match the unsplit reference.
  {
    auto seed_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "seed_a");
    auto seed_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "seed_b");
    auto work_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "work_a");
    auto work_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "work_b");
    auto out_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "out_a");
    auto out_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "out_b");

    init_with_sentinel_halo(dense, seed_a);
    init_with_sentinel_halo(dense, seed_b);
    fill_storage_host(dense, work_a, [](int64_t, int64_t, int64_t) {
      return kSentinel;
    });
    fill_storage_host(dense, work_b, [](int64_t, int64_t, int64_t) {
      return kSentinel;
    });
    fill_storage_host(dense, out_a, [](int64_t, int64_t, int64_t) {
      return kSentinel;
    });
    fill_storage_host(dense, out_b, [](int64_t, int64_t, int64_t) {
      return kSentinel;
    });

    carbon::Stream s_a = sys.stream();
    carbon::Stream s_b = sys.stream();
    if (!upload_field(sys, seed_a, s_a) || !upload_field(sys, seed_b, s_b) ||
        !upload_field(sys, work_a, s_a) || !upload_field(sys, work_b, s_b) ||
        !upload_field(sys, out_a, s_a) || !upload_field(sys, out_b, s_b) ||
        !sync_stream(s_a, "Ext parity upload a") || !sync_stream(s_b, "Ext parity upload b")) {
      return false;
    }

    auto build_stencil_avg = [&](carbon::Dense::Field<float, 1>& fin,
                                 carbon::Dense::Field<float, 1>& fout, std::string name) {
      return carbon::Kernel::compute(
          dense,
          [&fin, &fout](carbon::KernelLoader& loader) {
            const auto in = loader.read_stencil(fin);
            const auto out = loader.write(fout);
            return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
              const carbon::NeighbourValue<float> xp = in.neighbour(s, idx, {1, 0, 0}, 0);
              const carbon::NeighbourValue<float> xm = in.neighbour(s, idx, {-1, 0, 0}, 0);
              const carbon::NeighbourValue<float> yp = in.neighbour(s, idx, {0, 1, 0}, 0);
              const carbon::NeighbourValue<float> ym = in.neighbour(s, idx, {0, -1, 0}, 0);
              const carbon::NeighbourValue<float> zp = in.neighbour(s, idx, {0, 0, 1}, 0);
              const carbon::NeighbourValue<float> zm = in.neighbour(s, idx, {0, 0, -1}, 0);
              const float self = in.read(s, idx, 0);
              const bool all_valid =
                  xp.valid && xm.valid && yp.valid && ym.valid && zp.valid && zm.valid;
              if (!all_valid) {
                out.write(s, idx, 0, self);
                return;
              }
              const float sum = xp.value + xm.value + yp.value + ym.value + zp.value + zm.value;
              out.write(s, idx, 0, sum * (1.0F / 6.0F));
            };
          },
          std::move(name),
          /*shared_mem=*/0U);
    };

    constexpr int kExtIters = 4;
    carbon::Skeleton sk_a(sys, s_a, carbon::Skeleton::Options{carbon::OverlapMode::None});
    sk_a.add(build_copy_kernel(dense, seed_a, work_a, "ext_copy_none"));
    sk_a.add(build_stencil_avg(work_a, out_a, "ext_stencil_none"));

    carbon::Skeleton sk_b(sys, s_b, carbon::Skeleton::Options{carbon::OverlapMode::Extended});
    sk_b.add(build_copy_kernel(dense, seed_b, work_b, "ext_copy_ext"));
    sk_b.add(build_stencil_avg(work_b, out_b, "ext_stencil_ext"));

    try {
      for (int it = 0; it < kExtIters; ++it) {
        sk_a.run();
        sk_b.run();
      }
      sk_a.wait();
      sk_b.wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_skeleton: Extended OCC parity run threw: {}", e.what());
      return false;
    }

    if (!download_field(sys, out_a, s_a) || !download_field(sys, out_b, s_b) ||
        !sync_stream(s_a, "Ext parity download a") || !sync_stream(s_b, "Ext parity download b")) {
      return false;
    }
    if (!compare_fields(dense, out_a, out_b)) {
      CARBON_LOG_ERROR("test_skeleton: Extended OCC parity vs None failed");
      return false;
    }
  }

  // ----- G. TwoWayExtended OCC end-to-end parity vs OverlapMode::None -----
  // Pipeline: copy(seed -> work) ; stencil(work -> mid) ; copy(mid ->
  // out). The first copy is a Map-back candidate (writes what the
  // stencil Stencil-reads); the second copy is a Map-forward candidate
  // (reads what the stencil writes). TwoWayExtended should split all
  // three (copy_pre, stencil, copy_post) into Internal + Boundary
  // halves and still preserve byte-for-byte parity vs the unsplit
  // reference.
  {
    auto seed_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "seed2_a");
    auto seed_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "seed2_b");
    auto work_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "work2_a");
    auto work_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "work2_b");
    auto mid_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "mid2_a");
    auto mid_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "mid2_b");
    auto out_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "out2_a");
    auto out_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "out2_b");

    init_with_sentinel_halo(dense, seed_a);
    init_with_sentinel_halo(dense, seed_b);
    for (auto* f : {&work_a, &work_b, &mid_a, &mid_b, &out_a, &out_b}) {
      fill_storage_host(dense, *f, [](int64_t, int64_t, int64_t) {
        return kSentinel;
      });
    }

    carbon::Stream s_a = sys.stream();
    carbon::Stream s_b = sys.stream();
    if (!upload_field(sys, seed_a, s_a) || !upload_field(sys, seed_b, s_b) ||
        !upload_field(sys, work_a, s_a) || !upload_field(sys, work_b, s_b) ||
        !upload_field(sys, mid_a, s_a) || !upload_field(sys, mid_b, s_b) ||
        !upload_field(sys, out_a, s_a) || !upload_field(sys, out_b, s_b) ||
        !sync_stream(s_a, "TWE parity upload a") || !sync_stream(s_b, "TWE parity upload b")) {
      return false;
    }

    auto build_stencil_avg = [&](carbon::Dense::Field<float, 1>& fin,
                                 carbon::Dense::Field<float, 1>& fout, std::string name) {
      return carbon::Kernel::compute(
          dense,
          [&fin, &fout](carbon::KernelLoader& loader) {
            const auto in = loader.read_stencil(fin);
            const auto out = loader.write(fout);
            return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
              const carbon::NeighbourValue<float> xp = in.neighbour(s, idx, {1, 0, 0}, 0);
              const carbon::NeighbourValue<float> xm = in.neighbour(s, idx, {-1, 0, 0}, 0);
              const carbon::NeighbourValue<float> yp = in.neighbour(s, idx, {0, 1, 0}, 0);
              const carbon::NeighbourValue<float> ym = in.neighbour(s, idx, {0, -1, 0}, 0);
              const carbon::NeighbourValue<float> zp = in.neighbour(s, idx, {0, 0, 1}, 0);
              const carbon::NeighbourValue<float> zm = in.neighbour(s, idx, {0, 0, -1}, 0);
              const float self = in.read(s, idx, 0);
              const bool all_valid =
                  xp.valid && xm.valid && yp.valid && ym.valid && zp.valid && zm.valid;
              if (!all_valid) {
                out.write(s, idx, 0, self);
                return;
              }
              const float sum = xp.value + xm.value + yp.value + ym.value + zp.value + zm.value;
              out.write(s, idx, 0, sum * (1.0F / 6.0F));
            };
          },
          std::move(name),
          /*shared_mem=*/0U);
    };

    constexpr int kTweIters = 4;
    carbon::Skeleton sk_a(sys, s_a, carbon::Skeleton::Options{carbon::OverlapMode::None});
    sk_a.add(build_copy_kernel(dense, seed_a, work_a, "twe_pre_none"));
    sk_a.add(build_stencil_avg(work_a, mid_a, "twe_stencil_none"));
    sk_a.add(build_copy_kernel(dense, mid_a, out_a, "twe_post_none"));

    carbon::Skeleton sk_b(sys, s_b, carbon::Skeleton::Options{carbon::OverlapMode::TwoWayExtended});
    sk_b.add(build_copy_kernel(dense, seed_b, work_b, "twe_pre_two"));
    sk_b.add(build_stencil_avg(work_b, mid_b, "twe_stencil_two"));
    sk_b.add(build_copy_kernel(dense, mid_b, out_b, "twe_post_two"));

    try {
      for (int it = 0; it < kTweIters; ++it) {
        sk_a.run();
        sk_b.run();
      }
      sk_a.wait();
      sk_b.wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_skeleton: TwoWayExtended OCC parity run threw: {}", e.what());
      return false;
    }

    if (!download_field(sys, out_a, s_a) || !download_field(sys, out_b, s_b) ||
        !sync_stream(s_a, "TWE parity download a") || !sync_stream(s_b, "TWE parity download b")) {
      return false;
    }
    if (!compare_fields(dense, out_a, out_b)) {
      CARBON_LOG_ERROR("test_skeleton: TwoWayExtended OCC parity vs None failed");
      return false;
    }
  }

  return true;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_skeleton: no CUDA devices visible");
    return 1;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{6, 5, z_per_slab * static_cast<int64_t>(n_gpus)};
  carbon::Dense dense(sys, global, std::move(stencil));

  auto u_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice);
  auto u_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice);
  auto u_ref_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice);
  auto u_ref_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice);

  init_with_sentinel_halo(dense, u_a);
  init_with_sentinel_halo(dense, u_ref_a);
  fill_storage_host(dense, u_b, [](int64_t, int64_t, int64_t) {
    return kSentinel;
  });
  fill_storage_host(dense, u_ref_b, [](int64_t, int64_t, int64_t) {
    return kSentinel;
  });

  carbon::Stream s_ref = sys.stream();
  carbon::Stream s_test = sys.stream();

  if (!upload_field(sys, u_a, s_test) || !upload_field(sys, u_b, s_test) ||
      !upload_field(sys, u_ref_a, s_ref) || !upload_field(sys, u_ref_b, s_ref) ||
      !sync_stream(s_test, "test upload") || !sync_stream(s_ref, "ref upload")) {
    return 1;
  }

  // ---------- Reference path: manual halo + compute loop, single stream ----------
  {
    auto step_ref_ab = build_jacobi_step(dense, u_ref_a, u_ref_b, "ref_step_ab");
    auto step_ref_ba = build_jacobi_step(dense, u_ref_b, u_ref_a, "ref_step_ba");
    carbon::Halo halo_ref_a(u_ref_a);
    carbon::Halo halo_ref_b(u_ref_b);
    for (int it = 0; it < kRefIters; ++it) {
      auto& halo_in = (it % 2 == 0) ? halo_ref_a : halo_ref_b;
      auto& step = (it % 2 == 0) ? step_ref_ab : step_ref_ba;
      halo_in(s_ref);
      try {
        step.launch(s_ref, carbon::DataView::Standard);
      } catch (const std::exception& e) {
        CARBON_LOG_ERROR("test_skeleton: ref launch failed at iter {}: {}", it, e.what());
        return 1;
      }
    }
    if (!sync_stream(s_ref, "ref iterations")) {
      return 1;
    }
  }

  // ---------- Test path: Skeleton, ping-pong cycle as 4 nodes per iteration ----------
  carbon::Skeleton sk(sys, s_test);
  sk.add(u_a.make_halo_kernel("halo_a"));
  sk.add(build_jacobi_step(dense, u_a, u_b, "step_ab"));
  sk.add(u_b.make_halo_kernel("halo_b"));
  sk.add(build_jacobi_step(dense, u_b, u_a, "step_ba"));

  if (sk.compiled()) {
    CARBON_LOG_ERROR("test_skeleton: expected !compiled() before run/compile");
    return 1;
  }
  if (sk.size() != 4U) {
    CARBON_LOG_ERROR("test_skeleton: expected size()==4, got {}", sk.size());
    return 1;
  }

  sk.compile();
  if (!sk.compiled()) {
    CARBON_LOG_ERROR("test_skeleton: expected compiled() after compile()");
    return 1;
  }

  // Expected DAG (see test header):
  //   0:halo_a    -> {1, 3}   (touches u_a → conflicts with step_ab and step_ba)
  //   1:step_ab   -> {2, 3}   (touches u_b → conflicts with halo_b; touches u_a → step_ba)
  //   2:halo_b    -> {3}      (touches u_b → conflicts with step_ba)
  //   3:step_ba   -> {}
  const auto& edges = sk.edges();
  if (edges.size() != 4U) {
    CARBON_LOG_ERROR("test_skeleton: expected edges.size()==4, got {}", edges.size());
    return 1;
  }
  const std::vector<std::vector<size_t>> expected_edges = {{1U, 3U}, {2U, 3U}, {3U}, {}};
  if (edges != expected_edges) {
    for (size_t i = 0; i < edges.size(); ++i) {
      std::string row;
      for (size_t j : edges[i]) {
        row += std::to_string(j) + ",";
      }
      CARBON_LOG_ERROR("test_skeleton: edges[{}] = [{}]", i, row);
    }
    return 1;
  }

  // Jacobi ping-pong reduces to a linear chain — every transitive edge
  // (0 -> 3 implied by 0 -> 1 -> 2 -> 3, 1 -> 3 implied by 1 -> 2 -> 3)
  // drops out of min_edges() leaving consecutive predecessors only.
  const std::vector<std::vector<size_t>> expected_min = {{1U}, {2U}, {3U}, {}};
  if (sk.min_edges() != expected_min) {
    for (size_t i = 0; i < sk.min_edges().size(); ++i) {
      std::string row;
      for (size_t j : sk.min_edges()[i]) {
        row += std::to_string(j) + ",";
      }
      CARBON_LOG_ERROR("test_skeleton: jacobi min_edges[{}] = [{}]", i, row);
    }
    return 1;
  }

  // Jacobi ping-pong is fully serial — every step has the previous step as a
  // direct predecessor, so the greedy stream assignment extends stream 0 each
  // time and never opens stream 1.
  const std::vector<size_t> expected_streams = {0U, 0U, 0U, 0U};
  if (sk.kernel_streams() != expected_streams) {
    for (size_t i = 0; i < sk.kernel_streams().size(); ++i) {
      CARBON_LOG_ERROR("test_skeleton: jacobi kernel_streams[{}] = {}, expected 0", i,
                       sk.kernel_streams()[i]);
    }
    return 1;
  }
  if (sk.num_streams() != 1U) {
    CARBON_LOG_ERROR(
        "test_skeleton: jacobi num_streams() = {}, expected 1 (single-stream "
        "fast path)",
        sk.num_streams());
    return 1;
  }

  for (int it = 0; it < kCycles; ++it) {
    try {
      sk.run();  // run() returns immediately; work queued on s_test
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_skeleton: skeleton run failed at cycle {}: {}", it, e.what());
      return 1;
    }
  }
  sk.wait();

  // Download both final fields and compare cell-by-cell.
  if (!download_field(sys, u_a, s_test) || !download_field(sys, u_ref_a, s_ref) ||
      !sync_stream(s_test, "download test") || !sync_stream(s_ref, "download ref")) {
    return 1;
  }
  if (!compare_fields(dense, u_a, u_ref_a)) {
    return 1;
  }

  // ---------- reset() smoke test ----------
  sk.reset();
  if (!sk.empty() || sk.size() != 0U || sk.compiled() || !sk.edges().empty() ||
      !sk.min_edges().empty() || !sk.kernel_streams().empty()) {
    CARBON_LOG_ERROR("test_skeleton: reset() did not clear state cleanly");
    return 1;
  }

  // ---------- Empty-pipeline run is a no-op ----------
  try {
    sk.run();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton: empty-pipeline run threw: {}", e.what());
    return 1;
  }

  // ---------- Adding a single anchor + compute keeps a clean schedule ----------
  sk.add(carbon::Kernel::anchor("begin"));
  if (sk.compiled()) {
    CARBON_LOG_ERROR("test_skeleton: add() should have invalidated compile state");
    return 1;
  }
  sk.compile();
  if (sk.edges().size() != 1U || !sk.edges()[0].empty()) {
    CARBON_LOG_ERROR("test_skeleton: anchor-only DAG should have no edges");
    return 1;
  }

  // ---------- Parallel-sibling subtest ----------
  if (!run_parallel_sibling_subtest(sys, dense)) {
    CARBON_LOG_ERROR("test_skeleton: parallel-sibling subtest failed");
    return 1;
  }

  // ---------- Auto-halo + Options subtest ----------
  if (!run_auto_halo_subtest(sys, dense)) {
    CARBON_LOG_ERROR("test_skeleton: auto-halo subtest failed");
    return 1;
  }

  // ---------- reset() + recompile stream-stability subtest ----------
  // Locks the contract that running an OCC-rewriting pipeline twice
  // through the same skeleton (reset() in between) does not grow
  // streams_ monotonically. Without the OCC unused-stream-reuse path
  // each cycle would push a fresh halo stream onto streams_ and the
  // count would blow up over long-lived skeletons that periodically
  // re-build their pipeline (e.g. adaptive solvers that swap kernels
  // between phases).
  {
    auto u_rs = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "rs_u");
    auto v_rs = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "rs_v");
    init_with_sentinel_halo(dense, u_rs);
    init_with_sentinel_halo(dense, v_rs);
    carbon::Stream s_rs = sys.stream();
    if (!upload_field(sys, u_rs, s_rs) || !upload_field(sys, v_rs, s_rs) ||
        !sync_stream(s_rs, "reset-stream-stability upload")) {
      return 1;
    }

    carbon::Skeleton sk_rs(sys, s_rs, carbon::Skeleton::Options{carbon::OverlapMode::Standard});

    auto build_pipe = [&]() {
      sk_rs.add(carbon::Kernel::compute(
          dense,
          [&u_rs, &v_rs](carbon::KernelLoader& loader) {
            const auto in = loader.read_stencil(u_rs);
            const auto out = loader.write(v_rs);
            return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
              out.write(s, idx, 0, in.read(s, idx, 0));
            };
          },
          "rs_stencil_uv",
          /*shared_mem=*/0U));
    };

    build_pipe();
    sk_rs.compile();
    const size_t streams_after_first = sk_rs.num_streams();

    sk_rs.reset();
    build_pipe();
    sk_rs.compile();
    const size_t streams_after_second = sk_rs.num_streams();

    if (streams_after_second != streams_after_first) {
      CARBON_LOG_ERROR(
          "test_skeleton: reset()+recompile changed streams_.size(): {} -> {} (must be stable)",
          streams_after_first, streams_after_second);
      return 1;
    }

    sk_rs.reset();
    build_pipe();
    sk_rs.compile();
    if (sk_rs.num_streams() != streams_after_first) {
      CARBON_LOG_ERROR(
          "test_skeleton: reset()+recompile (3rd cycle) changed streams_.size(): {} -> {}",
          streams_after_first, sk_rs.num_streams());
      return 1;
    }
  }

  // ---------- Options.max_streams subtest ----------
  // A parallel-sibling pipeline (two independent copies + a join) uses
  // 2 streams when uncapped. With max_streams=1 the second sibling is
  // forced onto stream 0 with cross-stream events disabled (FIFO covers
  // the ordering); the schedule should report exactly 1 stream and
  // still produce the same final state as the uncapped reference.
  {
    auto u_cap = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_cap");
    auto v_cap = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v_cap");
    auto x_cap = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "x_cap");
    auto y_cap = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "y_cap");
    auto z_cap = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "z_cap");

    auto u_unc = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_unc");
    auto v_unc = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v_unc");
    auto x_unc = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "x_unc");
    auto y_unc = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "y_unc");
    auto z_unc = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "z_unc");

    init_with_sentinel_halo(dense, u_cap);
    init_with_sentinel_halo(dense, x_cap);
    init_with_sentinel_halo(dense, u_unc);
    init_with_sentinel_halo(dense, x_unc);
    for (auto* f : {&v_cap, &y_cap, &z_cap, &v_unc, &y_unc, &z_unc}) {
      fill_storage_host(dense, *f, [](int64_t, int64_t, int64_t) {
        return kSentinel;
      });
    }

    carbon::Stream s_cap = sys.stream();
    carbon::Stream s_unc = sys.stream();
    if (!upload_field(sys, u_cap, s_cap) || !upload_field(sys, v_cap, s_cap) ||
        !upload_field(sys, x_cap, s_cap) || !upload_field(sys, y_cap, s_cap) ||
        !upload_field(sys, z_cap, s_cap) || !upload_field(sys, u_unc, s_unc) ||
        !upload_field(sys, v_unc, s_unc) || !upload_field(sys, x_unc, s_unc) ||
        !upload_field(sys, y_unc, s_unc) || !upload_field(sys, z_unc, s_unc) ||
        !sync_stream(s_cap, "max_streams cap upload") ||
        !sync_stream(s_unc, "max_streams unc upload")) {
      return 1;
    }

    carbon::Skeleton sk_cap(sys, s_cap,
                            carbon::Skeleton::Options{carbon::OverlapMode::None,
                                                      /*strict=*/false,
                                                      /*max_streams=*/1U});
    sk_cap.add(build_copy_kernel(dense, u_cap, v_cap, "cap_uv"));
    sk_cap.add(build_copy_kernel(dense, x_cap, y_cap, "cap_xy"));
    sk_cap.add(build_combine_kernel(dense, u_cap, v_cap, x_cap, y_cap, z_cap, "cap_combine"));

    carbon::Skeleton sk_unc(sys, s_unc, carbon::Skeleton::Options{});
    sk_unc.add(build_copy_kernel(dense, u_unc, v_unc, "unc_uv"));
    sk_unc.add(build_copy_kernel(dense, x_unc, y_unc, "unc_xy"));
    sk_unc.add(build_combine_kernel(dense, u_unc, v_unc, x_unc, y_unc, z_unc, "unc_combine"));

    sk_cap.compile();
    sk_unc.compile();

    if (sk_cap.num_streams() != 1U) {
      CARBON_LOG_ERROR("test_skeleton: max_streams=1 expected num_streams()==1, got {}",
                       sk_cap.num_streams());
      return 1;
    }
    if (sk_unc.num_streams() < 2U) {
      CARBON_LOG_ERROR("test_skeleton: uncapped sibling pipeline expected num_streams()>=2, got {}",
                       sk_unc.num_streams());
      return 1;
    }

    try {
      sk_cap.run();
      sk_cap.wait();
      sk_unc.run();
      sk_unc.wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_skeleton: max_streams cap/uncapped run threw: {}", e.what());
      return 1;
    }

    if (!download_field(sys, z_cap, s_cap) || !download_field(sys, z_unc, s_unc) ||
        !sync_stream(s_cap, "max_streams cap download") ||
        !sync_stream(s_unc, "max_streams unc download")) {
      return 1;
    }
    if (!compare_fields(dense, z_cap, z_unc)) {
      CARBON_LOG_ERROR("test_skeleton: max_streams=1 vs uncapped final state mismatch");
      return 1;
    }
  }

  // ---------- DOT-export label refresh subtest ----------
  // AccessToken::field_name is captured by copy at KernelLoader record
  // time, so renaming a Field via Field::set_name AFTER add() does NOT
  // retroactively rewrite existing tokens. Two paths handle this:
  //   - The "rename then add MORE kernels" path is automatic — the new
  //     tokens carry the new name and export_dot's uid_to_label map
  //     prefers the most recent non-empty name.
  //   - The "rename AFTER all kernels are added" path goes through
  //     Skeleton::set_field_label as the explicit override.
  // This subtest covers both.
  {
    auto fa = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "fa_initial");
    auto fb = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "fb_initial");

    carbon::Skeleton sk_dot(sys);
    sk_dot.add(build_copy_kernel(dense, fa, fb, "rename_seed"));

    // Path 1: rename before adding the next kernel — token snapshot wins.
    fa.set_name("fa_renamed_via_set_name");
    fb.set_name("fb_renamed_via_set_name");
    sk_dot.add(build_copy_kernel(dense, fb, fa, "rename_swap"));

    const std::string dot_path1 = sk_dot.export_dot();
    if (dot_path1.find("fa_renamed_via_set_name") == std::string::npos ||
        dot_path1.find("fb_renamed_via_set_name") == std::string::npos) {
      CARBON_LOG_ERROR(
          "test_skeleton: DOT export missing renamed-then-added label (token-snapshot path "
          "did not refresh)");
      return 1;
    }

    // Path 2: explicit override after all kernels added.
    sk_dot.set_field_label(fa.gpu_buffers().get_uid(), "fa_explicit_override");
    sk_dot.set_field_label(fb.gpu_buffers().get_uid(), "fb_explicit_override");
    const std::string dot_path2 = sk_dot.export_dot();
    if (dot_path2.find("fa_explicit_override") == std::string::npos ||
        dot_path2.find("fb_explicit_override") == std::string::npos) {
      CARBON_LOG_ERROR("test_skeleton: DOT export missing set_field_label override");
      return 1;
    }
    if (dot_path2.find("fa_renamed_via_set_name") != std::string::npos ||
        dot_path2.find("fb_renamed_via_set_name") != std::string::npos) {
      CARBON_LOG_ERROR(
          "test_skeleton: DOT export still showing token-snapshot label after override "
          "applied");
      return 1;
    }

    // Empty-string clears the override (back to token-snapshot path).
    sk_dot.set_field_label(fa.gpu_buffers().get_uid(), "");
    const std::string dot_path3 = sk_dot.export_dot();
    if (dot_path3.find("fa_renamed_via_set_name") == std::string::npos) {
      CARBON_LOG_ERROR(
          "test_skeleton: clearing the override (empty label) did not fall back to "
          "token-snapshot label");
      return 1;
    }
  }

  CARBON_LOG_INFO(
      "test_skeleton: OK — {} cycles ({} stencil applications) across {} GPU(s); ping-pong DAG "
      "matches expectation and final field is bit-for-bit identical to the manual reference; "
      "parallel-sibling pipeline assigned to 2 streams with cross-stream + anchor fork-join sync "
      "validated against single-stream reference; auto-halo + strict_halo + OverlapMode gate "
      "all behave as documented",
      kCycles, kRefIters, n_gpus);
  return 0;
}
