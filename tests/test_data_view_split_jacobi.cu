// Multi-stream Jacobi with the Internal / Boundary DataView split overlapping halo exchange.
//
// What this test exercises
// ------------------------
// The orthogonal piece to test_data_view_split.cu: there we proved that
// "Internal + Boundary == Standard" within a single iteration on a single stream. Here we
// run a real iterative stencil (radius-1 7-point Jacobi smoothing) under the canonical
// overlap pattern that the data-view split exists for:
//
//   For each iteration i:
//     halo_stream:    halo(u_in)              // brings up-to-date neighbour-slab data into halos
//     compute_stream: kernel(u_in -> u_out, Internal)   // interior cells, no halo dependency
//     event evt_halo    : recorded on halo_stream after halo(...)
//     compute_stream waits on evt_halo
//     compute_stream: kernel(u_in -> u_out, Boundary)   // boundary cells, depend on halos
//     event evt_compute : recorded on compute_stream after Boundary
//     halo_stream waits on evt_compute (next iteration's halo of the swapped u_in)
//     swap (u_in, u_out)
//
// We compare the test path's final field against a reference path that runs the same kernel
// but with a single Standard launch per iteration on its own stream, halo exchange first.
// Both paths use the same loading lambda, the same stencil, and the same input data, so the
// math is deterministic and the comparison is bit-for-bit. Any missed sync between the two
// streams (Internal reading old u_in too early, halo reading half-written u_in, Boundary
// reading a not-yet-exchanged halo, etc.) would diverge the test field from the reference.
//
// This works in single-GPU runs too — Halo is a no-op when there are no neighbour slabs
// on this rank, so the test reduces to "Internal + Boundary on two streams with proper
// event synchronisation produces the same field as Standard on one stream", which still
// covers the orchestration mechanics (events, stream waits, swap).

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/halo.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/stencil.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

namespace {

constexpr int kIters = 20;  // Even, so the final field after ping-pong lands back in slot A.
static_assert(kIters % 2 == 0,
              "kIters must be even so the final result lives in slot A");

constexpr float kSentinel = -7777.0F;

// Per-global-cell initial value. Bounded well within float-exact integer range for the
// extents below, so a few Jacobi iterations don't accumulate enough rounding for the test
// path to drift away from the reference path through inexact-add reordering. (We don't
// rely on float-exactness here; the kernel reduces neighbours in a fixed order on both
// paths, so the comparison is bit-for-bit either way.)
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

[[nodiscard]] auto
sync_stream(const carbon::Stream& s,
            const char* tag) -> bool {
  try {
    s.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_data_view_split_jacobi: sync after {} failed: {}", tag, e.what());
    return false;
  }
  return true;
}

// Record the per-GPU event on the matching per-GPU stream. Carbon's `Stream` and `Event` are
// per-GPU groups; the per-entry record is just the natural CUDA / SYCL primitive applied
// device-by-device — `EventGroup::record(StreamGroup)` does the dispatch.
[[nodiscard]] auto
record(carbon::Event& evt,
       const carbon::Stream& s,
       const char* tag) -> bool {
  try {
    evt.record(s);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_data_view_split_jacobi: Event::record({}) failed: {}", tag, e.what());
    return false;
  }
  return true;
}

// Make `s` wait for `evt`. Carbon's @c StreamGroup::wait_for issues the per-GPU
// cross-stream barrier (cudaStreamWaitEvent under CUDA, AdaptiveCpp's depends_on host_task
// under SYCL); on-device only, no host-side sync.
[[nodiscard]] auto
wait(const carbon::Stream& s,
     const carbon::Event& evt,
     const char* tag) -> bool {
  try {
    s.wait_for(evt);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_data_view_split_jacobi: Stream::wait_for({}) failed: {}", tag, e.what());
    return false;
  }
  return true;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_data_view_split_jacobi: no CUDA devices visible");
    return 1;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  // Radius-1 face stencil so halo_z = 1; the kernel below reads exactly these six neighbours.
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  if (stencil.radius() != 1) {
    CARBON_LOG_ERROR("test_data_view_split_jacobi: unexpected stencil radius {}", stencil.radius());
    return 1;
  }

  // Per-slab z big enough for a meaningful Internal band (Lz - 2*halo_z = 2 layers minimum
  // at z_per_slab = 4) and small enough to keep the bit-for-bit verification cheap.
  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{6, 5, z_per_slab * static_cast<int64_t>(n_gpus)};

  // Default Open boundary mode: out-of-domain neighbour reads on the global top/bottom
  // (and x/y faces) return valid = false, and the kernel below falls back to "copy" there
  // — Dirichlet-style frozen boundary cells, no exterior halo data needed.
  carbon::Dense dense(sys, global, std::move(stencil));

  // Two ping-pong slots per path; HostDevice so we can fill on the host, upload, and
  // download the final field for the host-side comparison.
  auto u_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice);
  auto u_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice);
  auto u_ref_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice);
  auto u_ref_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice);

  // Initial state: owned cells start at encode(x, y, global_z); halo cells at sentinel.
  // Interior halos get overwritten by the first iteration's halo exchange; exterior halos
  // remain at sentinel because Open mode never writes them, and the kernel never
  // dereferences them either (Partition::neighbour returns valid = false out of domain).
  // Both paths must start from byte-identical state for the bit-for-bit comparison to be
  // a clean equivalence check.
  auto init_owned = [](int64_t x, int64_t y, int64_t gz) {
    return encode(x, y, gz);
  };
  auto init_with_sentinel_halo = [&](carbon::Dense::Field<float, 1>& f) {
    // First fill everything with sentinel, then overwrite owned cells with the encoder
    // — saves a per-cell halo / owned branch in the lambda.
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
            h[off] = init_owned(x, y, gz);
          }
        }
      }
    }
  };
  init_with_sentinel_halo(u_a);
  init_with_sentinel_halo(u_ref_a);
  // Slot B starts entirely sentinel — it's the iteration-0 output and gets fully written
  // by the first kernel pass on owned cells. Halo cells in slot B will be filled by iter-1
  // halo exchange (when slot B is the input).
  fill_storage_host(dense, u_b, [](int64_t, int64_t, int64_t) {
    return kSentinel;
  });
  fill_storage_host(dense, u_ref_b, [](int64_t, int64_t, int64_t) {
    return kSentinel;
  });

  // One Halo descriptor per ping-pong slot per path. We pick which one to fire based on
  // the iteration's input slot. Halo is bound to the field's underlying buffers via
  // shared_ptr, so building once and reusing across all iterations is correct.
  carbon::Halo halo_a(u_a);
  carbon::Halo halo_b(u_b);
  carbon::Halo halo_ref_a(u_ref_a);
  carbon::Halo halo_ref_b(u_ref_b);

  // 7-point Jacobi smoothing kernel with Dirichlet-style "freeze on boundary" fallback:
  // global x/y/z faces read at least one out-of-domain neighbour (Open mode), the
  // neighbour returns valid = false, and we copy the input value through. Interior owned
  // cells average all six neighbours.
  auto build_step = [&](carbon::Dense::Field<float, 1>& fin, carbon::Dense::Field<float, 1>& fout) {
    return dense.make_kernel(
        [&fin, &fout](carbon::KernelLoader& loader) {
          const auto in = loader(fin);
          const auto out = loader(fout);
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
        /*shared_mem=*/0U);
  };
  auto step_ab = build_step(u_a, u_b);
  auto step_ba = build_step(u_b, u_a);
  auto step_ref_ab = build_step(u_ref_a, u_ref_b);
  auto step_ref_ba = build_step(u_ref_b, u_ref_a);

  carbon::Stream s_ref = sys.stream();      // reference path: single stream
  carbon::Stream s_compute = sys.stream();  // test path: kernels (Internal + Boundary)
  carbon::Stream s_halo = sys.stream();     // test path: halo exchange
  carbon::Event evt_halo = sys.event();     // record after halo,    wait on s_compute
  carbon::Event evt_compute = sys.event();  // record after Boundary, wait on s_halo (next iter)

  // Upload all four fields on s_ref, then host-sync so every other stream sees the data.
  // This is the single host-device fence in the test; from here on everything is async on
  // device-side streams synchronised through events.
  try {
    carbon::Transfer<float>(sys, u_ref_a.cpu_buffers(), u_ref_a.gpu_buffers())(s_ref);
    carbon::Transfer<float>(sys, u_ref_b.cpu_buffers(), u_ref_b.gpu_buffers())(s_ref);
    carbon::Transfer<float>(sys, u_a.cpu_buffers(), u_a.gpu_buffers())(s_ref);
    carbon::Transfer<float>(sys, u_b.cpu_buffers(), u_b.gpu_buffers())(s_ref);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_data_view_split_jacobi: upload failed: {}", e.what());
    return 1;
  }
  if (!sync_stream(s_ref, "upload")) {
    return 1;
  }

  // ---------- Reference path: Standard launch + halo, single stream ----------
  for (int it = 0; it < kIters; ++it) {
    auto& halo_in = (it % 2 == 0) ? halo_ref_a : halo_ref_b;
    auto& step = (it % 2 == 0) ? step_ref_ab : step_ref_ba;
    halo_in(s_ref);
    try {
      step.launch(s_ref, carbon::DataView::Standard);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_data_view_split_jacobi: ref launch failed at iter {}: {}", it,
                       e.what());
      return 1;
    }
  }
  if (!sync_stream(s_ref, "ref iterations")) {
    return 1;
  }

  // ---------- Test path: Internal/Boundary across two streams ----------
  //
  // The orchestration:
  //
  //   it == 0: no prior compute, so s_halo runs halo(u_in) without waiting.
  //   it >  0: s_halo waits on evt_compute from iter (it-1) — last iter wrote into the
  //            field that's now this iter's u_in (after the swap), so halo on it must
  //            wait for those writes.
  //
  //   On s_compute we always launch Internal first (it has no halo dependency) so it can
  //   run in parallel with the halo exchange on s_halo.
  //
  //   evt_halo guards the Boundary launch: Boundary reads halo cells, so s_compute waits
  //   for evt_halo before issuing it.
  //
  //   evt_compute is recorded after Boundary so the next iteration's halo can wait on it.
  //
  // Note: Internal writes to u_out (not u_in), and halo writes to u_in's halo region —
  // disjoint memory, so they don't race. Boundary reads u_in halos, so it does need the
  // halo to be done first.
  for (int it = 0; it < kIters; ++it) {
    auto& halo_in = (it % 2 == 0) ? halo_a : halo_b;
    auto& step = (it % 2 == 0) ? step_ab : step_ba;

    if (it > 0) {
      if (!wait(s_halo, evt_compute, "iter wait halo<-compute")) {
        return 1;
      }
    }
    halo_in(s_halo);

    try {
      step.launch(s_compute, carbon::DataView::Internal);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_data_view_split_jacobi: Internal launch failed at iter {}: {}", it,
                       e.what());
      return 1;
    }

    if (!record(evt_halo, s_halo, "halo done")) {
      return 1;
    }
    if (!wait(s_compute, evt_halo, "compute wait halo")) {
      return 1;
    }

    try {
      step.launch(s_compute, carbon::DataView::Boundary);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_data_view_split_jacobi: Boundary launch failed at iter {}: {}", it,
                       e.what());
      return 1;
    }

    if (!record(evt_compute, s_compute, "compute done")) {
      return 1;
    }
  }
  if (!sync_stream(s_compute, "test compute") || !sync_stream(s_halo, "test halo")) {
    return 1;
  }

  // After kIters even iterations, the latest field is back in slot A on both paths.
  try {
    carbon::Transfer<float>(sys, u_a.gpu_buffers(), u_a.cpu_buffers())(s_compute);
    carbon::Transfer<float>(sys, u_ref_a.gpu_buffers(), u_ref_a.cpu_buffers())(s_ref);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_data_view_split_jacobi: download failed: {}", e.what());
    return 1;
  }
  if (!sync_stream(s_compute, "download test") || !sync_stream(s_ref, "download ref")) {
    return 1;
  }

  // Bit-for-bit comparison on every owned cell across every local slab. A drifting test
  // value here means a missed sync (the most likely root cause: Boundary running before
  // halo on s_compute, or halo on s_halo running before the previous iteration's compute
  // writes were complete).
  size_t owned_checked = 0;
  for (size_t g = 0; g < dense.gpu_count(); ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t lim_z = span.logic_extent().z;
    const int64_t z_origin = span.z_offset();
    const float* ht = u_a.cpu_buffers()[g].data();
    const float* hr = u_ref_a.cpu_buffers()[g].data();
    for (int64_t z = halo_z; z < halo_z + lim_z; ++z) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          if (ht[off] != hr[off]) {
            CARBON_LOG_ERROR(
                "test_data_view_split_jacobi: mismatch at gpu={} (x={}, y={}, gz={}): "
                "test={} ref={}",
                g, x, y, z_origin + z, ht[off], hr[off]);
            return 1;
          }
          ++owned_checked;
        }
      }
    }
  }

  const int64_t expected_owned = global.x * global.y * global.z;
  if (static_cast<int64_t>(owned_checked) != expected_owned) {
    CARBON_LOG_ERROR("test_data_view_split_jacobi: checked {} cells, expected {}", owned_checked,
                     expected_owned);
    return 1;
  }

  CARBON_LOG_INFO(
      "test_data_view_split_jacobi: OK — {} iterations across {} GPU(s); {} owned cells match "
      "between Standard-on-one-stream and Internal/Boundary-overlapped-with-halo",
      kIters, n_gpus, owned_checked);
  return 0;
}
