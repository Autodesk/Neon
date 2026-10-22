// Multi-GPU Jacobi-like stencil with int32 fields — bit-exact correctness check.
//
// Three subtests, all using int32_t fields and truncating integer division (sum / 6)
// so a halo / stream-ordering bug shows up as a wrong integer, not FP drift:
//
//   A. Reference — manual halo + Standard launch on one stream (kIters ping-pong steps).
//   B. Overlap   — Internal/Boundary across compute + halo streams with events
//                  (same kIters; mirrors test_data_view_split_jacobi.cu).
//   C. Skeleton  — four-kernel ping-pong cycle (halo_a, step_ab, halo_b, step_ba)
//                  replayed kIters/2 times via carbon::Skeleton (mirrors the Jacobi
//                  subtest in test_skeleton.cu, but int32 + bit-exact compare).
//
// Skipped (returns 0) when fewer than two GPUs are visible — the single-GPU case has no
// inter-slab halo exchange and is already covered elsewhere.

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

constexpr int kIters = 20;
static_assert(kIters % 2 == 0,
              "kIters must be even so the final result lives in slot A");

constexpr int kSkeletonCycles = kIters / 2;

constexpr int32_t kSentinel = -7777;

[[nodiscard]] CARBON_HD auto
encode(int64_t x,
       int64_t y,
       int64_t gz) noexcept -> int32_t {
  return static_cast<int32_t>(x + y * 11 + gz * 113);
}

template <typename Fn>
auto
fill_storage_host(const carbon::Dense& dense,
                  carbon::Dense::Field<int32_t,
                                       1>& f,
                  Fn fn) -> void {
  for (size_t g = 0; g < dense.gpu_count(); ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t z_origin = span.z_offset();
    int32_t* h = f.cpu_buffers()[g].data();
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
    CARBON_LOG_ERROR("test_jacobi_integer: sync after {} failed: {}", tag, e.what());
    return false;
  }
  return true;
}

[[nodiscard]] auto
record(carbon::Event& evt,
       const carbon::Stream& s,
       const char* tag) -> bool {
  try {
    evt.record(s);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_jacobi_integer: Event::record({}) failed: {}", tag, e.what());
    return false;
  }
  return true;
}

[[nodiscard]] auto
wait(const carbon::Stream& s,
     const carbon::Event& evt,
     const char* tag) -> bool {
  try {
    s.wait_for(evt);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_jacobi_integer: Stream::wait_for({}) failed: {}", tag, e.what());
    return false;
  }
  return true;
}

[[nodiscard]] auto
compare_owned_to_ref(const carbon::Dense& dense,
                     const carbon::int64_3d& global,
                     const carbon::Dense::Field<int32_t,
                                                1>& test,
                     const carbon::Dense::Field<int32_t,
                                                1>& ref,
                     const char* path_label) -> bool {
  size_t owned_checked = 0;
  for (size_t g = 0; g < dense.gpu_count(); ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t lim_z = span.logic_extent().z;
    const int64_t z_origin = span.z_offset();
    const int32_t* ht = test.cpu_buffers()[g].data();
    const int32_t* hr = ref.cpu_buffers()[g].data();
    for (int64_t z = halo_z; z < halo_z + lim_z; ++z) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          if (ht[off] != hr[off]) {
            CARBON_LOG_ERROR(
                "test_jacobi_integer: {} mismatch at gpu={} (x={}, y={}, gz={}): test={} ref={}",
                path_label, g, x, y, z_origin + z, ht[off], hr[off]);
            return false;
          }
          ++owned_checked;
        }
      }
    }
  }
  const int64_t expected_owned = global.x * global.y * global.z;
  if (static_cast<int64_t>(owned_checked) != expected_owned) {
    CARBON_LOG_ERROR("test_jacobi_integer: {} checked {} cells, expected {}", path_label,
                     owned_checked, expected_owned);
    return false;
  }
  return true;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 2) {
    CARBON_LOG_WARN(
        "test_jacobi_integer: needs >=2 visible CUDA devices for inter-slab halo exchange; "
        "have {} — skipping",
        carbon::device_count());
    return 0;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  if (stencil.radius() != 1) {
    CARBON_LOG_ERROR("test_jacobi_integer: unexpected stencil radius {}", stencil.radius());
    return 1;
  }

  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{6, 5, z_per_slab * static_cast<int64_t>(n_gpus)};
  carbon::Dense dense(sys, global, std::move(stencil));

  auto u_a = dense.newField<int32_t, 1>(carbon::MemorySide::HostDevice);
  auto u_b = dense.newField<int32_t, 1>(carbon::MemorySide::HostDevice);
  auto u_ref_a = dense.newField<int32_t, 1>(carbon::MemorySide::HostDevice);
  auto u_ref_b = dense.newField<int32_t, 1>(carbon::MemorySide::HostDevice);
  auto u_sk_a = dense.newField<int32_t, 1>(carbon::MemorySide::HostDevice);
  auto u_sk_b = dense.newField<int32_t, 1>(carbon::MemorySide::HostDevice);

  auto init_with_sentinel_halo = [&](carbon::Dense::Field<int32_t, 1>& f) {
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
      int32_t* h = f.cpu_buffers()[g].data();
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
  };
  auto init_slot_b = [&](carbon::Dense::Field<int32_t, 1>& f) {
    fill_storage_host(dense, f, [](int64_t, int64_t, int64_t) {
      return kSentinel;
    });
  };

  init_with_sentinel_halo(u_a);
  init_with_sentinel_halo(u_ref_a);
  init_with_sentinel_halo(u_sk_a);
  init_slot_b(u_b);
  init_slot_b(u_ref_b);
  init_slot_b(u_sk_b);

  carbon::Halo halo_a(u_a);
  carbon::Halo halo_b(u_b);
  carbon::Halo halo_ref_a(u_ref_a);
  carbon::Halo halo_ref_b(u_ref_b);

  auto build_step = [&](carbon::Dense::Field<int32_t, 1>& fin,
                        carbon::Dense::Field<int32_t, 1>& fout) {
    return dense.make_kernel(
        [&fin, &fout](carbon::KernelLoader& loader) {
          const auto in = loader.read(fin);
          const auto out = loader.write(fout);
          return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
            const carbon::NeighbourValue<int32_t> xp = in.neighbour(s, idx, {1, 0, 0}, 0);
            const carbon::NeighbourValue<int32_t> xm = in.neighbour(s, idx, {-1, 0, 0}, 0);
            const carbon::NeighbourValue<int32_t> yp = in.neighbour(s, idx, {0, 1, 0}, 0);
            const carbon::NeighbourValue<int32_t> ym = in.neighbour(s, idx, {0, -1, 0}, 0);
            const carbon::NeighbourValue<int32_t> zp = in.neighbour(s, idx, {0, 0, 1}, 0);
            const carbon::NeighbourValue<int32_t> zm = in.neighbour(s, idx, {0, 0, -1}, 0);
            const int32_t self = in.read(s, idx, 0);
            const bool all_valid =
                xp.valid && xm.valid && yp.valid && ym.valid && zp.valid && zm.valid;
            if (!all_valid) {
              out.write(s, idx, 0, self);
              return;
            }
            const int32_t sum = xp.value + xm.value + yp.value + ym.value + zp.value + zm.value;
            out.write(s, idx, 0, sum / 6);
          };
        },
        /*shared_mem=*/0U);
  };
  auto step_ab = build_step(u_a, u_b);
  auto step_ba = build_step(u_b, u_a);
  auto step_ref_ab = build_step(u_ref_a, u_ref_b);
  auto step_ref_ba = build_step(u_ref_b, u_ref_a);
  auto step_sk_ab = build_step(u_sk_a, u_sk_b);
  auto step_sk_ba = build_step(u_sk_b, u_sk_a);

  carbon::Stream s_ref = sys.stream();
  carbon::Stream s_compute = sys.stream();
  carbon::Stream s_halo = sys.stream();
  carbon::Stream s_sk = sys.stream();
  carbon::Event evt_halo = sys.event();
  carbon::Event evt_compute = sys.event();

  try {
    carbon::Transfer<int32_t>(sys, u_ref_a.cpu_buffers(), u_ref_a.gpu_buffers())(s_ref);
    carbon::Transfer<int32_t>(sys, u_ref_b.cpu_buffers(), u_ref_b.gpu_buffers())(s_ref);
    carbon::Transfer<int32_t>(sys, u_a.cpu_buffers(), u_a.gpu_buffers())(s_ref);
    carbon::Transfer<int32_t>(sys, u_b.cpu_buffers(), u_b.gpu_buffers())(s_ref);
    carbon::Transfer<int32_t>(sys, u_sk_a.cpu_buffers(), u_sk_a.gpu_buffers())(s_ref);
    carbon::Transfer<int32_t>(sys, u_sk_b.cpu_buffers(), u_sk_b.gpu_buffers())(s_ref);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_jacobi_integer: upload failed: {}", e.what());
    return 1;
  }
  if (!sync_stream(s_ref, "upload")) {
    return 1;
  }

  // ---------- A. Reference: halo + Standard on one stream ----------
  for (int it = 0; it < kIters; ++it) {
    auto& halo_in = (it % 2 == 0) ? halo_ref_a : halo_ref_b;
    auto& step = (it % 2 == 0) ? step_ref_ab : step_ref_ba;
    halo_in(s_ref);
    try {
      step.launch(s_ref, carbon::DataView::Standard);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_jacobi_integer: ref launch failed at iter {}: {}", it, e.what());
      return 1;
    }
  }
  if (!sync_stream(s_ref, "ref iterations")) {
    return 1;
  }

  // ---------- B. Overlap: Internal/Boundary + halo on two streams ----------
  for (int it = 0; it < kIters; ++it) {
    auto& halo_in = (it % 2 == 0) ? halo_a : halo_b;
    auto& step = (it % 2 == 0) ? step_ab : step_ba;

    if (it > 0) {
      if (!wait(s_halo, evt_compute, "overlap wait halo<-compute")) {
        return 1;
      }
    }
    halo_in(s_halo);

    try {
      step.launch(s_compute, carbon::DataView::Internal);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_jacobi_integer: overlap Internal launch failed at iter {}: {}", it,
                       e.what());
      return 1;
    }

    if (!record(evt_halo, s_halo, "overlap halo done")) {
      return 1;
    }
    if (!wait(s_compute, evt_halo, "overlap compute wait halo")) {
      return 1;
    }

    try {
      step.launch(s_compute, carbon::DataView::Boundary);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_jacobi_integer: overlap Boundary launch failed at iter {}: {}", it,
                       e.what());
      return 1;
    }

    if (!record(evt_compute, s_compute, "overlap compute done")) {
      return 1;
    }
  }
  if (!sync_stream(s_compute, "overlap compute") || !sync_stream(s_halo, "overlap halo")) {
    return 1;
  }

  // ---------- C. Skeleton: four-kernel ping-pong cycle ----------
  carbon::Skeleton sk(sys, s_sk);
  sk.add(u_sk_a.make_halo_kernel("halo_a"));
  sk.add(step_sk_ab);
  sk.add(u_sk_b.make_halo_kernel("halo_b"));
  sk.add(step_sk_ba);
  sk.compile();

  if (sk.size() != 4U) {
    CARBON_LOG_ERROR("test_jacobi_integer: skeleton expected size()==4, got {}", sk.size());
    return 1;
  }
  if (sk.num_streams() != 1U) {
    CARBON_LOG_ERROR(
        "test_jacobi_integer: skeleton Jacobi ping-pong expected num_streams()==1, got {}",
        sk.num_streams());
    return 1;
  }

  for (int cycle = 0; cycle < kSkeletonCycles; ++cycle) {
    try {
      sk.run();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_jacobi_integer: skeleton run failed at cycle {}: {}", cycle, e.what());
      return 1;
    }
  }
  sk.wait();

  // ---------- Download + bit-exact compare both test paths against reference ----------
  try {
    carbon::Transfer<int32_t>(sys, u_a.gpu_buffers(), u_a.cpu_buffers())(s_compute);
    carbon::Transfer<int32_t>(sys, u_sk_a.gpu_buffers(), u_sk_a.cpu_buffers())(s_sk);
    carbon::Transfer<int32_t>(sys, u_ref_a.gpu_buffers(), u_ref_a.cpu_buffers())(s_ref);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_jacobi_integer: download failed: {}", e.what());
    return 1;
  }
  if (!sync_stream(s_compute, "download overlap") || !sync_stream(s_sk, "download skeleton") ||
      !sync_stream(s_ref, "download ref")) {
    return 1;
  }

  if (!compare_owned_to_ref(dense, global, u_a, u_ref_a, "overlap")) {
    return 1;
  }
  if (!compare_owned_to_ref(dense, global, u_sk_a, u_ref_a, "skeleton")) {
    return 1;
  }

  CARBON_LOG_INFO(
      "test_jacobi_integer: OK — {} int32 Jacobi iterations across {} GPU(s); overlap + skeleton "
      "paths both match the Standard single-stream reference bit-for-bit (skeleton ran {} "
      "four-kernel cycles)",
      kIters, n_gpus, kSkeletonCycles);
  return 0;
}
