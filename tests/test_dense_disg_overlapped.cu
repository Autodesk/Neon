// OverlappedStencil + Skeleton coverage for carbon::DenseDisg.
//
// Locks that the grid-agnostic OverlappedStencil + Skeleton helpers run end-to-end
// on the disaggregated dense layout via the freshly-added
// DenseDisg::Field::make_halo_kernel adapter (which builds a HaloDisg<T,Components>
// and wraps it as a carbon::Kernel through Kernel::halo_disg).
//
// Two sub-tests in one binary:
//
//   A. Manual OverlappedStencil pipeline
//      - Run kIters Jacobi-like ping-pong steps where each step is a
//        compute_kernel that read_stencil()s the input field. The halo kernel
//        is the freshly-added DenseDisg::Field::make_halo_kernel adapter.
//      - Reference path runs the same kIters on a single stream
//        (halo.launch ; step.launch) with DataView::Standard.
//      - Final fields must match bit-for-bit (no halo cross-stream races for
//        DenseDisg since the OverlappedStencil halo runs on its own queue but
//        DenseDisg HaloDisg::operator() uses Transfer<T> on the halo stream,
//        identical contract to Dense::Halo).
//
//   B. Skeleton OverlapMode::Standard
//      - Build a Skeleton with OverlapMode::Standard, populate it with the
//        same step kernels (read_stencil tags the input, so the skeleton's
//        compile() pass auto-inserts a halo + splits Internal / Boundary).
//      - Skeleton::run() must produce the same final field as the manual
//        reference path above.
//
// Skipped (returns 0) when no CUDA devices are visible. Single-GPU runs are
// supported: the HaloDisg's intra-process plan collapses to zero pairs but
// the kernels still walk owned cells, so the orchestration still exercises
// the make_halo_kernel + KernelLoader::read_stencil + Skeleton OCC rewrite
// end-to-end.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense_disg.h>
#include <carbon/halo_disg.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/overlapped_stencil.h>
#include <carbon/skeleton.h>
#include <carbon/transfer.h>

#include <cstdint>
#include <cstdlib>
#include <exception>

namespace {

constexpr int kIters = 4;

[[nodiscard]] auto
disg_addr(const carbon::DenseDisg::Span& span,
          int64_t bz,
          int64_t x,
          int64_t y,
          int c,
          int components) -> int64_t {
  const int64_t lz = span.logic_extent().z;
  const int64_t xdim = span.haloed_extent().x;
  const int64_t xy = span.xy_slab_elements();
  int64_t region_first_z = 0;
  int64_t region_z_dim = 1;
  int64_t offset_local_no_card = x + y * xdim;
  if (bz <= 1 || bz >= lz) {
    region_first_z = bz;
    region_z_dim = 1;
  } else {
    region_first_z = 2;
    region_z_dim = lz - 2;
    offset_local_no_card += (bz - 2) * xy;
  }
  return offset_local_no_card + xy * (region_first_z * static_cast<int64_t>(components) +
                                      static_cast<int64_t>(c) * region_z_dim);
}

[[nodiscard]] auto
build_jacobi_step(carbon::DenseDisg& dense,
                  carbon::DenseDisg::Field<float,
                                           1>& fin,
                  carbon::DenseDisg::Field<float,
                                           1>& fout) -> carbon::Kernel {
  return dense.make_kernel(
      [&fin, &fout](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(fin);
        const auto out = loader.write(fout);
        return [in, out] CARBON_HOST_DEVICE(carbon::DenseDisg::Span s, carbon::DenseDisg::Idx idx) {
          const auto xp = in.neighbour(s, idx, {1, 0, 0}, 0);
          const auto xm = in.neighbour(s, idx, {-1, 0, 0}, 0);
          const auto yp = in.neighbour(s, idx, {0, 1, 0}, 0);
          const auto ym = in.neighbour(s, idx, {0, -1, 0}, 0);
          const auto zp = in.neighbour(s, idx, {0, 0, 1}, 0);
          const auto zm = in.neighbour(s, idx, {0, 0, -1}, 0);
          const float self = in.read(s, idx, 0);
          if (!(xp.valid && xm.valid && yp.valid && ym.valid && zp.valid && zm.valid)) {
            out.write(s, idx, 0, self);
            return;
          }
          const float sum = xp.value + xm.value + yp.value + ym.value + zp.value + zm.value;
          out.write(s, idx, 0, sum * (1.0F / 6.0F));
        };
      },
      /*shared_mem=*/0U);
}

auto
seed_field(carbon::System& sys,
           carbon::DenseDisg& dense,
           carbon::DenseDisg::Field<float,
                                    1>& field,
           float value,
           const carbon::Stream& stream) -> bool {
  const size_t n_gpus = sys.gpu_group().size();
  for (size_t i = 0; i < n_gpus; ++i) {
    const auto& span = dense.spans()[i];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t lim_z = span.logic_extent().z;
    float* h = field.cpu_buffers()[i].data();
    const size_t total = field.cpu_buffers()[i].size();
    for (size_t k = 0; k < total; ++k) {
      h[k] = 0.0F;
    }
    for (int64_t bz = 0; bz < haloed.z; ++bz) {
      const bool is_owned = (bz >= halo_z) && (bz < halo_z + lim_z);
      if (!is_owned) {
        continue;
      }
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = disg_addr(span, bz, x, y, 0, /*components=*/1);
          h[off] = value + 0.1F * static_cast<float>(bz);
        }
      }
    }
  }
  try {
    carbon::Transfer<float>(sys, field.cpu_buffers(), field.gpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("seed_field upload failed: {}", e.what());
    return false;
  }
  return true;
}

[[nodiscard]] auto
download_and_compare(carbon::System& sys,
                     carbon::DenseDisg& dense,
                     carbon::DenseDisg::Field<float,
                                              1>& ref,
                     carbon::DenseDisg::Field<float,
                                              1>& got,
                     const carbon::Stream& stream,
                     const char* tag) -> bool {
  const size_t n_gpus = sys.gpu_group().size();
  try {
    carbon::Transfer<float>(sys, ref.gpu_buffers(), ref.cpu_buffers())(stream);
    carbon::Transfer<float>(sys, got.gpu_buffers(), got.cpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("{}: download failed: {}", tag, e.what());
    return false;
  }
  size_t mismatches = 0;
  for (size_t i = 0; i < n_gpus; ++i) {
    const auto& span = dense.spans()[i];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t lim_z = span.logic_extent().z;
    const float* hr = ref.cpu_buffers()[i].data();
    const float* hg = got.cpu_buffers()[i].data();
    for (int64_t bz = halo_z; bz < halo_z + lim_z; ++bz) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = disg_addr(span, bz, x, y, 0, /*components=*/1);
          if (hr[off] != hg[off]) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR("{}: slab {} bz={} (x,y)=({},{}) got={} ref={}", tag, i, bz, x, y,
                               hg[off], hr[off]);
            }
            ++mismatches;
          }
        }
      }
    }
  }
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("{}: {} mismatches", tag, mismatches);
    return false;
  }
  return true;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_dense_disg_overlapped: no CUDA devices, skipping");
    return EXIT_SUCCESS;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();
  const int64_t z_per_slab = 6;
  const carbon::int64_3d global{8, 6, z_per_slab * static_cast<int64_t>(n_gpus)};
  carbon::DenseDisg dense(sys, global, /*halo_z=*/1);

  carbon::Stream stream = sys.stream();

  // ---- Path R: reference single-stream loop (halo + Standard step) ----
  auto u_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_ref");
  auto v_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v_ref");
  if (!seed_field(sys, dense, u_ref, 2.5F, stream)) {
    return EXIT_FAILURE;
  }
  if (!seed_field(sys, dense, v_ref, 0.0F, stream)) {
    return EXIT_FAILURE;
  }

  auto halo_u_ref = u_ref.make_halo_kernel("halo_u_ref");
  auto halo_v_ref = v_ref.make_halo_kernel("halo_v_ref");
  auto step_uv_ref = build_jacobi_step(dense, u_ref, v_ref);
  auto step_vu_ref = build_jacobi_step(dense, v_ref, u_ref);

  for (int it = 0; it < kIters; ++it) {
    const carbon::Kernel& halo = (it % 2 == 0) ? halo_u_ref : halo_v_ref;
    const carbon::Kernel& step = (it % 2 == 0) ? step_uv_ref : step_vu_ref;
    try {
      halo.launch(stream, carbon::DataView::Standard);
      step.launch(stream, carbon::DataView::Standard);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dense_disg_overlapped[ref]: launch failed: {}", e.what());
      return EXIT_FAILURE;
    }
  }
  stream.wait();
  carbon::DenseDisg::Field<float, 1>& final_ref = (kIters % 2 == 0) ? u_ref : v_ref;

  // ---- Path A: OverlappedStencil ----
  auto u_ovr = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_ovr");
  auto v_ovr = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v_ovr");
  if (!seed_field(sys, dense, u_ovr, 2.5F, stream)) {
    return EXIT_FAILURE;
  }
  if (!seed_field(sys, dense, v_ovr, 0.0F, stream)) {
    return EXIT_FAILURE;
  }

  carbon::OverlappedStencil overlap(sys);
  auto halo_u_ovr = u_ovr.make_halo_kernel("halo_u_ovr");
  auto halo_v_ovr = v_ovr.make_halo_kernel("halo_v_ovr");
  auto step_uv_ovr = build_jacobi_step(dense, u_ovr, v_ovr);
  auto step_vu_ovr = build_jacobi_step(dense, v_ovr, u_ovr);

  for (int it = 0; it < kIters; ++it) {
    const carbon::Kernel& halo = (it % 2 == 0) ? halo_u_ovr : halo_v_ovr;
    const carbon::Kernel& step = (it % 2 == 0) ? step_uv_ovr : step_vu_ovr;
    overlap.step(halo, step);
  }
  overlap.wait();
  carbon::DenseDisg::Field<float, 1>& final_ovr = (kIters % 2 == 0) ? u_ovr : v_ovr;

  if (!download_and_compare(sys, dense, final_ref, final_ovr, stream,
                            "test_dense_disg_overlapped[OverlappedStencil]")) {
    return EXIT_FAILURE;
  }
  CARBON_LOG_INFO("test_dense_disg_overlapped[OverlappedStencil]: matches reference");

  // ---- Path B: Skeleton OverlapMode::Standard ----
  auto u_sk = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_sk");
  auto v_sk = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v_sk");
  if (!seed_field(sys, dense, u_sk, 2.5F, stream)) {
    return EXIT_FAILURE;
  }
  if (!seed_field(sys, dense, v_sk, 0.0F, stream)) {
    return EXIT_FAILURE;
  }

  auto step_uv_sk = build_jacobi_step(dense, u_sk, v_sk);
  auto step_vu_sk = build_jacobi_step(dense, v_sk, u_sk);

  carbon::Skeleton sk(sys, carbon::Skeleton::Options(carbon::OverlapMode::Standard));
  for (int it = 0; it < kIters; ++it) {
    sk.add((it % 2 == 0) ? step_uv_sk : step_vu_sk);
  }
  try {
    sk.compile();
    sk.run(carbon::DataView::Standard);
    sk.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_overlapped[Skeleton]: compile/run failed: {}", e.what());
    return EXIT_FAILURE;
  }
  carbon::DenseDisg::Field<float, 1>& final_sk = (kIters % 2 == 0) ? u_sk : v_sk;

  if (!download_and_compare(sys, dense, final_ref, final_sk, stream,
                            "test_dense_disg_overlapped[Skeleton]")) {
    return EXIT_FAILURE;
  }
  CARBON_LOG_INFO("test_dense_disg_overlapped[Skeleton]: matches reference");

  CARBON_LOG_INFO("test_dense_disg_overlapped: PASS ({} GPUs)", n_gpus);
  return EXIT_SUCCESS;
}
