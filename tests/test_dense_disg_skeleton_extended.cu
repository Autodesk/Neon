// Skeleton OCC Extended + TwoWayExtended rewrites over carbon::DenseDisg.
//
// Mirrors test_skeleton_block_extended.cu but drives a DenseDisg (disaggregated
// dense) grid via the freshly-added DenseDisg::Field::make_halo_kernel adapter
// + the grid-agnostic Skeleton OCC rewrites. Validates that the rewrites
// produce numerically-equivalent results vs OverlapMode::None when the
// underlying field plumbing goes through DenseDisg::Field + HaloDisg.
//
// Section A (Extended): copy(seed -> work) ; stencil(work -> out).
//   - The copy is a Map predecessor that writes the field the next step
//     stencil-reads, so Extended's Map-back walk should find and split it
//     into Internal + Boundary halves. The unsplit reference and Extended
//     variant must agree on every owned cell.
//
// Section B (TwoWayExtended): copy(seed -> work) ; stencil(work -> mid) ;
//                             copy(mid -> out).
//   - Pre-copy is a Map-back candidate; post-copy is a Map-forward candidate.
//     TwoWayExtended should split all three into Internal + Boundary halves
//     and still preserve owned-cell parity vs the unsplit reference.
//
// Skipped (returns 0) when no CUDA devices are visible. Single-GPU runs are
// supported: HaloDisg's intra plan collapses to zero pairs but kernels still
// walk owned cells, exercising the full OCC rewrite end-to-end.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense_disg.h>
#include <carbon/halo_disg.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/skeleton.h>
#include <carbon/transfer.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <initializer_list>

namespace {

constexpr int kIters = 3;

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
build_copy(carbon::DenseDisg& dense,
           carbon::DenseDisg::Field<float,
                                    1>& fin,
           carbon::DenseDisg::Field<float,
                                    1>& fout) -> carbon::Kernel {
  return dense.make_kernel(
      [&fin, &fout](carbon::KernelLoader& loader) {
        const auto in = loader.read(fin);
        const auto out = loader.write(fout);
        return [in, out] CARBON_HOST_DEVICE(carbon::DenseDisg::Span s, carbon::DenseDisg::Idx idx) {
          out.write(s, idx, 0, in.read(s, idx, 0));
        };
      },
      /*shared_mem=*/0U);
}

[[nodiscard]] auto
build_stencil(carbon::DenseDisg& dense,
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
    for (int64_t bz = halo_z; bz < halo_z + lim_z; ++bz) {
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
    CARBON_LOG_INFO("test_dense_disg_skeleton_extended: no CUDA devices, skipping");
    return EXIT_SUCCESS;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();
  const int64_t z_per_slab = 6;
  const carbon::int64_3d global{8, 6, z_per_slab * static_cast<int64_t>(n_gpus)};
  carbon::DenseDisg dense(sys, global, /*halo_z=*/1);
  carbon::Stream stream = sys.stream();

  // ---- Section A: Extended ----
  {
    auto seed_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "seed_a");
    auto seed_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "seed_b");
    auto work_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "work_a");
    auto work_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "work_b");
    auto out_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "out_a");
    auto out_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "out_b");

    for (auto* f : std::initializer_list<carbon::DenseDisg::Field<float, 1>*>{
             &seed_a, &seed_b, &work_a, &work_b, &out_a, &out_b}) {
      if (!seed_field(sys, dense, *f, 0.0F, stream)) {
        return EXIT_FAILURE;
      }
    }
    if (!seed_field(sys, dense, seed_a, 2.5F, stream) ||
        !seed_field(sys, dense, seed_b, 2.5F, stream)) {
      return EXIT_FAILURE;
    }

    carbon::Skeleton sk_ref(sys, carbon::Skeleton::Options(carbon::OverlapMode::None));
    sk_ref.add(build_copy(dense, seed_a, work_a));
    sk_ref.add(build_stencil(dense, work_a, out_a));
    carbon::Skeleton sk_occ(sys, carbon::Skeleton::Options(carbon::OverlapMode::Extended));
    sk_occ.add(build_copy(dense, seed_b, work_b));
    sk_occ.add(build_stencil(dense, work_b, out_b));

    try {
      sk_ref.compile();
      sk_occ.compile();
      for (int it = 0; it < kIters; ++it) {
        sk_ref.run(carbon::DataView::Standard);
        sk_occ.run(carbon::DataView::Standard);
      }
      sk_ref.wait();
      sk_occ.wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dense_disg_skeleton_extended[Extended]: run failed: {}", e.what());
      return EXIT_FAILURE;
    }

    if (!download_and_compare(sys, dense, out_a, out_b, stream,
                              "test_dense_disg_skeleton_extended[Extended]")) {
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_dense_disg_skeleton_extended[Extended]: matches reference");
  }

  // ---- Section B: TwoWayExtended ----
  {
    auto seed_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "twe_seed_a");
    auto seed_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "twe_seed_b");
    auto work_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "twe_work_a");
    auto work_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "twe_work_b");
    auto mid_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "twe_mid_a");
    auto mid_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "twe_mid_b");
    auto out_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "twe_out_a");
    auto out_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "twe_out_b");

    for (auto* f : std::initializer_list<carbon::DenseDisg::Field<float, 1>*>{
             &work_a, &work_b, &mid_a, &mid_b, &out_a, &out_b}) {
      if (!seed_field(sys, dense, *f, 0.0F, stream)) {
        return EXIT_FAILURE;
      }
    }
    if (!seed_field(sys, dense, seed_a, 1.75F, stream) ||
        !seed_field(sys, dense, seed_b, 1.75F, stream)) {
      return EXIT_FAILURE;
    }

    carbon::Skeleton sk_ref(sys, carbon::Skeleton::Options(carbon::OverlapMode::None));
    sk_ref.add(build_copy(dense, seed_a, work_a));
    sk_ref.add(build_stencil(dense, work_a, mid_a));
    sk_ref.add(build_copy(dense, mid_a, out_a));
    carbon::Skeleton sk_occ(sys, carbon::Skeleton::Options(carbon::OverlapMode::TwoWayExtended));
    sk_occ.add(build_copy(dense, seed_b, work_b));
    sk_occ.add(build_stencil(dense, work_b, mid_b));
    sk_occ.add(build_copy(dense, mid_b, out_b));

    try {
      sk_ref.compile();
      sk_occ.compile();
      for (int it = 0; it < kIters; ++it) {
        sk_ref.run(carbon::DataView::Standard);
        sk_occ.run(carbon::DataView::Standard);
      }
      sk_ref.wait();
      sk_occ.wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dense_disg_skeleton_extended[TwoWayExtended]: run failed: {}",
                       e.what());
      return EXIT_FAILURE;
    }

    if (!download_and_compare(sys, dense, out_a, out_b, stream,
                              "test_dense_disg_skeleton_extended[TwoWayExtended]")) {
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_dense_disg_skeleton_extended[TwoWayExtended]: matches reference");
  }

  CARBON_LOG_INFO("test_dense_disg_skeleton_extended: PASS ({} GPUs)", n_gpus);
  return EXIT_SUCCESS;
}
