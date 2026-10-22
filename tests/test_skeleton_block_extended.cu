// Skeleton OCC Extended + TwoWayExtended rewrites over carbon::Block.
//
// Mirrors test_skeleton.cu sections F (Extended) and G (TwoWayExtended) but
// drives a Block (block-sparse) grid instead of Dense. Validates that the
// grid-agnostic Skeleton OCC rewrites still produce numerically-equivalent
// results when the underlying field plumbing goes through Block::Field +
// HaloBlock instead of Dense::Field + Halo.
//
// Section A (Extended): copy(seed -> work) ; stencil(work -> out).
//   - The copy is a Map predecessor that writes the field the next step
//     stencil-reads, so Extended's Map-back walk should find and split it
//     into Internal + Boundary halves. The unsplit reference (OverlapMode::None)
//     and the Extended variant must agree on every active voxel.
//
// Section B (TwoWayExtended): copy(seed -> work) ; stencil(work -> mid) ;
//                             copy(mid -> out).
//   - The pre-copy is a Map-back candidate (writes what the stencil
//     stencil-reads); the post-copy is a Map-forward candidate (reads what
//     the stencil writes). TwoWayExtended should split all three into
//     Internal + Boundary halves; the unsplit reference and the
//     TwoWayExtended variant must still agree on every active voxel.
//
// Skipped (returns 0) when fewer than 2 CUDA devices are visible — the OCC
// halo overlap path is only interesting on multi-GPU layouts where the halo
// kernel actually has cross-stream cudaMemcpyPeerAsyncs to dispatch.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/data_view.h>
#include <carbon/devices.h>
#include <carbon/halo_block.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/skeleton.h>
#include <carbon/system.h>

#include "test_domain.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using Span = BlockGrid::Span;
using Idx = BlockGrid::Idx;
using FieldT = BlockGrid::Field<float, 1>;

[[nodiscard]] auto
sphere_predicate(int64_t domain_n,
                 const carbon::int64_3d& gc) -> bool {
  const double cx = (domain_n - 1) * 0.5;
  const double cy = (domain_n - 1) * 0.5;
  const double cz = (domain_n - 1) * 0.5;
  const double r = domain_n * 0.42;
  const double dx = static_cast<double>(gc.x) - cx;
  const double dy = static_cast<double>(gc.y) - cy;
  const double dz = static_cast<double>(gc.z) - cz;
  return (dx * dx + dy * dy + dz * dz) <= (r * r);
}

[[nodiscard]] auto
build_copy(BlockGrid& block,
           FieldT& fin,
           FieldT& fout,
           std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      block,
      [&fin, &fout](carbon::KernelLoader& loader) {
        const auto in = loader.read(fin);
        const auto out = loader.write(fout);
        return [in, out] CARBON_HD(Span s, Idx idx) -> void {
          out.write(s, idx, 0, in.read(s, idx, 0));
        };
      },
      std::move(name));
}

[[nodiscard]] auto
build_stencil(BlockGrid& block,
              FieldT& fin,
              FieldT& fout,
              std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      block,
      [&fin, &fout](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(fin);
        const auto out = loader.write(fout);
        return [in, out] CARBON_HD(Span s, Idx idx) -> void {
          const auto xp = in.neighbour(s, idx, carbon::int8_3d{1, 0, 0}, 0);
          const auto xm = in.neighbour(s, idx, carbon::int8_3d{-1, 0, 0}, 0);
          const auto yp = in.neighbour(s, idx, carbon::int8_3d{0, 1, 0}, 0);
          const auto ym = in.neighbour(s, idx, carbon::int8_3d{0, -1, 0}, 0);
          const auto zp = in.neighbour(s, idx, carbon::int8_3d{0, 0, 1}, 0);
          const auto zm = in.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0);
          const float self = in.read(s, idx, 0);
          if (!(xp.valid && xm.valid && yp.valid && ym.valid && zp.valid && zm.valid)) {
            out.write(s, idx, 0, self);
            return;
          }
          const float sum = xp.value + xm.value + yp.value + ym.value + zp.value + zm.value;
          out.write(s, idx, 0, sum * (1.0F / 6.0F));
        };
      },
      std::move(name));
}

auto
zero_fields(carbon::System& sys,
            BlockGrid& block,
            std::initializer_list<FieldT*> fields,
            const carbon::Stream& stream) -> void {
  const size_t n_gpus = sys.gpu_group().size();
  for (size_t g = 0; g < n_gpus; ++g) {
    const size_t total =
        static_cast<size_t>(block.storage_block_count(g)) * static_cast<size_t>(SB::VoxelCount);
    if (total == 0U) {
      continue;
    }
    for (FieldT* f : fields) {
      sys.async_memset(f->gpu_buffers()[g], 0, total * sizeof(float), stream);
    }
  }
  stream.wait();
}

auto
seed_constant(carbon::System& sys,
              BlockGrid& block,
              FieldT& field,
              float value,
              const carbon::Stream& stream) -> void {
  const size_t n_gpus = sys.gpu_group().size();
  for (size_t g = 0; g < n_gpus; ++g) {
    const uint32_t n_owned = block.block_count(g);
    const size_t total =
        static_cast<size_t>(block.storage_block_count(g)) * static_cast<size_t>(SB::VoxelCount);
    if (total == 0U) {
      continue;
    }
    auto host_buf = carbon::Cpu::buffer<float>(total);
    std::vector<float> stage(total, 0.0F);
    for (uint32_t b = 0; b < n_owned; ++b) {
      const auto& bm = block.host_bitmasks(g)[b];
      for (int iz = 0; iz < SB::MemZ; ++iz) {
        for (int iy = 0; iy < SB::MemY; ++iy) {
          for (int ix = 0; ix < SB::MemX; ++ix) {
            if (!bm.is_active(ix, iy, iz)) {
              continue;
            }
            const size_t lin = (static_cast<size_t>(b) * SB::VoxelCount) +
                               (static_cast<size_t>(iz) * SB::MemY * SB::MemX) +
                               (static_cast<size_t>(iy) * SB::MemX) + static_cast<size_t>(ix);
            // Use a non-trivial value that depends on coordinates so the
            // stencil averaging produces nontrivial outputs across iters.
            stage[lin] = value + 0.01F * static_cast<float>(iz);
          }
        }
      }
    }
    std::memcpy(host_buf.data(), stage.data(), total * sizeof(float));
    sys.async_copy(host_buf, field.gpu_buffers()[g], total * sizeof(float), stream);
  }
  stream.wait();
}

[[nodiscard]] auto
compare_active_voxels(BlockGrid& block,
                      const std::vector<carbon::detail::Allocation<float>>& host_a,
                      const std::vector<carbon::detail::Allocation<float>>& host_b,
                      float tol,
                      const char* tag) -> bool {
  const size_t n_gpus = host_a.size();
  size_t out_of_tol = 0;
  size_t active_checked = 0;
  float max_abs_diff = 0.0F;
  for (size_t g = 0; g < n_gpus; ++g) {
    const uint32_t n_owned = block.block_count(g);
    const float* ha = host_a[g].data();
    const float* hb = host_b[g].data();
    for (uint32_t b = 0; b < n_owned; ++b) {
      const auto& bm = block.host_bitmasks(g)[b];
      for (int z = 0; z < SB::MemZ; ++z) {
        for (int y = 0; y < SB::MemY; ++y) {
          for (int x = 0; x < SB::MemX; ++x) {
            if (!bm.is_active(static_cast<uint8_t>(x), static_cast<uint8_t>(y),
                              static_cast<uint8_t>(z))) {
              continue;
            }
            const size_t off = (static_cast<size_t>(b) * SB::VoxelCount) +
                               (static_cast<size_t>(z) * SB::MemY * SB::MemX) +
                               (static_cast<size_t>(y) * SB::MemX) + static_cast<size_t>(x);
            const float diff = std::abs(ha[off] - hb[off]);
            if (diff > max_abs_diff) {
              max_abs_diff = diff;
            }
            if (diff > tol) {
              ++out_of_tol;
            }
            ++active_checked;
          }
        }
      }
    }
  }
  CARBON_LOG_INFO("{}: checked {} active voxels, max |a - b| = {:.6e}, out-of-tol (>{:.1e}) = {}",
                  tag, active_checked, static_cast<double>(max_abs_diff), static_cast<double>(tol),
                  out_of_tol);
  return out_of_tol == 0U;
}

constexpr int kIters = 3;
constexpr int64_t kDomainN = 48;
// Same race-aware tolerance test_skeleton_block uses; OCC + HaloBlock
// share the same cudaMemcpyPeerAsync cross-stream drift bound.
constexpr float kTol = 5.0e-3F;

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 2) {
    CARBON_LOG_INFO(
        "test_skeleton_block_extended: skipping (only {} CUDA device(s) visible, need >= 2)",
        carbon::device_count());
    return 0;
  }

  auto devs = carbon::get_all_devices();
  carbon::System sys(carbon::detail::GpuGroup{std::move(devs)});
  // Kept cubic as it grows, so the sphere scales with the domain instead of leaving the extra
  // brick rows empty.
  const int64_t domain_n = carbon::test::block_slab_z(kDomainN, sys.gpu_group().size(), SB::MemZ);
  const carbon::int64_3d global{domain_n, domain_n, domain_n};
  const auto predicate = [domain_n](const carbon::int64_3d& gc) -> bool {
    return sphere_predicate(domain_n, gc);
  };

  BlockGrid block(sys, global, predicate, "skeleton_block_extended");
  carbon::Stream stream = sys.stream();

  // ---- Section A: Extended ----
  {
    auto seed_a = block.template newField<float, 1>(carbon::MemorySide::Device, "seed_a");
    auto seed_b = block.template newField<float, 1>(carbon::MemorySide::Device, "seed_b");
    auto work_a = block.template newField<float, 1>(carbon::MemorySide::Device, "work_a");
    auto work_b = block.template newField<float, 1>(carbon::MemorySide::Device, "work_b");
    auto out_a = block.template newField<float, 1>(carbon::MemorySide::Device, "out_a");
    auto out_b = block.template newField<float, 1>(carbon::MemorySide::Device, "out_b");

    zero_fields(sys, block, {&work_a, &work_b, &out_a, &out_b}, stream);
    seed_constant(sys, block, seed_a, /*value=*/2.5F, stream);
    seed_constant(sys, block, seed_b, /*value=*/2.5F, stream);

    carbon::Skeleton sk_ref(sys, carbon::Skeleton::Options(carbon::OverlapMode::None));
    sk_ref.add(build_copy(block, seed_a, work_a, "ext_copy_none"));
    sk_ref.add(build_stencil(block, work_a, out_a, "ext_stencil_none"));
    carbon::Skeleton sk_occ(sys, carbon::Skeleton::Options(carbon::OverlapMode::Extended));
    sk_occ.add(build_copy(block, seed_b, work_b, "ext_copy_occ"));
    sk_occ.add(build_stencil(block, work_b, out_b, "ext_stencil_occ"));

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
      CARBON_LOG_ERROR("test_skeleton_block_extended[Extended]: skeleton run failed: {}", e.what());
      return 1;
    }

    auto host_ref = block.template download_to_host<float, 1>(out_a, stream);
    auto host_occ = block.template download_to_host<float, 1>(out_b, stream);
    stream.wait();
    if (!compare_active_voxels(block, host_ref, host_occ, kTol,
                               "test_skeleton_block_extended[Extended]")) {
      CARBON_LOG_ERROR("test_skeleton_block_extended[Extended]: parity vs None failed");
      return 1;
    }
  }

  // ---- Section B: TwoWayExtended ----
  {
    auto seed_a = block.template newField<float, 1>(carbon::MemorySide::Device, "twe_seed_a");
    auto seed_b = block.template newField<float, 1>(carbon::MemorySide::Device, "twe_seed_b");
    auto work_a = block.template newField<float, 1>(carbon::MemorySide::Device, "twe_work_a");
    auto work_b = block.template newField<float, 1>(carbon::MemorySide::Device, "twe_work_b");
    auto mid_a = block.template newField<float, 1>(carbon::MemorySide::Device, "twe_mid_a");
    auto mid_b = block.template newField<float, 1>(carbon::MemorySide::Device, "twe_mid_b");
    auto out_a = block.template newField<float, 1>(carbon::MemorySide::Device, "twe_out_a");
    auto out_b = block.template newField<float, 1>(carbon::MemorySide::Device, "twe_out_b");

    zero_fields(sys, block, {&work_a, &work_b, &mid_a, &mid_b, &out_a, &out_b}, stream);
    seed_constant(sys, block, seed_a, /*value=*/1.75F, stream);
    seed_constant(sys, block, seed_b, /*value=*/1.75F, stream);

    carbon::Skeleton sk_ref(sys, carbon::Skeleton::Options(carbon::OverlapMode::None));
    sk_ref.add(build_copy(block, seed_a, work_a, "twe_pre_none"));
    sk_ref.add(build_stencil(block, work_a, mid_a, "twe_stencil_none"));
    sk_ref.add(build_copy(block, mid_a, out_a, "twe_post_none"));
    carbon::Skeleton sk_occ(sys, carbon::Skeleton::Options(carbon::OverlapMode::TwoWayExtended));
    sk_occ.add(build_copy(block, seed_b, work_b, "twe_pre_occ"));
    sk_occ.add(build_stencil(block, work_b, mid_b, "twe_stencil_occ"));
    sk_occ.add(build_copy(block, mid_b, out_b, "twe_post_occ"));

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
      CARBON_LOG_ERROR("test_skeleton_block_extended[TwoWayExtended]: skeleton run failed: {}",
                       e.what());
      return 1;
    }

    auto host_ref = block.template download_to_host<float, 1>(out_a, stream);
    auto host_occ = block.template download_to_host<float, 1>(out_b, stream);
    stream.wait();
    if (!compare_active_voxels(block, host_ref, host_occ, kTol,
                               "test_skeleton_block_extended[TwoWayExtended]")) {
      CARBON_LOG_ERROR("test_skeleton_block_extended[TwoWayExtended]: parity vs None failed");
      return 1;
    }
  }

  CARBON_LOG_INFO("test_skeleton_block_extended: PASS (Extended + TwoWayExtended)");
  return 0;
}
