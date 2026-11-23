// MPI cross-rank Skeleton region masks.
//
// Every rank owns one z slab. The domain is labelled walls (x == 0 || x == nx - 1) and
// interior; each half step of a Jacobi ping-pong is an interior stencil plus a masked wall
// kernel writing the same field, so with masks honoured the two run side by side while the
// next stencil (and its cross-rank auto-halo) still waits on both.
//
//   A. Masked pipeline under OverlapMode::None and ::Standard is bit-identical to the same
//      pipeline with Options::ignore_region_masks, and every rank issues the same launch
//      order (hash compared across ranks).
//   B. compile() and the exporters stay rank-local: rank 0 alone exports a halo pipeline
//      without blocking (the rank-consistency collective lives in run(), not compile()).
//   C. A rank-dependent RegionMask makes the first run() throw std::logic_error on every
//      rank instead of issuing NCCL collectives in different orders and deadlocking.
//
// Skipped (returns 0) when world_size < 2 or no GPUs are visible.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/distributed.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/neighbour_value.h>
#include <carbon/region_mask.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Field = carbon::Dense::Field<float, 1>;
using carbon::RegionMask;

constexpr int kCycles = 4;
const RegionMask kInterior = RegionMask::from_label(0);
const RegionMask kWalls = RegionMask::from_label(1);

struct Geo {
  int64_t nx;

  [[nodiscard]] CARBON_HD auto
  label(const carbon::int64_3d& g) const -> std::uint8_t {
    return (g.x == 0 || g.x == nx - 1) ? 1U : 0U;
  }
};

auto
fail(const std::string& msg) -> bool {
  CARBON_LOG_ERROR("test_skeleton_region_mask_dist: {}", msg);
  return false;
}

[[nodiscard]] auto
build_interior(carbon::Dense& dense,
               Field& fin,
               Field& fout,
               Geo geo,
               std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&fin, &fout, geo](carbon::KernelLoader& loader) {
        loader.region(kInterior);
        const auto in = loader.read_stencil(fin);
        const auto out = loader.write(fout);
        return [in, out, geo] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          if (geo.label(s.global_location(idx)) != 0U) {
            return;
          }
          float sum = 0.0F;
          float n = 0.0F;
          for (const carbon::NeighbourValue<float>& nb :
               {in.neighbour(s, idx, {1, 0, 0}, 0), in.neighbour(s, idx, {-1, 0, 0}, 0),
                in.neighbour(s, idx, {0, 1, 0}, 0), in.neighbour(s, idx, {0, -1, 0}, 0),
                in.neighbour(s, idx, {0, 0, 1}, 0), in.neighbour(s, idx, {0, 0, -1}, 0)}) {
            if (nb.valid) {
              sum += nb.value;
              n += 1.0F;
            }
          }
          out.write(s, idx, 0, sum / n);
        };
      },
      std::move(name),
      /*shared_mem=*/0U);
}

[[nodiscard]] auto
build_walls(carbon::Dense& dense,
            Field& fin,
            Field& fout,
            RegionMask mask,
            Geo geo,
            std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&fin, &fout, mask, geo](carbon::KernelLoader& loader) {
        loader.region(mask);
        const auto in = loader.read(fin);
        const auto out = loader.write(fout);
        return [in, out, geo] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          if (geo.label(s.global_location(idx)) != 1U) {
            return;
          }
          out.write(s, idx, 0, 0.5F * in.read(s, idx, 0) + 1.0F);
        };
      },
      std::move(name),
      /*shared_mem=*/0U);
}

auto
add_pipeline(carbon::Skeleton& sk,
             carbon::Dense& dense,
             Field& u,
             Field& v,
             Geo geo,
             RegionMask wall_mask = kWalls) -> void {
  sk.add(build_interior(dense, u, v, geo, "interior_uv"));
  sk.add(build_walls(dense, u, v, wall_mask, geo, "walls_v"));
  sk.add(build_interior(dense, v, u, geo, "interior_vu"));
  sk.add(build_walls(dense, v, u, wall_mask, geo, "walls_u"));
}

auto
init_field(const carbon::Dense& dense,
           Field& f,
           float seed) -> void {
  const auto& span = dense.spans()[0];
  const auto haloed = span.haloed_extent();
  const int64_t pitch = span.xy_slab_elements();
  float* h = f.cpu_buffers()[0].data();
  for (int64_t z = 0; z < haloed.z; ++z) {
    for (int64_t y = 0; y < haloed.y; ++y) {
      for (int64_t x = 0; x < haloed.x; ++x) {
        h[z * pitch + y * haloed.x + x] =
            seed + static_cast<float>(x + y * 11 + (span.z_offset() + z) * 113);
      }
    }
  }
}

[[nodiscard]] auto
fnv1a(const std::string& s) -> std::uint64_t {
  std::uint64_t h = 1469598103934665603ULL;
  for (const char c : s) {
    h ^= static_cast<unsigned char>(c);
    h *= 1099511628211ULL;
  }
  return h;
}

// Runs the masked pipeline (honoured or ignored) and returns the owned cells of u.
[[nodiscard]] auto
run_masked(carbon::System& sys,
           carbon::Dense& dense,
           Geo geo,
           carbon::Skeleton::Options opts,
           std::vector<float>& owned,
           std::string& launch_order,
           size_t& streams) -> bool {
  auto u = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u");
  auto v = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v");
  init_field(dense, u, 0.0F);
  init_field(dense, v, 5000.0F);
  carbon::Stream stream = sys.stream();
  try {
    carbon::Transfer<float>(sys, u.cpu_buffers(), u.gpu_buffers())(stream);
    carbon::Transfer<float>(sys, v.cpu_buffers(), v.gpu_buffers())(stream);
    stream.wait();

    carbon::Skeleton sk(sys, stream, opts);
    add_pipeline(sk, dense, u, v, geo);
    for (int it = 0; it < kCycles; ++it) {
      sk.run();
    }
    sk.wait();
    launch_order = sk.export_launch_order();
    streams = sk.num_streams();

    carbon::Transfer<float>(sys, u.gpu_buffers(), u.cpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    return fail(std::string("masked pipeline threw: ") + e.what());
  }
  const auto& span = dense.spans()[0];
  const int64_t begin = span.halo_z() * span.xy_slab_elements();
  const int64_t end = begin + span.logic_extent().z * span.xy_slab_elements();
  const float* h = u.cpu_buffers()[0].data();
  owned.assign(h + begin, h + end);
  return true;
}

[[nodiscard]] auto
subtest_numerics(carbon::System& sys,
                 const carbon::Distributed& dist,
                 carbon::Dense& dense,
                 Geo geo) -> bool {
  carbon::Skeleton::Options ignored;
  ignored.ignore_region_masks = true;
  std::vector<float> ref;
  std::string ref_order;
  size_t ref_streams = 0;
  if (!run_masked(sys, dense, geo, ignored, ref, ref_order, ref_streams)) {
    return false;
  }
  for (const carbon::OverlapMode mode :
       {carbon::OverlapMode::None, carbon::OverlapMode::Standard}) {
    std::vector<float> got;
    std::string order;
    size_t streams = 0;
    if (!run_masked(sys, dense, geo, carbon::Skeleton::Options{mode}, got, order, streams)) {
      return false;
    }
    const std::string tag = "overlap=" + std::to_string(static_cast<int>(mode));
    if (mode == carbon::OverlapMode::None && streams < 2U) {
      return fail(tag + ": expected the wall kernel on its own stream");
    }
    if (got != ref) {
      return fail(tag + ": masked result differs from the whole-field schedule");
    }
    if (!dist.all_ranks_equal(fnv1a(order))) {
      return fail(tag + ": ranks issued different launch orders:\n" + order);
    }
  }
  return true;
}

[[nodiscard]] auto
subtest_rank_local_export(carbon::System& sys,
                          const carbon::Distributed& dist,
                          carbon::Dense& dense,
                          Geo geo) -> bool {
  auto u = dense.newField<float, 1>(carbon::MemorySide::Device, "u");
  auto v = dense.newField<float, 1>(carbon::MemorySide::Device, "v");
  carbon::Skeleton sk(sys);
  add_pipeline(sk, dense, u, v, geo);
  if (dist.world_rank() == 0) {
    const std::string dot = sk.export_kernel_list_dot("rank0_only");
    if (dot.find("region {1}") == std::string::npos) {
      return fail("rank-local export: missing region annotation");
    }
  }
  return true;
}

[[nodiscard]] auto
subtest_rank_mismatch(carbon::System& sys,
                      const carbon::Distributed& dist,
                      carbon::Dense& dense,
                      Geo geo) -> bool {
  auto u = dense.newField<float, 1>(carbon::MemorySide::Device, "u");
  auto v = dense.newField<float, 1>(carbon::MemorySide::Device, "v");
  carbon::Skeleton sk(sys);
  // Same cells, different declaration on rank 1: a contract violation the check must catch.
  add_pipeline(sk, dense, u, v, geo, dist.world_rank() == 1 ? RegionMask::bit(7) : kWalls);
  bool threw = false;
  try {
    sk.run();
    sk.wait();
  } catch (const std::logic_error& e) {
    threw = std::string(e.what()).find("ranks added different pipelines") != std::string::npos;
  }
  if (!dist.all_ranks_equal(threw ? 1U : 0U) || !threw) {
    return fail("rank-dependent mask: expected std::logic_error on every rank");
  }
  return true;
}

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_skeleton_region_mask_dist: world_size={} < 2, skipping",
                    distributed->world_size());
    return EXIT_SUCCESS;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_skeleton_region_mask_dist: no GPU devices, skipping");
    return EXIT_SUCCESS;
  }

  carbon::System sys(distributed);
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  const carbon::int64_3d global{8, 5, 4 * static_cast<int64_t>(distributed->world_size())};
  carbon::Dense dense(sys, global, std::move(stencil));
  const Geo geo{global.x};

  const bool ok = subtest_numerics(sys, *distributed, dense, geo) &&
                  subtest_rank_local_export(sys, *distributed, dense, geo) &&
                  subtest_rank_mismatch(sys, *distributed, dense, geo);
  // Agree on the verdict so a one-rank failure cannot leave the peer blocked or green.
  const bool all_ok = distributed->all_ranks_equal(ok ? 1U : 0U) && ok;
  if (!all_ok) {
    return EXIT_FAILURE;
  }
  CARBON_LOG_INFO("test_skeleton_region_mask_dist: OK on rank {}/{}", distributed->world_rank(),
                  distributed->world_size());
  return EXIT_SUCCESS;
}
