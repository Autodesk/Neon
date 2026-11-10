// Device-side test for spatially varying boundary data on carbon::DenseDistributed (single process,
// every visible device; on the SYCL omp backend the host is the device).
//
// A six-point neighbour-sum kernel reads every neighbour through Partition::neighbour_bc with a
// BoundaryData view, inside a real Skeleton, so the whole path is exercised: BoundaryData
// allocation, fill, upload, the per-GPU view taken in the loading lambda, and the device lambda. The
// result over every owned cell is compared with an independent host oracle.
//
// The conditions vary across faces: a Dirichlet rectangle in an otherwise Neumann ZHigh face, a
// Dirichlet disc in an otherwise Neumann XLow face, a Neumann strip on YHigh, and nothing on the
// remaining faces (those reads must come back invalid).

#include <carbon/boundary_data_distributed.h>
#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense_distributed.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <vector>

namespace {

using carbon::BcKind;
using carbon::DenseDistributed;
using carbon::Face;
using carbon::FaceBc;
using carbon::int64_3d;

constexpr double kInvalid = -999.0;
constexpr double kSentinel = 1.0e30;

[[nodiscard]] auto
field(int64_t x,
      int64_t y,
      int64_t z) -> double {
  const double fx = static_cast<double>(x);
  const double fy = static_cast<double>(y);
  const double fz = static_cast<double>(z);
  return 1.0 + 0.37 * fx + 0.11 * fy + 0.053 * fz + 0.013 * fx * fy + 0.007 * fy * fz -
         0.003 * fx * fz;
}

[[nodiscard]] auto
condition(Face f,
          const int64_3d& g) -> FaceBc {
  switch (f) {
    case Face::ZHigh:
      if (g.x >= 3 && g.x < 7 && g.y >= 2 && g.y < 6) {
        return carbon::dirichlet(2.0 + 0.1 * static_cast<double>(g.x));
      }
      return carbon::neumann(0.5);
    case Face::XLow: {
      const double dy = static_cast<double>(g.y) - 5.0;
      const double dz = static_cast<double>(g.z) - 5.0;
      if (dy * dy + dz * dz <= 3.2 * 3.2) {
        return carbon::dirichlet(-1.0 + 0.2 * static_cast<double>(g.z));
      }
      return carbon::neumann(0.0);
    }
    case Face::YHigh:
      return (g.z >= 3 && g.z < 6) ? carbon::neumann(-0.75) : FaceBc{};
    default:
      return {};
  }
}

// Independent reference for one neighbour read of owned cell g in direction d (one axis only).
[[nodiscard]] auto
expected_read(const int64_3d& ge,
              const int64_t g[3],
              const int d[3],
              double h,
              double& out) -> bool {
  const int64_t ext[3] = {ge.x, ge.y, ge.z};
  int64_t t[3] = {g[0] + d[0], g[1] + d[1], g[2] + d[2]};
  for (int a = 0; a < 3; ++a) {
    if (t[a] >= 0 && t[a] < ext[a]) {
      continue;
    }
    const bool low = t[a] < 0;
    const int64_t k = low ? -t[a] : t[a] - (ext[a] - 1);
    int64_t m[3] = {t[0], t[1], t[2]};
    m[a] = low ? k : (ext[a] - 1) - k;
    int64_t fc[3] = {t[0], t[1], t[2]};
    fc[a] = low ? 0 : ext[a] - 1;
    const FaceBc bc =
        condition(static_cast<Face>(2 * a + (low ? 0 : 1)), int64_3d{fc[0], fc[1], fc[2]});
    if (bc.kind == BcKind::None) {
      return false;
    }
    const double u_m = field(m[0], m[1], m[2]);
    out = bc.kind == BcKind::Dirichlet ? 2.0 * bc.value - u_m
                                       : u_m + 2.0 * static_cast<double>(k) * h * bc.value;
    return true;
  }
  out = field(t[0], t[1], t[2]);
  return true;
}

}  // namespace

auto
main() -> int {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_boundary_data: no devices visible");
    return 1;
  }
  carbon::System sys(carbon::get_all_devices());
  const int64_3d global{12, 12, 12};
  const float spacing = 0.5F;
  const carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  DenseDistributed grid(sys, global, stencil, carbon::BoundaryModeDistributed::Open,
                        carbon::float32_3d{}, spacing);

  carbon::BoundaryData<1> bnd(grid, "bc");
  bnd.fill([](Face f, int64_3d g, int) {
    return condition(f, g);
  });
  bnd.upload();

  auto in = grid.newField<double, 1>(carbon::MemorySide::HostDevice, "in");
  auto out = grid.newField<double, 1>(carbon::MemorySide::HostDevice, "out");
  for (size_t gpu = 0; gpu < grid.gpu_count(); ++gpu) {
    const auto& span = grid.spans()[gpu];
    const int64_3d halo = span.halo();
    const int64_3d owned = span.logic_extent();
    const int64_3d haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    double* hi = in.cpu_buffers()[gpu].data();
    double* ho = out.cpu_buffers()[gpu].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const bool own = x >= halo.x && x < halo.x + owned.x && y >= halo.y &&
                           y < halo.y + owned.y && z >= halo.z && z < halo.z + owned.z;
          const int64_3d g = span.global_location(span.make_idx(x, y, z));
          const int64_t off = z * pitch + y * haloed.x + x;
          hi[off] = own ? field(g.x, g.y, g.z) : kSentinel;
          ho[off] = kSentinel;
        }
      }
    }
  }
  carbon::Stream stream = sys.stream();
  try {
    carbon::Transfer<double>(sys, in.cpu_buffers(), in.gpu_buffers())(stream);
    carbon::Transfer<double>(sys, out.cpu_buffers(), out.gpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_boundary_data: upload failed: {}", e.what());
    return 1;
  }

  using Span = DenseDistributed::Span;
  using Idx = DenseDistributed::Idx;
  carbon::Skeleton sk(sys, carbon::Skeleton::Options(carbon::OverlapMode::None));
  sk.add(carbon::Kernel::compute(
      grid,
      [in, out, &bnd](carbon::KernelLoader& loader) {
        const auto pin = loader.read_stencil(in);
        const auto pout = loader.write(out);
        const auto bv = bnd.view(loader.relative_gpu_id());
        return [pin, pout, bv] CARBON_HOST_DEVICE(Span s, Idx idx) {
          const int dirs[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                  {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
          double acc = 0.0;
          for (const auto& d : dirs) {
            const auto n = pin.neighbour_bc(
                s, bv, idx,
                carbon::int8_3d{static_cast<int8_t>(d[0]), static_cast<int8_t>(d[1]),
                                static_cast<int8_t>(d[2])},
                0);
            acc += n.valid ? n.value : kInvalid;
          }
          pout.write(s, idx, 0, acc);
        };
      },
      "bc_neighbour_sum"));
  try {
    sk.compile();
    sk.run();
    sk.wait();
    carbon::Transfer<double>(sys, out.gpu_buffers(), out.cpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_boundary_data: skeleton or download failed: {}", e.what());
    return 1;
  }

  size_t cells = 0;
  size_t mismatches = 0;
  size_t boundary_reads = 0;
  size_t invalid_reads = 0;
  for (size_t gpu = 0; gpu < grid.gpu_count(); ++gpu) {
    const auto& span = grid.spans()[gpu];
    const int64_3d halo = span.halo();
    const int64_3d owned = span.logic_extent();
    const int64_3d haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const double* ho = out.cpu_buffers()[gpu].data();
    for (int64_t z = 0; z < owned.z; ++z) {
      for (int64_t y = 0; y < owned.y; ++y) {
        for (int64_t x = 0; x < owned.x; ++x) {
          const int64_3d g3 =
              span.global_location(span.make_idx(halo.x + x, halo.y + y, halo.z + z));
          const int64_t g[3] = {g3.x, g3.y, g3.z};
          static constexpr int kDirs[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                              {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
          double want = 0.0;
          for (const auto& d : kDirs) {
            double v = 0.0;
            if (expected_read(global, g, d, static_cast<double>(spacing), v)) {
              want += v;
              const int64_t t[3] = {g[0] + d[0], g[1] + d[1], g[2] + d[2]};
              boundary_reads += (t[0] < 0 || t[0] >= global.x || t[1] < 0 || t[1] >= global.y ||
                                 t[2] < 0 || t[2] >= global.z)
                                    ? 1U
                                    : 0U;
            } else {
              want += kInvalid;
              ++invalid_reads;
            }
          }
          const double got = ho[(halo.z + z) * pitch + (halo.y + y) * haloed.x + (halo.x + x)];
          ++cells;
          if (std::fabs(got - want) > 1e-9 * (1.0 + std::fabs(want))) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR("test_boundary_data: cell ({},{},{}) got {} want {}", g[0], g[1],
                               g[2], got, want);
            }
            ++mismatches;
          }
        }
      }
    }
  }
  if (boundary_reads == 0U || invalid_reads == 0U) {
    CARBON_LOG_ERROR(
        "test_boundary_data: the case must exercise both boundary reads ({}) and unconditioned "
        "faces ({})",
        boundary_reads, invalid_reads);
    return 1;
  }
  if (mismatches != 0U) {
    CARBON_LOG_ERROR("test_boundary_data: {} of {} cells differ from the oracle", mismatches,
                     cells);
    return 1;
  }
  CARBON_LOG_INFO(
      "test_boundary_data: {} cells match the oracle ({} boundary reads, {} invalid reads)", cells,
      boundary_reads, invalid_reads);
  return 0;
}
