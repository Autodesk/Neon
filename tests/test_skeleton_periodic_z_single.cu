// Single-GPU PeriodicZ through the Skeleton: the user kernels only touch owned cells and
// read z neighbours with read_stencil; periodicity must come entirely from the auto-inserted
// halo (single-slab self-wrap pair). Each run does
//   b = a(z+1) + 2 * a(z-1)   (asymmetric so a swapped wrap direction is visible)
//   a = b                     (re-dirties a, so the next run needs a fresh halo)
// and the result is compared against a host reference with wrapped z.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/skeleton.h>
#include <carbon/system.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using FieldT = carbon::Dense::Field<float, 1>;

constexpr carbon::int64_3d kGlobal{6, 3, 5};
constexpr int kRuns = 3;

[[nodiscard]] auto
initial_value(carbon::int64_3d g) -> float {
  return static_cast<float>(g.z * 100 + g.y * 10 + g.x);
}

[[nodiscard]] auto
host_index(carbon::int64_3d g) -> size_t {
  return static_cast<size_t>((g.z * kGlobal.y + g.y) * kGlobal.x + g.x);
}

[[nodiscard]] auto
host_reference() -> std::vector<float> {
  std::vector<float> a(static_cast<size_t>(kGlobal.x * kGlobal.y * kGlobal.z));
  for (int64_t z = 0; z < kGlobal.z; ++z) {
    for (int64_t y = 0; y < kGlobal.y; ++y) {
      for (int64_t x = 0; x < kGlobal.x; ++x) {
        a[host_index({x, y, z})] = initial_value({x, y, z});
      }
    }
  }
  std::vector<float> b(a.size());
  for (int r = 0; r < kRuns; ++r) {
    for (int64_t z = 0; z < kGlobal.z; ++z) {
      const int64_t zp = (z + 1) % kGlobal.z;
      const int64_t zm = (z + kGlobal.z - 1) % kGlobal.z;
      for (int64_t y = 0; y < kGlobal.y; ++y) {
        for (int64_t x = 0; x < kGlobal.x; ++x) {
          b[host_index({x, y, z})] = a[host_index({x, y, zp})] + 2.0F * a[host_index({x, y, zm})];
        }
      }
    }
    a = b;
  }
  return a;
}

auto
run_case(carbon::System& sys,
         carbon::OverlapMode overlap,
         const std::vector<float>& expected) -> bool {
  const char* tag = overlap == carbon::OverlapMode::None ? "none" : "standard";
  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, 1}, {0, 0, -1}},
                          /*remove_center=*/true);
  carbon::Dense dense(sys, kGlobal, std::move(stencil), carbon::BoundaryMode::PeriodicZ);
  FieldT a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "a");
  FieldT b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "b");
  a.fill_owned_host([](carbon::int64_3d g, int) {
    return initial_value(g);
  });
  b.fill_owned_host([](carbon::int64_3d, int) {
    return 0.0F;
  });
  carbon::Stream up = sys.stream();
  carbon::Transfer<float>(sys, a.cpu_buffers(), a.gpu_buffers())(up);
  carbon::Transfer<float>(sys, b.cpu_buffers(), b.gpu_buffers())(up);
  up.wait();

  carbon::Skeleton sk(sys, carbon::Skeleton::Options{overlap});
  sk.add(carbon::Kernel::compute(
      dense,
      [&a, &b](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(a);
        const auto out = loader.write(b);
        return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          const auto zp = in.neighbour(s, idx, carbon::int8_3d{0, 0, 1}, 0);
          const auto zm = in.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0);
          out.write(s, idx, 0,
                    (zp.valid ? zp.value : -1.0e6F) + 2.0F * (zm.valid ? zm.value : -1.0e6F));
        };
      },
      "shift_z"));
  sk.add(carbon::Kernel::compute(
      dense,
      [&a, &b](carbon::KernelLoader& loader) {
        const auto in = loader.read(b);
        const auto out = loader.write(a);
        return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          out.write(s, idx, 0, in.read(s, idx, 0));
        };
      },
      "copy_back"));
  sk.compile();

  const std::string order = sk.export_launch_order();
  if (order.find("auto_halo_") == std::string::npos) {
    CARBON_LOG_ERROR("test_skeleton_periodic_z_single[{}]: no auto halo in launch order\n{}", tag,
                     order);
    return false;
  }

  for (int r = 0; r < kRuns; ++r) {
    sk.run();
  }
  sk.wait();

  carbon::Stream down = sys.stream();
  carbon::Transfer<float>(sys, a.gpu_buffers(), a.cpu_buffers())(down);
  down.wait();

  const auto& span = dense.spans()[0];
  const auto haloed = span.haloed_extent();
  const int64_t pitch = span.xy_slab_elements();
  const int64_t halo_z = span.halo_z();
  const float* h = a.cpu_buffers()[0].data();
  size_t mismatches = 0;
  for (int64_t z = 0; z < kGlobal.z; ++z) {
    for (int64_t y = 0; y < kGlobal.y; ++y) {
      for (int64_t x = 0; x < kGlobal.x; ++x) {
        const float got = h[(z + halo_z) * pitch + y * haloed.x + x];
        const float want = expected[host_index({x, y, z})];
        if (std::fabs(got - want) > 1e-3F * std::fabs(want) + 1e-3F) {
          if (mismatches < 8U) {
            CARBON_LOG_ERROR("test_skeleton_periodic_z_single[{}]: ({}, {}, {}) got {} want {}",
                             tag, x, y, z, got, want);
          }
          ++mismatches;
        }
      }
    }
  }
  if (mismatches != 0U) {
    CARBON_LOG_ERROR("test_skeleton_periodic_z_single[{}]: {} mismatches", tag, mismatches);
    return false;
  }
  return true;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_WARN("test_skeleton_periodic_z_single: no visible CUDA devices; skipping");
    return EXIT_SUCCESS;
  }
  std::vector<carbon::detail::Gpu> one = carbon::get_all_devices();
  one.erase(one.begin() + 1, one.end());
  carbon::System sys(one);

  const std::vector<float> expected = host_reference();
  bool ok = true;
  ok = run_case(sys, carbon::OverlapMode::None, expected) && ok;
  ok = run_case(sys, carbon::OverlapMode::Standard, expected) && ok;
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
