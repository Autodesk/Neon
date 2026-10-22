// Single-GPU 7-point discrete Laplacian on a quadratic input — first end-to-end numerical
// kernel test in carbon.
//
// Setup
// -----
// Forces a one-GPU System (slices the first device off carbon::get_all_devices()) so this test
// does not depend on any halo-exchange machinery: the slab covers the whole logical domain. We
// still build Dense with a radius-1 Stencil so storage carries one z-halo on each side. Those
// halos are populated on the host as a no-op smoke check of halo-region writes; today they are
// not read, because Partition::neighbour treats the top/bottom slab's halo cells as
// out-of-domain in a non-toroidal grid (global z would be -1 or global.z).
//
// The input field is the quadratic
//     f(x, y, z) = a * x^2 + b * y^2 + c * z^2
// whose 7-point discrete Laplacian
//     Δf = f(x+1) + f(x-1) + f(y+1) + f(y-1) + f(z+1) + f(z-1) - 6 * f(x, y, z)
// equals exactly 2 * (a + b + c) at every cell — independent of (x, y, z), and exact in
// integer arithmetic (and exactly representable in float for our small ranges).
//
// The kernel cannot read across any slab boundary in any axis: x/y carry no halo at all, and
// for z the only slab is the boundary slab in both directions (single-GPU), so the upper /
// lower halos are out-of-domain. The kernel substitutes 0 there via NeighbourValue::value_or;
// the verifier skips those boundary cells (their computed Laplacian is intentionally garbage)
// and checks the strict (x, y, z) interior exhaustively.
//
// Two runs
// --------
// One pass with T = int32_t and one with T = float, against the same expected value. The
// integer pass is a tight reference (no rounding ever happens), and the float pass cross-checks
// that float arithmetic produces the same result for these small magnitudes (< 2^14, well below
// 2^24 where float32 still represents integers exactly).

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/stencil.h>

#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

namespace {

constexpr int64_t kA = 1;
constexpr int64_t kB = 2;
constexpr int64_t kC = 3;
constexpr int64_t kExpected = 2 * (kA + kB + kC);

// Analytic input over global integer coordinates. Kept as a function so the host fill and the
// "what should the Laplacian be" reference share the exact same encoding.
template <typename T>
[[nodiscard]] CARBON_HD auto
analytic_input(int64_t global_x,
               int64_t global_y,
               int64_t global_z) -> T {
  return static_cast<T>(kA * global_x * global_x + kB * global_y * global_y +
                        kC * global_z * global_z);
}

// Run one pass of the test for element type T. Returns true on success.
template <typename T>
[[nodiscard]] auto
run_laplacian_test(carbon::System& sys,
                   const char* tag) -> bool {
  // 7-point stencil with center dropped (the kernel handles the center explicitly).
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  if (stencil.radius() != 1) {
    CARBON_LOG_ERROR("{}: unexpected stencil radius {}", tag, stencil.radius());
    return false;
  }

  // Modest size, divisible by 2/4/8 so the launch grid is regular. The strict interior in all
  // three axes (skipping each global-domain boundary) is 14 * 14 * 14 = 2744 verified cells.
  const carbon::int64_3d global{16, 16, 16};
  const carbon::Dense dense(sys, global, stencil);
  if (dense.gpu_count() != 1U) {
    CARBON_LOG_ERROR("{}: expected single-GPU run, got gpu_count={}", tag, dense.gpu_count());
    return false;
  }

  const auto& span = dense.spans()[0];
  if (span.halo_z() != 1) {
    CARBON_LOG_ERROR("{}: expected halo_z=1, got {}", tag, span.halo_z());
    return false;
  }

  // Two scalar fields, both Host+Device, so we can fill on host, upload, run, download, verify.
  // Components=1 follows the Field<T, Components> docstring (1 = scalar, 3 = vector, ...).
  auto u_in = dense.newField<T, 1>(carbon::MemorySide::HostDevice);
  auto u_out = dense.newField<T, 1>(carbon::MemorySide::HostDevice);

  // Initialize u_in's host mirror over the full haloed extent (lower halo + owned + upper
  // halo). The analytic polynomial is defined for negative / over-extent z too, so we fill the
  // ghost layers with their "natural" values; Partition::neighbour ignores them on this single
  // -GPU run (top/bottom slab in a non-toroidal grid → out-of-domain), but the prefill is left
  // in to exercise the halo-region storage and document what an exchanged halo would carry.
  {
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t z_origin = span.z_offset();
    T* h = u_in.cpu_buffers()[0].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      const int64_t global_z = z_origin + z;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          h[off] = analytic_input<T>(x, y, global_z);
        }
      }
    }
  }

  carbon::Stream stream = sys.stream();

  carbon::Transfer<T> upload(sys, u_in.cpu_buffers(), u_in.gpu_buffers());
  try {
    upload(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("{}: upload failed: {}", tag, e.what());
    return false;
  }

  // Launch the 7-point Laplacian. The launch grid iterates owned cells only. Each cell reads
  // its 6 neighbours through Partition::neighbour, which returns NeighbourValue<T> and marks
  // reads that fall outside the @b global Cartesian grid as invalid:
  //
  // - x, y: no halo, so cells at cell.x ∈ {0, lim.x-1} or cell.y ∈ {0, lim.y-1} get an invalid
  //   read on at least one neighbour.
  // - z: this is a single-GPU run, so the only slab is also the top and bottom slab; the
  //   upper/lower halo cells map to global z = global.z / global z = -1, both out of domain
  //   in a non-toroidal grid. Owned z=0 and z=lim.z-1 get invalid -z / +z reads. Interior
  //   slabs in a future multi-GPU run would instead see valid halo reads after exchange.
  //
  // value_or(T{0}) substitutes 0 for invalid reads; the verifier skips boundary cells in all
  // three axes (their computed Laplacian is intentionally garbage). No manual boundary check
  // in the kernel: neighbour() does the bounds work.
  const auto kernel = dense.make_kernel(
      [&u_in, &u_out](carbon::KernelLoader& loader) {
        const auto in = loader(u_in);
        const auto out = loader(u_out);
        return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          const T self = in.read(s, idx, 0);
          const T xp = in.neighbour(s, idx, {1, 0, 0}, 0).value_or(T{0});
          const T xm = in.neighbour(s, idx, {-1, 0, 0}, 0).value_or(T{0});
          const T yp = in.neighbour(s, idx, {0, 1, 0}, 0).value_or(T{0});
          const T ym = in.neighbour(s, idx, {0, -1, 0}, 0).value_or(T{0});
          const T zp = in.neighbour(s, idx, {0, 0, 1}, 0).value_or(T{0});
          const T zm = in.neighbour(s, idx, {0, 0, -1}, 0).value_or(T{0});

          const T lap = static_cast<T>(xp + xm + yp + ym + zp + zm - T{6} * self);
          out.write(s, idx, 0, lap);
        };
      },
      /*shared_mem=*/0U);

  try {
    kernel.launch(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("{}: kernel launch failed: {}", tag, e.what());
    return false;
  }

  try {
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("{}: stream.wait after kernel failed: {}", tag, e.what());
    return false;
  }

  // Sentinel-fill the host mirror so a missing download is caught immediately.
  {
    T* h = u_out.cpu_buffers()[0].data();
    const size_t n = u_out.cpu_buffers()[0].size();
    for (size_t i = 0; i < n; ++i) {
      h[i] = static_cast<T>(-1);
    }
  }

  carbon::Transfer<T> download(sys, u_out.gpu_buffers(), u_out.cpu_buffers());
  try {
    download(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("{}: download failed: {}", tag, e.what());
    return false;
  }
  try {
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("{}: stream.wait after download failed: {}", tag, e.what());
    return false;
  }

  // Verify: every owned cell on the strict interior in all three axes must equal 2*(a+b+c).
  // Boundary cells (x ∈ {0, lim.x-1}, y ∈ {0, lim.y-1}, z ∈ {0, lim.z-1}) had at least one
  // out-of-domain neighbour read substituted with 0 by value_or — we skip them on purpose.
  const T expected = static_cast<T>(kExpected);
  const auto haloed = span.haloed_extent();
  const auto lim = span.logic_extent();
  const int64_t pitch = span.xy_slab_elements();
  const int64_t halo_z = span.halo_z();
  const T* h = u_out.cpu_buffers()[0].data();
  size_t mismatches = 0;
  for (int64_t z = 1; z < lim.z - 1; ++z) {
    const int64_t cell_z = halo_z + z;  // buffer-local z
    for (int64_t y = 1; y < lim.y - 1; ++y) {
      for (int64_t x = 1; x < lim.x - 1; ++x) {
        const int64_t off = cell_z * pitch + y * haloed.x + x;
        const T got = h[off];
        if (got != expected) {
          if (mismatches < 4U) {
            CARBON_LOG_ERROR("{}: mismatch at (x,y,z)=({},{},{}) got={} want={}", tag, x, y, z,
                             static_cast<double>(got), static_cast<double>(expected));
          }
          ++mismatches;
        }
      }
    }
  }
  const size_t interior_count = static_cast<size_t>((lim.x - 2) * (lim.y - 2) * (lim.z - 2));
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("{}: {} interior mismatches out of {} cells checked", tag, mismatches,
                     interior_count);
    return false;
  }
  CARBON_LOG_INFO("{}: all {} interior cells == {}", tag, interior_count,
                  static_cast<double>(expected));
  return true;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_kernel_laplacian_single_gpu: no CUDA devices");
    return 1;
  }

  // Force a single-GPU System: take the first visible device only. This isolates the test
  // from any inter-GPU halo behaviour that does not exist yet.
  std::vector<carbon::detail::Gpu> all = carbon::get_all_devices();
  if (all.empty()) {
    CARBON_LOG_ERROR("test_kernel_laplacian_single_gpu: get_all_devices returned empty");
    return 1;
  }
  std::vector<carbon::detail::Gpu> single;
  single.push_back(std::move(all.front()));
  carbon::System sys(std::move(single));

  const bool ok_int = run_laplacian_test<std::int32_t>(sys, "laplacian<int32_t>");
  const bool ok_flt = run_laplacian_test<float>(sys, "laplacian<float>");
  return (ok_int && ok_flt) ? 0 : 1;
}
