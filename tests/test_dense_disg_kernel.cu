// Single-GPU compute-kernel + neighbour smoke test for the disaggregated dense grid.
//
// Mirrors test_kernel_laplacian_single_gpu.cu: a 7-point discrete Laplacian over a
// quadratic input that has an exact analytic Laplacian (2 * (a + b + c)). The novelty
// here is exercising:
//   * carbon::DenseDisg::make_kernel — wires DisgPolicy through the compile-time
//     dispatch inside DeviceKernelImpl and reaches detail::dense_disg::launch_disg_grid.
//   * carbon::DenseDisg::Partition::neighbour — uses detail::dense_disg::neighbour_idx
//     (with the per-region pitch metadata populated on the returned Idx) and the
//     disaggregated Span::linear_offset.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense_disg.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>

#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

namespace {

constexpr int64_t kA = 1;
constexpr int64_t kB = 2;
constexpr int64_t kC = 3;
constexpr int64_t kExpected = 2 * (kA + kB + kC);

template <typename T>
[[nodiscard]] CARBON_HD auto
analytic_input(int64_t global_x,
               int64_t global_y,
               int64_t global_z) -> T {
  return static_cast<T>(kA * global_x * global_x + kB * global_y * global_y +
                        kC * global_z * global_z);
}

// Address for cell (bz, x, y, c) in slab `span`'s disaggregated buffer; copied from
// test_dense_disg_multi_gpu.cu so the host-side prefill matches what the device
// kernel reads through Partition::read / Partition::neighbour.
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

template <typename T>
[[nodiscard]] auto
run_laplacian_test(carbon::System& sys,
                   const char* tag) -> bool {
  const carbon::int64_3d global{16, 16, 16};
  const carbon::DenseDisg dense(sys, global, /*halo_z=*/1);
  if (dense.gpu_count() != 1U) {
    CARBON_LOG_ERROR("{}: expected single-GPU run, got gpu_count={}", tag, dense.gpu_count());
    return false;
  }

  const auto& span = dense.spans()[0];
  if (span.halo_z() != 1) {
    CARBON_LOG_ERROR("{}: expected halo_z=1, got {}", tag, span.halo_z());
    return false;
  }

  auto u_in = dense.newField<T, 1>(carbon::MemorySide::HostDevice);
  auto u_out = dense.newField<T, 1>(carbon::MemorySide::HostDevice);

  // Fill u_in's host mirror over the full haloed extent using the disaggregated address
  // formula so every region's storage carries the analytic value at the right (x,y,gz).
  {
    const auto haloed = span.haloed_extent();
    const int64_t z_origin = span.z_offset();
    T* h = u_in.cpu_buffers()[0].data();
    for (int64_t bz = 0; bz < haloed.z; ++bz) {
      const int64_t gz = z_origin + bz;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = disg_addr(span, bz, x, y, 0, /*components=*/1);
          h[off] = analytic_input<T>(x, y, gz);
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

  const auto kernel = dense.make_kernel(
      [&u_in, &u_out](carbon::KernelLoader& loader) {
        const auto in = loader(u_in);
        const auto out = loader(u_out);
        return [in, out] CARBON_HOST_DEVICE(carbon::DenseDisg::Span s, carbon::DenseDisg::Idx idx) {
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

  // Sentinel-fill then download.
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
  // Boundary cells get an out-of-domain neighbour read substituted with 0 — skip them.
  const T expected = static_cast<T>(kExpected);
  const auto lim = span.logic_extent();
  const int64_t halo_z = span.halo_z();
  const T* h = u_out.cpu_buffers()[0].data();
  size_t mismatches = 0;
  for (int64_t z = 1; z < lim.z - 1; ++z) {
    const int64_t bz = halo_z + z;
    for (int64_t y = 1; y < lim.y - 1; ++y) {
      for (int64_t x = 1; x < lim.x - 1; ++x) {
        const int64_t off = disg_addr(span, bz, x, y, 0, /*components=*/1);
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
    CARBON_LOG_ERROR("test_dense_disg_kernel: no CUDA devices");
    return 1;
  }

  // Force single-GPU; multi-GPU coverage lives in test_dense_disg_kernel_multi_gpu.cu.
  std::vector<carbon::detail::Gpu> all = carbon::get_all_devices();
  std::vector<carbon::detail::Gpu> single;
  single.push_back(std::move(all.front()));
  carbon::System sys(std::move(single));

  const bool ok_int = run_laplacian_test<std::int32_t>(sys, "disg_laplacian<int32_t>");
  const bool ok_flt = run_laplacian_test<float>(sys, "disg_laplacian<float>");
  return (ok_int && ok_flt) ? 0 : 1;
}
