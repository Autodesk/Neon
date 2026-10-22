// Host (OpenMP) compute + reduction parity test for the disaggregated dense grid.
//
// Exercises:
//   - carbon::DenseDisg::make_host_kernel : routes through Kernel::host_compute,
//     which dispatches on Grid::BlockPolicy (absent here) → DenseHostKernelImpl,
//     which uses @c auto Span / @c auto ValidatedIdx so the disg Span's
//     iter_extent / set_and_validate / Idx work without a parallel impl.
//   - carbon::DenseDisg::make_host_reduce_kernel : same idea for the host reduce
//     path (ReduceHostKernelImpl + thread-local partials + sequential combine).
//   - carbon::DenseDisg::make_universal_kernel + launch_on_host : the dual-target
//     compute kernel; its host body shares the same generic launch logic.
//
// Single-GPU (forced) so we can hold the per-cell expected value in the host
// reference without worrying about cross-slab fences.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense_disg.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>

#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

namespace {

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

constexpr int kComponents = 1;

[[nodiscard]] CARBON_HD auto
encode(int64_t x,
       int64_t y,
       int64_t gz) -> int64_t {
  return gz * 100000 + y * 100 + x;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_dense_disg_host: no CUDA devices");
    return 1;
  }

  std::vector<carbon::detail::Gpu> all = carbon::get_all_devices();
  std::vector<carbon::detail::Gpu> single;
  single.push_back(std::move(all.front()));
  carbon::System sys(std::move(single));

  const carbon::int64_3d global{8, 6, 6};
  const carbon::DenseDisg dense(sys, global, /*halo_z=*/1);

  auto u_in = dense.newField<int64_t, kComponents>(carbon::MemorySide::Host);
  auto u_out = dense.newField<int64_t, kComponents>(carbon::MemorySide::Host);
  auto u_out_universal = dense.newField<int64_t, kComponents>(carbon::MemorySide::Host);

  const auto& span = dense.spans()[0];
  const auto haloed = span.haloed_extent();
  const int64_t halo_z = span.halo_z();
  const int64_t z_origin = span.z_offset();
  const int64_t lim_z = span.logic_extent().z;

  {
    int64_t* h = u_in.cpu_buffers()[0].data();
    for (size_t j = 0; j < u_in.cpu_buffers()[0].size(); ++j) {
      h[j] = 0;
    }
    for (int64_t bz = halo_z; bz < halo_z + lim_z; ++bz) {
      const int64_t gz = z_origin + bz;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = disg_addr(span, bz, x, y, 0, kComponents);
          h[off] = encode(x, y, gz);
        }
      }
    }
  }

  carbon::Stream stream = sys.stream();

  // (1) Host compute: out[i] = 2 * in[i].
  const auto host_kernel = dense.make_host_kernel([&u_in, &u_out](carbon::KernelLoader& loader) {
    const auto in = loader.read(u_in);
    const auto out = loader.write(u_out);
    return [in, out](carbon::DenseDisg::Span s, carbon::DenseDisg::Idx idx) -> void {
      out.write(s, idx, 0, static_cast<int64_t>(2) * in.read(s, idx, 0));
    };
  });

  try {
    host_kernel.launch(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_host: host compute kernel failed: {}", e.what());
    return 1;
  }

  // (2) Universal compute on host side — same formula, different factory; confirms the
  //     UniversalKernelImpl host branch routes through the disg span correctly.
  const auto universal =
      dense.make_universal_kernel([&u_in, &u_out_universal](carbon::KernelLoader& loader) {
        const auto in = loader.read(u_in);
        const auto out = loader.write(u_out_universal);
        return [in, out] CARBON_HOST_DEVICE(carbon::DenseDisg::Span s,
                                            carbon::DenseDisg::Idx idx) -> void {
          out.write(s, idx, 0, static_cast<int64_t>(2) * in.read(s, idx, 0));
        };
      });

  try {
    universal.launch_on_host(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_host: universal host launch failed: {}", e.what());
    return 1;
  }

  // Verify both outputs match 2 * encode(x, y, gz) on every owned cell.
  size_t mismatches = 0;
  size_t mismatches_u = 0;
  const int64_t* hh = u_out.cpu_buffers()[0].data();
  const int64_t* hu = u_out_universal.cpu_buffers()[0].data();
  for (int64_t bz = halo_z; bz < halo_z + lim_z; ++bz) {
    const int64_t gz = z_origin + bz;
    for (int64_t y = 0; y < haloed.y; ++y) {
      for (int64_t x = 0; x < haloed.x; ++x) {
        const int64_t off = disg_addr(span, bz, x, y, 0, kComponents);
        const int64_t want = 2 * encode(x, y, gz);
        if (hh[off] != want) {
          ++mismatches;
        }
        if (hu[off] != want) {
          ++mismatches_u;
        }
      }
    }
  }
  if (mismatches > 0U || mismatches_u > 0U) {
    CARBON_LOG_ERROR("test_dense_disg_host: host kernel mismatches: compute={} universal={}",
                     mismatches, mismatches_u);
    return 1;
  }

  // (3) Host reduction: sum of u_out, computed both on host and against a reference.
  int64_t reference = 0;
  for (int64_t bz = halo_z; bz < halo_z + lim_z; ++bz) {
    const int64_t gz = z_origin + bz;
    for (int64_t y = 0; y < haloed.y; ++y) {
      for (int64_t x = 0; x < haloed.x; ++x) {
        reference += 2 * encode(x, y, gz);
      }
    }
  }

  carbon::ReductionResult<int64_t> dst(sys);
  const auto host_reduce = dense.make_host_reduce_kernel<int64_t>(
      [&u_out](carbon::KernelLoader& loader) {
        const auto in = loader.read(u_out);
        return [in](carbon::DenseDisg::Span s, carbon::DenseDisg::Idx idx) -> int64_t {
          return in.read(s, idx, 0);
        };
      },
      dst, carbon::reduce::Sum<int64_t>{});

  try {
    host_reduce.launch(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_host: host reduce failed: {}", e.what());
    return 1;
  }
  const int64_t got = dst.host_value();
  if (got != reference) {
    CARBON_LOG_ERROR("test_dense_disg_host: host reduce got={} want={}", got, reference);
    return 1;
  }

  CARBON_LOG_INFO("test_dense_disg_host: ok (host compute + universal-host + host reduce, sum={})",
                  got);
  return 0;
}
