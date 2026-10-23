#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/dense_disg.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/transfer.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace {

using DenseSpan = carbon::Dense::Span;
using DenseIdx = carbon::Dense::Idx;

using DisgSpan = carbon::DenseDisg::Span;
using DisgIdx = carbon::DenseDisg::Idx;

CARBON_HD auto
encoded(int64_t x,
        int64_t y,
        int64_t z,
        int c) -> std::uint32_t {
  return static_cast<std::uint32_t>(1000000U * static_cast<unsigned>(c) +
                                    10000U * static_cast<unsigned>(z) +
                                    100U * static_cast<unsigned>(y) + static_cast<unsigned>(x));
}

template <typename Part,
          typename Span,
          typename Idx>
__device__ void
check_cell_dev(const Part& part,
               const Span& span,
               const Idx& idx,
               int* err_count) {
  const auto cur = part.cursor(span, idx);
  for (int dz = -2; dz <= 2; ++dz) {
    for (int dy = -2; dy <= 2; ++dy) {
      for (int dx = -2; dx <= 2; ++dx) {
        const carbon::int8_3d off{static_cast<int8_t>(dx), static_cast<int8_t>(dy),
                                  static_cast<int8_t>(dz)};
        const auto cell = idx.cell();
        const int64_t bx = cell.x + dx;
        const int64_t by = cell.y + dy;
        const int64_t bz = cell.z + dz;
        const auto haloed = span.haloed_extent();
        if (bx < 0 || bx >= haloed.x || by < 0 || by >= haloed.y || bz < 0 || bz >= haloed.z) {
          continue;
        }
        for (int c = 0; c < 2; ++c) {
          const std::uint32_t want = part.read_pull(span, idx, off, c);
          const std::uint32_t got = cur.pull(off, c);
          if (got != want) {
            atomicAdd(err_count, 1);
          }
        }
      }
    }
  }
  for (int c = 0; c < 2; ++c) {
    if (cur.value(c) != part.read(span, idx, c)) {
      atomicAdd(err_count, 1);
    }
  }
}

auto
make_host_int(carbon::System& sys) -> carbon::Buffer<int> {
  std::vector<carbon::detail::Allocation<int>> v;
  v.reserve(sys.gpu_group().size());
  for (size_t i = 0; i < sys.gpu_group().size(); ++i) {
    v.push_back(carbon::Cpu::buffer<int>(1U));
  }
  return carbon::detail::BufferGroup<int>::from_vector(std::move(v));
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    return 0;
  }
  std::vector<carbon::detail::Gpu> one = carbon::get_all_devices();
  one.erase(one.begin() + 1, one.end());
  carbon::System sys(one);
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}, {1, 0, 0}, {-1, 0, 0}},
      /*remove_center=*/true);

  carbon::Dense dense(sys, carbon::int64_3d{8, 4, 8}, stencil);
  auto field = dense.newField<std::uint32_t, 2>(carbon::MemorySide::HostDevice);
  field.fill_owned_host([](carbon::int64_3d g, int c) {
    return encoded(g.x, g.y, g.z, c);
  });
  carbon::Stream up = sys.stream();
  carbon::Transfer<std::uint32_t>(sys, field.cpu_buffers(), field.gpu_buffers())(up);
  up.wait();

  carbon::Buffer<int> d_err = sys.gpu_group().buffer<int>({1U});
  carbon::Buffer<int> h_err = make_host_int(sys);

  int* const d_err_ptr = d_err[0].data();
  const auto launch_dense = [&](carbon::DataView view) {
    h_err[0].data()[0] = 0;
    carbon::Transfer<int>(sys, h_err, d_err)(up);
    up.wait();
    carbon::Kernel::compute(
        dense,
        [&field, d_err_ptr](carbon::KernelLoader& loader) {
          const auto part = loader.read(field);
          return [part, d_err_ptr] CARBON_HOST_DEVICE(DenseSpan s, DenseIdx idx) {
#if defined(__CUDA_ARCH__)
            check_cell_dev(part, s, idx, d_err_ptr);
#endif
          };
        },
        "check_dense_cursor")
        .launch(up, view);
    up.wait();
    carbon::Transfer<int>(sys, d_err, h_err)(up);
    up.wait();
    return h_err[0].data()[0];
  };

  if (launch_dense(carbon::DataView::Standard) != 0 ||
      launch_dense(carbon::DataView::Internal) != 0 ||
      launch_dense(carbon::DataView::Boundary) != 0) {
    CARBON_LOG_ERROR("test_partition_cursor: Dense cursor mismatch");
    return 1;
  }

  carbon::DenseDisg disg(sys, carbon::int64_3d{8, 4, 8}, /*halo_z=*/1);
  auto dfield = disg.newField<std::uint32_t, 2>(carbon::MemorySide::HostDevice);
  dfield.fill_owned_host([](carbon::int64_3d g, int c) {
    return encoded(g.x, g.y, g.z, c);
  });
  carbon::Transfer<std::uint32_t>(sys, dfield.cpu_buffers(), dfield.gpu_buffers())(up);
  up.wait();

  const auto launch_disg = [&](carbon::DataView view) {
    h_err[0].data()[0] = 0;
    carbon::Transfer<int>(sys, h_err, d_err)(up);
    up.wait();
    carbon::Kernel::compute(
        disg,
        [&dfield, d_err_ptr](carbon::KernelLoader& loader) {
          const auto part = loader.read(dfield);
          return [part, d_err_ptr] CARBON_HOST_DEVICE(DisgSpan s, DisgIdx idx) {
#if defined(__CUDA_ARCH__)
            check_cell_dev(part, s, idx, d_err_ptr);
#endif
          };
        },
        "check_disg_cursor")
        .launch(up, view);
    up.wait();
    carbon::Transfer<int>(sys, d_err, h_err)(up);
    up.wait();
    return h_err[0].data()[0];
  };

  if (launch_disg(carbon::DataView::Standard) != 0 ||
      launch_disg(carbon::DataView::Internal) != 0 ||
      launch_disg(carbon::DataView::Boundary) != 0) {
    CARBON_LOG_ERROR("test_partition_cursor: DenseDisg cursor mismatch");
    return 1;
  }

  return 0;
}
