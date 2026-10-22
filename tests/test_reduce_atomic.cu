// ReduceFinalize::BlockAtomic vs default DeviceReduce on Dense sum reductions.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/stencil.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <utility>

namespace {

constexpr int64_t kWidth = 24;
constexpr int64_t kHeight = 32;
constexpr int64_t kDepth = 16;

[[nodiscard]] CARBON_HD auto
encode(int64_t x,
       int64_t y,
       int64_t z) -> int64_t {
  return z * 100000 + y * 100 + x - 50000;
}

[[nodiscard]] auto
host_reference_sum(int64_t W,
                   int64_t H,
                   int64_t D) -> int64_t {
  int64_t s = 0;
  for (int64_t z = 0; z < D; ++z) {
    for (int64_t y = 0; y < H; ++y) {
      for (int64_t x = 0; x < W; ++x) {
        s += encode(x, y, z);
      }
    }
  }
  return s;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_reduce_atomic: no CUDA devices visible");
    return EXIT_FAILURE;
  }

  carbon::System sys(carbon::get_all_devices());
  carbon::Stencil stencil({}, /*remove_center=*/false);
  carbon::Dense dense(sys, {kWidth, kHeight, kDepth}, stencil);
  auto field = dense.newField<int64_t, 1>(carbon::MemorySide::HostDevice, "f");

  field.fill_owned_host([](carbon::int64_3d g, int) {
    return encode(g.x, g.y, g.z);
  });

  carbon::Stream upload = sys.stream();
  try {
    carbon::Transfer<int64_t>(sys, field.cpu_buffers(), field.gpu_buffers())(upload);
    upload.wait();
  } catch (const std::exception& ex) {
    CARBON_LOG_ERROR("test_reduce_atomic: upload failed: {}", ex.what());
    return EXIT_FAILURE;
  }

  const int64_t expected = host_reference_sum(kWidth, kHeight, kDepth);

  carbon::ReductionResult<int64_t> dev_result(sys);
  carbon::ReductionResult<int64_t> atomic_result(sys);

  auto loader = [&field](carbon::KernelLoader& kl) {
    const auto in = kl.read(field);
    return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> int64_t {
      return in.read(s, idx, 0);
    };
  };

  auto dev_kernel = dense.make_reduce_kernel<int64_t>(loader, dev_result, carbon::reduce::Sum<int64_t>{});
  carbon::reduce::ReduceOptions atomic_opts{};
  atomic_opts.finalize = carbon::reduce::ReduceFinalize::BlockAtomic;
  auto atomic_kernel =
      dense.make_reduce_kernel<int64_t>(loader, atomic_result, carbon::reduce::Sum<int64_t>{},
                                        /*shared_mem=*/0U, atomic_opts);

  carbon::Stream st = sys.stream();
  try {
    dev_kernel.launch(st);
    dev_result.sync(st);
    st.wait();
    atomic_kernel.launch(st);
    atomic_result.sync(st);
    st.wait();
  } catch (const std::exception& ex) {
    CARBON_LOG_ERROR("test_reduce_atomic: launch failed: {}", ex.what());
    return EXIT_FAILURE;
  }

  if (dev_result.host_value() != expected) {
    CARBON_LOG_ERROR("test_reduce_atomic: device path got {} expected {}", dev_result.host_value(),
                     expected);
    return EXIT_FAILURE;
  }
  if (atomic_result.host_value() != expected) {
    CARBON_LOG_ERROR("test_reduce_atomic: atomic path got {} expected {}", atomic_result.host_value(),
                     expected);
    return EXIT_FAILURE;
  }

  // Float: atomic should match within tolerance vs default path.
  auto f_field = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "ff");
  f_field.fill_owned_host([](carbon::int64_3d g, int) {
    return static_cast<float>(encode(g.x, g.y, g.z));
  });
  try {
    carbon::Transfer<float>(sys, f_field.cpu_buffers(), f_field.gpu_buffers())(upload);
    upload.wait();
  } catch (const std::exception& ex) {
    CARBON_LOG_ERROR("test_reduce_atomic: float upload failed: {}", ex.what());
    return EXIT_FAILURE;
  }

  carbon::ReductionResult<float> f_dev(sys);
  carbon::ReductionResult<float> f_atomic(sys);
  auto f_loader = [&f_field](carbon::KernelLoader& kl) {
    const auto in = kl.read(f_field);
    return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> float {
      return in.read(s, idx, 0);
    };
  };
  auto f_dev_k = dense.make_reduce_kernel<float>(f_loader, f_dev);
  auto f_atomic_k = dense.make_reduce_kernel<float>(f_loader, f_atomic, carbon::reduce::Sum<float>{},
                                                    0U, atomic_opts);
  try {
    f_dev_k.launch(st);
    f_dev.sync(st);
    f_atomic_k.launch(st);
    f_atomic.sync(st);
    st.wait();
  } catch (const std::exception& ex) {
    CARBON_LOG_ERROR("test_reduce_atomic: float launch failed: {}", ex.what());
    return EXIT_FAILURE;
  }
  const float rel =
      std::fabs(f_dev.host_value() - f_atomic.host_value()) /
      std::fmax(1.0F, std::fabs(f_dev.host_value()));
  if (rel > 1.0e-5F) {
    CARBON_LOG_ERROR("test_reduce_atomic: float atomic vs device rel err {}", rel);
    return EXIT_FAILURE;
  }

  // Invalid: Min + BlockAtomic throws at construction.
  carbon::ReductionResult<int64_t> bad_dst(sys);
  bool threw = false;
  try {
    (void)dense.make_reduce_kernel<int64_t>(
        loader, bad_dst, carbon::reduce::Min<int64_t>{}, 0U, atomic_opts);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) {
    CARBON_LOG_ERROR("test_reduce_atomic: expected invalid_argument for Min+BlockAtomic");
    return EXIT_FAILURE;
  }

  if (carbon::device_count() >= 2) {
    carbon::ReductionResult<int64_t> mg_dev(sys);
    carbon::ReductionResult<int64_t> mg_atomic(sys);
    auto mg_dev_k = dense.make_reduce_kernel<int64_t>(loader, mg_dev);
    auto mg_atomic_k = dense.make_reduce_kernel<int64_t>(loader, mg_atomic, carbon::reduce::Sum<int64_t>{},
                                                         0U, atomic_opts);
    try {
      mg_dev_k.launch(st);
      mg_dev.sync(st);
      mg_atomic_k.launch(st);
      mg_atomic.sync(st);
      st.wait();
    } catch (const std::exception& ex) {
      CARBON_LOG_ERROR("test_reduce_atomic: multi-GPU launch failed: {}", ex.what());
      return EXIT_FAILURE;
    }
    if (mg_dev.host_value() != expected || mg_atomic.host_value() != expected) {
      CARBON_LOG_ERROR("test_reduce_atomic: multi-GPU mismatch dev={} atomic={} expected={}",
                       mg_dev.host_value(), mg_atomic.host_value(), expected);
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
