// Coverage for the built-in carbon::reduce::{Min, Max, Prod} operators on a single-GPU
// Dense field. Mirrors test_reduce_sum.cu in shape: device launch via
// Dense::make_reduce_kernel with the matching Op, host reference computed over the
// same per-cell encoder, exact bit comparison for integer types, and the empty-domain
// branch (which must produce the operator's identity element verbatim).
//
// Sanity-checks the identity values in particular — Min over an empty domain must return
// std::numeric_limits<T>::max(), Max over an empty domain must return
// std::numeric_limits<T>::lowest(), Prod over an empty domain must return T{1} — which
// is the contract every cross-rank op_kind specialisation also relies on.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/stencil.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <utility>
#include <vector>

namespace {

constexpr int64_t kWidth = 8;
constexpr int64_t kHeight = 6;
constexpr int64_t kDepth = 4;

// Per-cell encoder chosen so each cell carries a unique signed value that fits well
// inside int32 when summed/multiplied at this extent. We deliberately drop the
// `- 50000` bias from test_reduce_sum's encoder so values stay positive — Prod of
// zero would short-circuit and hide bugs in the per-block reduce, and for Min/Max
// the comparison logic is trivial either way.
[[nodiscard]] CARBON_HD auto
encode(int64_t x,
       int64_t y,
       int64_t z) -> std::int32_t {
  return static_cast<std::int32_t>(z * 100 + y * 10 + x + 1);
}

// Float encoder for Prod — keeps each contribution slightly above 1 so the running
// product stays in finite range. With 192 cells at extents (8, 6, 4) and per-cell
// values in [1.0, 1.04], the host reference product fits comfortably in double.
[[nodiscard]] CARBON_HD auto
encode_double(int64_t x,
              int64_t y,
              int64_t z) -> double {
  return 1.0 + (static_cast<double>(x + y * kWidth + z * kWidth * kHeight) /
                static_cast<double>(kWidth * kHeight * kDepth)) *
                   0.04;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_reduce_ops: no CUDA devices visible");
    return EXIT_FAILURE;
  }

  std::vector<carbon::detail::Gpu> all = carbon::get_all_devices();
  std::vector<carbon::detail::Gpu> single;
  single.push_back(std::move(all.front()));
  carbon::System sys(std::move(single));

  carbon::Stencil stencil{};
  const carbon::int64_3d global{kWidth, kHeight, kDepth};
  carbon::Dense dense(sys, global, stencil);

  // Pre-compute host references over the global cell list.
  std::int32_t host_min = std::numeric_limits<std::int32_t>::max();
  std::int32_t host_max = std::numeric_limits<std::int32_t>::lowest();
  for (int64_t z = 0; z < kDepth; ++z) {
    for (int64_t y = 0; y < kHeight; ++y) {
      for (int64_t x = 0; x < kWidth; ++x) {
        host_min = std::min(host_min, encode(x, y, z));
        host_max = std::max(host_max, encode(x, y, z));
      }
    }
  }
  double host_prod = 1.0;
  for (int64_t z = 0; z < kDepth; ++z) {
    for (int64_t y = 0; y < kHeight; ++y) {
      for (int64_t x = 0; x < kWidth; ++x) {
        host_prod *= encode_double(x, y, z);
      }
    }
  }

  // Shared int32 field for Min/Max.
  auto i_field = dense.newField<std::int32_t, 1>(carbon::MemorySide::HostDevice);
  i_field.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
    return encode(gc.x, gc.y, gc.z);
  });
  carbon::Stream stream = sys.stream();
  try {
    carbon::Transfer<std::int32_t>(sys, i_field.cpu_buffers(), i_field.gpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_reduce_ops: int32 upload failed: {}", e.what());
    return EXIT_FAILURE;
  }

  // ---- Min ----
  {
    carbon::ReductionResult<std::int32_t> r(sys);
    auto k = dense.make_reduce_kernel<std::int32_t>(
        [&i_field](carbon::KernelLoader& loader) {
          const auto in = loader(i_field);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s,
                                         carbon::Dense::Idx idx) -> std::int32_t {
            return in.read(s, idx, 0);
          };
        },
        r, carbon::reduce::Min<std::int32_t>{});
    try {
      k.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_ops: Min launch failed: {}", e.what());
      return EXIT_FAILURE;
    }
    r.sync(stream);
    if (r.host_value() != host_min) {
      CARBON_LOG_ERROR("test_reduce_ops: Min mismatch: got {}, expected {}", r.host_value(),
                       host_min);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_reduce_ops: Min = {}", r.host_value());
  }

  // ---- Max ----
  {
    carbon::ReductionResult<std::int32_t> r(sys);
    auto k = dense.make_reduce_kernel<std::int32_t>(
        [&i_field](carbon::KernelLoader& loader) {
          const auto in = loader(i_field);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s,
                                         carbon::Dense::Idx idx) -> std::int32_t {
            return in.read(s, idx, 0);
          };
        },
        r, carbon::reduce::Max<std::int32_t>{});
    try {
      k.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_ops: Max launch failed: {}", e.what());
      return EXIT_FAILURE;
    }
    r.sync(stream);
    if (r.host_value() != host_max) {
      CARBON_LOG_ERROR("test_reduce_ops: Max mismatch: got {}, expected {}", r.host_value(),
                       host_max);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_reduce_ops: Max = {}", r.host_value());
  }

  // ---- Prod over a double field ----
  {
    auto d_field = dense.newField<double, 1>(carbon::MemorySide::HostDevice);
    d_field.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
      return encode_double(gc.x, gc.y, gc.z);
    });
    try {
      carbon::Transfer<double>(sys, d_field.cpu_buffers(), d_field.gpu_buffers())(stream);
      stream.wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_ops: double upload failed: {}", e.what());
      return EXIT_FAILURE;
    }

    carbon::ReductionResult<double> r(sys);
    auto k = dense.make_reduce_kernel<double>(
        [&d_field](carbon::KernelLoader& loader) {
          const auto in = loader(d_field);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> double {
            return in.read(s, idx, 0);
          };
        },
        r, carbon::reduce::Prod<double>{});
    try {
      k.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_ops: Prod launch failed: {}", e.what());
      return EXIT_FAILURE;
    }
    r.sync(stream);
    const double got = r.host_value();
    const double rel = std::abs(got - host_prod) / std::abs(host_prod);
    constexpr double kRelTol = 1e-9;
    if (rel > kRelTol) {
      CARBON_LOG_ERROR(
          "test_reduce_ops: Prod mismatch: got {:.10g}, expected {:.10g} (rel diff {:.3e})", got,
          host_prod, rel);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_reduce_ops: Prod = {:.10g} (rel diff {:.3e})", got, rel);
  }

  // ---- Empty-domain branches ----
  // A zero-extent grid must return the operator's identity element. This locks the
  // identity contract every NCCL op_kind specialisation relies on.
  {
    const carbon::int64_3d empty_global{kWidth, kHeight, 0};
    carbon::Dense empty_dense(sys, empty_global, carbon::Stencil{});
    auto field = empty_dense.newField<std::int32_t, 1>(carbon::MemorySide::Device);

    carbon::ReductionResult<std::int32_t> r_min(sys);
    auto k_min = empty_dense.make_reduce_kernel<std::int32_t>(
        [&field](carbon::KernelLoader& loader) {
          const auto in = loader(field);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s,
                                         carbon::Dense::Idx idx) -> std::int32_t {
            return in.read(s, idx, 0);
          };
        },
        r_min, carbon::reduce::Min<std::int32_t>{});
    k_min.launch(stream);
    r_min.sync(stream);
    if (r_min.host_value() != std::numeric_limits<std::int32_t>::max()) {
      CARBON_LOG_ERROR(
          "test_reduce_ops: Min empty-domain expected std::numeric_limits<int32>::max() ({}), "
          "got {}",
          std::numeric_limits<std::int32_t>::max(), r_min.host_value());
      return EXIT_FAILURE;
    }

    carbon::ReductionResult<std::int32_t> r_max(sys);
    auto k_max = empty_dense.make_reduce_kernel<std::int32_t>(
        [&field](carbon::KernelLoader& loader) {
          const auto in = loader(field);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s,
                                         carbon::Dense::Idx idx) -> std::int32_t {
            return in.read(s, idx, 0);
          };
        },
        r_max, carbon::reduce::Max<std::int32_t>{});
    k_max.launch(stream);
    r_max.sync(stream);
    if (r_max.host_value() != std::numeric_limits<std::int32_t>::lowest()) {
      CARBON_LOG_ERROR(
          "test_reduce_ops: Max empty-domain expected std::numeric_limits<int32>::lowest() "
          "({}), got {}",
          std::numeric_limits<std::int32_t>::lowest(), r_max.host_value());
      return EXIT_FAILURE;
    }

    auto df = empty_dense.newField<double, 1>(carbon::MemorySide::Device);
    carbon::ReductionResult<double> r_prod(sys);
    auto k_prod = empty_dense.make_reduce_kernel<double>(
        [&df](carbon::KernelLoader& loader) {
          const auto in = loader(df);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> double {
            return in.read(s, idx, 0);
          };
        },
        r_prod, carbon::reduce::Prod<double>{});
    k_prod.launch(stream);
    r_prod.sync(stream);
    if (r_prod.host_value() != 1.0) {
      CARBON_LOG_ERROR("test_reduce_ops: Prod empty-domain expected 1.0, got {:.10g}",
                       r_prod.host_value());
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_reduce_ops: empty-domain identities verified");
  }

  // ---- Host (OpenMP) reduce parity for Min ----
  // Validates make_host_reduce_kernel routes through the new Min op the same way the
  // device path does.
  {
    auto field = dense.newField<std::int32_t, 1>(carbon::MemorySide::HostDevice);
    field.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
      return encode(gc.x, gc.y, gc.z);
    });
    carbon::ReductionResult<std::int32_t> r(sys);
    auto k = dense.make_host_reduce_kernel<std::int32_t>(
        [&field](carbon::KernelLoader& loader) {
          const auto in = loader(field);
          return [in](carbon::Dense::Span s, carbon::Dense::Idx idx) -> std::int32_t {
            return in.read(s, idx, 0);
          };
        },
        r, carbon::reduce::Min<std::int32_t>{});
    try {
      k.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_ops: host Min launch failed: {}", e.what());
      return EXIT_FAILURE;
    }
    if (r.host_value() != host_min) {
      CARBON_LOG_ERROR("test_reduce_ops: host Min mismatch: got {}, expected {}", r.host_value(),
                       host_min);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_reduce_ops: host Min = {}", r.host_value());
  }

  CARBON_LOG_INFO("test_reduce_ops: PASS");
  return EXIT_SUCCESS;
}
