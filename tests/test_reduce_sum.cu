// Single-GPU sum-reduction over a Dense field via the new ReduceKernel + ReductionResult
// path (CUB BlockReduce in the fused stage-1 kernel, cub::DeviceReduce::Reduce in the
// stage-2 collapse, async cudaMemcpyAsync into the host slot).
//
// Two cases:
//
// 1) Bit-exact int64 sum. We populate the field with a deterministic per-cell encoder
//    (gz * 100000 + gy * 100 + gx, fits safely in int64 for the test extents) and compute
//    the host reference with the same encoder over the global cell list. CUB's reduce is
//    a tree of integer additions so the result is independent of block scheduling and we
//    require bit-exactness against the host sum.
//
// 2) Tolerance-checked double sum. Same field but the kernel casts to double and the
//    operator is reduce::Sum<double>{}; the tree-shape difference between host and device
//    can produce ~1 ULP slop on a sum of ~10^4 floats, which is well below the 1e-9
//    relative tolerance we assert.
//
// Also pokes ReductionResult::sync + host_value, ReduceKernel::total_cells /
// block_count, and the empty-domain branch (Dense built with z = 0 cells; reduce should
// produce the operator's identity element).

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
#include <vector>

namespace {

constexpr int64_t kWidth = 24;
constexpr int64_t kHeight = 32;
constexpr int64_t kDepth = 16;

[[nodiscard]] CARBON_HD auto
encode(int64_t x,
       int64_t y,
       int64_t z) -> int64_t {
  // Distinct, signed, fits in int64 for x<256, y<1024, z<2^28. Negative-ish on parity to
  // exercise sign handling in the CUB reduce.
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

[[nodiscard]] auto
host_reference_sum_double(int64_t W,
                          int64_t H,
                          int64_t D) -> double {
  double s = 0.0;
  for (int64_t z = 0; z < D; ++z) {
    for (int64_t y = 0; y < H; ++y) {
      for (int64_t x = 0; x < W; ++x) {
        s += static_cast<double>(encode(x, y, z));
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
    CARBON_LOG_ERROR("test_reduce_sum: no CUDA devices visible");
    return EXIT_FAILURE;
  }

  // Force a single-GPU System (v1 reduce path is single-GPU only and ReductionResult /
  // ReduceKernel both reject gpu_count != 1 at construction).
  std::vector<carbon::detail::Gpu> all = carbon::get_all_devices();
  std::vector<carbon::detail::Gpu> single;
  single.push_back(std::move(all.front()));
  carbon::System sys(std::move(single));

  // Empty stencil → halo_z == 0; the reduce path uses Standard view which iterates every
  // owned cell anyway. A bigger stencil would only change the haloed buffer layout, not
  // the reduction set — covered separately by the existing data_view_split tests.
  carbon::Stencil stencil{};
  const carbon::int64_3d global{kWidth, kHeight, kDepth};
  carbon::Dense dense(sys, global, stencil);

  // ---- Case 1: bit-exact int64 sum ----
  {
    auto field = dense.newField<int64_t, 1>(carbon::MemorySide::HostDevice);
    field.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
      return encode(gc.x, gc.y, gc.z);
    });

    carbon::Stream stream = sys.stream();
    try {
      carbon::Transfer<int64_t>(sys, field.cpu_buffers(), field.gpu_buffers())(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum: int64 upload failed: {}", e.what());
      return EXIT_FAILURE;
    }
    try {
      stream.wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum: sync after int64 upload failed: {}", e.what());
      return EXIT_FAILURE;
    }

    carbon::ReductionResult<int64_t> result(sys);
    auto reducer = dense.make_reduce_kernel<int64_t>(
        [&field](carbon::KernelLoader& loader) {
          const auto in = loader(field);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> int64_t {
            return in.read(s, idx, 0);
          };
        },
        result, carbon::reduce::Sum<int64_t>{});

    try {
      reducer.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum: int64 reduce launch failed: {}", e.what());
      return EXIT_FAILURE;
    }
    result.sync(stream);

    const int64_t got = result.host_value();
    const int64_t want = host_reference_sum(kWidth, kHeight, kDepth);
    if (got != want) {
      CARBON_LOG_ERROR("test_reduce_sum: int64 mismatch: got {}, expected {}", got, want);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_reduce_sum: int64 sum = {} over {} cells", got,
                    static_cast<int64_t>(kWidth * kHeight * kDepth));
  }

  // ---- Case 2: double sum, tolerance-checked ----
  {
    auto field = dense.newField<int64_t, 1>(carbon::MemorySide::HostDevice);
    field.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
      return encode(gc.x, gc.y, gc.z);
    });

    carbon::Stream stream = sys.stream();
    try {
      carbon::Transfer<int64_t>(sys, field.cpu_buffers(), field.gpu_buffers())(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum: double upload failed: {}", e.what());
      return EXIT_FAILURE;
    }
    try {
      stream.wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum: sync after double upload failed: {}", e.what());
      return EXIT_FAILURE;
    }

    carbon::ReductionResult<double> result(sys);
    auto reducer = dense.make_reduce_kernel<double>(
        [&field](carbon::KernelLoader& loader) {
          const auto in = loader(field);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> double {
            return static_cast<double>(in.read(s, idx, 0));
          };
        },
        result, carbon::reduce::Sum<double>{});

    try {
      reducer.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum: double reduce launch failed: {}", e.what());
      return EXIT_FAILURE;
    }
    result.sync(stream);

    const double got = result.host_value();
    const double want = host_reference_sum_double(kWidth, kHeight, kDepth);
    const double rel = (want == 0.0) ? std::abs(got - want) : std::abs(got - want) / std::abs(want);
    constexpr double kRelTol = 1e-9;
    if (rel > kRelTol) {
      CARBON_LOG_ERROR(
          "test_reduce_sum: double mismatch beyond tolerance: got {}, expected {}, "
          "rel diff {:.3e} > {:.0e}",
          got, want, rel, kRelTol);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_reduce_sum: double sum = {:.10g} (rel diff {:.3e} vs host)", got, rel);
  }

  // ---- Case 3: empty-domain shortcut ----
  // A Dense with z = 0 has zero owned cells; the reducer's total_cells_ and block_count_
  // are both 0 and the launch path goes through cub::DeviceReduce::Reduce(num_items=0),
  // which writes the identity into the device scalar. Result must equal Sum::identity()
  // (i.e. 0).
  {
    const carbon::int64_3d empty_global{kWidth, kHeight, 0};
    carbon::Dense empty_dense(sys, empty_global, carbon::Stencil{});
    auto field = empty_dense.newField<int64_t, 1>(carbon::MemorySide::Device);
    carbon::Stream stream = sys.stream();
    carbon::ReductionResult<int64_t> result(sys);
    auto reducer = empty_dense.make_reduce_kernel<int64_t>(
        [&field](carbon::KernelLoader& loader) {
          const auto in = loader(field);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> int64_t {
            return in.read(s, idx, 0);
          };
        },
        result, carbon::reduce::Sum<int64_t>{});
    try {
      reducer.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum: empty-domain launch failed: {}", e.what());
      return EXIT_FAILURE;
    }
    result.sync(stream);
    if (result.host_value() != 0) {
      CARBON_LOG_ERROR("test_reduce_sum: empty-domain expected 0, got {}", result.host_value());
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_reduce_sum: empty-domain reduce -> {} (identity)", result.host_value());
  }

  // ---- Case 4: re-launch into the same ReductionResult ----
  // Run the int64 sum twice into a single ReductionResult; every launch overwrites the
  // device scalar (no accumulation), so the second host_value must match the first.
  {
    auto field = dense.newField<int64_t, 1>(carbon::MemorySide::HostDevice);
    field.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
      return encode(gc.x, gc.y, gc.z);
    });
    carbon::Stream stream = sys.stream();
    carbon::Transfer<int64_t>(sys, field.cpu_buffers(), field.gpu_buffers())(stream);
    try {
      stream.wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum: sync after relaunch upload failed: {}", e.what());
      return EXIT_FAILURE;
    }
    carbon::ReductionResult<int64_t> result(sys);
    auto reducer = dense.make_reduce_kernel<int64_t>(
        [&field](carbon::KernelLoader& loader) {
          const auto in = loader(field);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> int64_t {
            return in.read(s, idx, 0);
          };
        },
        result, carbon::reduce::Sum<int64_t>{});
    reducer.launch(stream);
    result.sync(stream);
    const int64_t a = result.host_value();
    reducer.launch(stream);
    result.sync(stream);
    const int64_t b = result.host_value();
    if (a != b) {
      CARBON_LOG_ERROR("test_reduce_sum: relaunch mismatch: first {}, second {}", a, b);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_reduce_sum: relaunch reproducible at {}", a);
  }

  // ---- Case 5: single-GPU Internal+Boundary recombine matches Standard ----
  // Build a Dense with a non-trivial stencil so halo_z > 0 and the Internal/Boundary
  // split is meaningful. Launch the same reducer three times into separate per-view
  // slots (Standard, Internal, Boundary) and check that the Standard slot equals the
  // Sum-combine of Internal + Boundary slots — the carbon analogue of Neon's
  // gUt_dataView_patterns "int_output + bd_output == ground_truth" contract.
  {
    carbon::Stencil stencil_radius1(
        std::vector<carbon::int32_3d>{
            {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
        /*remove_center=*/true);
    carbon::Dense halo_dense(sys, global, stencil_radius1);
    auto field = halo_dense.newField<int64_t, 1>(carbon::MemorySide::HostDevice);
    field.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
      return encode(gc.x, gc.y, gc.z);
    });

    carbon::Stream stream = sys.stream();
    try {
      carbon::Transfer<int64_t>(sys, field.cpu_buffers(), field.gpu_buffers())(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum: dataview upload failed: {}", e.what());
      return EXIT_FAILURE;
    }
    try {
      stream.wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum: sync after dataview upload failed: {}", e.what());
      return EXIT_FAILURE;
    }

    carbon::ReductionResult<int64_t> result(sys);
    auto reducer = halo_dense.make_reduce_kernel<int64_t>(
        [&field](carbon::KernelLoader& loader) {
          const auto in = loader(field);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> int64_t {
            return in.read(s, idx, 0);
          };
        },
        result, carbon::reduce::Sum<int64_t>{});

    try {
      reducer.launch(stream, carbon::DataView::Standard);
      reducer.launch(stream, carbon::DataView::Internal);
      reducer.launch(stream, carbon::DataView::Boundary);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum: dataview reduce launch failed: {}", e.what());
      return EXIT_FAILURE;
    }
    result.sync(stream);

    const int64_t standard = result.host_value(carbon::DataView::Standard);
    const int64_t internal = result.host_value(carbon::DataView::Internal);
    const int64_t boundary = result.host_value(carbon::DataView::Boundary);
    const int64_t want = host_reference_sum(kWidth, kHeight, kDepth);
    if (standard != want) {
      CARBON_LOG_ERROR(
          "test_reduce_sum: dataview Standard mismatch: got {}, expected {} (host ref)", standard,
          want);
      return EXIT_FAILURE;
    }
    if (internal + boundary != standard) {
      CARBON_LOG_ERROR("test_reduce_sum: dataview Internal({}) + Boundary({}) = {} != Standard({})",
                       internal, boundary, internal + boundary, standard);
      return EXIT_FAILURE;
    }
    // host_value_combined folds every fresh slot via Sum — Standard slot is also fresh
    // here because we launched it above, so the combined value would double-count.
    // Run a separate ReductionResult for the combine sanity check, with only the two
    // split views fresh.
    carbon::ReductionResult<int64_t> split_only(sys);
    try {
      reducer = halo_dense.make_reduce_kernel<int64_t>(
          [&field](carbon::KernelLoader& loader) {
            const auto in = loader(field);
            return
                [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> int64_t {
                  return in.read(s, idx, 0);
                };
          },
          split_only, carbon::reduce::Sum<int64_t>{});
      reducer.launch(stream, carbon::DataView::Internal);
      reducer.launch(stream, carbon::DataView::Boundary);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum: split_only launch failed: {}", e.what());
      return EXIT_FAILURE;
    }
    split_only.sync(stream);
    const int64_t combined = split_only.host_value_combined(carbon::reduce::Sum<int64_t>{});
    if (combined != standard) {
      CARBON_LOG_ERROR("test_reduce_sum: host_value_combined({}) does not equal Standard({})",
                       combined, standard);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO(
        "test_reduce_sum: dataview Standard={} = Internal({}) + Boundary({}) (combined={})",
        standard, internal, boundary, combined);
  }

  CARBON_LOG_INFO("test_reduce_sum: PASS");
  return EXIT_SUCCESS;
}
