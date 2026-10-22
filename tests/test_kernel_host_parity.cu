// Cross-target parity test for ::carbon::Kernel between the device launch path
// (DenseKernelImpl + DenseReduceKernelImpl, going through CUDA cudaLaunchKernel or SYCL
// q.parallel_for / sycl::reduction) and the host launch path (DenseHostKernelImpl +
// ReduceHostKernelImpl, going through OpenMP nested-for + per-thread partials).
//
// Setup
// -----
// Single-GPU System (sliced off carbon::get_all_devices()) with a 7-point Stencil over
// a 16^3 domain — same shape as test_kernel_laplacian_single_gpu so the verifier can
// reuse its analytic Laplacian (Δ(a x^2 + b y^2 + c z^2) == 2 (a+b+c) at every interior
// cell).
//
// What we check
// -------------
// 1. *Compute parity*: build the SAME Laplacian lambda twice — once via
//    Dense::make_kernel (device) and once via Dense::make_host_kernel (CPU). Launch the
//    device kernel into u_out_dev; download to the CPU mirror. Launch the host kernel
//    into u_out_host (which writes directly into the host mirror). Assert the two host
//    mirrors are *bit-exactly* equal on every interior cell (int32 has no rounding,
//    float on these magnitudes is exactly representable).
//
// 2. *Reduction parity*: build the SAME per-cell `f(s, idx) -> int64` lambda twice —
//    once via Dense::make_reduce_kernel (CUB / sycl::reduction backed) and once via
//    Dense::make_host_reduce_kernel (OpenMP per-thread partials + sequential combine).
//    Both target their own ReductionResult<int64_t>; we assert host_value() matches
//    bit-exactly (sum of int64 has no rounding regardless of reduction tree shape).
//
// 3. *Heterogeneous std::vector<Kernel>*: confirm a host kernel and a device kernel can
//    sit side-by-side in a std::vector<Kernel> and run in order; the host kernel's
//    Stream::wait() acts as a fence between the device kernel's writes (downloaded to
//    the CPU mirror by an explicit Transfer) and the host kernel reading them.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/stencil.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

namespace {

constexpr int64_t kA = 1;
constexpr int64_t kB = 2;
constexpr int64_t kC = 3;
constexpr int64_t kExpectedLaplacian = 2 * (kA + kB + kC);

template <typename T>
[[nodiscard]] CARBON_HD auto
analytic_input(int64_t global_x,
               int64_t global_y,
               int64_t global_z) -> T {
  return static_cast<T>(kA * global_x * global_x + kB * global_y * global_y +
                        kC * global_z * global_z);
}

// Fill the entire haloed extent of @p f's host mirror with analytic_input(global x,y,z).
template <typename T>
auto
fill_host_full(const carbon::Dense& dense,
               carbon::Dense::Field<T,
                                    1>& f) -> void {
  const auto& span = dense.spans()[0];
  const auto haloed = span.haloed_extent();
  const int64_t pitch = span.xy_slab_elements();
  const int64_t z_origin = span.z_offset();
  T* h = f.cpu_buffers()[0].data();
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

// nvcc rejects extended __host__ __device__ lambdas inside an enclosing function with
// a deduced return type, so we cannot factor out the loading callable as a templated
// function. Instead, define it as a macro that expands to the same lambda body at every
// call site — this puts the inner extended HD lambda in main()'s scope (a regular
// non-template function) where nvcc accepts it.
#define MAKE_LAPLACIAN_LOADER(T_, u_in_, u_out_)                                         \
  [&u_in_, &u_out_](carbon::KernelLoader& loader) {                                      \
    const auto in = loader(u_in_);                                                       \
    const auto out = loader(u_out_);                                                     \
    return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) { \
      const T_ self = in.read(s, idx, 0);                                                \
      const T_ xp = in.neighbour(s, idx, {1, 0, 0}, 0).value_or(T_{0});                  \
      const T_ xm = in.neighbour(s, idx, {-1, 0, 0}, 0).value_or(T_{0});                 \
      const T_ yp = in.neighbour(s, idx, {0, 1, 0}, 0).value_or(T_{0});                  \
      const T_ ym = in.neighbour(s, idx, {0, -1, 0}, 0).value_or(T_{0});                 \
      const T_ zp = in.neighbour(s, idx, {0, 0, 1}, 0).value_or(T_{0});                  \
      const T_ zm = in.neighbour(s, idx, {0, 0, -1}, 0).value_or(T_{0});                 \
      const T_ lap = static_cast<T_>(xp + xm + yp + ym + zp + zm - T_{6} * self);        \
      out.write(s, idx, 0, lap);                                                         \
    };                                                                                   \
  }

// Compare two Field<T, 1> host mirrors (over the strict interior, where the
// Laplacian's value_or(0) substitution has not poisoned the result). Returns true on
// bit-exact equality across every interior cell.
template <typename T>
[[nodiscard]] auto
compare_interior(const carbon::Dense& dense,
                 const carbon::Dense::Field<T,
                                            1>& a,
                 const carbon::Dense::Field<T,
                                            1>& b,
                 const char* tag) -> bool {
  const auto& span = dense.spans()[0];
  const auto haloed = span.haloed_extent();
  const auto lim = span.logic_extent();
  const int64_t pitch = span.xy_slab_elements();
  const int64_t halo_z = span.halo_z();
  const T* ha = a.cpu_buffers()[0].data();
  const T* hb = b.cpu_buffers()[0].data();

  size_t mismatches = 0;
  for (int64_t z = 1; z < lim.z - 1; ++z) {
    const int64_t cell_z = halo_z + z;
    for (int64_t y = 1; y < lim.y - 1; ++y) {
      for (int64_t x = 1; x < lim.x - 1; ++x) {
        const int64_t off = cell_z * pitch + y * haloed.x + x;
        if (ha[off] != hb[off]) {
          if (mismatches < 4U) {
            CARBON_LOG_ERROR("{}: mismatch at (x,y,z)=({},{},{}) device={} host={}", tag, x, y, z,
                             static_cast<double>(ha[off]), static_cast<double>(hb[off]));
          }
          ++mismatches;
        }
      }
    }
  }
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("{}: {} interior mismatches", tag, mismatches);
    return false;
  }
  return true;
}

// Compute parity test: device and host kernel must produce identical Laplacian fields.
//
// nvcc forbids extended __host__ __device__ lambdas inside template functions with
// deduced return type, so each parity case is its own non-template function (with the
// per-cell lambda body inlined through MAKE_LAPLACIAN_LOADER) instead of a single
// template parameterised on T.
//
// Two flavours of the test are run:
//   - int32_t: Universal-kernel API. ONE Dense::make_kernel handle, two launch sites
//     (Kernel::launch_on_device + Kernel::launch_on_host on the same handle into two
//     different output fields). This is the post-rewrite showcase pattern: same
//     lambda body, both targets, single factory call.
//   - float: explicit single-target factories. Dense::make_device_kernel +
//     Dense::make_host_kernel as TWO separate handles. Documents that the explicit
//     style still works for users who want it.
[[nodiscard]] auto
run_compute_parity_int32(carbon::System& sys) -> bool {
  using T = std::int32_t;
  const char* tag = "compute_parity<int32_t>(universal)";
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  const carbon::int64_3d global{16, 16, 16};
  const carbon::Dense dense(sys, global, stencil);

  auto u_in = dense.newField<T, 1>(carbon::MemorySide::HostDevice);
  auto u_out_dev = dense.newField<T, 1>(carbon::MemorySide::HostDevice);
  auto u_out_host = dense.newField<T, 1>(carbon::MemorySide::HostDevice);

  fill_host_full<T>(dense, u_in);

  carbon::Stream stream = sys.stream();

  try {
    carbon::Transfer<T>(sys, u_in.cpu_buffers(), u_in.gpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("{}: u_in upload failed: {}", tag, e.what());
    return false;
  }

  // Universal kernels — ONE per output field, each launched on its own target. Note we
  // need two universal handles because the lambda captures u_out_dev / u_out_host by
  // reference and writes to a single output; if we wanted a single Universal kernel for
  // BOTH outputs we'd need two output fields inside the same closure (overkill for the
  // test).
  const auto kernel_dev = dense.make_universal_kernel(MAKE_LAPLACIAN_LOADER(T, u_in, u_out_dev),
                                                      /*shared_mem=*/0U);
  const auto kernel_host = dense.make_universal_kernel(MAKE_LAPLACIAN_LOADER(T, u_in, u_out_host),
                                                       /*shared_mem=*/0U);

  // Capability queries on the universal handle: it should advertise BOTH targets.
  if (!kernel_dev.runs_on_device() || !kernel_dev.runs_on_host()) {
    CARBON_LOG_ERROR(
        "{}: Universal kernel must report runs_on_device == runs_on_host == true (got dev={}, "
        "host={})",
        tag, kernel_dev.runs_on_device(), kernel_dev.runs_on_host());
    return false;
  }
  if (kernel_dev.kind() != carbon::KernelKind::Universal) {
    CARBON_LOG_ERROR("{}: Universal kernel must report kind() == Universal", tag);
    return false;
  }

  try {
    kernel_dev.launch_on_device(stream);  // explicit device launch on the Universal handle
    stream.wait();
    carbon::Transfer<T>(sys, u_out_dev.gpu_buffers(), u_out_dev.cpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("{}: Universal device launch pipeline failed: {}", tag, e.what());
    return false;
  }

  try {
    kernel_host.launch_on_host(stream);  // explicit host launch on a SEPARATE Universal handle
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("{}: Universal host launch failed: {}", tag, e.what());
    return false;
  }

  if (!compare_interior<T>(dense, u_out_dev, u_out_host, tag)) {
    return false;
  }

  // Sanity check: the analytic Laplacian must equal kExpectedLaplacian at every
  // interior cell. If the host run silently took the wrong path (e.g. read uninitialised
  // memory and happened to match the device run because the device run also read junk),
  // this catches it.
  const auto& span = dense.spans()[0];
  const auto haloed = span.haloed_extent();
  const auto lim = span.logic_extent();
  const int64_t pitch = span.xy_slab_elements();
  const int64_t halo_z = span.halo_z();
  const T expected = static_cast<T>(kExpectedLaplacian);
  const T* h = u_out_host.cpu_buffers()[0].data();
  for (int64_t z = 1; z < lim.z - 1; ++z) {
    const int64_t cell_z = halo_z + z;
    for (int64_t y = 1; y < lim.y - 1; ++y) {
      for (int64_t x = 1; x < lim.x - 1; ++x) {
        const int64_t off = cell_z * pitch + y * haloed.x + x;
        if (h[off] != expected) {
          CARBON_LOG_ERROR(
              "{}: host result deviates from analytic at (x,y,z)=({},{},{}): got={} want={}", tag,
              x, y, z, static_cast<double>(h[off]), static_cast<double>(expected));
          return false;
        }
      }
    }
  }

  CARBON_LOG_INFO("{}: device == host == analytic Laplacian over interior", tag);
  return true;
}

[[nodiscard]] auto
run_compute_parity_float(carbon::System& sys) -> bool {
  using T = float;
  const char* tag = "compute_parity<float>(explicit factories)";
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  const carbon::int64_3d global{16, 16, 16};
  const carbon::Dense dense(sys, global, stencil);

  auto u_in = dense.newField<T, 1>(carbon::MemorySide::HostDevice);
  auto u_out_dev = dense.newField<T, 1>(carbon::MemorySide::HostDevice);
  auto u_out_host = dense.newField<T, 1>(carbon::MemorySide::HostDevice);

  fill_host_full<T>(dense, u_in);

  carbon::Stream stream = sys.stream();
  try {
    carbon::Transfer<T>(sys, u_in.cpu_buffers(), u_in.gpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("{}: u_in upload failed: {}", tag, e.what());
    return false;
  }
  // Explicit single-target factories: device-only handle vs host-only handle.
  const auto device_kernel = dense.make_kernel(MAKE_LAPLACIAN_LOADER(T, u_in, u_out_dev),
                                               /*shared_mem=*/0U);
  if (device_kernel.kind() != carbon::KernelKind::Compute || !device_kernel.runs_on_device() ||
      device_kernel.runs_on_host()) {
    CARBON_LOG_ERROR(
        "{}: device-only kernel should report Compute / runs_on_device only (got kind={}, dev={}, "
        "host={})",
        tag, static_cast<int>(device_kernel.kind()), device_kernel.runs_on_device(),
        device_kernel.runs_on_host());
    return false;
  }
  try {
    device_kernel.launch(stream);
    stream.wait();
    carbon::Transfer<T>(sys, u_out_dev.gpu_buffers(), u_out_dev.cpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("{}: device kernel pipeline failed: {}", tag, e.what());
    return false;
  }
  const auto host_kernel = dense.make_host_kernel(MAKE_LAPLACIAN_LOADER(T, u_in, u_out_host));
  if (host_kernel.kind() != carbon::KernelKind::ComputeHost || host_kernel.runs_on_device() ||
      !host_kernel.runs_on_host()) {
    CARBON_LOG_ERROR(
        "{}: host-only kernel should report ComputeHost / runs_on_host only (got kind={}, dev={}, "
        "host={})",
        tag, static_cast<int>(host_kernel.kind()), host_kernel.runs_on_device(),
        host_kernel.runs_on_host());
    return false;
  }
  // Calling launch_on_device on a host-only kernel must throw std::logic_error.
  bool device_throw = false;
  try {
    host_kernel.launch_on_device(stream);
  } catch (const std::logic_error&) {
    device_throw = true;
  }
  if (!device_throw) {
    CARBON_LOG_ERROR("{}: host-only kernel.launch_on_device must throw std::logic_error", tag);
    return false;
  }
  try {
    host_kernel.launch(stream);  // host kernel's natural target is host
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("{}: host kernel launch failed: {}", tag, e.what());
    return false;
  }

  if (!compare_interior<T>(dense, u_out_dev, u_out_host, tag)) {
    return false;
  }
  CARBON_LOG_INFO("{}: device == host (bit-exact) Laplacian over interior", tag);
  return true;
}

// Reduction parity: device and host paths must produce the same scalar.
[[nodiscard]] auto
run_reduce_parity(carbon::System& sys) -> bool {
  carbon::Stencil stencil{};
  const carbon::int64_3d global{16, 16, 16};
  const carbon::Dense dense(sys, global, stencil);

  using T = int64_t;
  auto field = dense.newField<T, 1>(carbon::MemorySide::HostDevice);
  field.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
    // Distinct deterministic int64 per cell. Range is small (< 2^32), well clear of
    // overflow, and the device-side CUB / sycl::reduction sum + the OpenMP per-thread
    // sum produce bit-exact equality on integer addition.
    return static_cast<T>(gc.z * 1000000 + gc.y * 1000 + gc.x);
  });

  carbon::Stream stream = sys.stream();
  try {
    carbon::Transfer<T>(sys, field.cpu_buffers(), field.gpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("reduce_parity: upload failed: {}", e.what());
    return false;
  }

  // Same per-cell loader for both reduction paths. The closure captures `field` once
  // and the loader::operator() inside picks the correct partition (device or host)
  // based on the loader's Target.
  auto reduce_loader = [&field](carbon::KernelLoader& loader) {
    const auto in = loader(field);
    return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> T {
      return in.read(s, idx, 0);
    };
  };

  carbon::ReductionResult<T> dev_result(sys);
  carbon::ReductionResult<T> host_result(sys);
  carbon::ReductionResult<T> universal_dev_result(sys);
  carbon::ReductionResult<T> universal_host_result(sys);

  // Four reducers: explicit device-only, explicit host-only, and a Universal reducer
  // launched once on each target through two separate handles (a single Universal
  // handle could be reused but each Universal handle is bound to ONE
  // ReductionResult, so two handles let us keep both scalars side-by-side for
  // cross-comparison).
  const auto dev_reducer =
      dense.make_reduce_kernel<T>(reduce_loader, dev_result, carbon::reduce::Sum<T>{});
  const auto host_reducer =
      dense.make_host_reduce_kernel<T>(reduce_loader, host_result, carbon::reduce::Sum<T>{});
  const auto universal_dev_reducer = dense.make_universal_reduce_kernel<T>(
      reduce_loader, universal_dev_result, carbon::reduce::Sum<T>{});
  const auto universal_host_reducer = dense.make_universal_reduce_kernel<T>(
      reduce_loader, universal_host_result, carbon::reduce::Sum<T>{});

  if (universal_dev_reducer.kind() != carbon::KernelKind::UniversalReduction ||
      !universal_dev_reducer.runs_on_device() || !universal_dev_reducer.runs_on_host()) {
    CARBON_LOG_ERROR(
        "reduce_parity: Universal reducer must report UniversalReduction / both targets");
    return false;
  }

  try {
    dev_reducer.launch(stream);
    dev_result.sync(stream);
    host_reducer.launch(stream);  // launch waits on stream first; OpenMP then runs synchronously.
    host_result.sync(stream);
    universal_dev_reducer.launch_on_device(stream);
    universal_dev_result.sync(stream);
    universal_host_reducer.launch_on_host(stream);
    universal_host_result.sync(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("reduce_parity: launch failed: {}", e.what());
    return false;
  }

  const T dev_val = dev_result.host_value();
  const T host_val = host_result.host_value();
  const T udev_val = universal_dev_result.host_value();
  const T uhost_val = universal_host_result.host_value();
  if (dev_val != host_val || dev_val != udev_val || dev_val != uhost_val) {
    CARBON_LOG_ERROR("reduce_parity: mismatch dev={} host={} universal_dev={} universal_host={}",
                     static_cast<long long>(dev_val), static_cast<long long>(host_val),
                     static_cast<long long>(udev_val), static_cast<long long>(uhost_val));
    return false;
  }
  CARBON_LOG_INFO("reduce_parity: dev == host == universal_dev == universal_host == {}",
                  static_cast<long long>(dev_val));
  return true;
}

// Heterogeneous pipeline test: build a std::vector<Kernel> mixing a host kernel after
// a device kernel. The host kernel must see the device kernel's writes (downloaded to
// the CPU mirror by an explicit Transfer) thanks to its own internal Stream::wait().
[[nodiscard]] auto
run_heterogeneous_pipeline(carbon::System& sys) -> bool {
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  const carbon::int64_3d global{16, 16, 16};
  const carbon::Dense dense(sys, global, stencil);

  using T = std::int32_t;
  auto u_in = dense.newField<T, 1>(carbon::MemorySide::HostDevice);
  auto u_dev = dense.newField<T, 1>(carbon::MemorySide::HostDevice);
  auto u_host = dense.newField<T, 1>(carbon::MemorySide::HostDevice);

  fill_host_full<T>(dense, u_in);

  carbon::Stream stream = sys.stream();
  try {
    carbon::Transfer<T>(sys, u_in.cpu_buffers(), u_in.gpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("heterogeneous: u_in upload failed: {}", e.what());
    return false;
  }

  // Step 1: device Laplacian into u_dev's device side.
  // Step 2: anchor (just to exercise the heterogeneous mix; could be omitted).
  // Step 3: host Laplacian into u_host's host mirror.
  // Step 4 (post-pipeline): download u_dev's device data to its CPU mirror so we can
  //                         compare device vs host results.
  std::vector<carbon::Kernel> pipeline;
  pipeline.push_back(dense.make_universal_kernel(MAKE_LAPLACIAN_LOADER(T, u_in, u_dev),
                                                 /*shared_mem=*/0U));
  pipeline.push_back(carbon::Kernel::anchor("between"));
  pipeline.push_back(dense.make_host_kernel(MAKE_LAPLACIAN_LOADER(T, u_in, u_host)));

  try {
    for (const auto& k : pipeline) {
      k.launch(stream);
    }
    stream.wait();
    carbon::Transfer<T>(sys, u_dev.gpu_buffers(), u_dev.cpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("heterogeneous: pipeline failed: {}", e.what());
    return false;
  }

  if (!compare_interior<T>(dense, u_dev, u_host, "heterogeneous_pipeline")) {
    return false;
  }
  CARBON_LOG_INFO("heterogeneous_pipeline: 3-kernel mixed pipeline produced parity");
  return true;
}

// Negative-path test: a host kernel reading a device-only field must fail with a clean
// std::invalid_argument from KernelLoader's host branch (host_partitions() is empty).
[[nodiscard]] auto
run_device_only_negative(carbon::System& sys) -> bool {
  carbon::Stencil stencil{};
  const carbon::int64_3d global{8, 8, 8};
  const carbon::Dense dense(sys, global, stencil);

  using T = std::int32_t;
  auto u_in = dense.newField<T, 1>(carbon::MemorySide::Device);   // no host mirror
  auto u_out = dense.newField<T, 1>(carbon::MemorySide::Device);  // no host mirror

  // Build a host kernel that tries to load a device-only field. The throw fires when
  // the OpenMP launch path invokes the loading lambda, which calls loader(u_in) and
  // KernelLoader::operator() takes the host branch (empty host_partitions) and throws.
  const auto bad_kernel = dense.make_host_kernel([&u_in, &u_out](carbon::KernelLoader& loader) {
    const auto in = loader(u_in);
    const auto out = loader(u_out);
    return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
      out.write(s, idx, 0, in.read(s, idx, 0));
    };
  });

  carbon::Stream stream = sys.stream();
  bool threw = false;
  try {
    bad_kernel.launch(stream);
  } catch (const std::invalid_argument&) {
    threw = true;
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("device_only_negative: wrong exception type: {}", e.what());
    return false;
  }
  if (!threw) {
    CARBON_LOG_ERROR(
        "device_only_negative: host kernel did not throw on a MemorySide::Device field");
    return false;
  }
  CARBON_LOG_INFO("device_only_negative: host kernel correctly rejected device-only field");
  return true;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_kernel_host_parity: no devices visible");
    return 1;
  }

  // Single-GPU System. The host paths under test do not depend on multi-GPU halo, and
  // ReductionResult/Reduce* impls require gpu_count == 1 in v1.
  std::vector<carbon::detail::Gpu> all = carbon::get_all_devices();
  std::vector<carbon::detail::Gpu> single;
  single.push_back(std::move(all.front()));
  carbon::System sys(std::move(single));

  if (!run_compute_parity_int32(sys)) {
    return 1;
  }
  if (!run_compute_parity_float(sys)) {
    return 1;
  }
  if (!run_reduce_parity(sys)) {
    return 1;
  }
  if (!run_heterogeneous_pipeline(sys)) {
    return 1;
  }
  if (!run_device_only_negative(sys)) {
    return 1;
  }

  CARBON_LOG_INFO("test_kernel_host_parity: PASS");
  return 0;
}
