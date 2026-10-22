// SYCL-backend implementation of the non-inline parts of carbon::detail::Gpu plus the
// host-side helpers (carbon::device_count, carbon::get_all_devices, carbon::detail::GpuGroup
// methods, gpu_track_* accounting). Mirrors src/gpu.cpp's structure 1:1 so the two backend
// .cpp files stay easy to keep in sync; the only differences are the device enumeration
// (sycl::platform vs cudaGetDeviceCount), the property mapping
// (sycl::info::device::* vs cudaDeviceProp), and the smoke kernel (parallel_for vs
// __global__).

#include <carbon/detail/gpu_group.h>
#include <carbon/devices.h>
#include <carbon/logging.h>
#include <carbon/report.h>

#include <carbon/detail/gpu_allocation_tracking.h>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace carbon {
namespace detail {

namespace {

/**
 * Stable per-process ordering of GPU+accelerator SYCL devices. SYCL itself does not
 * promise a stable ordering across calls, so we build one once on first access (via the
 * function-local static) and reuse it. Enumeration goes platform-by-platform and within
 * each platform device-by-device, filtering out CPU devices. Order is then deterministic
 * for a given AdaptiveCpp + driver combination.
 *
 * Returning by const reference is safe: the vector is constructed once and never
 * mutated afterwards.
 */
auto
all_sycl_devices() -> const std::vector<sycl::device>& {
  static const std::vector<sycl::device> devices = []() {
    std::vector<sycl::device> out;
    for (const auto& platform : sycl::platform::get_platforms()) {
      for (const auto& dev : platform.get_devices(sycl::info::device_type::all)) {
        // Skip the host CPU device exposed by SYCL — carbon's "device" concept is the
        // GPU/accelerator side; the host is reached via Cpu::buffer instead.
        if (!dev.is_cpu()) {
          out.push_back(dev);
        }
      }
    }
    return out;
  }();
  return devices;
}

/**
 * Single process-wide @c sycl::context that spans every GPU/accelerator SYCL device the
 * runtime can see. Lifetime is the program's; the context is built lazily on first
 * @c shared_sycl_context() call from the same device list @ref all_sycl_devices returns.
 *
 * @par Why one shared context
 * Cross-device USM peer copies (@c queue.memcpy from a queue bound to GPU 0 into a
 * USM allocation owned by GPU 1, or vice versa) require both devices to live inside the
 * same @c sycl::context. AdaptiveCpp's CUDA target maps this to the underlying
 * @c cuMemcpyPeerAsync; on SPIR-V/Level Zero it is the normal cross-device USM path. By
 * giving every @c carbon::detail::Gpu the same @c sycl::context here, halo exchanges
 * and arbitrary @c System::async_copy calls between two Gpus become a one-line
 * @c queue.memcpy with no explicit peer-access enable step.
 *
 * @par Failure mode
 * If @c sycl::context construction over the multi-device list throws (some runtimes
 * reject mixed-platform device lists), we fall back to a context over the first device
 * only and log a warning. In that mode multi-GPU peer copies will surface as a
 * @c sycl::exception out of @c queue.memcpy and @c System::async_copy will propagate it
 * — there is no graceful peer fallback under SYCL the way @c cudaMemcpyPeerAsync has.
 */
auto
shared_sycl_context() -> const sycl::context& {
  static const sycl::context ctx = []() {
    const auto& devs = all_sycl_devices();
    if (devs.empty()) {
      // Default-constructed context (binds to the SYCL default selector). Queues built
      // against it will fail the moment they try to bind to a missing device, which is
      // the same observable behaviour as the no-device CUDA path.
      return sycl::context{};
    }
    try {
      return sycl::context(devs);
    } catch (const sycl::exception& ex) {
      CARBON_LOG_WARN(
          "carbon::detail: multi-device sycl::context construction failed ({}); falling back to "
          "a single-device context — peer copies will throw at submission time",
          ex.what());
      return sycl::context(devs.front());
    }
  }();
  return ctx;
}

/**
 * Build a Properties snapshot from a sycl::device. Fields not exposed by SYCL are left
 * at zero with a comment documenting the gap; everything carbon's tests inspect (name,
 * total_global_mem_bytes, multiprocessor_count, max_threads_per_block) is populated.
 */
auto
make_properties_sycl(const sycl::device& dev) -> Properties {
  Properties p;
  p.name = dev.get_info<sycl::info::device::name>();
  // SYCL has no portable "compute capability" concept. AdaptiveCpp exposes the CUDA
  // arch as a custom info query, but only on cuda targets. Leave at 0 for now; consumers
  // wanting arch can read it from p.name.
  p.compute_major = 0;
  p.compute_minor = 0;
  p.total_global_mem_bytes =
      static_cast<size_t>(dev.get_info<sycl::info::device::global_mem_size>());
  p.multiprocessor_count = static_cast<int>(dev.get_info<sycl::info::device::max_compute_units>());
  p.max_threads_per_block =
      static_cast<int>(dev.get_info<sycl::info::device::max_work_group_size>());
  // sub_group_sizes returns a vector; the first element matches the warp/sub-group used
  // by parallel_for under AdaptiveCpp's CUDA target (32 on NVIDIA, 64 on AMD CDNA).
  const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
  p.warp_size = sg_sizes.empty() ? 0 : static_cast<int>(sg_sizes.front());
  // Not exposed by SYCL portably; consumers that depended on this on the CUDA backend
  // for occupancy heuristics fall back to multiprocessor_count * max_threads_per_block.
  p.max_threads_per_multiprocessor = 0;
  // Clock rates and bus widths are CUDA-driver internals that SYCL does not expose.
  p.clock_rate_khz = 0;
  p.memory_clock_rate_khz = 0;
  p.memory_bus_width_bits = 0;
  p.l2_cache_size_bytes =
      static_cast<int>(dev.get_info<sycl::info::device::global_mem_cache_size>());
  // SYCL does not expose constant memory size separately from global. Leave at 0.
  p.total_const_mem_bytes = 0U;
  p.shared_mem_per_block_bytes =
      static_cast<size_t>(dev.get_info<sycl::info::device::local_mem_size>());
  // Not exposed by SYCL; left at 0 (the field is rarely used in carbon's report path).
  p.regs_per_block = 0;
  return p;
}

}  // namespace

/**
 * Adds member gpu_<id> with device fields under @p parent (the JSON object under "carbon").
 * Backend-symmetric copy of the function in src/gpu.cpp.
 */
auto
add_gpu_device_subdoc(const Gpu& gpu,
                      Report& report,
                      Report::SubBlock& parent) -> void {
  const std::string key = "gpu_" + std::to_string(gpu.device_id());
  Report::SubBlock sub = report.get_subdoc();
  const auto& p = gpu.properties();
  report.add_member("device_id", gpu.device_id(), &sub);
  report.add_member("name", p.name, &sub);
  report.add_member("compute_major", p.compute_major, &sub);
  report.add_member("compute_minor", p.compute_minor, &sub);
  report.add_member("total_global_mem_bytes", static_cast<std::uint64_t>(p.total_global_mem_bytes),
                    &sub);
  report.add_member("allocated_bytes", static_cast<std::uint64_t>(gpu.allocated_bytes()), &sub);
  report.add_member("multiprocessor_count", p.multiprocessor_count, &sub);
  report.add_member("max_threads_per_block", p.max_threads_per_block, &sub);
  report.add_member("warp_size", p.warp_size, &sub);
  report.add_member("max_threads_per_multiprocessor", p.max_threads_per_multiprocessor, &sub);
  report.add_member("clock_rate_khz", p.clock_rate_khz, &sub);
  report.add_member("memory_clock_rate_khz", p.memory_clock_rate_khz, &sub);
  report.add_member("memory_bus_width_bits", p.memory_bus_width_bits, &sub);
  report.add_member("l2_cache_size_bytes", p.l2_cache_size_bytes, &sub);
  report.add_member("total_const_mem_bytes", static_cast<std::uint64_t>(p.total_const_mem_bytes),
                    &sub);
  report.add_member("shared_mem_per_block_bytes",
                    static_cast<std::uint64_t>(p.shared_mem_per_block_bytes), &sub);
  report.add_member("regs_per_block", p.regs_per_block, &sub);
  report.add_subdoc(key, sub, &parent);
}

Gpu::Gpu(int device_id)
    : device_id_(device_id) {
  const auto& devs = all_sycl_devices();
  const int count = static_cast<int>(devs.size());
  if (device_id < 0 || device_id >= count) {
    carbon::log::throw_with_log<std::runtime_error>(
        "carbon::detail::Gpu: device_id out of range (device_id={}, count={})", device_id, count);
  }
  device_ = devs[static_cast<size_t>(device_id)];
  // Shared multi-device context (see shared_sycl_context above). All Gpus in the
  // process now share one context, which is what enables peer-style USM memcpy across
  // devices via queue.memcpy. The sycl::context handle is refcounted internally so
  // copying it into context_ here is cheap.
  context_ = shared_sycl_context();
  props_ = make_properties_sycl(device_);
}

auto
Gpu::device_count() noexcept -> int {
  // Catch SYCL platform-init exceptions: if AdaptiveCpp can't enumerate any backend, we
  // return 0 to match the CUDA backend's "no devices visible" semantics rather than
  // propagating the exception out of a noexcept function.
  try {
    return static_cast<int>(all_sycl_devices().size());
  } catch (...) {
    return 0;
  }
}

auto
Gpu::report() const -> void {
  report(carbon::global_report());
}

auto
Gpu::report(Report& report) const -> void {
  Report::SubBlock carbon = report.get_subdoc();
  report.add_member("gpu_count", static_cast<std::int32_t>(1), &carbon);
  add_gpu_device_subdoc(*this, report, carbon);
  report.add_subdoc("carbon", carbon);
}

auto
Gpu::set_device() const -> void {
  // No-op under SYCL (devices are bound per-queue, not per-thread). Kept for API
  // symmetry with the CUDA backend where it calls cudaSetDevice.
}

auto
Gpu::launch_smoke_kernel() const -> void {
  // Submit an empty 1-thread parallel_for and wait. Exercises the queue / submission /
  // wait path on this device, so any toolchain misconfiguration shows up here.
  sycl::queue q(context_, device_, sycl::property::queue::in_order{});
  q.parallel_for(sycl::range<1>(1), [](sycl::id<1>) {
   }).wait();
}

auto
Gpu::allocated_bytes() const noexcept -> size_t {
  return gpu_tracked_bytes(device_id_);
}

}  // namespace detail

CARBON_API auto
device_count() noexcept -> int {
  return detail::Gpu::device_count();
}

CARBON_API auto
get_all_devices() -> std::vector<detail::Gpu> {
  const int n = detail::Gpu::device_count();
  std::vector<detail::Gpu> gpus;
  gpus.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    gpus.emplace_back(i);
  }
  return gpus;
}

namespace detail {

// Empty by default; enumerate explicitly via GpuGroup(carbon::get_all_devices()).
// See the CUDA gpu.cpp twin for the full rationale (avoids per-launch device
// re-enumeration through System's default / move paths).
GpuGroup::GpuGroup() = default;

GpuGroup::GpuGroup(std::vector<Gpu> gpus)
    : gpus_(std::move(gpus)) {
  const int n = static_cast<int>(gpus_.size());
  if (n == 0) {
    CARBON_LOG_INFO("GpuGroup: no SYCL GPU/accelerator devices visible");
    return;
  }

  CARBON_LOG_INFO("GpuGroup: {} SYCL device(s)", n);
  for (const auto& g : gpus_) {
    const auto& p = g.properties();
    CARBON_LOG_INFO(
        "  GPU {}: {} | {} MiB global | {} compute units | sub-group {} | max work-group {}",
        g.device_id(), g.name(), p.total_global_mem_bytes / (1024U * 1024U), p.multiprocessor_count,
        p.warp_size, p.max_threads_per_block);
  }
}

auto
GpuGroup::size() const noexcept -> size_t {
  return gpus_.size();
}

auto
GpuGroup::report() const -> void {
  report(carbon::global_report());
}

auto
GpuGroup::report(Report& report) const -> void {
  Report::SubBlock carbon = report.get_subdoc();
  report.add_member("gpu_count", static_cast<std::int32_t>(size()), &carbon);
  for (const auto& gpu : gpus_) {
    add_gpu_device_subdoc(gpu, report, carbon);
  }
  report.add_subdoc("carbon", carbon);
}

auto
GpuGroup::for_each_device(const std::function<void(const Gpu&)>& f) const -> void {
  for (const auto& g : gpus_) {
    g.set_device();
    f(g);
  }
}

auto
GpuGroup::parallel_for_each_device(const std::function<void(const Gpu&)>& f) const -> void {
  parallel_for_each_device([&f](const Gpu& g) {
    f(g);
  });
}

auto
GpuGroup::launch_smoke_kernel_on_each() const -> void {
  for (const auto& g : gpus_) {
    g.launch_smoke_kernel();
  }
}

namespace {

std::mutex g_gpu_alloc_mutex;
/** Indexed by carbon device ordinal; grown on demand when tracking allocations. */
std::vector<size_t> g_gpu_alloc_bytes;

}  // namespace

auto
gpu_track_allocation(int device_id,
                     size_t bytes) -> void {
  if (bytes == 0U || device_id < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_gpu_alloc_mutex);
  const auto idx = static_cast<size_t>(device_id);
  if (idx >= g_gpu_alloc_bytes.size()) {
    g_gpu_alloc_bytes.resize(idx + 1U, 0U);
  }
  g_gpu_alloc_bytes[idx] += bytes;
}

auto
gpu_track_deallocation(int device_id,
                       size_t bytes) -> void {
  if (bytes == 0U || device_id < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_gpu_alloc_mutex);
  const auto idx = static_cast<size_t>(device_id);
  if (idx >= g_gpu_alloc_bytes.size()) {
    return;
  }
  if (g_gpu_alloc_bytes[idx] >= bytes) {
    g_gpu_alloc_bytes[idx] -= bytes;
  } else {
    g_gpu_alloc_bytes[idx] = 0;
  }
}

auto
gpu_tracked_bytes(int device_id) -> size_t {
  std::lock_guard<std::mutex> lock(g_gpu_alloc_mutex);
  if (device_id < 0) {
    return 0U;
  }
  const auto idx = static_cast<size_t>(device_id);
  return idx >= g_gpu_alloc_bytes.size() ? 0U : g_gpu_alloc_bytes[idx];
}

}  // namespace detail

}  // namespace carbon
