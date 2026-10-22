#include <carbon/detail/gpu_group.h>
#include <carbon/devices.h>
#include <carbon/logging.h>
#include <carbon/report.h>

#include <carbon/detail/backend/cudalike/peer_access.h>
#include <carbon/detail/gpu_allocation_tracking.h>

#include <cuda_runtime.h>

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" cudaError_t
carbon_launch_smoke_kernel(int device_id);

namespace carbon {
namespace detail {

namespace {

auto
make_properties(const cudaDeviceProp& prop) -> Properties {
  Properties p;
  p.name = prop.name;
  p.compute_major = prop.major;
  p.compute_minor = prop.minor;
  p.total_global_mem_bytes = static_cast<size_t>(prop.totalGlobalMem);
  p.multiprocessor_count = prop.multiProcessorCount;
  p.max_threads_per_block = prop.maxThreadsPerBlock;
  p.warp_size = prop.warpSize;
  p.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
#if CUDART_VERSION >= 13000
  p.clock_rate_khz = 0;
  p.memory_clock_rate_khz = 0;
#else
  p.clock_rate_khz = prop.clockRate;
  p.memory_clock_rate_khz = prop.memoryClockRate;
#endif
  p.memory_bus_width_bits = prop.memoryBusWidth;
  p.l2_cache_size_bytes = prop.l2CacheSize;
  p.total_const_mem_bytes = static_cast<size_t>(prop.totalConstMem);
  p.shared_mem_per_block_bytes = static_cast<size_t>(prop.sharedMemPerBlock);
  p.regs_per_block = prop.regsPerBlock;
  return p;
}

}  // namespace

/** Adds member gpu_<id> with device fields under @p parent (the JSON object under "carbon"). */
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
  int count = 0;
  const cudaError_t count_err = cudaGetDeviceCount(&count);
  if (count_err != cudaSuccess) {
    carbon::log::throw_with_log<std::runtime_error>("cudaGetDeviceCount: {}",
                                                    cudaGetErrorString(count_err));
  }
  if (device_id < 0 || device_id >= count) {
    carbon::log::throw_with_log<std::runtime_error>(
        "carbon::detail::Gpu: device_id out of range (device_id={}, count={})", device_id, count);
  }

  cudaDeviceProp prop{};
  const cudaError_t prop_err = cudaGetDeviceProperties(&prop, device_id_);
  if (prop_err != cudaSuccess) {
    carbon::log::throw_with_log<std::runtime_error>("cudaGetDeviceProperties: {}",
                                                    cudaGetErrorString(prop_err));
  }

  props_ = make_properties(prop);
}

auto
Gpu::device_count() noexcept -> int {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0;
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
  const cudaError_t err = cudaSetDevice(device_id_);
  if (err != cudaSuccess) {
    carbon::log::throw_with_log<std::runtime_error>("cudaSetDevice: {}", cudaGetErrorString(err));
  }
}

auto
Gpu::launch_smoke_kernel() const -> void {
  const cudaError_t err = carbon_launch_smoke_kernel(device_id_);
  if (err != cudaSuccess) {
    carbon::log::throw_with_log<std::runtime_error>("carbon_launch_smoke_kernel: {}",
                                                    cudaGetErrorString(err));
  }
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

// Empty by default. Enumerating every visible GPU is heavyweight (one
// cudaGetDeviceProperties per device, ~1ms each), so it must be requested
// explicitly via GpuGroup(carbon::get_all_devices()). A default / moved-from /
// unattached GpuGroup carries no devices, which keeps System's default and
// move paths off the device-probe path (a non-empty default here was the root
// cause of the per-launch device re-enumeration stall — see docs/CHANGELOG.md).
GpuGroup::GpuGroup() = default;

GpuGroup::GpuGroup(std::vector<Gpu> gpus)
    : gpus_(std::move(gpus)) {
  const int n = static_cast<int>(gpus_.size());
  if (n == 0) {
    CARBON_LOG_INFO("GpuGroup: no CUDA devices visible (cudaGetDeviceCount returned 0)");
    return;
  }

  CARBON_LOG_INFO("GpuGroup: {} CUDA device(s)", n);
  for (const auto& g : gpus_) {
    const auto& p = g.properties();
    CARBON_LOG_INFO(
        "  GPU {}: {} | compute {}.{} | {} MiB global | {} multiprocessors | warp {} | max "
        "threads/block {}",
        g.device_id(), g.name(), p.compute_major, p.compute_minor,
        p.total_global_mem_bytes / (1024U * 1024U), p.multiprocessor_count, p.warp_size,
        p.max_threads_per_block);
  }

  // Peer access, subject to the shared CUDA/HIP policy: validated rather than taken on the
  // runtime's word, because a platform can claim P2P capability it does not have and then
  // silently transfer nothing. See detail/backend/cudalike/peer_access.h.
  std::vector<int> device_ids;
  device_ids.reserve(gpus_.size());
  for (const auto& g : gpus_) {
    device_ids.push_back(g.device_id());
  }
  configure_peer_access(device_ids);
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
/** Indexed by CUDA device ordinal; grown on demand when tracking allocations. */
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
