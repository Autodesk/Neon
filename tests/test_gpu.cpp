#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/detail/backend/select.h>
#include <carbon/logging.h>
#include <carbon/report.h>

#include <cassert>
#include <cstdint>
#include <vector>

namespace {

// The backends expose the underlying handle under different names on purpose — see the
// "Public surface vs the CUDA wrapper" note in detail/backend/sycl/stream.h — so the
// shallow-copy identity checks below go through these rather than naming one backend's
// accessor. Comparing addresses is enough: the point is that a copy shares the original's
// handle, not what the handle is.
[[nodiscard]] auto
stream_handle(const carbon::detail::CudaStream& s) noexcept -> const void* {
#if CARBON_BACKEND_IS_SYCL
  return static_cast<const void*>(&s.queue());
#else
  return static_cast<const void*>(s.get());
#endif
}

[[nodiscard]] auto
event_handle(const carbon::detail::Event& e) noexcept -> const void* {
#if CARBON_BACKEND_IS_SYCL
  return static_cast<const void*>(&e.event());
#else
  return static_cast<const void*>(e.get());
#endif
}

}  // namespace

int
main(int argc,
     char** argv) {
  carbon::init();
  const int device_count = carbon::device_count();
  if (device_count == 0) {
    CARBON_LOG_ERROR("test_gpu: no CUDA devices reported (cudaGetDeviceCount)");
    return 1;
  }

  carbon::report::init("carbon_test_gpu", argc, argv);
  CARBON_LOG_INFO("CUDA device count: {}", device_count);

  const carbon::detail::GpuGroup gpu_group(carbon::get_all_devices());
  assert(gpu_group.size() == static_cast<size_t>(device_count));
  assert(!gpu_group.empty());
  const carbon::detail::Gpu& gpu = gpu_group.devices().front();
  assert(!gpu.name().empty());
  assert(!gpu.properties().name.empty());
  assert(gpu.properties().total_global_mem_bytes > 0U);
  assert(gpu.properties().warp_size > 0);
  assert(gpu.device_id() == 0);

  {
    carbon::detail::CudaStream st = gpu.stream();
    assert(st);
    carbon::detail::CudaStream st_copy = st;
    assert(stream_handle(st_copy) == stream_handle(st));
    carbon::detail::Event ev = gpu.event();
    assert(ev);
    carbon::detail::Event ev_copy = ev;
    assert(event_handle(ev_copy) == event_handle(ev));
  }

  const size_t alloc_base = gpu.allocated_bytes();
  {
    carbon::detail::Allocation<std::uint8_t> tmp = gpu.buffer<std::uint8_t>(64U, 0);
    assert(gpu.allocated_bytes() == alloc_base + 64U);
  }
  assert(gpu.allocated_bytes() == alloc_base);

  carbon::detail::Allocation<std::uint8_t> cpu_buf = carbon::Cpu::buffer<std::uint8_t>(64U);
  assert(cpu_buf);
  assert(cpu_buf.size() == 64U);
  assert(cpu_buf.size_bytes() == 64U);
  assert(cpu_buf.location() == carbon::detail::AllocationLocation::Cpu);
  assert(cpu_buf.gpu_id() == -1);
  assert(cpu_buf.relative_gpu_id() == -1);
  assert(cpu_buf.data() != nullptr);

  carbon::detail::Allocation<std::uint8_t> gpu_buf = gpu.buffer<std::uint8_t>(64U, 0);
  assert(gpu.allocated_bytes() == alloc_base + 64U);
  assert(gpu_buf);
  assert(gpu_buf.size() == 64U);
  assert(gpu_buf.size_bytes() == 64U);
  assert(gpu_buf.location() == carbon::detail::AllocationLocation::Cuda);
  assert(gpu_buf.gpu_id() == gpu.device_id());
  assert(gpu_buf.relative_gpu_id() == 0);
  assert(gpu_buf.data() != nullptr);

  {
    std::vector<size_t> counts(static_cast<size_t>(device_count), 32U);
    carbon::detail::BufferGroup<std::uint8_t> multi = gpu_group.buffer<std::uint8_t>(counts);
    assert(multi.size() == static_cast<size_t>(device_count));
    for (size_t i = 0; i < multi.size(); ++i) {
      assert(multi[i].size() == 32U);
      assert(multi[i].relative_gpu_id() == static_cast<int>(i));
      assert(multi[i].gpu_id() == gpu_group.devices()[i].device_id());
    }
  }

  {
    carbon::StreamGroup sg = gpu_group.stream();
    assert(sg.size() == static_cast<size_t>(device_count));
    carbon::StreamGroup sg_copy = sg;
    assert(sg_copy.get_uid() == sg.get_uid());
    for (size_t i = 0; i < sg.size(); ++i) {
      assert(stream_handle(sg[i]) == stream_handle(sg_copy[i]));
    }
    carbon::EventGroup eg = gpu_group.event();
    assert(eg.size() == static_cast<size_t>(device_count));
    carbon::EventGroup eg_copy = eg;
    assert(eg_copy.get_uid() == eg.get_uid());
    for (size_t i = 0; i < eg.size(); ++i) {
      assert(event_handle(eg[i]) == event_handle(eg_copy[i]));
    }
  }

  gpu_group.for_each_device([](const carbon::detail::Gpu& g) {
    CARBON_LOG_INFO("for_each_device: current target is GPU {}", g.device_id());
  });
  gpu_group.parallel_for_each_device([](const carbon::detail::Gpu& g) {
    CARBON_LOG_INFO("parallel_for_each_device: thread for GPU {}", g.device_id());
  });
  gpu_group.launch_smoke_kernel_on_each();
  CARBON_LOG_INFO("smoke kernel completed on each GPU");

  gpu_group.report();

  carbon::report::write("carbon_gpu_test", true, ".");

  const auto& p = gpu.properties();
  CARBON_LOG_INFO("--- device {} ---", gpu.device_id());
  CARBON_LOG_INFO("name: {}", gpu.name());
  CARBON_LOG_INFO("compute capability: {}.{}", p.compute_major, p.compute_minor);
  CARBON_LOG_INFO("total global memory: {} bytes", p.total_global_mem_bytes);
  CARBON_LOG_INFO("multiprocessor count: {}", p.multiprocessor_count);
  CARBON_LOG_INFO("max threads / block: {}", p.max_threads_per_block);
  CARBON_LOG_INFO("warp size: {}", p.warp_size);
  CARBON_LOG_INFO("max threads / multiprocessor: {}", p.max_threads_per_multiprocessor);
  CARBON_LOG_INFO("clock rate: {} kHz", p.clock_rate_khz);
  CARBON_LOG_INFO("memory clock rate: {} kHz", p.memory_clock_rate_khz);
  CARBON_LOG_INFO("memory bus width: {} bits", p.memory_bus_width_bits);
  CARBON_LOG_INFO("L2 cache size: {} bytes", p.l2_cache_size_bytes);
  CARBON_LOG_INFO("total constant memory: {} bytes", p.total_const_mem_bytes);
  CARBON_LOG_INFO("shared memory / block: {} bytes", p.shared_mem_per_block_bytes);
  CARBON_LOG_INFO("registers / block: {}", p.regs_per_block);

  return 0;
}
