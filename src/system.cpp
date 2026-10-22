#include <carbon/system.h>

#include <carbon/detail/backend/select.h>
#include <carbon/logging.h>

#if CARBON_BACKEND_IS_CUDA
#include <cuda_runtime.h>
#elif CARBON_BACKEND_IS_HIP
#include <hip/hip_runtime.h>
#elif CARBON_BACKEND_IS_SYCL
#include <sycl/sycl.hpp>
#endif

#include <cstring>
#include <stdexcept>
#include <utility>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/detail/communicator.h>
#include <carbon/distributed.h>
#endif

namespace carbon {

#ifdef CARBON_HAVE_MPI_NCCL
System::System(std::shared_ptr<const Distributed> distributed) {
  if (!distributed) {
    log::throw_with_log<std::invalid_argument>("System: Distributed handle is null");
  }
  const int chosen = distributed->pin_local_device();
  detail::GpuGroup group(std::vector<detail::Gpu>{detail::Gpu{chosen}});
  auto comm = detail::Communicator::from_mpi(*distributed);
  state_ = std::make_shared<State>(
      State{std::move(distributed), std::move(comm), std::move(group), {}, {}, {}});
}
#endif

auto
System::async_copy(const detail::Allocation<std::byte>& src,
                   const detail::Allocation<std::byte>& dst,
                   size_t size_bytes,
                   const Stream& stream) const -> void {
  if (size_bytes == 0U) {
    return;
  }
  if (src.size_bytes() < size_bytes || dst.size_bytes() < size_bytes) {
    log::throw_with_log<std::out_of_range>(
        "System::async_copy: size_bytes {} exceeds src.size_bytes {} or dst.size_bytes {}",
        size_bytes, src.size_bytes(), dst.size_bytes());
  }
  const auto sl = src.location();
  const auto dl = dst.location();
  const void* s = static_cast<const void*>(src.data());
  void* d = static_cast<void*>(dst.data());

  if (sl == detail::AllocationLocation::Cpu && dl == detail::AllocationLocation::Cpu) {
    // Host-to-host: streams do not order host work, so a synchronous memcpy is the correct
    // semantic here (no fence introduced). The Stream argument is irrelevant in this branch
    // and may legitimately be empty (size 0).
    std::memcpy(d, s, size_bytes);
    return;
  }

  // Pick the per-GPU stream out of the group following the same rule the previous
  // pick_stream_ helper in transfer.h used: destination's relative_gpu_id for H2D / D2D /
  // peer copies (the writer side runs the copy), source's for D2H. The check is also the
  // single place we validate that the Stream group is large enough to cover the device
  // side of the transfer — callers no longer need to worry about it.
  int rel = -1;
  if (dl == detail::AllocationLocation::Cuda) {
    rel = dst.relative_gpu_id();
  } else if (sl == detail::AllocationLocation::Cuda) {
    rel = src.relative_gpu_id();
  }
  if (rel < 0 || static_cast<size_t>(rel) >= stream.size()) {
    log::throw_with_log<std::out_of_range>(
        "System::async_copy: relative_gpu_id {} is out of range for the provided Stream "
        "(size={})",
        rel, stream.size());
  }

#if CARBON_BACKEND_IS_CUDA
  const cudaStream_t cs = stream[static_cast<size_t>(rel)].get();

  cudaError_t err = cudaSuccess;
  if (sl == detail::AllocationLocation::Cpu && dl == detail::AllocationLocation::Cuda) {
    err = cudaMemcpyAsync(d, s, size_bytes, cudaMemcpyHostToDevice, cs);
  } else if (sl == detail::AllocationLocation::Cuda && dl == detail::AllocationLocation::Cpu) {
    err = cudaMemcpyAsync(d, s, size_bytes, cudaMemcpyDeviceToHost, cs);
  } else {
    // Both CUDA.
    if (src.gpu_id() == dst.gpu_id()) {
      err = cudaMemcpyAsync(d, s, size_bytes, cudaMemcpyDeviceToDevice, cs);
    } else {
      // Cross-device copy. Use cudaMemcpyAsync with cudaMemcpyDefault rather than
      // cudaMemcpyPeerAsync: under UVA (always on for these 64-bit builds) the
      // driver infers both endpoints' devices from the pointers, so this is the
      // exact same peer DMA when peer access is enabled (GpuGroup enables it for
      // every P2P-capable pair) — but, unlike cudaMemcpyPeerAsync, it is legal
      // during CUDA-graph stream capture (the peer variant returns
      // cudaErrorStreamCaptureUnsupported). This is what lets Skeleton capture a
      // multi-GPU pipeline whose halo issues cross-device copies.
      err = cudaMemcpyAsync(d, s, size_bytes, cudaMemcpyDefault, cs);
    }
  }
  if (err != cudaSuccess) {
    log::throw_with_log<std::runtime_error>("System::async_copy: CUDA error {} ({})",
                                            static_cast<int>(err), cudaGetErrorString(err));
  }
#elif CARBON_BACKEND_IS_HIP
  const hipStream_t cs = stream[static_cast<size_t>(rel)].get();

  hipError_t err = hipSuccess;
  if (sl == detail::AllocationLocation::Cpu && dl == detail::AllocationLocation::Cuda) {
    err = hipMemcpyAsync(d, s, size_bytes, hipMemcpyHostToDevice, cs);
  } else if (sl == detail::AllocationLocation::Cuda && dl == detail::AllocationLocation::Cpu) {
    err = hipMemcpyAsync(d, s, size_bytes, hipMemcpyDeviceToHost, cs);
  } else {
    // Both device.
    if (src.gpu_id() == dst.gpu_id()) {
      err = hipMemcpyAsync(d, s, size_bytes, hipMemcpyDeviceToDevice, cs);
    } else {
      // Cross-device copy via hipMemcpyDefault (UVA infers both devices) rather
      // than hipMemcpyPeerAsync — identical peer DMA when peer access is enabled,
      // but legal during HIP-graph stream capture. See the CUDA branch above.
      err = hipMemcpyAsync(d, s, size_bytes, hipMemcpyDefault, cs);
    }
  }
  if (err != hipSuccess) {
    log::throw_with_log<std::runtime_error>("System::async_copy: HIP error {} ({})",
                                            static_cast<int>(err), hipGetErrorString(err));
  }
#elif CARBON_BACKEND_IS_SYCL
  sycl::queue& q = stream[static_cast<size_t>(rel)].queue();
  try {
    auto ev = q.memcpy(d, s, size_bytes);
    if (sl == detail::AllocationLocation::Cpu && dl == detail::AllocationLocation::Cuda) {
      // AdaptiveCpp 25.02 may not pin the host source for H2D memcpy in a shared
      // multi-device context; wait so the host buffer stays valid for the transfer.
      ev.wait();
    }
  } catch (const sycl::exception& ex) {
    log::throw_with_log<std::runtime_error>("System::async_copy: SYCL exception: {}", ex.what());
  } catch (const std::exception& ex) {
    log::throw_with_log<std::runtime_error>("System::async_copy: {}", ex.what());
  }
#endif
}

auto
System::async_memset(const detail::Allocation<std::byte>& dst,
                     int value,
                     size_t size_bytes,
                     const Stream& stream) const -> void {
  if (size_bytes == 0U) {
    return;
  }
  if (dst.size_bytes() < size_bytes) {
    log::throw_with_log<std::out_of_range>(
        "System::async_memset: size_bytes {} exceeds dst.size_bytes {}", size_bytes,
        dst.size_bytes());
  }
  void* d = static_cast<void*>(dst.data());

  if (dst.location() == detail::AllocationLocation::Cpu) {
    // Host destinations: streams do not order host work, so a synchronous memset is the
    // correct semantic (no fence introduced). The Stream argument is irrelevant in this
    // branch and may legitimately be empty (size 0). Mirrors the host-to-host short-circuit
    // in async_copy. memset takes an int value but only writes the low byte; the cast keeps
    // the public API symmetrical with cudaMemsetAsync's signature.
    std::memset(d, value, size_bytes);
    return;
  }

  // Device destination: pick the per-GPU stream the same way async_copy does (writer side
  // owns the operation, which here is the destination), and validate the Stream group is
  // wide enough to cover dst.relative_gpu_id().
  const int rel = dst.relative_gpu_id();
  if (rel < 0 || static_cast<size_t>(rel) >= stream.size()) {
    log::throw_with_log<std::out_of_range>(
        "System::async_memset: relative_gpu_id {} is out of range for the provided Stream "
        "(size={})",
        rel, stream.size());
  }

#if CARBON_BACKEND_IS_CUDA
  const cudaStream_t cs = stream[static_cast<size_t>(rel)].get();
  const cudaError_t err = cudaMemsetAsync(d, value, size_bytes, cs);
  if (err != cudaSuccess) {
    log::throw_with_log<std::runtime_error>("System::async_memset: CUDA error {} ({})",
                                            static_cast<int>(err), cudaGetErrorString(err));
  }
#elif CARBON_BACKEND_IS_HIP
  const hipStream_t cs = stream[static_cast<size_t>(rel)].get();
  const hipError_t err = hipMemsetAsync(d, value, size_bytes, cs);
  if (err != hipSuccess) {
    log::throw_with_log<std::runtime_error>("System::async_memset: HIP error {} ({})",
                                            static_cast<int>(err), hipGetErrorString(err));
  }
#elif CARBON_BACKEND_IS_SYCL
  sycl::queue& q = stream[static_cast<size_t>(rel)].queue();
  try {
    q.memset(d, value, size_bytes);
  } catch (const sycl::exception& ex) {
    log::throw_with_log<std::runtime_error>("System::async_memset: SYCL exception: {}", ex.what());
  } catch (const std::exception& ex) {
    log::throw_with_log<std::runtime_error>("System::async_memset: {}", ex.what());
  }
#endif
}

}  // namespace carbon
