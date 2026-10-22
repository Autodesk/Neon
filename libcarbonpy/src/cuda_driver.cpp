#include <carbonpy/container_token.h>
#include <carbonpy/cuda_driver.h>

#include <carbonpy/detail/field_box.h>

#include <carbon/access_token.h>
#include <carbon/boundary_mode.h>
#include <carbon/data_view.h>
#include <carbon/dense.h>
#include <carbon/block.h>
#include <carbon/detail/block/static_block.h>
#include <carbon/kernel.h>
#include <carbon/mgrid.h>
#include <carbon/system.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

using carbonpy::detail::dispatch_dtype;
using carbonpy::detail::FieldBox;
using carbonpy::detail::MgridFieldBox;

auto
check_cu(CUresult res,
         const char* op) -> int {
  if (res == CUDA_SUCCESS) {
    return 0;
  }
  const char* name = nullptr;
  const char* msg = nullptr;
  cuGetErrorName(res, &name);
  cuGetErrorString(res, &msg);
  (void)name;
  (void)msg;
  return static_cast<int>(res);
}

auto
int_to_view(int v) -> carbon::DataView {
  switch (v) {
    case 1:
      return carbon::DataView::Internal;
    case 2:
      return carbon::DataView::Boundary;
    default:
      return carbon::DataView::Standard;
  }
}

auto
make_bounds(int64_t nx,
            int64_t ny,
            int64_t nz,
            CarbonLaunchBounds* out) -> void {
  out->ndim = 3;
  out->shape[0] = static_cast<int32_t>(nx);
  out->shape[1] = static_cast<int32_t>(ny);
  out->shape[2] = static_cast<int32_t>(nz);
  out->shape[3] = 1;
  out->size = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
}

class CudaDriver {
public:
  explicit CudaDriver(carbon::System* system)
      : system_(system) {
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
      throw std::runtime_error("cuInit failed");
    }
    const size_t n = system_->gpu_group().size();
    devices_.resize(n);
    contexts_.resize(n);
    streams_.resize(n);
    for (size_t i = 0; i < n; ++i) {
      const int ord = system_->gpu_group().devices()[i].device_id();
      res = cuDeviceGet(&devices_[i], ord);
      if (res != CUDA_SUCCESS) {
        throw std::runtime_error("cuDeviceGet failed");
      }
      res = cuDevicePrimaryCtxRetain(&contexts_[i], devices_[i]);
      if (res != CUDA_SUCCESS) {
        throw std::runtime_error("cuDevicePrimaryCtxRetain failed");
      }
      res = cuCtxSetCurrent(contexts_[i]);
      if (res != CUDA_SUCCESS) {
        throw std::runtime_error("cuCtxSetCurrent failed");
      }
      cudaStream_t stream{};
      if (cudaStreamCreate(&stream) != cudaSuccess) {
        throw std::runtime_error("cudaStreamCreate failed");
      }
      streams_[i] = stream;
    }
  }

  ~CudaDriver() {
    for (size_t i = 0; i < streams_.size(); ++i) {
      if (streams_[i] != nullptr) {
        cudaStreamDestroy(streams_[i]);
      }
      if (contexts_[i] != nullptr) {
        cuDevicePrimaryCtxRelease(devices_[i]);
      }
    }
  }

  [[nodiscard]] auto
  gpu_count() const -> size_t {
    return system_->gpu_group().size();
  }

  [[nodiscard]] auto
  stream(size_t gpu_id) const -> cudaStream_t {
    return streams_.at(gpu_id);
  }

  auto
  launch(int gpu_id,
         CUfunction function,
         CarbonLaunchBounds bounds,
         int block_dim,
         int shared_memory_bytes,
         void** kernel_params) const -> int {
    return launch_on(0U, gpu_id, function, bounds, block_dim, shared_memory_bytes, kernel_params);
  }

  auto
  ensure_stream_slots(size_t count) -> void {
    if (count <= stream_slots_) {
      return;
    }
    while (extra_streams_.size() + 1U < count) {
      std::vector<cudaStream_t> row(streams_.size(), nullptr);
      for (size_t g = 0; g < streams_.size(); ++g) {
        CUresult res = cuCtxSetCurrent(contexts_[g]);
        if (res != CUDA_SUCCESS) {
          throw std::runtime_error("cuCtxSetCurrent failed");
        }
        if (cudaStreamCreate(&row[g]) != cudaSuccess) {
          throw std::runtime_error("cudaStreamCreate failed");
        }
      }
      extra_streams_.push_back(std::move(row));
    }
    stream_slots_ = count;
  }

  [[nodiscard]] auto
  stream_at(size_t slot,
            size_t gpu) const -> cudaStream_t {
    if (slot == 0U) {
      return streams_.at(gpu);
    }
    return extra_streams_.at(slot - 1U).at(gpu);
  }

  auto
  set_device(size_t gpu) const -> int {
    return check_cu(cuCtxSetCurrent(contexts_.at(gpu)), "cuCtxSetCurrent");
  }

  auto
  sync_stream_slots(size_t slot_count) const -> void {
    const size_t n = streams_.size();
    for (size_t s = 0; s < slot_count; ++s) {
      for (size_t g = 0; g < n; ++g) {
        (void)set_device(g);
        cudaStreamSynchronize(stream_at(s, g));
      }
    }
  }

  auto
  launch_on_native_stream(int gpu_id,
                          CUfunction function,
                          CarbonLaunchBounds bounds,
                          int block_dim,
                          int shared_memory_bytes,
                          void** kernel_params,
                          cudaStream_t stream) const -> int {
    if (block_dim <= 0) {
      block_dim = 256;
    }
    int grid_dim = static_cast<int>((bounds.size + static_cast<size_t>(block_dim) - 1U) /
                                    static_cast<size_t>(block_dim));
    if (grid_dim < 0) {
      grid_dim = 2147483647;
    }

    CUresult res = cuCtxSetCurrent(contexts_[static_cast<size_t>(gpu_id)]);
    if (res != CUDA_SUCCESS) {
      return check_cu(res, "cuCtxSetCurrent");
    }

    res = cuLaunchKernel(function, grid_dim, 1, 1, block_dim, 1, 1, shared_memory_bytes,
                         static_cast<CUstream>(stream), kernel_params, nullptr);
    return check_cu(res, "cuLaunchKernel");
  }

  auto
  launch_on(size_t stream_slot,
            int gpu_id,
            CUfunction function,
            CarbonLaunchBounds bounds,
            int block_dim,
            int shared_memory_bytes,
            void** kernel_params) const -> int {
    return launch_on_native_stream(gpu_id, function, bounds, block_dim, shared_memory_bytes,
                                   kernel_params, stream_at(stream_slot, static_cast<size_t>(gpu_id)));
  }

private:
  carbon::System* system_;
  std::vector<CUdevice> devices_;
  std::vector<CUcontext> contexts_;
  std::vector<cudaStream_t> streams_;
  size_t stream_slots_{1U};
  std::vector<std::vector<cudaStream_t>> extra_streams_;
};

struct DenseContainer {
  CudaDriver* driver{};
  carbon::Dense* dense{};
  std::vector<void*> hooks;
  int data_views{3};
  int field_count{1};
  // Capture kernels (built by carbon.kernel / build_bound_kernel) bake all
  // partitions + span as Warp constants and take no user params, so their
  // launch passes only the bounds. Non-capture kernels take (part, span[, ...]).
  bool capture{false};
  std::vector<CarbonDensePartition> parts;
  std::vector<CarbonDenseSpan> spans;
  std::vector<carbon::AccessToken> tokens;

  DenseContainer(CudaDriver* drv,
                 carbon::Dense* grid,
                 void** kernel_hooks,
                 int hook_count,
                 int fields,
                 bool capture_kernels)
      : driver(drv),
        dense(grid),
        field_count(fields > 0 ? fields : 1),
        capture(capture_kernels) {
    hooks.assign(static_cast<size_t>(hook_count), nullptr);
    for (int i = 0; i < hook_count; ++i) {
      hooks[static_cast<size_t>(i)] = kernel_hooks[i];
    }
    const size_t n_gpus = driver->gpu_count();
    parts.assign(n_gpus * static_cast<size_t>(field_count), CarbonDensePartition{});
    spans.assign(n_gpus, CarbonDenseSpan{});
  }

  [[nodiscard]] auto
  hook_index(size_t gpu,
             carbon::DataView view) const -> size_t {
    return gpu * static_cast<size_t>(data_views) + static_cast<size_t>(view);
  }
};

auto
fill_span(carbon::Dense* dense,
          int device,
          carbon::DataView view,
          CarbonDenseSpan* out) -> int {
  const auto& spans = dense->spans();
  if (device < 0 || static_cast<size_t>(device) >= spans.size()) {
    return -1;
  }
  const auto& span = spans[static_cast<size_t>(device)];
  const auto ext = span.iter_extent(view);
  out->iter_x = ext.x;
  out->iter_y = ext.y;
  out->iter_z = ext.z;
  const auto logic = span.logic_extent();
  out->logic_x = logic.x;
  out->logic_y = logic.y;
  out->logic_z = logic.z;
  out->pitch_xy = span.xy_slab_elements();
  out->z_offset = span.z_offset();
  out->halo_x = 0;
  out->halo_y = 0;
  out->halo_z = static_cast<int32_t>(span.halo_z());
  const auto global = span.global_extent();
  out->global_x = global.x;
  out->global_y = global.y;
  out->global_z = global.z;
  out->boundary_mode =
      span.boundary_mode() == carbon::BoundaryMode::PeriodicZ ? 1 : 0;
  out->data_view = static_cast<int32_t>(view);
  return 0;
}

// Component counts instantiated for Dense fields in the container path. Mirrors
// the list in dense.cpp: 1-4 for Cartesian scalar/vector fields, 7/9/15/19/27
// for the LBM lattice sizes (D3Q7, D2Q9, D3Q15, D3Q19, D3Q27).
#define CARBON_DENSE_FOR_EACH_COMPONENT_COUNT(X) \
  X(1)                                           \
  X(2)                                           \
  X(3)                                           \
  X(4)                                           \
  X(7)                                           \
  X(9)                                           \
  X(15)                                          \
  X(19)                                          \
  X(27)

auto
fill_partition(FieldBox* box,
               int device,
               CarbonDensePartition* out) -> int {
  if (box == nullptr || box->field == nullptr) {
    return -1;
  }
  auto set_part = [&](auto* field) -> int {
    const auto& parts = field->partitions();
    if (device < 0 || static_cast<size_t>(device) >= parts.size()) {
      return -1;
    }
    const auto& part = parts[static_cast<size_t>(device)];
    out->data_ptr = reinterpret_cast<uint64_t>(part.ptr());
    out->span = CarbonDenseSpan{};
    if (box->dense != nullptr) {
      fill_span(static_cast<carbon::Dense*>(box->dense), device,
                carbon::DataView::Standard, &out->span);
    }
    return 0;
  };
  return dispatch_dtype(box->dtype, [&]<typename T>() -> int {
    switch (box->components) {
#define CARBON_CASE(C) \
  case C:              \
    return set_part(static_cast<carbon::Dense::Field<T, C>*>(box->field));
      CARBON_DENSE_FOR_EACH_COMPONENT_COUNT(CARBON_CASE)
#undef CARBON_CASE
      default:
        return -1;
    }
  });
}

auto
launch_dense_gpu(DenseContainer* container,
                 size_t gpu,
                 carbon::DataView view,
                 cudaStream_t stream) -> int {
  if (container == nullptr || container->driver == nullptr) {
    return -1;
  }
  if (fill_span(container->dense, static_cast<int>(gpu), view, &container->spans[gpu]) != 0) {
    return -1;
  }
  const size_t idx = container->hook_index(gpu, view);
  if (idx >= container->hooks.size() || container->hooks[idx] == nullptr) {
    return 0;
  }
  const auto& span = container->spans[gpu];
  CarbonLaunchBounds bounds{};
  make_bounds(span.iter_x, span.iter_y, span.iter_z, &bounds);
  if (container->capture) {
    // No user params: partitions + span are baked into the kernel as constants.
    void* params[1];
    params[0] = &bounds;
    return container->driver->launch_on_native_stream(static_cast<int>(gpu),
                                                    static_cast<CUfunction>(container->hooks[idx]),
                                                    bounds, 256, 0, params, stream);
  }
  if (container->field_count == 1) {
    const auto& part = container->parts[gpu];
    void* params[3];
    params[0] = &bounds;
    params[1] = const_cast<CarbonDensePartition*>(&part);
    params[2] = const_cast<CarbonDenseSpan*>(&span);
    return container->driver->launch_on_native_stream(static_cast<int>(gpu),
                                                    static_cast<CUfunction>(container->hooks[idx]),
                                                    bounds, 256, 0, params, stream);
  }
  if (container->field_count == 2) {
    const size_t base = gpu * 2U;
    void* params[5];
    params[0] = &bounds;
    params[1] = const_cast<CarbonDensePartition*>(&container->parts[base]);
    params[2] = const_cast<CarbonDenseSpan*>(&span);
    params[3] = const_cast<CarbonDensePartition*>(&container->parts[base + 1U]);
    params[4] = const_cast<CarbonDenseSpan*>(&span);
    return container->driver->launch_on_native_stream(static_cast<int>(gpu),
                                                    static_cast<CUfunction>(container->hooks[idx]),
                                                    bounds, 256, 0, params, stream);
  }
  return -1;
}

auto
run_dense_container(DenseContainer* container,
                    size_t stream_slot,
                    carbon::DataView view,
                    bool sync_at_end) -> int {
  const size_t n = container->driver->gpu_count();
  for (size_t g = 0; g < n; ++g) {
    const int rc = launch_dense_gpu(container, g, view,
                                    container->driver->stream_at(stream_slot, g));
    if (rc != 0) {
      return rc;
    }
  }
  if (sync_at_end) {
    container->driver->sync_stream_slots(stream_slot + 1U);
  }
  return 0;
}

}  // namespace

extern "C" auto
carbonpy_cuda_driver_new(void** handle,
                         void* system_handle) -> int {
  if (handle == nullptr || system_handle == nullptr) {
    return -1;
  }
  try {
    auto* sys = static_cast<carbon::System*>(system_handle);
    *handle = new CudaDriver(sys);
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_cuda_driver_delete(void** handle) -> int {
  if (handle == nullptr || *handle == nullptr) {
    return -1;
  }
  delete static_cast<CudaDriver*>(*handle);
  *handle = nullptr;
  return 0;
}

extern "C" auto
carbonpy_cuda_driver_launch(void* driver_handle,
                            int gpu_id,
                            void* cuda_stream,
                            void* cu_function,
                            CarbonLaunchBounds bounds,
                            int block_dim,
                            int shared_memory_bytes,
                            void** kernel_params,
                            int /*param_count*/) -> int {
  if (driver_handle == nullptr || cu_function == nullptr) {
    return -1;
  }
  try {
    auto* driver = static_cast<CudaDriver*>(driver_handle);
    (void)cuda_stream;
    return driver->launch(gpu_id, static_cast<CUfunction>(cu_function), bounds, block_dim,
                          shared_memory_bytes, kernel_params);
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_dense_container_new(void** handle,
                             void* driver_handle,
                             void* dense_handle,
                             void** kernel_hooks,
                             int hook_count,
                             int field_count,
                             int capture) -> int {
  if (handle == nullptr || driver_handle == nullptr || dense_handle == nullptr ||
      kernel_hooks == nullptr) {
    return -1;
  }
  try {
    auto* driver = static_cast<CudaDriver*>(driver_handle);
    auto* dense = static_cast<carbon::Dense*>(dense_handle);
    *handle = new DenseContainer(driver, dense, kernel_hooks, hook_count, field_count,
                                 capture != 0);
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_dense_container_delete(void** handle) -> int {
  if (handle == nullptr || *handle == nullptr) {
    return -1;
  }
  delete static_cast<DenseContainer*>(*handle);
  *handle = nullptr;
  return 0;
}

extern "C" auto
carbonpy_dense_container_bind_field(void* container_handle,
                                    void* field_handle,
                                    int device,
                                    int field_slot) -> int {
  if (container_handle == nullptr || field_handle == nullptr) {
    return -1;
  }
  try {
    auto* container = static_cast<DenseContainer*>(container_handle);
    auto* box = static_cast<FieldBox*>(field_handle);
    const size_t n_gpus = container->driver->gpu_count();
    if (device < 0 || static_cast<size_t>(device) >= n_gpus || field_slot < 0 ||
        field_slot >= container->field_count) {
      return -1;
    }
    const size_t part_idx =
        static_cast<size_t>(device) * static_cast<size_t>(container->field_count) +
        static_cast<size_t>(field_slot);
    if (fill_partition(box, device, &container->parts[part_idx]) != 0) {
      return -1;
    }
    return fill_span(container->dense, device, carbon::DataView::Standard,
                     &container->spans[static_cast<size_t>(device)]);
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_dense_container_run(void* container_handle,
                             int stream_idx,
                             int data_view) -> int {
  if (container_handle == nullptr) {
    return -1;
  }
  try {
    auto* container = static_cast<DenseContainer*>(container_handle);
    const auto view = int_to_view(data_view);
    const size_t slot = stream_idx < 0 ? 0U : static_cast<size_t>(stream_idx);
    return run_dense_container(container, slot, view, true);
  } catch (...) {
    return -1;
  }
}

namespace {

using BlockSBlock = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<BlockSBlock>;

struct BlockContainer {
  CudaDriver* driver{};
  BlockGrid* block{};
  std::vector<void*> hooks;
  int data_views{3};
  std::vector<CarbonBlockPartition> parts;
  std::vector<CarbonBlockSpan> spans;
  std::vector<carbon::AccessToken> tokens;

  BlockContainer(CudaDriver* drv,
                 BlockGrid* grid,
                 void** kernel_hooks,
                 int hook_count)
      : driver(drv),
        block(grid) {
    hooks.assign(static_cast<size_t>(hook_count), nullptr);
    for (int i = 0; i < hook_count; ++i) {
      hooks[static_cast<size_t>(i)] = kernel_hooks[i];
    }
    const size_t n_gpus = driver->gpu_count();
    parts.assign(n_gpus, CarbonBlockPartition{});
    spans.assign(n_gpus, CarbonBlockSpan{});
  }

  [[nodiscard]] auto
  hook_index(size_t gpu,
             carbon::DataView view) const -> size_t {
    return gpu * static_cast<size_t>(data_views) + static_cast<size_t>(view);
  }
};

auto
fill_block_span(BlockGrid* block,
                int device,
                carbon::DataView view,
                CarbonBlockSpan* out) -> int {
  const auto& spans = block->spans();
  if (device < 0 || static_cast<size_t>(device) >= spans.size()) {
    return -1;
  }
  const auto& span = spans[static_cast<size_t>(device)];
  out->bitmask_ptr = reinterpret_cast<uint64_t>(span.bitmask());
  out->connectivity_ptr = reinterpret_cast<uint64_t>(span.connectivity());
  out->origins_ptr = reinterpret_cast<uint64_t>(span.origins());
  out->block_count = span.block_count();
  out->n_internal = span.n_internal();
  out->iter_block_count = span.iter_block_count(view);
  out->mem_x = BlockSBlock::MemX;
  out->mem_y = BlockSBlock::MemY;
  out->mem_z = BlockSBlock::MemZ;
  return 0;
}

auto
fill_block_partition(FieldBox* box,
                     int device,
                     CarbonBlockPartition* out) -> int {
  if (box == nullptr || box->field == nullptr) {
    return -1;
  }
  auto set_part = [&](auto* field) -> int {
    const auto& parts = field->partitions();
    if (device < 0 || static_cast<size_t>(device) >= parts.size()) {
      return -1;
    }
    const auto& part = parts[static_cast<size_t>(device)];
    out->data_ptr = reinterpret_cast<uint64_t>(part.ptr());
    out->components = box->components;
    return 0;
  };
  return dispatch_dtype(box->dtype, [&]<typename T>() -> int {
    switch (box->components) {
      case 1:
        return set_part(static_cast<BlockGrid::Field<T, 1>*>(box->field));
      case 2:
        return set_part(static_cast<BlockGrid::Field<T, 2>*>(box->field));
      case 3:
        return set_part(static_cast<BlockGrid::Field<T, 3>*>(box->field));
      case 4:
        return set_part(static_cast<BlockGrid::Field<T, 4>*>(box->field));
      default:
        return -1;
    }
  });
}

auto
make_bounds_block(const CarbonBlockSpan& span,
                  CarbonLaunchBounds* out) -> void {
  out->ndim = 4;
  out->shape[0] = static_cast<int32_t>(span.iter_block_count);
  out->shape[1] = span.mem_x;
  out->shape[2] = span.mem_y;
  out->shape[3] = span.mem_z;
  out->size = static_cast<size_t>(span.iter_block_count) * static_cast<size_t>(span.mem_x) *
              static_cast<size_t>(span.mem_y) *               static_cast<size_t>(span.mem_z);
}

}  // namespace

extern "C" auto
carbonpy_block_container_new(void** handle,
                               void* driver_handle,
                               void* block_handle,
                               void** kernel_hooks,
                               int hook_count) -> int {
  if (handle == nullptr || driver_handle == nullptr || block_handle == nullptr ||
      kernel_hooks == nullptr) {
    return -1;
  }
  try {
    auto* driver = static_cast<CudaDriver*>(driver_handle);
    auto* block = static_cast<BlockGrid*>(block_handle);
    *handle = new BlockContainer(driver, block, kernel_hooks, hook_count);
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_block_container_delete(void** handle) -> int {
  if (handle == nullptr || *handle == nullptr) {
    return -1;
  }
  delete static_cast<BlockContainer*>(*handle);
  *handle = nullptr;
  return 0;
}

extern "C" auto
carbonpy_block_container_bind_field(void* container_handle,
                                    void* field_handle,
                                    int device) -> int {
  if (container_handle == nullptr || field_handle == nullptr) {
    return -1;
  }
  try {
    auto* container = static_cast<BlockContainer*>(container_handle);
    auto* box = static_cast<FieldBox*>(field_handle);
    if (device < 0 || static_cast<size_t>(device) >= container->parts.size()) {
      return -1;
    }
    if (fill_block_partition(box, device, &container->parts[static_cast<size_t>(device)]) != 0) {
      return -1;
    }
    return fill_block_span(container->block, device, carbon::DataView::Standard,
                           &container->spans[static_cast<size_t>(device)]);
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_block_container_run(void* container_handle,
                             int stream_idx,
                             int data_view) -> int {
  if (container_handle == nullptr) {
    return -1;
  }
  try {
    auto* container = static_cast<BlockContainer*>(container_handle);
    const auto view = int_to_view(data_view);
    const size_t slot = stream_idx < 0 ? 0U : static_cast<size_t>(stream_idx);
    const size_t n = container->driver->gpu_count();
    for (size_t g = 0; g < n; ++g) {
      if (fill_block_span(container->block, static_cast<int>(g), view, &container->spans[g]) != 0) {
        return -1;
      }
    }
    for (size_t g = 0; g < n; ++g) {
      const size_t idx = container->hook_index(g, view);
      if (idx >= container->hooks.size() || container->hooks[idx] == nullptr) {
        continue;
      }
      const auto& span = container->spans[g];
      const auto& part = container->parts[g];
      CarbonLaunchBounds bounds{};
      make_bounds_block(span, &bounds);
      void* params[3];
      params[0] = &bounds;
      params[1] = const_cast<CarbonBlockPartition*>(&part);
      params[2] = const_cast<CarbonBlockSpan*>(&span);
      const int rc = container->driver->launch_on(slot, static_cast<int>(g),
                                                  static_cast<CUfunction>(container->hooks[idx]),
                                                  bounds, 256, 0, params);
      if (rc != 0) {
        return rc;
      }
    }
    container->driver->sync_stream_slots(slot + 1U);
    return 0;
  } catch (...) {
    return -1;
  }
}


using MGridSBlock = carbon::DefaultMultiResBlock;
using MGridGrid = carbon::MultiResBlock<MGridSBlock>;

struct MgridContainer {
  CudaDriver* driver{};
  MGridGrid* mgrid{};
  int level{0};
  std::vector<void*> hooks;
  int data_views{3};
  std::vector<CarbonBlockPartition> parts;
  std::vector<CarbonBlockSpan> spans;
  std::vector<carbon::AccessToken> tokens;

  MgridContainer(CudaDriver* drv,
                 MGridGrid* grid,
                 int grid_level,
                 void** kernel_hooks,
                 int hook_count)
      : driver(drv),
        mgrid(grid),
        level(grid_level) {
    hooks.assign(static_cast<size_t>(hook_count), nullptr);
    for (int i = 0; i < hook_count; ++i) {
      hooks[static_cast<size_t>(i)] = kernel_hooks[i];
    }
    const size_t n_gpus = driver->gpu_count();
    parts.assign(n_gpus, CarbonBlockPartition{});
    spans.assign(n_gpus, CarbonBlockSpan{});
  }

  [[nodiscard]] auto
  hook_index(size_t gpu,
             carbon::DataView view) const -> size_t {
    return gpu * static_cast<size_t>(data_views) + static_cast<size_t>(view);
  }
};

auto
fill_mgrid_span(MGridGrid* mgrid,
                int level,
                int device,
                carbon::DataView view,
                CarbonBlockSpan* out) -> int {
  if (mgrid == nullptr || level < 0 || level >= mgrid->depth()) {
    return -1;
  }
  const auto& grid = mgrid->grid(level);
  const auto& spans = grid.spans();
  if (device < 0 || static_cast<size_t>(device) >= spans.size()) {
    return -1;
  }
  const auto& span = spans[static_cast<size_t>(device)];
  out->bitmask_ptr = reinterpret_cast<uint64_t>(span.bitmask());
  out->connectivity_ptr = reinterpret_cast<uint64_t>(span.connectivity());
  out->origins_ptr = reinterpret_cast<uint64_t>(span.origins());
  out->block_count = span.block_count();
  out->n_internal = span.n_internal();
  out->iter_block_count = span.iter_block_count(view);
  out->mem_x = MGridSBlock::MemX;
  out->mem_y = MGridSBlock::MemY;
  out->mem_z = MGridSBlock::MemZ;
  return 0;
}

auto
fill_mgrid_partition(MgridFieldBox* box,
                     int level,
                     int device,
                     CarbonBlockPartition* out) -> int {
  if (box == nullptr || box->field == nullptr) {
    return -1;
  }
  auto set_part = [&](auto* field) -> int {
    if (level < 0 || level >= field->depth()) {
      return -1;
    }
    const auto part = field->make_partition(level, static_cast<size_t>(device));
    out->data_ptr = reinterpret_cast<uint64_t>(part.ptr());
    out->components = box->components;
    return 0;
  };
  return dispatch_dtype(box->dtype, [&]<typename T>() -> int {
    switch (box->components) {
      case 1:
        return set_part(static_cast<MGridGrid::Field<T, 1>*>(box->field));
      case 2:
        return set_part(static_cast<MGridGrid::Field<T, 2>*>(box->field));
      case 3:
        return set_part(static_cast<MGridGrid::Field<T, 3>*>(box->field));
      case 4:
        return set_part(static_cast<MGridGrid::Field<T, 4>*>(box->field));
      default:
        return -1;
    }
  });
}

extern "C" auto
carbonpy_mgrid_container_new(void** handle,
                             void* driver_handle,
                             void* mgrid_handle,
                             int level,
                             void** kernel_hooks,
                             int hook_count) -> int {
  if (handle == nullptr || driver_handle == nullptr || mgrid_handle == nullptr ||
      kernel_hooks == nullptr) {
    return -1;
  }
  try {
    auto* driver = static_cast<CudaDriver*>(driver_handle);
    auto* mgrid = static_cast<MGridGrid*>(mgrid_handle);
    *handle = new MgridContainer(driver, mgrid, level, kernel_hooks, hook_count);
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_mgrid_container_delete(void** handle) -> int {
  if (handle == nullptr || *handle == nullptr) {
    return -1;
  }
  delete static_cast<MgridContainer*>(*handle);
  *handle = nullptr;
  return 0;
}

extern "C" auto
carbonpy_mgrid_container_bind_field(void* container_handle,
                                    void* field_handle,
                                    int device) -> int {
  if (container_handle == nullptr || field_handle == nullptr) {
    return -1;
  }
  try {
    auto* container = static_cast<MgridContainer*>(container_handle);
    auto* box = static_cast<MgridFieldBox*>(field_handle);
    if (device < 0 || static_cast<size_t>(device) >= container->parts.size()) {
      return -1;
    }
    if (fill_mgrid_partition(box, container->level, device,
                             &container->parts[static_cast<size_t>(device)]) != 0) {
      return -1;
    }
    return fill_mgrid_span(container->mgrid, container->level, device, carbon::DataView::Standard,
                           &container->spans[static_cast<size_t>(device)]);
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_mgrid_container_run(void* container_handle,
                             int stream_idx,
                             int data_view) -> int {
  if (container_handle == nullptr) {
    return -1;
  }
  try {
    auto* container = static_cast<MgridContainer*>(container_handle);
    const auto view = int_to_view(data_view);
    const size_t slot = stream_idx < 0 ? 0U : static_cast<size_t>(stream_idx);
    const size_t n = container->driver->gpu_count();
    for (size_t g = 0; g < n; ++g) {
      if (fill_mgrid_span(container->mgrid, container->level, static_cast<int>(g), view,
                          &container->spans[g]) != 0) {
        return -1;
      }
    }
    for (size_t g = 0; g < n; ++g) {
      const size_t idx = container->hook_index(g, view);
      if (idx >= container->hooks.size() || container->hooks[idx] == nullptr) {
        continue;
      }
      const auto& span = container->spans[g];
      const auto& part = container->parts[g];
      CarbonLaunchBounds bounds{};
      make_bounds_block(span, &bounds);
      void* params[3];
      params[0] = &bounds;
      params[1] = const_cast<CarbonBlockPartition*>(&part);
      params[2] = const_cast<CarbonBlockSpan*>(&span);
      const int rc = container->driver->launch_on(slot, static_cast<int>(g),
                                                  static_cast<CUfunction>(container->hooks[idx]),
                                                  bounds, 256, 0, params);
      if (rc != 0) {
        return rc;
      }
    }
    container->driver->sync_stream_slots(slot + 1U);
    return 0;
  } catch (...) {
    return -1;
  }
}

namespace {

auto
append_dense_token(DenseContainer* container,
                   FieldBox* box,
                   carbon::AccessMode mode,
                   carbon::AccessPattern pattern) -> int {
  if (container == nullptr || box == nullptr || box->field == nullptr) {
    return -1;
  }
  auto push = [&](auto* field) -> int {
    carbon::AccessToken token{field->gpu_buffers().get_uid(), mode, pattern, field->name(), {}};
    if (pattern == carbon::AccessPattern::Stencil &&
        (mode == carbon::AccessMode::Read || mode == carbon::AccessMode::ReadWrite)) {
      using FieldT = std::decay_t<decltype(*field)>;
      token.halo_factory = std::make_shared<carbon::detail::TypedHaloFactory<FieldT>>(*field);
    }
    container->tokens.push_back(std::move(token));
    return 0;
  };
  return dispatch_dtype(box->dtype, [&]<typename T>() -> int {
    switch (box->components) {
#define CARBON_CASE(C) \
  case C:              \
    return push(static_cast<carbon::Dense::Field<T, C>*>(box->field));
      CARBON_DENSE_FOR_EACH_COMPONENT_COUNT(CARBON_CASE)
#undef CARBON_CASE
      default:
        return -1;
    }
  });
}

auto
append_block_token(BlockContainer* container,
                   FieldBox* box,
                   carbon::AccessMode mode,
                   carbon::AccessPattern pattern) -> int {
  if (container == nullptr || box == nullptr || box->field == nullptr) {
    return -1;
  }
  auto push = [&](auto* field) -> int {
    carbon::AccessToken token{field->gpu_buffers().get_uid(), mode, pattern, field->name(), {}};
    if (pattern == carbon::AccessPattern::Stencil &&
        (mode == carbon::AccessMode::Read || mode == carbon::AccessMode::ReadWrite)) {
      using FieldT = std::decay_t<decltype(*field)>;
      token.halo_factory = std::make_shared<carbon::detail::TypedHaloFactory<FieldT>>(*field);
    }
    container->tokens.push_back(std::move(token));
    return 0;
  };
  return dispatch_dtype(box->dtype, [&]<typename T>() -> int {
    switch (box->components) {
      case 1:
        return push(static_cast<BlockGrid::Field<T, 1>*>(box->field));
      case 2:
        return push(static_cast<BlockGrid::Field<T, 2>*>(box->field));
      case 3:
        return push(static_cast<BlockGrid::Field<T, 3>*>(box->field));
      case 4:
        return push(static_cast<BlockGrid::Field<T, 4>*>(box->field));
      default:
        return -1;
    }
  });
}

auto
append_mgrid_token(MgridContainer* container,
                   MgridFieldBox* box,
                   carbon::AccessMode mode,
                   carbon::AccessPattern pattern) -> int {
  if (container == nullptr || box == nullptr || box->field == nullptr) {
    return -1;
  }
  auto push = [&](auto* field) -> int {
    const auto view = field->level_view(container->level);
    container->tokens.push_back(carbon::AccessToken{view.gpu_buffers().get_uid(),
                                                    carbon::ResourceKind::Field,
                                                    mode,
                                                    pattern,
                                                    view.name(),
                                                    {},
                                                    {}});
    return 0;
  };
  return dispatch_dtype(box->dtype, [&]<typename T>() -> int {
    switch (box->components) {
      case 1:
        return push(static_cast<MGridGrid::Field<T, 1>*>(box->field));
      case 2:
        return push(static_cast<MGridGrid::Field<T, 2>*>(box->field));
      case 3:
        return push(static_cast<MGridGrid::Field<T, 3>*>(box->field));
      case 4:
        return push(static_cast<MGridGrid::Field<T, 4>*>(box->field));
      default:
        return -1;
    }
  });
}

auto
launch_block_gpu(BlockContainer* container,
                 size_t gpu,
                 carbon::DataView view,
                 cudaStream_t stream) -> int {
  if (container == nullptr || container->driver == nullptr) {
    return -1;
  }
  if (fill_block_span(container->block, static_cast<int>(gpu), view, &container->spans[gpu]) != 0) {
    return -1;
  }
  const size_t idx = container->hook_index(gpu, view);
  if (idx >= container->hooks.size() || container->hooks[idx] == nullptr) {
    return 0;
  }
  const auto& span = container->spans[gpu];
  const auto& part = container->parts[gpu];
  CarbonLaunchBounds bounds{};
  make_bounds_block(span, &bounds);
  void* params[3];
  params[0] = &bounds;
  params[1] = const_cast<CarbonBlockPartition*>(&part);
  params[2] = const_cast<CarbonBlockSpan*>(&span);
  return container->driver->launch_on_native_stream(
      static_cast<int>(gpu), static_cast<CUfunction>(container->hooks[idx]), bounds, 256, 0,
      params, stream);
}

auto
launch_mgrid_gpu(MgridContainer* container,
                 size_t gpu,
                 carbon::DataView view,
                 cudaStream_t stream) -> int {
  if (container == nullptr || container->driver == nullptr) {
    return -1;
  }
  if (fill_mgrid_span(container->mgrid, container->level, static_cast<int>(gpu), view,
                      &container->spans[gpu]) != 0) {
    return -1;
  }
  const size_t idx = container->hook_index(gpu, view);
  if (idx >= container->hooks.size() || container->hooks[idx] == nullptr) {
    return 0;
  }
  const auto& span = container->spans[gpu];
  const auto& part = container->parts[gpu];
  CarbonLaunchBounds bounds{};
  make_bounds_block(span, &bounds);
  void* params[3];
  params[0] = &bounds;
  params[1] = const_cast<CarbonBlockPartition*>(&part);
  params[2] = const_cast<CarbonBlockSpan*>(&span);
  return container->driver->launch_on_native_stream(
      static_cast<int>(gpu), static_cast<CUfunction>(container->hooks[idx]), bounds, 256, 0,
      params, stream);
}

auto
run_container_entry(int kind,
                    void* handle,
                    size_t stream_slot,
                    int data_view,
                    bool sync_at_end) -> int {
  const auto view = int_to_view(data_view);
  switch (kind) {
    case 0:
      return run_dense_container(static_cast<DenseContainer*>(handle), stream_slot, view,
                                 sync_at_end);
    case 1: {
      auto* container = static_cast<BlockContainer*>(handle);
      const size_t n = container->driver->gpu_count();
      for (size_t g = 0; g < n; ++g) {
        const int rc = launch_block_gpu(container, g, view, container->driver->stream_at(stream_slot, g));
        if (rc != 0) {
          return rc;
        }
      }
      if (sync_at_end) {
        container->driver->sync_stream_slots(stream_slot + 1U);
      }
      return 0;
    }
    case 2: {
      auto* container = static_cast<MgridContainer*>(handle);
      const size_t n = container->driver->gpu_count();
      for (size_t g = 0; g < n; ++g) {
        const int rc =
            launch_mgrid_gpu(container, g, view, container->driver->stream_at(stream_slot, g));
        if (rc != 0) {
          return rc;
        }
      }
      if (sync_at_end) {
        container->driver->sync_stream_slots(stream_slot + 1U);
      }
      return 0;
    }
    default:
      return -1;
  }
}

auto
container_tokens(int kind,
                 void* handle) -> const std::vector<carbon::AccessToken>* {
  switch (kind) {
    case 0:
      return &static_cast<DenseContainer*>(handle)->tokens;
    case 1:
      return &static_cast<BlockContainer*>(handle)->tokens;
    case 2:
      return &static_cast<MgridContainer*>(handle)->tokens;
    default:
      return nullptr;
  }
}

auto
container_driver(int kind,
                 void* handle) -> CudaDriver* {
  switch (kind) {
    case 0:
      return static_cast<DenseContainer*>(handle)->driver;
    case 1:
      return static_cast<BlockContainer*>(handle)->driver;
    case 2:
      return static_cast<MgridContainer*>(handle)->driver;
    default:
      return nullptr;
  }
}

}  // namespace

extern "C" auto
carbonpy_dense_container_add_token(void* container_handle,
                                   void* field_handle,
                                   int access_mode,
                                   int access_pattern) -> int {
  if (container_handle == nullptr || field_handle == nullptr) {
    return -1;
  }
  try {
    const auto mode = static_cast<carbon::AccessMode>(access_mode);
    const auto pattern = static_cast<carbon::AccessPattern>(access_pattern);
    return append_dense_token(static_cast<DenseContainer*>(container_handle),
                              static_cast<FieldBox*>(field_handle), mode, pattern);
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_block_container_add_token(void* container_handle,
                                   void* field_handle,
                                   int access_mode,
                                   int access_pattern) -> int {
  if (container_handle == nullptr || field_handle == nullptr) {
    return -1;
  }
  try {
    const auto mode = static_cast<carbon::AccessMode>(access_mode);
    const auto pattern = static_cast<carbon::AccessPattern>(access_pattern);
    return append_block_token(static_cast<BlockContainer*>(container_handle),
                              static_cast<FieldBox*>(field_handle), mode, pattern);
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_mgrid_container_add_token(void* container_handle,
                                   void* field_handle,
                                   int access_mode,
                                   int access_pattern) -> int {
  if (container_handle == nullptr || field_handle == nullptr) {
    return -1;
  }
  try {
    const auto mode = static_cast<carbon::AccessMode>(access_mode);
    const auto pattern = static_cast<carbon::AccessPattern>(access_pattern);
    return append_mgrid_token(static_cast<MgridContainer*>(container_handle),
                              static_cast<MgridFieldBox*>(field_handle), mode, pattern);
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_warp_container_run_scheduled(void* container_handle,
                                      int kind,
                                      int stream_idx,
                                      int data_view,
                                      int sync_at_end) -> int {
  if (container_handle == nullptr) {
    return -1;
  }
  try {
    const size_t slot = stream_idx < 0 ? 0U : static_cast<size_t>(stream_idx);
    return run_container_entry(kind, container_handle, slot, data_view, sync_at_end != 0);
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_warp_container_token_count(void* container_handle,
                                    int kind) -> int {
  if (container_handle == nullptr) {
    return -1;
  }
  try {
    const auto* tokens = container_tokens(kind, container_handle);
    if (tokens == nullptr) {
      return -1;
    }
    return static_cast<int>(tokens->size());
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_warp_container_token_at(void* container_handle,
                                 int kind,
                                 int index,
                                 uint64_t* out_uid,
                                 int* out_mode,
                                 int* out_pattern) -> int {
  if (container_handle == nullptr || out_uid == nullptr || out_mode == nullptr ||
      out_pattern == nullptr) {
    return -1;
  }
  try {
    const auto* tokens = container_tokens(kind, container_handle);
    if (tokens == nullptr || index < 0 || static_cast<size_t>(index) >= tokens->size()) {
      return -1;
    }
    const auto& token = (*tokens)[static_cast<size_t>(index)];
    *out_uid = token.field_uid;
    *out_mode = static_cast<int>(token.mode);
    *out_pattern = static_cast<int>(token.pattern);
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_warp_driver_ensure_streams(void* driver_handle,
                                    int stream_count) -> int {
  if (driver_handle == nullptr || stream_count < 1) {
    return -1;
  }
  try {
    static_cast<CudaDriver*>(driver_handle)->ensure_stream_slots(static_cast<size_t>(stream_count));
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_warp_container_driver(void* container_handle,
                               int kind,
                               void** out_driver) -> int {
  if (container_handle == nullptr || out_driver == nullptr) {
    return -1;
  }
  try {
    auto* driver = container_driver(kind, container_handle);
    if (driver == nullptr) {
      return -1;
    }
    *out_driver = driver;
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_warp_driver_stream_at(void* driver_handle,
                               int stream_idx,
                               int gpu_id,
                               void** out_stream) -> int {
  if (driver_handle == nullptr || out_stream == nullptr || gpu_id < 0) {
    return -1;
  }
  try {
    auto* driver = static_cast<CudaDriver*>(driver_handle);
    const size_t slot = stream_idx < 0 ? 0U : static_cast<size_t>(stream_idx);
    *out_stream = driver->stream_at(slot, static_cast<size_t>(gpu_id));
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_warp_driver_set_device(void* driver_handle,
                                int gpu_id) -> int {
  if (driver_handle == nullptr || gpu_id < 0) {
    return -1;
  }
  try {
    return static_cast<CudaDriver*>(driver_handle)->set_device(static_cast<size_t>(gpu_id));
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_warp_driver_sync_streams(void* driver_handle,
                                  int stream_count) -> int {
  if (driver_handle == nullptr || stream_count < 1) {
    return -1;
  }
  try {
    static_cast<CudaDriver*>(driver_handle)->sync_stream_slots(static_cast<size_t>(stream_count));
    return 0;
  } catch (...) {
    return -1;
  }
}

namespace carbonpy::detail {

auto
launch_dense_container_on_carbon_stream(void* dense_container_handle,
                                        size_t gpu,
                                        void* native_stream,
                                        carbon::DataView view) -> int {
  if (dense_container_handle == nullptr || native_stream == nullptr) {
    return -1;
  }
  try {
    return launch_dense_gpu(static_cast<DenseContainer*>(dense_container_handle), gpu, view,
                            static_cast<cudaStream_t>(native_stream));
  } catch (...) {
    return -1;
  }
}

auto
launch_block_container_on_carbon_stream(void* block_container_handle,
                                        size_t gpu,
                                        void* native_stream,
                                        carbon::DataView view) -> int {
  if (block_container_handle == nullptr || native_stream == nullptr) {
    return -1;
  }
  try {
    return launch_block_gpu(static_cast<BlockContainer*>(block_container_handle), gpu, view,
                            static_cast<cudaStream_t>(native_stream));
  } catch (...) {
    return -1;
  }
}

auto
launch_mgrid_container_on_carbon_stream(void* mgrid_container_handle,
                                        size_t gpu,
                                        void* native_stream,
                                        carbon::DataView view) -> int {
  if (mgrid_container_handle == nullptr || native_stream == nullptr) {
    return -1;
  }
  try {
    return launch_mgrid_gpu(static_cast<MgridContainer*>(mgrid_container_handle), gpu, view,
                            static_cast<cudaStream_t>(native_stream));
  } catch (...) {
    return -1;
  }
}

auto
dense_container_tokens(void* dense_container_handle) -> const std::vector<carbon::AccessToken>& {
  return static_cast<DenseContainer*>(dense_container_handle)->tokens;
}

auto
block_container_tokens(void* block_container_handle) -> const std::vector<carbon::AccessToken>& {
  return static_cast<BlockContainer*>(block_container_handle)->tokens;
}

auto
mgrid_container_tokens(void* mgrid_container_handle) -> const std::vector<carbon::AccessToken>& {
  return static_cast<MgridContainer*>(mgrid_container_handle)->tokens;
}

}  // namespace carbonpy::detail
