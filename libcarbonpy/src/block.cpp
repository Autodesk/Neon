#include <carbonpy/carbonpy.h>

#include <carbonpy/detail/field_box.h>

#include <carbon/block.h>
#include <carbon/data_view.h>
#include <carbon/detail/block/static_block.h>
#include <carbon/system.h>

#include <memory>

namespace {

using SBlock = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SBlock>;

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

using carbonpy::detail::dispatch_dtype;
using carbonpy::detail::FieldBox;

template <typename T>
auto
block_field_new(BlockGrid* block,
                int components,
                void** out_field) -> int {
  switch (components) {
    case 1:
      *out_field = new BlockGrid::Field<T, 1>(
          block->newField<T, 1>(carbon::MemorySide::Device, "f"));
      return 0;
    case 2:
      *out_field = new BlockGrid::Field<T, 2>(
          block->newField<T, 2>(carbon::MemorySide::Device, "f"));
      return 0;
    case 3:
      *out_field = new BlockGrid::Field<T, 3>(
          block->newField<T, 3>(carbon::MemorySide::Device, "f"));
      return 0;
    case 4:
      *out_field = new BlockGrid::Field<T, 4>(
          block->newField<T, 4>(carbon::MemorySide::Device, "f"));
      return 0;
    default:
      return -1;
  }
}

template <typename T>
auto
block_field_delete(void* field,
                   int components) -> int {
  switch (components) {
    case 1:
      delete static_cast<BlockGrid::Field<T, 1>*>(field);
      return 0;
    case 2:
      delete static_cast<BlockGrid::Field<T, 2>*>(field);
      return 0;
    case 3:
      delete static_cast<BlockGrid::Field<T, 3>*>(field);
      return 0;
    case 4:
      delete static_cast<BlockGrid::Field<T, 4>*>(field);
      return 0;
    default:
      return -1;
  }
}

auto
block_set_partition(auto* field,
                    int device,
                    int components,
                    CarbonBlockPartition* out_part) -> int {
  const auto& parts = field->partitions();
  if (device < 0 || static_cast<size_t>(device) >= parts.size()) {
    return -1;
  }
  const auto& part = parts[static_cast<size_t>(device)];
  out_part->data_ptr = reinterpret_cast<uint64_t>(part.ptr());
  out_part->components = components;
  return 0;
}

template <typename T>
auto
block_get_partition(void* field,
                    int components,
                    int device,
                    CarbonBlockPartition* out_part) -> int {
  switch (components) {
    case 1:
      return block_set_partition(static_cast<BlockGrid::Field<T, 1>*>(field), device, components,
                                 out_part);
    case 2:
      return block_set_partition(static_cast<BlockGrid::Field<T, 2>*>(field), device, components,
                                 out_part);
    case 3:
      return block_set_partition(static_cast<BlockGrid::Field<T, 3>*>(field), device, components,
                                 out_part);
    case 4:
      return block_set_partition(static_cast<BlockGrid::Field<T, 4>*>(field), device, components,
                                 out_part);
    default:
      return -1;
  }
}

}  // namespace

extern "C" auto
carbonpy_block_new(void** handle,
                   void* system_handle,
                   int64_t extent_x,
                   int64_t extent_y,
                   int64_t extent_z) -> int {
  if (handle == nullptr || system_handle == nullptr) {
    return -1;
  }
  try {
    auto* sys = static_cast<carbon::System*>(system_handle);
    const auto all_active = [](const carbon::int64_3d& /*gc*/) -> bool {
      return true;
    };
    auto block = std::make_unique<BlockGrid>(
        *sys, carbon::int64_3d{extent_x, extent_y, extent_z}, all_active, "carbonpy_block");
    *handle = block.release();
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_block_delete(void** handle) -> int {
  if (handle == nullptr || *handle == nullptr) {
    return -1;
  }
  delete static_cast<BlockGrid*>(*handle);
  *handle = nullptr;
  return 0;
}

extern "C" auto
carbonpy_block_field_new(void** handle,
                         void* block_handle,
                         int components,
                         int dtype) -> int {
  if (handle == nullptr || block_handle == nullptr || components <= 0) {
    return -1;
  }
  try {
    auto* block = static_cast<BlockGrid*>(block_handle);
    auto* box = new FieldBox{};
    box->dtype = dtype;
    box->components = components;
    const int rc = dispatch_dtype(dtype, [&]<typename T>() -> int {
      return block_field_new<T>(block, components, &box->field);
    });
    if (rc != 0) {
      delete box;
      return -1;
    }
    *handle = box;
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_block_field_delete(void** handle) -> int {
  if (handle == nullptr || *handle == nullptr) {
    return -1;
  }
  auto* box = static_cast<FieldBox*>(*handle);
  if (box->field != nullptr) {
    dispatch_dtype(box->dtype, [&]<typename T>() -> int {
      return block_field_delete<T>(box->field, box->components);
    });
  }
  delete box;
  *handle = nullptr;
  return 0;
}

extern "C" auto
carbonpy_block_get_span(void* block_handle,
                        int device,
                        int data_view,
                        CarbonBlockSpan* out_span) -> int {
  if (block_handle == nullptr || out_span == nullptr) {
    return -1;
  }
  try {
    auto* block = static_cast<BlockGrid*>(block_handle);
    const auto& spans = block->spans();
    if (device < 0 || static_cast<size_t>(device) >= spans.size()) {
      return -1;
    }
    const auto& span = spans[static_cast<size_t>(device)];
    const auto view = int_to_view(data_view);
    out_span->bitmask_ptr = reinterpret_cast<uint64_t>(span.bitmask());
    out_span->connectivity_ptr = reinterpret_cast<uint64_t>(span.connectivity());
    out_span->origins_ptr = reinterpret_cast<uint64_t>(span.origins());
    out_span->block_count = span.block_count();
    out_span->n_internal = span.n_internal();
    out_span->iter_block_count = span.iter_block_count(view);
    out_span->mem_x = carbon::detail::block::DefaultStaticBlock::MemX;
    out_span->mem_y = carbon::detail::block::DefaultStaticBlock::MemY;
    out_span->mem_z = carbon::detail::block::DefaultStaticBlock::MemZ;
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_block_field_get_partition(void* field_handle,
                                   int device,
                                   int /*data_view*/,
                                   CarbonBlockPartition* out_part) -> int {
  if (field_handle == nullptr || out_part == nullptr) {
    return -1;
  }
  try {
    auto* box = static_cast<FieldBox*>(field_handle);
    if (box->field == nullptr) {
      return -1;
    }
    return dispatch_dtype(box->dtype, [&]<typename T>() -> int {
      return block_get_partition<T>(box->field, box->components, device, out_part);
    });
  } catch (...) {
    return -1;
  }
}
