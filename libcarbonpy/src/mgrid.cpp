#include <carbonpy/mgrid.h>

#include <carbonpy/detail/field_box.h>

#include <carbon/data_view.h>
#include <carbon/detail/block/static_block.h>
#include <carbon/memory_side.h>
#include <carbon/mgrid.h>
#include <carbon/system.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace {

using SBlock = carbon::DefaultMultiResBlock;
using MGrid = carbon::MultiResBlock<SBlock>;

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
using carbonpy::detail::MgridFieldBox;

template <typename T>
auto
mgrid_field_new(const MGrid* mgrid,
                int components,
                void** out_field) -> int {
  switch (components) {
    case 1:
      *out_field = new MGrid::Field<T, 1>(mgrid->newField<T, 1>(carbon::MemorySide::Device, "f"));
      return 0;
    case 2:
      *out_field = new MGrid::Field<T, 2>(mgrid->newField<T, 2>(carbon::MemorySide::Device, "f"));
      return 0;
    case 3:
      *out_field = new MGrid::Field<T, 3>(mgrid->newField<T, 3>(carbon::MemorySide::Device, "f"));
      return 0;
    case 4:
      *out_field = new MGrid::Field<T, 4>(mgrid->newField<T, 4>(carbon::MemorySide::Device, "f"));
      return 0;
    default:
      return -1;
  }
}

template <typename T>
auto
mgrid_field_delete(void* field,
                   int components) -> int {
  switch (components) {
    case 1:
      delete static_cast<MGrid::Field<T, 1>*>(field);
      return 0;
    case 2:
      delete static_cast<MGrid::Field<T, 2>*>(field);
      return 0;
    case 3:
      delete static_cast<MGrid::Field<T, 3>*>(field);
      return 0;
    case 4:
      delete static_cast<MGrid::Field<T, 4>*>(field);
      return 0;
    default:
      return -1;
  }
}

auto
blit_block_span(const typename MGrid::BlockGrid& grid,
                int device,
                carbon::DataView view,
                CarbonBlockSpan* out) -> int {
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
  out->mem_x = SBlock::MemX;
  out->mem_y = SBlock::MemY;
  out->mem_z = SBlock::MemZ;
  return 0;
}

auto
fill_span(const MGrid& mgrid,
          int level,
          int device,
          carbon::DataView view,
          CarbonBlockSpan* out) -> int {
  if (level < 0 || level >= mgrid.depth()) {
    return -1;
  }
  return blit_block_span(mgrid.grid(level), device, view, out);
}

template <typename Field>
auto
fill_cross_partition(Field* field,
                     const MGrid* mgrid,
                     int level,
                     int device,
                     int components,
                     CarbonMgridPartition* out) -> int {
  if (field == nullptr || mgrid == nullptr || out == nullptr) {
    return -1;
  }
  if (level < 0 || level >= field->depth()) {
    return -1;
  }
  const auto level_uid = [](const typename Field::LevelField& f, size_t g) -> uint64_t {
    const auto& parts = f.partitions();
    return g < parts.size() ? reinterpret_cast<uint64_t>(parts[g].ptr()) : 0U;
  };

  const size_t gpu_id = static_cast<size_t>(device);
  out->data_ptr = level_uid(field->level_field(level), gpu_id);
  out->ref_factor = mgrid->ref_factor();
  out->components = components;
  if (blit_block_span(mgrid->grid(level), device, carbon::DataView::Standard, &out->own_span) !=
      0) {
    return -1;
  }
  out->parent_data_ptr = 0U;
  out->parent_block_id_ptr = 0U;
  out->child_data_ptr = 0U;
  out->child_block_id_ptr = 0U;
  out->parent_span = CarbonBlockSpan{};
  out->child_span = CarbonBlockSpan{};

  if (level + 1 < mgrid->depth()) {
    out->parent_data_ptr = level_uid(field->level_field(level + 1), gpu_id);
    out->parent_block_id_ptr =
        reinterpret_cast<uint64_t>(mgrid->parent_block_id_dev(level, gpu_id));
    if (blit_block_span(mgrid->grid(level + 1), device, carbon::DataView::Standard,
                        &out->parent_span) != 0) {
      return -1;
    }
  }

  if (level >= 1) {
    out->child_data_ptr = level_uid(field->level_field(level - 1), gpu_id);
    out->child_block_id_ptr = reinterpret_cast<uint64_t>(mgrid->child_block_id_dev(level, gpu_id));
    if (blit_block_span(mgrid->grid(level - 1), device, carbon::DataView::Standard,
                        &out->child_span) != 0) {
      return -1;
    }
  }
  return 0;
}

}  // namespace

extern "C" auto
carbonpy_mgrid_new(void** handle,
                   void* system_handle,
                   int64_t finest_x,
                   int64_t finest_y,
                   int64_t finest_z,
                   int depth,
                   int ref_factor) -> int {
  if (handle == nullptr || system_handle == nullptr || depth <= 0) {
    return -1;
  }
  try {
    auto* sys = static_cast<carbon::System*>(system_handle);
    const auto is_active = [](int /*level*/, const carbon::int64_3d& /*v*/) -> bool {
      return true;
    };
    auto mgrid = std::make_unique<MGrid>(*sys, carbon::int64_3d{finest_x, finest_y, finest_z},
                                         depth, is_active, ref_factor, "carbonpy_mgrid");
    *handle = mgrid.release();
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_mgrid_new_sparse(void** handle,
                          void* system_handle,
                          int64_t finest_x,
                          int64_t finest_y,
                          int64_t finest_z,
                          int depth,
                          int ref_factor,
                          const int32_t* masks_flat,
                          const int64_t* mask_shapes,
                          const int64_t* mask_origins) -> int {
  if (handle == nullptr || system_handle == nullptr || depth <= 0 || mask_shapes == nullptr ||
      mask_origins == nullptr) {
    return -1;
  }
  try {
    auto* sys = static_cast<carbon::System*>(system_handle);

    // Copy each level's mask + shape + origin into a shared payload the predicate owns, so the
    // caller's buffers can be freed as soon as this call returns.
    struct LevelMask {
      carbon::int64_3d shape{};
      carbon::int64_3d origin{};
      std::vector<int32_t> data{};
    };
    auto levels = std::make_shared<std::vector<LevelMask>>();
    levels->reserve(static_cast<size_t>(depth));
    size_t cursor = 0;
    for (int l = 0; l < depth; ++l) {
      LevelMask lm;
      lm.shape = {mask_shapes[3 * l + 0], mask_shapes[3 * l + 1], mask_shapes[3 * l + 2]};
      lm.origin = {mask_origins[3 * l + 0], mask_origins[3 * l + 1], mask_origins[3 * l + 2]};
      const size_t count = static_cast<size_t>(std::max<int64_t>(lm.shape.x, 0)) *
                           static_cast<size_t>(std::max<int64_t>(lm.shape.y, 0)) *
                           static_cast<size_t>(std::max<int64_t>(lm.shape.z, 0));
      if (count > 0) {
        if (masks_flat == nullptr) {
          return -1;
        }
        lm.data.assign(masks_flat + cursor, masks_flat + cursor + count);
        cursor += count;
      }
      levels->push_back(std::move(lm));
    }

    const auto is_active = [levels](int level, const carbon::int64_3d& v) -> bool {
      if (level < 0 || level >= static_cast<int>(levels->size())) {
        return false;
      }
      const auto& lm = (*levels)[static_cast<size_t>(level)];
      const carbon::int64_3d p{v.x - lm.origin.x, v.y - lm.origin.y, v.z - lm.origin.z};
      if (p.x < 0 || p.y < 0 || p.z < 0 || p.x >= lm.shape.x || p.y >= lm.shape.y ||
          p.z >= lm.shape.z) {
        return false;
      }
      const size_t idx =
          (static_cast<size_t>(p.x) * static_cast<size_t>(lm.shape.y) + static_cast<size_t>(p.y)) *
              static_cast<size_t>(lm.shape.z) +
          static_cast<size_t>(p.z);
      return lm.data[idx] != 0;
    };

    auto mgrid = std::make_unique<MGrid>(*sys, carbon::int64_3d{finest_x, finest_y, finest_z},
                                         depth, is_active, ref_factor, "carbonpy_mgrid_sparse");
    *handle = mgrid.release();
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_mgrid_delete(void** handle) -> int {
  if (handle == nullptr || *handle == nullptr) {
    return -1;
  }
  delete static_cast<MGrid*>(*handle);
  *handle = nullptr;
  return 0;
}

extern "C" auto
carbonpy_mgrid_depth(void* mgrid_handle) -> int {
  if (mgrid_handle == nullptr) {
    return -1;
  }
  try {
    return static_cast<MGrid*>(mgrid_handle)->depth();
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_mgrid_field_new(void** handle,
                         void* mgrid_handle,
                         int components,
                         int dtype) -> int {
  if (handle == nullptr || mgrid_handle == nullptr || components <= 0) {
    return -1;
  }
  try {
    auto* mgrid = static_cast<MGrid*>(mgrid_handle);
    auto* box = new MgridFieldBox{};
    box->dtype = dtype;
    box->components = components;
    box->mgrid = mgrid_handle;
    const int rc = dispatch_dtype(dtype, [&]<typename T>() -> int {
      return mgrid_field_new<T>(mgrid, components, &box->field);
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
carbonpy_mgrid_field_delete(void** handle) -> int {
  if (handle == nullptr || *handle == nullptr) {
    return -1;
  }
  auto* box = static_cast<MgridFieldBox*>(*handle);
  if (box->field != nullptr) {
    dispatch_dtype(box->dtype, [&]<typename T>() -> int {
      return mgrid_field_delete<T>(box->field, box->components);
    });
  }
  delete box;
  *handle = nullptr;
  return 0;
}

extern "C" auto
carbonpy_mgrid_get_span(void* mgrid_handle,
                        int level,
                        int device,
                        int data_view,
                        CarbonBlockSpan* out_span) -> int {
  if (mgrid_handle == nullptr || out_span == nullptr) {
    return -1;
  }
  try {
    auto* mgrid = static_cast<MGrid*>(mgrid_handle);
    return fill_span(*mgrid, level, device, int_to_view(data_view), out_span);
  } catch (...) {
    return -1;
  }
}

namespace {

auto
mgrid_set_partition(auto* field,
                    int level,
                    int device,
                    int components,
                    CarbonBlockPartition* out_part) -> int {
  if (level < 0 || level >= field->depth()) {
    return -1;
  }
  const auto part = field->make_partition(level, static_cast<size_t>(device));
  out_part->data_ptr = reinterpret_cast<uint64_t>(part.ptr());
  out_part->components = components;
  return 0;
}

template <typename T>
auto
mgrid_get_partition(void* field,
                    int components,
                    int level,
                    int device,
                    CarbonBlockPartition* out_part) -> int {
  switch (components) {
    case 1:
      return mgrid_set_partition(static_cast<MGrid::Field<T, 1>*>(field), level, device, components,
                                 out_part);
    case 2:
      return mgrid_set_partition(static_cast<MGrid::Field<T, 2>*>(field), level, device, components,
                                 out_part);
    case 3:
      return mgrid_set_partition(static_cast<MGrid::Field<T, 3>*>(field), level, device, components,
                                 out_part);
    case 4:
      return mgrid_set_partition(static_cast<MGrid::Field<T, 4>*>(field), level, device, components,
                                 out_part);
    default:
      return -1;
  }
}

template <typename T>
auto
mgrid_get_cross_partition(void* field,
                          MGrid* mgrid,
                          int components,
                          int level,
                          int device,
                          CarbonMgridPartition* out_part) -> int {
  switch (components) {
    case 1:
      return fill_cross_partition(static_cast<MGrid::Field<T, 1>*>(field), mgrid, level, device,
                                  components, out_part);
    case 2:
      return fill_cross_partition(static_cast<MGrid::Field<T, 2>*>(field), mgrid, level, device,
                                  components, out_part);
    case 3:
      return fill_cross_partition(static_cast<MGrid::Field<T, 3>*>(field), mgrid, level, device,
                                  components, out_part);
    case 4:
      return fill_cross_partition(static_cast<MGrid::Field<T, 4>*>(field), mgrid, level, device,
                                  components, out_part);
    default:
      return -1;
  }
}

}  // namespace

extern "C" auto
carbonpy_mgrid_field_get_partition(void* field_handle,
                                   int level,
                                   int device,
                                   int /*data_view*/,
                                   CarbonBlockPartition* out_part) -> int {
  if (field_handle == nullptr || out_part == nullptr) {
    return -1;
  }
  try {
    auto* box = static_cast<MgridFieldBox*>(field_handle);
    if (box->field == nullptr) {
      return -1;
    }
    return dispatch_dtype(box->dtype, [&]<typename T>() -> int {
      return mgrid_get_partition<T>(box->field, box->components, level, device, out_part);
    });
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_mgrid_field_get_cross_partition(void* field_handle,
                                         int level,
                                         int device,
                                         int /*data_view*/,
                                         CarbonMgridPartition* out_part) -> int {
  if (field_handle == nullptr || out_part == nullptr) {
    return -1;
  }
  try {
    auto* box = static_cast<MgridFieldBox*>(field_handle);
    if (box->field == nullptr || box->mgrid == nullptr) {
      return -1;
    }
    auto* mgrid = static_cast<MGrid*>(box->mgrid);
    return dispatch_dtype(box->dtype, [&]<typename T>() -> int {
      return mgrid_get_cross_partition<T>(box->field, mgrid, box->components, level, device,
                                          out_part);
    });
  } catch (...) {
    return -1;
  }
}
