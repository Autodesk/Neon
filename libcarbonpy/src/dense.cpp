#include <carbonpy/carbonpy.h>

#include <carbonpy/detail/field_box.h>

#include <carbon/boundary_mode.h>
#include <carbon/data_view.h>
#include <carbon/dense.h>
#include <carbon/halo.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <cstdio>
#include <exception>
#include <memory>
#include <vector>

namespace {

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
make_dense_stencil(int halo_radius) -> carbon::Stencil {
  if (halo_radius <= 0) {
    return carbon::Stencil{};
  }
  const std::vector<carbon::int32_3d> points{
      {-1, 0, 0},
      {1, 0, 0},
      {0, -1, 0},
      {0, 1, 0},
      {0, 0, -1},
      {0, 0, 1},
  };
  return carbon::Stencil{points};
}

auto
fill_dense_span(carbon::Dense* dense,
                int device,
                int data_view,
                CarbonDenseSpan* out_span) -> int {
  const auto& spans = dense->spans();
  if (device < 0 || static_cast<size_t>(device) >= spans.size()) {
    return -1;
  }
  const auto& span = spans[static_cast<size_t>(device)];
  const auto ext = span.iter_extent(int_to_view(data_view));
  out_span->iter_x = ext.x;
  out_span->iter_y = ext.y;
  out_span->iter_z = ext.z;
  const auto logic = span.logic_extent();
  out_span->logic_x = logic.x;
  out_span->logic_y = logic.y;
  out_span->logic_z = logic.z;
  out_span->pitch_xy = span.xy_slab_elements();
  out_span->z_offset = span.z_offset();
  out_span->halo_x = 0;
  out_span->halo_y = 0;
  out_span->halo_z = static_cast<int32_t>(span.halo_z());
  const auto global = span.global_extent();
  out_span->global_x = global.x;
  out_span->global_y = global.y;
  out_span->global_z = global.z;
  out_span->boundary_mode =
      span.boundary_mode() == carbon::BoundaryMode::PeriodicZ ? 1 : 0;
  out_span->data_view = static_cast<int32_t>(int_to_view(data_view));
  return 0;
}

}  // namespace

extern "C" auto
carbonpy_dense_new(void** handle,
                   void* system_handle,
                   int64_t extent_x,
                   int64_t extent_y,
                   int64_t extent_z,
                   int halo_radius) -> int {
  if (handle == nullptr || system_handle == nullptr) {
    return -1;
  }
  try {
    auto* sys = static_cast<carbon::System*>(system_handle);
    auto grid = std::make_unique<carbon::Dense>(
        *sys, carbon::int64_3d{extent_x, extent_y, extent_z}, make_dense_stencil(halo_radius),
        carbon::BoundaryMode::Open);
    *handle = grid.release();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "carbonpy_dense_new: %s\n", e.what());
    return -1;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_dense_delete(void** handle) -> int {
  if (handle == nullptr || *handle == nullptr) {
    return -1;
  }
  delete static_cast<carbon::Dense*>(*handle);
  *handle = nullptr;
  return 0;
}

namespace {

using carbonpy::detail::dispatch_dtype;
using carbonpy::detail::FieldBox;

// Component counts instantiated for Dense fields. 1-4 cover the Cartesian
// scalar/vector fields; 7/9/15/19/27 cover the LBM lattice sizes (D3Q7, D2Q9,
// D3Q15, D3Q19, D3Q27) so XLB distribution / missing-mask fields allocate.
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

template <typename T>
auto
dense_field_new(carbon::Dense* dense,
                int components,
                void** out_field) -> int {
  switch (components) {
#define CARBON_CASE(C)                                        \
  case C:                                                     \
    *out_field = new carbon::Dense::Field<T, C>(              \
        dense->newField<T, C>(carbon::MemorySide::Device, "f")); \
    return 0;
    CARBON_DENSE_FOR_EACH_COMPONENT_COUNT(CARBON_CASE)
#undef CARBON_CASE
    default:
      return -1;
  }
}

struct HaloBox {
  int dtype{};
  int components{};
  void* halo{};
};

template <typename T>
auto
dense_halo_new(void* field,
               int components,
               void** out_halo) -> int {
  switch (components) {
#define CARBON_CASE(C)                                                          \
  case C:                                                                       \
    *out_halo = new carbon::Halo<T, C>(*static_cast<carbon::Dense::Field<T, C>*>(field)); \
    return 0;
    CARBON_DENSE_FOR_EACH_COMPONENT_COUNT(CARBON_CASE)
#undef CARBON_CASE
    default:
      return -1;
  }
}

template <typename T>
auto
dense_halo_delete(void* halo,
                    int components) -> int {
  switch (components) {
#define CARBON_CASE(C)                              \
  case C:                                           \
    delete static_cast<carbon::Halo<T, C>*>(halo);  \
    return 0;
    CARBON_DENSE_FOR_EACH_COMPONENT_COUNT(CARBON_CASE)
#undef CARBON_CASE
    default:
      return -1;
  }
}

template <typename T>
auto
dense_halo_run(void* halo,
               int components,
               carbon::System* system) -> int {
  switch (components) {
#define CARBON_CASE(C)                                     \
  case C: {                                                \
    auto* h = static_cast<carbon::Halo<T, C>*>(halo);      \
    carbon::Stream stream = system->stream();              \
    (*h)(stream);                                          \
    stream.wait();                                         \
    return 0;                                              \
  }
    CARBON_DENSE_FOR_EACH_COMPONENT_COUNT(CARBON_CASE)
#undef CARBON_CASE
    default:
      return -1;
  }
}

template <typename T>
auto
dense_field_delete(void* field,
                   int components) -> int {
  switch (components) {
#define CARBON_CASE(C)                                        \
  case C:                                                     \
    delete static_cast<carbon::Dense::Field<T, C>*>(field);   \
    return 0;
    CARBON_DENSE_FOR_EACH_COMPONENT_COUNT(CARBON_CASE)
#undef CARBON_CASE
    default:
      return -1;
  }
}

}  // namespace

extern "C" auto
carbonpy_dense_field_new(void** handle,
                         void* dense_handle,
                         int components,
                         int dtype) -> int {
  if (handle == nullptr || dense_handle == nullptr || components <= 0) {
    return -1;
  }
  try {
    auto* dense = static_cast<carbon::Dense*>(dense_handle);
    auto* box = new FieldBox{};
    box->dtype = dtype;
    box->components = components;
    box->dense = dense;
    const int rc = dispatch_dtype(dtype, [&]<typename T>() -> int {
      return dense_field_new<T>(dense, components, &box->field);
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
carbonpy_dense_field_delete(void** handle) -> int {
  if (handle == nullptr || *handle == nullptr) {
    return -1;
  }
  auto* box = static_cast<FieldBox*>(*handle);
  if (box->field != nullptr) {
    dispatch_dtype(box->dtype, [&]<typename T>() -> int {
      return dense_field_delete<T>(box->field, box->components);
    });
  }
  delete box;
  *handle = nullptr;
  return 0;
}

extern "C" auto
carbonpy_dense_get_span(void* dense_handle,
                        int device,
                        int data_view,
                        CarbonDenseSpan* out_span) -> int {
  if (dense_handle == nullptr || out_span == nullptr) {
    return -1;
  }
  try {
    auto* dense = static_cast<carbon::Dense*>(dense_handle);
    return fill_dense_span(dense, device, data_view, out_span);
  } catch (...) {
    return -1;
  }
}

namespace {

auto
dense_set_partition(auto* field,
                    int device,
                    CarbonDensePartition* out_part) -> int {
  const auto& parts = field->partitions();
  if (device < 0 || static_cast<size_t>(device) >= parts.size()) {
    return -1;
  }
  const auto& part = parts[static_cast<size_t>(device)];
  out_part->data_ptr = reinterpret_cast<uint64_t>(part.ptr());
  return 0;
}

template <typename T>
auto
dense_get_partition(void* field,
                    int components,
                    int device,
                    CarbonDensePartition* out_part) -> int {
  switch (components) {
#define CARBON_CASE(C)                                                              \
  case C:                                                                           \
    return dense_set_partition(static_cast<carbon::Dense::Field<T, C>*>(field), device, out_part);
    CARBON_DENSE_FOR_EACH_COMPONENT_COUNT(CARBON_CASE)
#undef CARBON_CASE
    default:
      return -1;
  }
}

}  // namespace

extern "C" auto
carbonpy_dense_field_get_partition(void* field_handle,
                                   int device,
                                   int data_view,
                                   CarbonDensePartition* out_part) -> int {
  if (field_handle == nullptr || out_part == nullptr) {
    return -1;
  }
  try {
    auto* box = static_cast<FieldBox*>(field_handle);
    if (box->field == nullptr) {
      return -1;
    }
    const int rc = dispatch_dtype(box->dtype, [&]<typename T>() -> int {
      return dense_get_partition<T>(box->field, box->components, device, out_part);
    });
    if (rc != 0) {
      return rc;
    }
    if (box->dense != nullptr) {
      return fill_dense_span(static_cast<carbon::Dense*>(box->dense), device, data_view,
                             &out_part->span);
    }
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_dense_halo_new(void** handle,
                        void* field_handle) -> int {
  if (handle == nullptr || field_handle == nullptr) {
    return -1;
  }
  try {
    auto* field_box = static_cast<FieldBox*>(field_handle);
    if (field_box->field == nullptr) {
      return -1;
    }
    auto* halo_box = new HaloBox{};
    halo_box->dtype = field_box->dtype;
    halo_box->components = field_box->components;
    const int rc = dispatch_dtype(field_box->dtype, [&]<typename T>() -> int {
      return dense_halo_new<T>(field_box->field, field_box->components, &halo_box->halo);
    });
    if (rc != 0) {
      delete halo_box;
      return -1;
    }
    *handle = halo_box;
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_dense_halo_run(void* halo_handle,
                        void* system_handle) -> int {
  if (halo_handle == nullptr || system_handle == nullptr) {
    return -1;
  }
  try {
    auto* halo_box = static_cast<HaloBox*>(halo_handle);
    if (halo_box->halo == nullptr) {
      return -1;
    }
    auto* system = static_cast<carbon::System*>(system_handle);
    return dispatch_dtype(halo_box->dtype, [&]<typename T>() -> int {
      return dense_halo_run<T>(halo_box->halo, halo_box->components, system);
    });
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_dense_halo_delete(void** handle) -> int {
  if (handle == nullptr || *handle == nullptr) {
    return -1;
  }
  auto* halo_box = static_cast<HaloBox*>(*handle);
  if (halo_box->halo != nullptr) {
    dispatch_dtype(halo_box->dtype, [&]<typename T>() -> int {
      return dense_halo_delete<T>(halo_box->halo, halo_box->components);
    });
  }
  delete halo_box;
  *handle = nullptr;
  return 0;
}
