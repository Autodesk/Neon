#pragma once

#include <stdint.h>
#include <string.h>

#include "carbonpy/warp_partition_types.h"

#if defined(__CUDACC__)
#define CARBON_WARP_HD __device__ __host__ __forceinline__
#else
#define CARBON_WARP_HD inline
#endif

namespace wp {

// CarbonDenseSpan is defined in warp_partition_types.h (included above) so the dense
// partitions can embed it; the spanless helpers below address memory from part.span.

CARBON_WARP_HD auto
carbon_dense_haloed_z(const CarbonDenseSpan& span) -> int64_t {
  return span.logic_z + 2 * static_cast<int64_t>(span.halo_z);
}

CARBON_WARP_HD auto
carbon_dense_storage_elements(const CarbonDenseSpan& span) -> int64_t {
  return carbon_dense_haloed_z(span) * span.pitch_xy;
}

CARBON_WARP_HD auto
carbon_dense_linear_offset(const CarbonDenseSpan& span,
                           int64_t ix,
                           int64_t iy,
                           int64_t iz,
                           int component_id) -> int64_t {
  const int64_t bz = static_cast<int64_t>(span.halo_z) + iz;
  return static_cast<int64_t>(component_id) * carbon_dense_storage_elements(span) +
         bz * span.pitch_xy + iy * span.logic_x + ix;
}

CARBON_WARP_HD auto
carbon_dense_partition_read_f32(const CarbonDensePartitionF32& part,
                                const CarbonDenseSpan& span,
                                int64_t ix,
                                int64_t iy,
                                int64_t iz,
                                int component_id) -> float {
  auto* mem = reinterpret_cast<float*>(static_cast<uintptr_t>(part.data_ptr));
  return mem[carbon_dense_linear_offset(span, ix, iy, iz, component_id)];
}

CARBON_WARP_HD void
carbon_dense_partition_write_f32(const CarbonDensePartitionF32& part,
                                 const CarbonDenseSpan& span,
                                 int64_t ix,
                                 int64_t iy,
                                 int64_t iz,
                                 int component_id,
                                 float value) {
  auto* mem = reinterpret_cast<float*>(static_cast<uintptr_t>(part.data_ptr));
  mem[carbon_dense_linear_offset(span, ix, iy, iz, component_id)] = value;
}

CARBON_WARP_HD auto
carbon_dense_partition_read_f64(const CarbonDensePartitionF64& part,
                                const CarbonDenseSpan& span,
                                int64_t ix,
                                int64_t iy,
                                int64_t iz,
                                int component_id) -> double {
  auto* mem = reinterpret_cast<double*>(static_cast<uintptr_t>(part.data_ptr));
  return mem[carbon_dense_linear_offset(span, ix, iy, iz, component_id)];
}

CARBON_WARP_HD void
carbon_dense_partition_write_f64(const CarbonDensePartitionF64& part,
                                 const CarbonDenseSpan& span,
                                 int64_t ix,
                                 int64_t iy,
                                 int64_t iz,
                                 int component_id,
                                 double value) {
  auto* mem = reinterpret_cast<double*>(static_cast<uintptr_t>(part.data_ptr));
  mem[carbon_dense_linear_offset(span, ix, iy, iz, component_id)] = value;
}

CARBON_WARP_HD auto
carbon_dense_partition_read_i32(const CarbonDensePartitionI32& part,
                                const CarbonDenseSpan& span,
                                int64_t ix,
                                int64_t iy,
                                int64_t iz,
                                int component_id) -> int32_t {
  auto* mem = reinterpret_cast<int32_t*>(static_cast<uintptr_t>(part.data_ptr));
  return mem[carbon_dense_linear_offset(span, ix, iy, iz, component_id)];
}

CARBON_WARP_HD void
carbon_dense_partition_write_i32(const CarbonDensePartitionI32& part,
                                 const CarbonDenseSpan& span,
                                 int64_t ix,
                                 int64_t iy,
                                 int64_t iz,
                                 int component_id,
                                 int32_t value) {
  auto* mem = reinterpret_cast<int32_t*>(static_cast<uintptr_t>(part.data_ptr));
  mem[carbon_dense_linear_offset(span, ix, iy, iz, component_id)] = value;
}

CARBON_WARP_HD auto
carbon_dense_partition_read_i64(const CarbonDensePartitionI64& part,
                                const CarbonDenseSpan& span,
                                int64_t ix,
                                int64_t iy,
                                int64_t iz,
                                int component_id) -> int64_t {
  auto* mem = reinterpret_cast<int64_t*>(static_cast<uintptr_t>(part.data_ptr));
  return mem[carbon_dense_linear_offset(span, ix, iy, iz, component_id)];
}

CARBON_WARP_HD void
carbon_dense_partition_write_i64(const CarbonDensePartitionI64& part,
                                 const CarbonDenseSpan& span,
                                 int64_t ix,
                                 int64_t iy,
                                 int64_t iz,
                                 int component_id,
                                 int64_t value) {
  auto* mem = reinterpret_cast<int64_t*>(static_cast<uintptr_t>(part.data_ptr));
  mem[carbon_dense_linear_offset(span, ix, iy, iz, component_id)] = value;
}

CARBON_WARP_HD auto
carbon_dense_partition_read_u8(const CarbonDensePartitionU8& part,
                               const CarbonDenseSpan& span,
                               int64_t ix,
                               int64_t iy,
                               int64_t iz,
                               int component_id) -> uint8_t {
  auto* mem = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(part.data_ptr));
  return mem[carbon_dense_linear_offset(span, ix, iy, iz, component_id)];
}

CARBON_WARP_HD void
carbon_dense_partition_write_u8(const CarbonDensePartitionU8& part,
                                const CarbonDenseSpan& span,
                                int64_t ix,
                                int64_t iy,
                                int64_t iz,
                                int component_id,
                                uint8_t value) {
  auto* mem = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(part.data_ptr));
  mem[carbon_dense_linear_offset(span, ix, iy, iz, component_id)] = value;
}

struct CarbonIndex3 {
  int64_t x;
  int64_t y;
  int64_t z;

  CARBON_WARP_HD CarbonIndex3() = default;

  CARBON_WARP_HD CarbonIndex3(const char* bytes,
                              size_t n) {
    if (n == sizeof(*this)) {
      memcpy(this, bytes, n);
    }
  }
};

struct CarbonDenseIdx {
  int64_t ix;
  int64_t iy;
  int64_t iz;

  CARBON_WARP_HD CarbonDenseIdx() = default;

  CARBON_WARP_HD CarbonDenseIdx(const char* bytes,
                                size_t n) {
    if (n == sizeof(*this)) {
      memcpy(this, bytes, n);
    }
  }
};

// Map a launch-grid thread (ix, iy, iz) into a buffer-relative owned-cell Idx for the
// span's DataView, mirroring carbon::detail::dense::Span::set_and_validate. The launch
// grid for Internal / Boundary is thinner on z, so the raw thread iz must be shifted back
// onto the owned-cell z axis; downstream carbon_dense_linear_offset adds halo_z to reach
// the haloed buffer row. Standard: iz passes through. Internal: skip the lower boundary
// band (owned_z = halo_z + iz). Boundary: the [0, 2*halo_z) thread-z domain folds the two
// halo-thick bands — lower band (iz < halo_z) → owned_z = iz; upper band → owned_z lands in
// the top halo_z owned cells (owned_z = logic_z - 2*halo_z + iz).
CARBON_WARP_HD auto
carbon_dense_set(const CarbonDenseSpan& span,
                 int64_t ix,
                 int64_t iy,
                 int64_t iz) -> CarbonDenseIdx {
  const int64_t hz = static_cast<int64_t>(span.halo_z);
  int64_t owned_z = iz;
  if (span.data_view == 1) {
    owned_z = hz + iz;
  } else if (span.data_view == 2) {
    owned_z = (iz < hz) ? iz : (span.logic_z - 2 * hz + iz);
  }
  CarbonDenseIdx idx{};
  idx.ix = ix;
  idx.iy = iy;
  idx.iz = owned_z;
  return idx;
}

CARBON_WARP_HD auto
carbon_dense_global_idx(const CarbonDenseSpan& span,
                        const CarbonDenseIdx& idx) -> CarbonIndex3 {
  CarbonIndex3 out{};
  out.x = idx.ix;
  out.y = idx.iy;
  // z_offset is measured from the haloed buffer row (see carbon_dense_neighbour_idx_valid,
  // which forms the global z as halo_z + idx.iz + z_offset). idx.iz here is the owned-cell
  // z, so the halo_z term must be added to recover the true global z. Omitting it shifted
  // every global z down by halo_z (e.g. the top owned plane reported global_z - 1), which
  // made the indices boundary masker miss the max-z plane entirely.
  out.z = static_cast<int64_t>(span.halo_z) + idx.iz + span.z_offset;
  return out;
}

CARBON_WARP_HD auto
carbon_dense_partition_read_f32(const CarbonDensePartitionF32& part,
                                const CarbonDenseSpan& span,
                                const CarbonDenseIdx& idx,
                                int component_id) -> float {
  return carbon_dense_partition_read_f32(part, span, idx.ix, idx.iy, idx.iz, component_id);
}

CARBON_WARP_HD auto
carbon_dense_partition_read_f64(const CarbonDensePartitionF64& part,
                                const CarbonDenseSpan& span,
                                const CarbonDenseIdx& idx,
                                int component_id) -> double {
  return carbon_dense_partition_read_f64(part, span, idx.ix, idx.iy, idx.iz, component_id);
}

CARBON_WARP_HD auto
carbon_dense_partition_read_i32(const CarbonDensePartitionI32& part,
                                const CarbonDenseSpan& span,
                                const CarbonDenseIdx& idx,
                                int component_id) -> int32_t {
  return carbon_dense_partition_read_i32(part, span, idx.ix, idx.iy, idx.iz, component_id);
}

CARBON_WARP_HD auto
carbon_dense_partition_read_i64(const CarbonDensePartitionI64& part,
                                const CarbonDenseSpan& span,
                                const CarbonDenseIdx& idx,
                                int component_id) -> int64_t {
  return carbon_dense_partition_read_i64(part, span, idx.ix, idx.iy, idx.iz, component_id);
}

CARBON_WARP_HD void
carbon_dense_partition_write_f32(const CarbonDensePartitionF32& part,
                                   const CarbonDenseSpan& span,
                                   const CarbonDenseIdx& idx,
                                   int component_id,
                                   float value) {
  carbon_dense_partition_write_f32(part, span, idx.ix, idx.iy, idx.iz, component_id, value);
}

CARBON_WARP_HD void
carbon_dense_partition_write_f64(const CarbonDensePartitionF64& part,
                                   const CarbonDenseSpan& span,
                                   const CarbonDenseIdx& idx,
                                   int component_id,
                                   double value) {
  carbon_dense_partition_write_f64(part, span, idx.ix, idx.iy, idx.iz, component_id, value);
}

CARBON_WARP_HD void
carbon_dense_partition_write_i32(const CarbonDensePartitionI32& part,
                                   const CarbonDenseSpan& span,
                                   const CarbonDenseIdx& idx,
                                   int component_id,
                                   int32_t value) {
  carbon_dense_partition_write_i32(part, span, idx.ix, idx.iy, idx.iz, component_id, value);
}

CARBON_WARP_HD void
carbon_dense_partition_write_i64(const CarbonDensePartitionI64& part,
                                   const CarbonDenseSpan& span,
                                   const CarbonDenseIdx& idx,
                                   int component_id,
                                   int64_t value) {
  carbon_dense_partition_write_i64(part, span, idx.ix, idx.iy, idx.iz, component_id, value);
}

CARBON_WARP_HD auto
carbon_dense_partition_read_u8(const CarbonDensePartitionU8& part,
                               const CarbonDenseSpan& span,
                               const CarbonDenseIdx& idx,
                               int component_id) -> uint8_t {
  return carbon_dense_partition_read_u8(part, span, idx.ix, idx.iy, idx.iz, component_id);
}

CARBON_WARP_HD void
carbon_dense_partition_write_u8(const CarbonDensePartitionU8& part,
                                  const CarbonDenseSpan& span,
                                  const CarbonDenseIdx& idx,
                                  int component_id,
                                  uint8_t value) {
  carbon_dense_partition_write_u8(part, span, idx.ix, idx.iy, idx.iz, component_id, value);
}

static constexpr int32_t kCarbonBoundaryOpen = 0;
static constexpr int32_t kCarbonBoundaryPeriodicZ = 1;

struct CarbonNeighbourValueF32 {
  float value;
  int32_t valid;

  CARBON_WARP_HD CarbonNeighbourValueF32() = default;

  CARBON_WARP_HD CarbonNeighbourValueF32(const char* bytes,
                                         size_t n) {
    if (n == sizeof(*this)) {
      memcpy(this, bytes, n);
    }
  }
};

struct CarbonNeighbourValueF64 {
  double value;
  int32_t valid;

  CARBON_WARP_HD CarbonNeighbourValueF64() = default;

  CARBON_WARP_HD CarbonNeighbourValueF64(const char* bytes,
                                         size_t n) {
    if (n == sizeof(*this)) {
      memcpy(this, bytes, n);
    }
  }
};

struct CarbonNeighbourValueI32 {
  int32_t value;
  int32_t valid;

  CARBON_WARP_HD CarbonNeighbourValueI32() = default;

  CARBON_WARP_HD CarbonNeighbourValueI32(const char* bytes,
                                         size_t n) {
    if (n == sizeof(*this)) {
      memcpy(this, bytes, n);
    }
  }
};

struct CarbonNeighbourValueI64 {
  int64_t value;
  int32_t valid;

  CARBON_WARP_HD CarbonNeighbourValueI64() = default;

  CARBON_WARP_HD CarbonNeighbourValueI64(const char* bytes,
                                         size_t n) {
    if (n == sizeof(*this)) {
      memcpy(this, bytes, n);
    }
  }
};

struct CarbonNeighbourValueU8 {
  uint8_t value;
  int32_t valid;

  CARBON_WARP_HD CarbonNeighbourValueU8() = default;

  CARBON_WARP_HD CarbonNeighbourValueU8(const char* bytes,
                                        size_t n) {
    if (n == sizeof(*this)) {
      memcpy(this, bytes, n);
    }
  }
};

CARBON_WARP_HD auto
carbon_dense_neighbour_idx_valid(const CarbonDenseSpan& span,
                                 const CarbonDenseIdx& idx,
                                 int ox,
                                 int oy,
                                 int oz) -> bool {
  const int64_t bx = idx.ix + static_cast<int64_t>(ox);
  const int64_t by = idx.iy + static_cast<int64_t>(oy);
  const int64_t bz =
      static_cast<int64_t>(span.halo_z) + idx.iz + static_cast<int64_t>(oz);
  const int64_t haloed_x = span.logic_x;
  const int64_t haloed_y = span.logic_y;
  const int64_t haloed_z =
      span.logic_z + 2 * static_cast<int64_t>(span.halo_z);
  if (bx < 0 || bx >= haloed_x || by < 0 || by >= haloed_y || bz < 0 || bz >= haloed_z) {
    return false;
  }
  const int64_t gz = bz + span.z_offset;
  if (gz < 0 || gz >= span.global_z) {
    return span.boundary_mode == kCarbonBoundaryPeriodicZ;
  }
  return true;
}

#define CARBON_DEFINE_DENSE_NEIGHBOUR(TypeName, ValueType, ReadFn)                     \
  CARBON_WARP_HD auto carbon_dense_neighbour_##TypeName(                               \
      const CarbonDensePartition##TypeName& part,                                      \
      const CarbonDenseSpan& span,                                                     \
      const CarbonDenseIdx& idx,                                                       \
      int ox,                                                                          \
      int oy,                                                                          \
      int oz,                                                                          \
      int component_id) -> CarbonNeighbourValue##TypeName {                            \
    CarbonNeighbourValue##TypeName out{};                                              \
    if (!carbon_dense_neighbour_idx_valid(span, idx, ox, oy, oz)) {                    \
      return out;                                                                      \
    }                                                                                  \
    out.value = ReadFn(part, span, idx.ix + ox, idx.iy + oy, idx.iz + oz,              \
                       component_id);                                                  \
    out.valid = 1;                                                                     \
    return out;                                                                        \
  }                                                                                    \
  CARBON_WARP_HD auto carbon_neighbour_value_or_##TypeName(                            \
      const CarbonNeighbourValue##TypeName& nv,                                         \
      ValueType alternative) -> ValueType {                                            \
    return nv.valid ? nv.value : alternative;                                          \
  }                                                                                    \
  CARBON_WARP_HD auto carbon_neighbour_value_##TypeName(                               \
      const CarbonNeighbourValue##TypeName& nv) -> ValueType {                           \
    return nv.value;                                                                   \
  }                                                                                    \
  CARBON_WARP_HD auto carbon_neighbour_valid_##TypeName(                               \
      const CarbonNeighbourValue##TypeName& nv) -> bool {                              \
    return nv.valid != 0;                                                              \
  }                                                                                    \
  CARBON_WARP_HD auto carbon_dense_read_ngh_##TypeName(                                \
      const CarbonDensePartition##TypeName& part,                                      \
      const CarbonDenseSpan& span,                                                     \
      const CarbonDenseIdx& idx,                                                       \
      int ox,                                                                          \
      int oy,                                                                          \
      int oz,                                                                          \
      int component_id,                                                                \
      ValueType alternative) -> ValueType {                                            \
    return carbon_neighbour_value_or_##TypeName(                                       \
        carbon_dense_neighbour_##TypeName(part, span, idx, ox, oy, oz, component_id), \
        alternative);                                                                  \
  }

CARBON_DEFINE_DENSE_NEIGHBOUR(F32, float, carbon_dense_partition_read_f32)
CARBON_DEFINE_DENSE_NEIGHBOUR(F64, double, carbon_dense_partition_read_f64)
CARBON_DEFINE_DENSE_NEIGHBOUR(I32, int32_t, carbon_dense_partition_read_i32)
CARBON_DEFINE_DENSE_NEIGHBOUR(I64, int64_t, carbon_dense_partition_read_i64)
CARBON_DEFINE_DENSE_NEIGHBOUR(U8, uint8_t, carbon_dense_partition_read_u8)

#undef CARBON_DEFINE_DENSE_NEIGHBOUR

CARBON_WARP_HD auto
carbon_index3_get_x(const CarbonIndex3& idx) -> int64_t {
  return idx.x;
}

CARBON_WARP_HD auto
carbon_index3_get_y(const CarbonIndex3& idx) -> int64_t {
  return idx.y;
}

CARBON_WARP_HD auto
carbon_index3_get_z(const CarbonIndex3& idx) -> int64_t {
  return idx.z;
}

}  // namespace wp

#if !defined(__CUDA_ARCH__)
#include "carbonpy/pod_layout_checks.h"
#endif
