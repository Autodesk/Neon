#pragma once

#include <stdint.h>
#include <string.h>

#if defined(__CUDACC__)
#define CARBON_PART_WARP_HD __device__ __host__ __forceinline__
#else
#define CARBON_PART_WARP_HD inline
#endif

namespace wp {

#define CARBON_DEFINE_POD_BINARY_CTOR(TypeName)                                        \
  CARBON_PART_WARP_HD TypeName() = default;                                            \
  CARBON_PART_WARP_HD TypeName(const char* bytes,                                      \
                               size_t n) {                                             \
    if (n == sizeof(*this)) {                                                          \
      memcpy(this, bytes, n);                                                          \
    }                                                                                  \
  }

// Defined here (ahead of the partitions) so the dense partitions can embed it; the
// spanless read/write helpers in warp_dense.h address memory from part.span alone.
struct CarbonDenseSpan {
  int64_t iter_x;
  int64_t iter_y;
  int64_t iter_z;
  int64_t logic_x;
  int64_t logic_y;
  int64_t logic_z;
  int64_t pitch_xy;
  int64_t z_offset;
  int32_t halo_x;
  int32_t halo_y;
  int32_t halo_z;
  int64_t global_x;
  int64_t global_y;
  int64_t global_z;
  int32_t boundary_mode;
  int32_t data_view;
  CARBON_DEFINE_POD_BINARY_CTOR(CarbonDenseSpan)
};

struct CarbonDensePartitionF32 {
  uint64_t data_ptr;
  CarbonDenseSpan span;
  CARBON_DEFINE_POD_BINARY_CTOR(CarbonDensePartitionF32)
};

struct CarbonDensePartitionF64 {
  uint64_t data_ptr;
  CarbonDenseSpan span;
  CARBON_DEFINE_POD_BINARY_CTOR(CarbonDensePartitionF64)
};

struct CarbonDensePartitionI32 {
  uint64_t data_ptr;
  CarbonDenseSpan span;
  CARBON_DEFINE_POD_BINARY_CTOR(CarbonDensePartitionI32)
};

struct CarbonDensePartitionI64 {
  uint64_t data_ptr;
  CarbonDenseSpan span;
  CARBON_DEFINE_POD_BINARY_CTOR(CarbonDensePartitionI64)
};

struct CarbonDensePartitionU8 {
  uint64_t data_ptr;
  CarbonDenseSpan span;
  CARBON_DEFINE_POD_BINARY_CTOR(CarbonDensePartitionU8)
};

using CarbonDensePartition = CarbonDensePartitionF32;

struct CarbonBlockPartitionF32 {
  uint64_t data_ptr;
  int32_t components;
  CARBON_DEFINE_POD_BINARY_CTOR(CarbonBlockPartitionF32)
};

struct CarbonBlockPartitionF64 {
  uint64_t data_ptr;
  int32_t components;
  CARBON_DEFINE_POD_BINARY_CTOR(CarbonBlockPartitionF64)
};

struct CarbonBlockPartitionI32 {
  uint64_t data_ptr;
  int32_t components;
  CARBON_DEFINE_POD_BINARY_CTOR(CarbonBlockPartitionI32)
};

struct CarbonBlockPartitionI64 {
  uint64_t data_ptr;
  int32_t components;
  CARBON_DEFINE_POD_BINARY_CTOR(CarbonBlockPartitionI64)
};

using CarbonBlockPartition = CarbonBlockPartitionF32;

#undef CARBON_DEFINE_POD_BINARY_CTOR

}  // namespace wp
