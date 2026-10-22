#pragma once

#include <cstddef>
#include <cstdint>

namespace carbonpy::detail {

enum CarbonDtype : int {
  Dtype_F32 = 0,
  Dtype_F64 = 1,
  Dtype_I32 = 2,
  Dtype_I64 = 3,
  Dtype_U8 = 4,
};

/**
 * Field handle shared by the Dense and Block C-ABI shims. The layout must be
 * byte-identical in every translation unit that casts a field @c void* (the
 * grid TU that creates it and @c cuda_driver.cpp that binds it), so both sides
 * include this one definition rather than redeclaring it.
 */
struct FieldBox {
  int dtype;
  int components;
  void* field;
  void* dense;  // owning carbon::Dense* (Dense fields only; null for Block), used to
                // fill the partition's embedded span per (device, data_view).
};

/** mGrid field handle: carries the owning grid for cross-level partitions. */
struct MgridFieldBox {
  int dtype;
  int components;
  void* field;
  void* mgrid;
};

/**
 * Invoke @c fn.operator()<T>() for the scalar type named by @p dtype. @c fn is
 * a C++20 templated lambda returning @c int (0 on success). Returns -1 for an
 * unrecognized dtype.
 */
template <typename Fn>
auto
dispatch_dtype(int dtype,
               Fn&& fn) -> int {
  switch (dtype) {
    case Dtype_F32:
      return fn.template operator()<float>();
    case Dtype_F64:
      return fn.template operator()<double>();
    case Dtype_I32:
      return fn.template operator()<int32_t>();
    case Dtype_I64:
      return fn.template operator()<int64_t>();
    case Dtype_U8:
      return fn.template operator()<uint8_t>();
    default:
      return -1;
  }
}

}  // namespace carbonpy::detail
