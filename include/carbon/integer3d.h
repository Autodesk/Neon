#pragma once

#include <carbon/cuda_annotation.h>

#include <array>
#include <cstdint>
#include <type_traits>

namespace carbon {

namespace detail {

template <typename T>
using integer3d_volume_t = std::conditional_t<std::is_signed_v<T>, std::intmax_t, std::uintmax_t>;

}  // namespace detail

/**
 * Three-component integer tuple (x, y, z) — grid extents, cell indices, block shapes, etc.
 *
 * @c CARBON_HD on every member so the type is usable from @c __global__ / @c __device__ code
 * when compiled with nvcc. @ref volume widens to @c std::intmax_t (signed @p T) or
 * @c std::uintmax_t (unsigned @p T) to reduce overflow risk on the product.
 *
 * @tparam T Integral type for each axis (enforced via @c static_assert).
 */
template <typename T>
struct Integer3d {
  static_assert(std::is_integral_v<T>,
                "Integer3d requires an integral T");

  // Alias hoisted out of member signatures so the formatter doesn't split
  // `std::array<T, 3>` when `BinPackParameters: AlwaysOnePerLine` is on.
  using array_type = std::array<T, 3>;

  T x{};
  T y{};
  T z{};

  CARBON_HD constexpr Integer3d() noexcept = default;

  CARBON_HD constexpr Integer3d(T x_,
                                T y_,
                                T z_) noexcept
      : x(x_),
        y(y_),
        z(z_) {}

  /** { x, y, z } for generic algorithms. */
  [[nodiscard]] CARBON_HD constexpr auto
  as_array() const -> array_type {
    return {x, y, z};
  }

  /** Product x * y * z widened to @c std::intmax_t / @c std::uintmax_t (see class comment). */
  [[nodiscard]] CARBON_HD constexpr auto
  volume() const -> detail::integer3d_volume_t<T> {
    using W = detail::integer3d_volume_t<T>;
    return static_cast<W>(x) * static_cast<W>(y) * static_cast<W>(z);
  }

  /** Same x and y; z replaced by @p new_z. */
  [[nodiscard]] CARBON_HD constexpr auto
  with_z(T new_z) const noexcept -> Integer3d {
    return {x, y, new_z};
  }

  [[nodiscard]] friend CARBON_HD constexpr auto
  operator==(const Integer3d& a,
             const Integer3d& b) noexcept -> bool {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  }

  [[nodiscard]] friend CARBON_HD constexpr auto
  operator!=(const Integer3d& a,
             const Integer3d& b) noexcept -> bool {
    return !(a == b);
  }
};

/**
 * 3D tuple of signed 8-bit integers — typical for per-thread neighbour offsets passed into
 * device kernels (radius up to ±127). Same role as Neon's @c NghIdx; pairs naturally with
 * @c carbon::detail::dense::Partition::neighbour.
 */
using int8_3d = Integer3d<std::int8_t>;

/** 3D tuple of signed 32-bit integers. */
using int32_3d = Integer3d<std::int32_t>;

/** 3D tuple of signed 64-bit integers — the default for host/device grid extents. */
using int64_3d = Integer3d<int64_t>;

#if !defined(__CUDACC__) && !defined(__HIPCC__)
namespace detail {

// Hoisted out of static_assert expressions to keep the asserts themselves one-line readable
// after clang-format breaks the brace-initialised arguments.
inline constexpr int64_3d kInt64_3d_2_3_4{2, 3, 4};
inline constexpr int32_3d kInt32_3d_1_2_3{1, 2, 3};

}  // namespace detail

static_assert(detail::kInt64_3d_2_3_4.volume() == 24);
static_assert(detail::kInt64_3d_2_3_4.z == 4);
static_assert(detail::kInt32_3d_1_2_3.with_z(9).z == 9);
#endif

}  // namespace carbon
