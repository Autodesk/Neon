#pragma once

#include <carbon/cuda_annotation.h>

#include <array>
#include <type_traits>

namespace carbon {

/**
 * Three-component floating-point tuple (x, y, z) — typically physical sizes / spacings / origins
 * rather than cell counts (use @ref Integer3d for the latter).
 *
 * @c CARBON_HD on every member so the type is usable from @c __global__ / @c __device__ code
 * when compiled with nvcc.
 *
 * @tparam T Floating-point type for each axis (enforced via @c static_assert).
 */
template <typename T>
struct Floating3d {
  static_assert(std::is_floating_point_v<T>,
                "Floating3d requires a floating-point T");

  // Alias hoisted out of member signatures so the formatter doesn't split
  // `std::array<T, 3>` when `BinPackParameters: AlwaysOnePerLine` is on.
  using array_type = std::array<T, 3>;

  T x{};
  T y{};
  T z{};

  CARBON_HD constexpr Floating3d() noexcept = default;

  CARBON_HD constexpr Floating3d(T x_,
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

  /**
   * Product @c x * @c y * @c z in @p T.
   *
   * @note Unlike @ref Integer3d::volume, this overload does @b not widen the element type:
   *       the result keeps the same precision and dynamic range as @p T (e.g. @c float64_3d
   *       returns @c double, not a higher-precision intermediate). Callers that need an exact
   *       product over a wider type or precision must perform the widening themselves before
   *       multiplying — for floating-point types we deliberately avoid silently picking
   *       @c long double, which is platform-dependent and disallowed in CUDA device code.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  volume() const noexcept -> T {
    return x * y * z;
  }

  /** Same x and y; z replaced by @p new_z. */
  [[nodiscard]] CARBON_HD constexpr auto
  with_z(T new_z) const noexcept -> Floating3d {
    return {x, y, new_z};
  }

  /** Exact component-wise equality. Callers needing tolerance must compare themselves. */
  [[nodiscard]] friend CARBON_HD constexpr auto
  operator==(const Floating3d& a,
             const Floating3d& b) noexcept -> bool {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  }

  [[nodiscard]] friend CARBON_HD constexpr auto
  operator!=(const Floating3d& a,
             const Floating3d& b) noexcept -> bool {
    return !(a == b);
  }
};

/** 3D tuple of IEEE-754 single-precision floats. */
using float32_3d = Floating3d<float>;

/** 3D tuple of IEEE-754 double-precision floats. */
using float64_3d = Floating3d<double>;

#if !defined(__CUDACC__) && !defined(__HIPCC__)
namespace detail {

// Hoisted out of static_assert expressions for the same readability reason as in integer3d.h.
inline constexpr float64_3d kFloat64_3d_2_3_4{2.0, 3.0, 4.0};
inline constexpr float64_3d kFloat64_3d_1_2_3{1.0, 2.0, 3.0};
inline constexpr float32_3d kFloat32_3d_1_2_3{1.0F, 2.0F, 3.0F};

}  // namespace detail

static_assert(detail::kFloat64_3d_2_3_4.volume() == 24.0);
static_assert(detail::kFloat32_3d_1_2_3.with_z(9.0F).z == 9.0F);
static_assert(detail::kFloat64_3d_1_2_3 == detail::kFloat64_3d_1_2_3);
#endif

}  // namespace carbon
