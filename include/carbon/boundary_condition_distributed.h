#pragma once

#include <carbon/cuda_annotation.h>

#include <cstdint>

namespace carbon {

/**
 * Kind of boundary condition applied on one face of a @ref DenseDistributed grid.
 *
 * Opt-in: a face with @ref BcKind::None keeps the historical behaviour of an open face
 * (out-of-domain reads report @c valid = false through @c Partition::neighbour).
 */
enum class BcKind : std::uint8_t {
  /** No boundary condition on this face (default). */
  None = 0,

  /** Fixed value @c u = g on the boundary node. */
  Dirichlet = 1,

  /** Fixed outward-normal derivative @c du/dn = g on the boundary node. */
  Neumann = 2,
};

/** One of the six faces of the global box. "Low" is the face at coordinate 0, "High" at extent-1. */
enum class Face : std::uint8_t {
  XLow = 0,
  XHigh = 1,
  YLow = 2,
  YHigh = 3,
  ZLow = 4,
  ZHigh = 5,
};

/** Number of faces of the global box. */
inline constexpr int kFaceCount = 6;

/**
 * Boundary condition on a single face: a kind and one constant scalar.
 *
 * For @ref BcKind::Dirichlet, @c value is the field value @c g on the boundary node.
 * For @ref BcKind::Neumann, @c value is the @b outward-normal derivative @c du/dn (the normal
 * points out of the domain), so the same sign convention holds on low and high faces.
 */
struct FaceBc {
  BcKind kind{BcKind::None};
  double value{0.0};
};

/** Dirichlet condition @c u = g. */
[[nodiscard]] constexpr auto
dirichlet(double g) noexcept -> FaceBc {
  return FaceBc{BcKind::Dirichlet, g};
}

/** Neumann condition with outward-normal derivative @c du/dn = g. */
[[nodiscard]] constexpr auto
neumann(double g = 0.0) noexcept -> FaceBc {
  return FaceBc{BcKind::Neumann, g};
}

/**
 * Per-face boundary conditions of a @ref DenseDistributed grid.
 *
 * A trivially-copyable value type (it rides inside the by-value kernel @c Span), constexpr
 * throughout. Every face defaults to @ref BcKind::None, so a default-constructed
 * @ref BoundaryConditions changes nothing.
 *
 * @par Node-centred convention
 * The boundary lies @b on the first / last in-domain cell (cell @c (0,0,0) sits at the grid
 * origin). A read @c k cells outside a BC face is a @b virtual boundary value: it is computed on
 * the spot from the in-domain @b mirror cell, @c k cells inside the face (reflection through the
 * boundary node), and nothing is stored for it. With @c u_m the mirror cell value and @c h the
 * grid spacing:
 * - Dirichlet: @c 2*g - u_m
 * - Neumann:   @c u_m + 2*k*h*g
 *
 * The boundary node itself is not set by the grid; the solver pins or solves it.
 *
 * @par Not combinable with periodic axes
 * A BC on an axis that is periodic in @ref BoundaryModeDistributed is rejected when the grid is
 * constructed.
 *
 * @code
 *   auto bc = carbon::BoundaryConditions::none()
 *                 .with(carbon::Face::XLow, carbon::neumann())
 *                 .with(carbon::Face::ZHigh, carbon::dirichlet(-1.0));
 * @endcode
 */
struct BoundaryConditions {
  FaceBc faces[kFaceCount]{};

  /** No boundary condition on any face. */
  [[nodiscard]] static constexpr auto
  none() noexcept -> BoundaryConditions {
    return BoundaryConditions{};
  }

  /** Copy of @c *this with @p face set to @p bc. */
  [[nodiscard]] constexpr auto
  with(Face face,
       FaceBc bc) const noexcept -> BoundaryConditions {
    BoundaryConditions out = *this;
    out.faces[static_cast<int>(face)] = bc;
    return out;
  }

  /** Condition on @p face. */
  [[nodiscard]] constexpr CARBON_HD auto
  at(Face face) const noexcept -> const FaceBc& {
    return faces[static_cast<int>(face)];
  }

  /** True when at least one face carries a condition. */
  [[nodiscard]] constexpr CARBON_HD auto
  any() const noexcept -> bool {
    for (const FaceBc& f : faces) {
      if (f.kind != BcKind::None) {
        return true;
      }
    }
    return false;
  }
};

}  // namespace carbon
