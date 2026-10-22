#pragma once

// Shared wind-tunnel LBM simulation + ray-march render helpers for the Carbon live viewer.
// Extracted from render_wind_tunnel.cu; the example TU includes this header unchanged for now.

#include "cell_type.h"
#include "d3q19.h"
#include "d3q27.h"
#include "lbm_kernels.h"

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/halo.h>
#include <carbon/kernel.h>
#include <carbon/memory_side.h>
#include <carbon/qcriterion.h>
#include <carbon/render.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/distributed.h>
#endif

namespace carbon::examples::wind_tunnel {

enum class Layers : unsigned char {
  Speed,
  Q,
  /** @c |u - u_inf| : the free stream subtracted away, so only the body's wake is drawn. */
  Wake,
  /** Q-criterion opacity, local-speed hue. */
  QTinted,
};

using carbon::bench::lbm::BgkCollision;
using carbon::bench::lbm::Classification;
using carbon::bench::lbm::D3Q27;
using carbon::bench::lbm::encode_cell;
using carbon::bench::lbm::LbmParameters;
using carbon::bench::lbm::make_compute_wall_mask_kernel;
using carbon::bench::lbm::make_iteration_kernel;
using carbon::bench::lbm::make_macroscopic_kernel;

[[nodiscard]] inline auto
world_rank(const carbon::System& sys) noexcept -> int {
#ifdef CARBON_HAVE_MPI_NCCL
  if (const auto* d = sys.distributed(); d != nullptr) {
    return d->world_rank();
  }
#endif
  return 0;
}

[[nodiscard]] CARBON_HD inline auto
cylinder_sdf(float px,
             float /*py*/,
             float pz,
             float cx,
             float /*cy*/,
             float cz,
             float radius) noexcept -> float {
  const float dx = px - cx;
  const float dz = pz - cz;
  return std::sqrt(dx * dx + dz * dz) - radius;
}

[[nodiscard]] inline auto
inside_cylinder(const carbon::int64_3d& gc,
                float cx,
                float cz,
                float radius) noexcept -> bool {
  const float dx = static_cast<float>(gc.x) - cx;
  const float dz = static_cast<float>(gc.z) - cz;
  return (dx * dx + dz * dz) <= radius * radius;
}

enum class ObstacleShape : int32_t {
  Cylinder = 0,
  Sphere = 1,
  Box = 2,
};

[[nodiscard]] CARBON_HD inline auto
sphere_sdf(float px,
           float py,
           float pz,
           float cx,
           float cy,
           float cz,
           float radius) noexcept -> float {
  const float dx = px - cx;
  const float dy = py - cy;
  const float dz = pz - cz;
  return std::sqrt(dx * dx + dy * dy + dz * dz) - radius;
}

[[nodiscard]] CARBON_HD inline auto
box_sdf(float px,
        float py,
        float pz,
        float cx,
        float cy,
        float cz,
        float half) noexcept -> float {
  const float dx = std::fabs(px - cx) - half;
  const float dy = std::fabs(py - cy) - half;
  const float dz = std::fabs(pz - cz) - half;
  const float ax = (dx > 0.0F) ? dx : 0.0F;
  const float ay = (dy > 0.0F) ? dy : 0.0F;
  const float az = (dz > 0.0F) ? dz : 0.0F;
  const float mx = (dx > dy) ? dx : dy;
  const float mxyz = (mx > dz) ? mx : dz;
  const float inner = (mxyz < 0.0F) ? mxyz : 0.0F;
  return std::sqrt(ax * ax + ay * ay + az * az) + inner;
}

[[nodiscard]] CARBON_HD inline auto
inside_obstacle(const carbon::int64_3d& gc,
                ObstacleShape shape,
                float cx,
                float cy,
                float cz,
                float radius) noexcept -> bool {
  const float dx = static_cast<float>(gc.x) - cx;
  const float dy = static_cast<float>(gc.y) - cy;
  switch (shape) {
    case ObstacleShape::Sphere: {
      const float dz = static_cast<float>(gc.z) - cz;
      return (dx * dx + dy * dy + dz * dz) <= radius * radius;
    }
    case ObstacleShape::Box:
      return std::fabs(dx) <= radius && std::fabs(dy) <= radius &&
             std::fabs(static_cast<float>(gc.z) - cz) <= radius;
    case ObstacleShape::Cylinder:
    default: {
      const float dz = static_cast<float>(gc.z) - cz;
      return (dx * dx + dz * dz) <= radius * radius;
    }
  }
}

[[nodiscard]] CARBON_HD inline auto
obstacle_sdf(float px,
             float py,
             float pz,
             ObstacleShape shape,
             float cx,
             float cy,
             float cz,
             float radius) noexcept -> float {
  switch (shape) {
    case ObstacleShape::Sphere:
      return sphere_sdf(px, py, pz, cx, cy, cz, radius);
    case ObstacleShape::Box:
      return box_sdf(px, py, pz, cx, cy, cz, radius);
    case ObstacleShape::Cylinder:
    default:
      return cylinder_sdf(px, py, pz, cx, cy, cz, radius);
  }
}

[[nodiscard]] CARBON_HD inline auto
obstacle_surface_normal(float px,
                        float py,
                        float pz,
                        ObstacleShape shape,
                        float cx,
                        float cy,
                        float cz) noexcept -> carbon::float32_3d {
  switch (shape) {
    case ObstacleShape::Sphere: {
      const float dx = px - cx;
      const float dy = py - cy;
      const float dz = pz - cz;
      const float r = std::sqrt(dx * dx + dy * dy + dz * dz);
      return (r > 1.0e-6F) ? carbon::float32_3d{dx / r, dy / r, dz / r}
                           : carbon::float32_3d{1.0F, 0.0F, 0.0F};
    }
    case ObstacleShape::Box: {
      const float ax = std::fabs(px - cx);
      const float ay = std::fabs(py - cy);
      const float az = std::fabs(pz - cz);
      if (ax >= ay && ax >= az) {
        return carbon::float32_3d{(px >= cx) ? 1.0F : -1.0F, 0.0F, 0.0F};
      }
      if (ay >= az) {
        return carbon::float32_3d{0.0F, (py >= cy) ? 1.0F : -1.0F, 0.0F};
      }
      return carbon::float32_3d{0.0F, 0.0F, (pz >= cz) ? 1.0F : -1.0F};
    }
    case ObstacleShape::Cylinder:
    default: {
      const float dx = px - cx;
      const float dz = pz - cz;
      const float r = std::sqrt(dx * dx + dz * dz);
      return (r > 1.0e-6F) ? carbon::float32_3d{dx / r, 0.0F, dz / r}
                           : carbon::float32_3d{1.0F, 0.0F, 0.0F};
    }
  }
}

template <typename Lattice>
[[nodiscard]] CARBON_HD inline auto
feq3(int q,
     float ux,
     float uy,
     float uz) noexcept -> float {
  const float w = Lattice::w(q);
  const auto cq = Lattice::c(q);
  const float cu =
      static_cast<float>(cq.x) * ux + static_cast<float>(cq.y) * uy + static_cast<float>(cq.z) * uz;
  const float usqr = ux * ux + uy * uy + uz * uz;
  return w * (1.0F + 3.0F * cu + 4.5F * cu * cu - 1.5F * usqr);
}

template <typename Lattice>
[[nodiscard]] CARBON_HD inline auto
feq_z(int q,
      float uz) noexcept -> float {
  return feq3<Lattice>(q, 0.0F, 0.0F, uz);
}

[[nodiscard]] CARBON_HD inline auto
wt_is_wall(const carbon::int64_3d& gc,
           float cx,
           float cy,
           float cz,
           float radius,
           ObstacleShape shape) -> bool {
  return inside_obstacle(gc, shape, cx, cy, cz, radius);
}

[[nodiscard]] CARBON_HD inline auto
wt_cell_value(const carbon::int64_3d& gc,
              float cx,
              float cy,
              float cz,
              float radius,
              ObstacleShape shape) -> int32_t {
  return wt_is_wall(gc, cx, cy, cz, radius, shape) ? encode_cell(Classification::BounceBack)
                                                   : encode_cell(Classification::Bulk);
}

template <typename Lattice>
[[nodiscard]] CARBON_HD inline auto
wt_population_value(const carbon::int64_3d& gc,
                    int q,
                    float u_max,
                    float cx,
                    float cy,
                    float cz,
                    float radius,
                    ObstacleShape shape,
                    int64_t nspan,
                    float pert_amp) -> float {
  if (wt_is_wall(gc, cx, cy, cz, radius, shape)) {
    return 0.0F;
  }
  float ux = 0.0F;
  float uy = 0.0F;
  if (nspan > 1 && pert_amp > 0.0F) {
    constexpr float two_pi = 6.2831853F;
    const float yf = static_cast<float>(gc.y) / static_cast<float>(nspan);
    ux = pert_amp * u_max * std::sin(two_pi * yf);
    uy = pert_amp * u_max * std::sin(2.0F * two_pi * yf + 1.0F);
  }
  return feq3<Lattice>(q, ux, uy, u_max);
}

template <typename Lattice>
[[nodiscard]] inline auto
build_inout_bc_kernel(const carbon::Dense& dense,
                      carbon::Dense::Field<float,
                                           Lattice::Q>& field,
                      carbon::int64_3d global,
                      float u_max,
                      float cx,
                      float cy,
                      float cz,
                      float radius,
                      ObstacleShape shape) -> carbon::Kernel {
  return dense.make_kernel(
      [&field, global, u_max, cx, cy, cz, radius, shape](carbon::KernelLoader& loader) {
        auto io = loader.read_write(field);
        return [io, global, u_max, cx, cy, cz, radius, shape]
               CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> void {
                 const carbon::int64_3d gc = span.global_location(idx);
                 // A cylinder is infinite along the span, so it reaches both span-end
                 // faces. Those faces carry a free-stream Dirichlet condition, which
                 // would otherwise inject fluid into the solid every step and hollow
                 // the body out from its end caps inwards.
                 if (inside_obstacle(gc, shape, cx, cy, cz, radius)) {
                   return;
                 }
                 if (gc.z == global.z - 1) {
                   const carbon::int8_3d back{0, 0, -1};
                   float rho = 0.0F;
                   float mx = 0.0F;
                   float my = 0.0F;
                   float mz = 0.0F;
#pragma unroll
                   for (int q = 0; q < Lattice::Q; ++q) {
                     const auto ngh = io.neighbour(span, idx, back, q);
                     const float fq = ngh.valid ? ngh.value : feq_z<Lattice>(q, u_max);
                     const auto cq = Lattice::c(q);
                     rho += fq;
                     mx += static_cast<float>(cq.x) * fq;
                     my += static_cast<float>(cq.y) * fq;
                     mz += static_cast<float>(cq.z) * fq;
                   }
                   const float inv_rho = (rho > 1.0e-6F) ? (1.0F / rho) : 0.0F;
                   const float ux = mx * inv_rho;
                   const float uy = my * inv_rho;
                   const float uz = mz * inv_rho;
#pragma unroll
                   for (int q = 0; q < Lattice::Q; ++q) {
                     io.write(span, idx, q, feq3<Lattice>(q, ux, uy, uz));
                   }
                 } else if (gc.z == 0 || gc.y == 0 || gc.y == global.y - 1 || gc.x == 0 ||
                            gc.x == global.x - 1) {
#pragma unroll
                   for (int q = 0; q < Lattice::Q; ++q) {
                     io.write(span, idx, q, feq_z<Lattice>(q, u_max));
                   }
                 }
               };
      });
}

/**
 * Seeds a population field with the free-stream equilibrium (and zero inside the obstacle) on
 * the device.
 *
 * The host equivalent — @c Field::fill_owned_host with @c wt_population_value — is a serial
 * loop over cells x 27 components, which at the xlarge preset is ~1.4e10 evaluations and takes
 * minutes on one core while the UI thread waits on @c g_app_mtx . Seeding on the device
 * also lets the field stay @c MemorySide::Device , which is ~120 GB of host mirror saved at
 * that preset.
 */
template <typename Lattice>
[[nodiscard]] inline auto
build_pop_init_kernel(const carbon::Dense& dense,
                      carbon::Dense::Field<float,
                                           Lattice::Q>& pop,
                      float u_max,
                      float cx,
                      float cy,
                      float cz,
                      float radius,
                      ObstacleShape shape,
                      int64_t nspan,
                      float pert_amp) -> carbon::Kernel {
  return dense.make_kernel(
      [&pop, u_max, cx, cy, cz, radius, shape, nspan, pert_amp](carbon::KernelLoader& loader) {
        auto out = loader.write(pop);
        return [out, u_max, cx, cy, cz, radius, shape, nspan, pert_amp]
               CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> void {
                 const carbon::int64_3d gc = span.global_location(idx);
#pragma unroll
                 for (int q = 0; q < Lattice::Q; ++q) {
                   out.write(span, idx, q,
                             wt_population_value<Lattice>(gc, q, u_max, cx, cy, cz, radius, shape,
                                                          nspan, pert_amp));
                 }
               };
      });
}

[[nodiscard]] inline auto
build_cells_init_kernel(const carbon::Dense& dense,
                        carbon::Dense::Field<int32_t,
                                             1>& cells,
                        float cx,
                        float cy,
                        float cz,
                        float radius,
                        ObstacleShape shape) -> carbon::Kernel {
  return dense.make_kernel([&cells, cx, cy, cz, radius, shape](carbon::KernelLoader& loader) {
    auto out = loader.write(cells);
    return [out, cx, cy, cz, radius, shape]
           CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> void {
             const carbon::int64_3d gc = span.global_location(idx);
             out.write(span, idx, 0, wt_cell_value(gc, cx, cy, cz, radius, shape));
           };
  });
}

[[nodiscard]] inline auto
build_sdf_init_kernel(const carbon::Dense& dense,
                      carbon::Dense::Field<float,
                                           1>& sdf,
                      float cx,
                      float cy,
                      float cz,
                      float radius,
                      ObstacleShape shape) -> carbon::Kernel {
  return dense.make_kernel([&sdf, cx, cy, cz, radius, shape](carbon::KernelLoader& loader) {
    auto out = loader.write(sdf);
    return [out, cx, cy, cz, radius, shape]
           CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> void {
             const carbon::int64_3d gc = span.global_location(idx);
             const float sp = span.spacing();
             const carbon::float32_3d origin = span.origin();
             const float px = origin.x + static_cast<float>(gc.x) * sp;
             const float py = origin.y + static_cast<float>(gc.y) * sp;
             const float pz = origin.z + static_cast<float>(gc.z) * sp;
             out.write(span, idx, 0, obstacle_sdf(px, py, pz, shape, cx, cy, cz, radius));
           };
  });
}

[[nodiscard]] inline auto
pick_render_devices(bool want_multi) -> std::vector<carbon::detail::Gpu> {
  auto devices = carbon::get_all_devices();
  if (!want_multi && devices.size() > 1U) {
    return std::vector<carbon::detail::Gpu>{devices.front()};
  }
  return devices;
}

[[nodiscard]] constexpr auto
compute_nu(float ulb,
           float diameter,
           float re) noexcept -> float {
  return ulb * diameter / re;
}

[[nodiscard]] constexpr auto
compute_omega(float ulb,
              float diameter,
              float re) noexcept -> float {
  return 1.0F / (3.0F * compute_nu(ulb, diameter, re) + 0.5F);
}

[[nodiscard]] constexpr auto
max_re_for_omega(float ulb,
                 float diameter,
                 float omega_max = 1.98F) noexcept -> float {
  const float denom = 1.0F / omega_max - 0.5F;
  if (denom <= 0.0F) {
    return std::numeric_limits<float>::max();
  }
  return 3.0F * ulb * diameter / denom;
}

[[nodiscard]] constexpr auto
clamp_re(float ulb,
         float diameter,
         float re,
         float omega_max = 1.98F) noexcept -> float {
  const float cap = max_re_for_omega(ulb, diameter, omega_max);
  return (re > cap) ? cap : re;
}

/**
 * Domain-size ladder, each rung ~2x the cell count of the one below except @c XLarge , which
 * doubles every axis of @c Large and so is 8x it (537M cells). The nine live fields cost
 * 252 B/cell, so @c XLarge needs ~139 GiB of device memory — measured 17.8 GiB per GPU across
 * 8, and more than any single card holds, so it is a multi-GPU-only rung.
 * @c radius() keys off @c nz , so the blockage ratio is the same on every rung.
 */
enum class Preset : int32_t {
  Small = 0,
  Medium = 1,
  Large = 2,
  XLarge = 3,
};

struct LiveSimConfig {
  Preset preset = Preset::Large;
  int64_t nx = 256;
  int64_t ny = 256;
  int64_t nz = 1024;
  float re = 450.0F;
  float ulb = 0.10F;
  float pert_amp = 0.05F;
  ObstacleShape shape = ObstacleShape::Cylinder;

  [[nodiscard]] static auto
  from_preset(Preset p) noexcept -> LiveSimConfig {
    LiveSimConfig cfg;
    cfg.preset = p;
    switch (p) {
      case Preset::Small:
        cfg.nx = 160;
        cfg.ny = 160;
        cfg.nz = 512;
        break;
      case Preset::Medium:
        cfg.nx = 192;
        cfg.ny = 192;
        cfg.nz = 768;
        break;
      case Preset::XLarge:
        cfg.nx = 512;
        cfg.ny = 512;
        cfg.nz = 2048;
        break;
      case Preset::Large:
      default:
        cfg.nx = 256;
        cfg.ny = 256;
        cfg.nz = 1024;
        break;
    }
    return cfg;
  }

  [[nodiscard]] constexpr auto
  extent() const noexcept -> carbon::int64_3d {
    return {nx, ny, nz};
  }

  [[nodiscard]] constexpr auto
  radius() const noexcept -> float {
    return static_cast<float>(nz) / 24.0F;
  }

  [[nodiscard]] constexpr auto
  cx() const noexcept -> float {
    return 0.5F * static_cast<float>(nx - 1) + 2.5F;
  }

  [[nodiscard]] constexpr auto
  cy() const noexcept -> float {
    return 0.5F * static_cast<float>(ny - 1);
  }

  [[nodiscard]] constexpr auto
  cz() const noexcept -> float {
    return static_cast<float>(nz) / 4.0F;
  }

  [[nodiscard]] constexpr auto
  diameter() const noexcept -> float {
    return 2.0F * radius();
  }
};

struct ParamPacket {
  float re;
  float ulb;
  int32_t preset;
  int32_t shape;
  int32_t restart;
};

namespace detail {

[[nodiscard]] inline auto
make_d3q27_stencil() -> carbon::Stencil {
  std::vector<carbon::int32_3d> stencil_points;
  stencil_points.reserve(D3Q27::Q - 1);
  for (int q = 0; q < D3Q27::Q; ++q) {
    if (q == D3Q27::kCenter) {
      continue;
    }
    const auto cq = D3Q27::c(q);
    stencil_points.push_back({cq.x, cq.y, cq.z});
  }
  return carbon::Stencil(stencil_points, /*remove_center=*/true);
}

}  // namespace detail

class LiveSim {
public:
  LiveSim(carbon::System& sys,
          LiveSimConfig config)
      : sys_(sys),
        config_(std::move(config)),
        re_(clamp_re(config_.ulb,
                     config_.diameter(),
                     config_.re)),
        params_{config_.ulb,
                compute_omega(config_.ulb,
                              config_.diameter(),
                              re_)},
        stream_(sys_.default_stream()),
        dense_(sys_,
               config_.extent(),
               detail::make_d3q27_stencil(),
               carbon::BoundaryMode::Open),
        pop_a_(dense_.newField<float,
                               D3Q27::Q>(carbon::MemorySide::Device,
                                         "pop_a")),
        pop_b_(dense_.newField<float,
                               D3Q27::Q>(carbon::MemorySide::Device,
                                         "pop_b")),
        cells_(dense_.newField<int32_t,
                               1>(carbon::MemorySide::Device,
                                  "cells")),
        cells_scratch_(dense_.newField<int32_t,
                                       1>(carbon::MemorySide::Device,
                                          "cells_scratch")),
        velocity_(dense_
                      .newField<float,
                                3>(carbon::MemorySide::HostDevice)
                      .set_name("u")),
        density_(dense_
                     .newField<float,
                               1>(carbon::MemorySide::HostDevice)
                     .set_name("rho")),
        qfield_(dense_
                    .newField<float,
                              1>(carbon::MemorySide::HostDevice)
                    .set_name("q")),
        qpos_(dense_
                  .newField<float,
                            1>(carbon::MemorySide::Device)
                  .set_name("qpos")),
        sdf_(dense_
                 .newField<float,
                           1>(carbon::MemorySide::Device)
                 .set_name("sdf")),
        sk_ab_(sys_),
        sk_ba_(sys_),
        vel_halo_(velocity_),
        sdf_halo_(sdf_),
        q_halo_(qfield_),
        qpos_halo_(qpos_),
        rho_halo_(density_) {
    reset();
  }

  auto
  step(int n) -> void {
    for (int i = 0; i < n; ++i) {
      if (fresh_in_a_) {
        sk_ab_.run();
      } else {
        sk_ba_.run();
      }
      fresh_in_a_ = !fresh_in_a_;
      ++iteration_;
    }
    sk_ab_.wait();
    sk_ba_.wait();
    stream_.wait();
  }

  auto
  refresh_macroscopic() -> void {
    const carbon::Kernel& macro = fresh_in_a_ ? macro_a_ : macro_b_;
    macro.launch(stream_);
    stream_.wait();
    vel_halo_(stream_);
    stream_.wait();
  }

  /**
   * Derives Q-criterion from the current macroscopic fields. @ref refresh_macroscopic is a
   * precondition rather than something this does itself: every caller already runs it as part of
   * the normal per-frame sequence, so folding it in here ran the full-field macroscopic kernel
   * and its halo twice per frame in exactly the modes that are already the most expensive.
   */
  auto
  refresh_q() -> void {
    const float inv_ulb = 1.0F / params_.ulb;
    carbon::make_qcriterion_kernel(dense_, velocity_, qfield_, inv_ulb).launch(stream_);
    stream_.wait();
    q_halo_(stream_);
    stream_.wait();
    qpos_k_.launch(stream_);
    stream_.wait();
    qpos_halo_(stream_);
    stream_.wait();
    rho_halo_(stream_);
    stream_.wait();

    carbon::ReductionResult<float> q_pos_max(sys_);
    const float wake_z0 = config_.cz() - 1.5F * config_.radius();
    const float wake_z1 = config_.cz() + 20.0F * config_.radius();
    const float wake_x0 = std::max(0.0F, config_.cx() - 6.0F * config_.radius());
    const float wake_x1 =
        std::min(static_cast<float>(config_.nx), config_.cx() + 6.0F * config_.radius());
    const float wake_y0 = -1.0F;
    const float wake_y1 = static_cast<float>(config_.ny);
    const auto q_wake_k = dense_.make_reduce_kernel<float>(
        [this, wake_x0, wake_x1, wake_y0, wake_y1, wake_z0, wake_z1](carbon::KernelLoader& loader) {
          const auto field = loader.read(qpos_);
          return [field, wake_x0, wake_x1, wake_y0, wake_y1, wake_z0, wake_z1]
                 CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                   const carbon::int64_3d gc = span.global_location(idx);
                   const float x = static_cast<float>(gc.x);
                   const float y = static_cast<float>(gc.y);
                   const float z = static_cast<float>(gc.z);
                   if (x < wake_x0 || x > wake_x1 || y < wake_y0 || y > wake_y1 || z < wake_z0 ||
                       z > wake_z1) {
                     return 0.0F;
                   }
                   return field.read(span, idx, 0);
                 };
        },
        q_pos_max, carbon::reduce::Max<float>{});
    q_wake_k.launch(stream_);
    q_pos_max.sync(stream_);
    // This drives the Q colormap ceiling, and the raw per-frame wake maximum swings with vortex
    // shedding, so using it directly makes the whole image pulse in brightness. Track it with a
    // decaying peak-hold: jump straight to a new high, ease back down from it.
    const float wake_max = q_pos_max.host_value();
    q_wake_max_ = (wake_max >= q_wake_max_)
                      ? wake_max
                      : (k_q_ceiling_decay * q_wake_max_ + (1.0F - k_q_ceiling_decay) * wake_max);
  }

  auto
  refresh_rho_inf() -> void {
    carbon::ReductionResult<float> rho_sum(sys_);
    const float ref_z0 = config_.cz() - 5.0F * config_.radius();
    const float ref_z1 = config_.cz() - 3.0F * config_.radius();
    const auto rho_ref_k = dense_.make_reduce_kernel<float>(
        [this, ref_z0, ref_z1](carbon::KernelLoader& loader) {
          const auto field = loader.read(density_);
          return [field, ref_z0, ref_z1]
                 CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                   const carbon::int64_3d gc = span.global_location(idx);
                   const float z = static_cast<float>(gc.z);
                   return (z < ref_z0 || z > ref_z1) ? 0.0F : field.read(span, idx, 0);
                 };
        },
        rho_sum, carbon::reduce::Sum<float>{});
    rho_ref_k.launch(stream_);
    rho_sum.sync(stream_);
    const auto z_lo = static_cast<int64_t>(std::ceil(ref_z0));
    const auto z_hi = static_cast<int64_t>(std::floor(ref_z1));
    const double cells = static_cast<double>(z_hi - z_lo + 1) * static_cast<double>(config_.nx) *
                         static_cast<double>(config_.ny);
    rho_inf_ = static_cast<float>(static_cast<double>(rho_sum.host_value()) / cells);
    k_cp_scale_ = 1.0F / (1.5F * rho_inf_ * params_.ulb * params_.ulb);
  }

  auto
  set_physics(float re,
              float ulb) -> void {
    config_.re = re;
    config_.ulb = ulb;
    rebuild_kernels();
  }

  auto
  rebuild_kernels() -> void {
    re_ = clamp_re(config_.ulb, config_.diameter(), config_.re);
    params_ = LbmParameters{config_.ulb, compute_omega(config_.ulb, config_.diameter(), re_)};

    wall_k_ = make_compute_wall_mask_kernel<carbon::Dense, D3Q27>(dense_, cells_, cells_scratch_);
    macro_a_ = make_macroscopic_kernel<carbon::Dense, D3Q27>(dense_, pop_a_, cells_, density_,
                                                             velocity_, params_);
    macro_b_ = make_macroscopic_kernel<carbon::Dense, D3Q27>(dense_, pop_b_, cells_, density_,
                                                             velocity_, params_);
    qpos_k_ = dense_.make_kernel([this](carbon::KernelLoader& loader) {
      const auto q = loader.read(qfield_);
      auto out = loader.write(qpos_);
      return [q, out] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> void {
        const float v = q.read(span, idx, 0);
        out.write(span, idx, 0, (v > 0.0F) ? v : 0.0F);
      };
    });

    sk_ab_ = carbon::Skeleton(sys_);
    sk_ba_ = carbon::Skeleton(sys_);
    sk_ab_.add(make_iteration_kernel<carbon::Dense, D3Q27, BgkCollision>(
        dense_, pop_a_, pop_b_, cells_, params_, "wt_step_ab"));
    sk_ab_.add(build_inout_bc_kernel<D3Q27>(dense_, pop_b_, config_.extent(), params_.ulb,
                                            config_.cx(), config_.cy(), config_.cz(),
                                            config_.radius(), config_.shape));
    sk_ba_.add(make_iteration_kernel<carbon::Dense, D3Q27, BgkCollision>(
        dense_, pop_b_, pop_a_, cells_, params_, "wt_step_ba"));
    sk_ba_.add(build_inout_bc_kernel<D3Q27>(dense_, pop_a_, config_.extent(), params_.ulb,
                                            config_.cx(), config_.cy(), config_.cz(),
                                            config_.radius(), config_.shape));
    sk_ab_.compile();
    sk_ba_.compile();
  }

  auto
  reset() -> void {
    iteration_ = 0;
    fresh_in_a_ = true;
    rho_inf_ = 1.0F;
    k_cp_scale_ = 1.0F / (1.5F * rho_inf_ * params_.ulb * params_.ulb);
    q_wake_max_ = 0.0F;

    const float cx = config_.cx();
    const float cy = config_.cy();
    const float cz = config_.cz();
    const float radius = config_.radius();
    const ObstacleShape shape = config_.shape;
    const int64_t nspan = config_.ny;
    const float pert = config_.pert_amp;
    const float u_max = params_.ulb;

    // The seeding kernels write owned cells only. Zero the halo planes first so the pre-exchange
    // state is defined — the host-fill path this replaced inherited zeroed halos from the
    // zero-initialized host mirror, and the wall-mask stencil reads cells halos on a slab edge.
    zero_storage_<float, D3Q27::Q>(pop_a_);
    zero_storage_<float, D3Q27::Q>(pop_b_);
    zero_storage_<int32_t, 1>(cells_);
    zero_storage_<int32_t, 1>(cells_scratch_);
    stream_.wait();

    const auto pop_init_a =
        build_pop_init_kernel<D3Q27>(dense_, pop_a_, u_max, cx, cy, cz, radius, shape, nspan, pert);
    const auto pop_init_b =
        build_pop_init_kernel<D3Q27>(dense_, pop_b_, u_max, cx, cy, cz, radius, shape, nspan, pert);
    const auto cells_init = build_cells_init_kernel(dense_, cells_, cx, cy, cz, radius, shape);
    const auto scratch_init =
        build_cells_init_kernel(dense_, cells_scratch_, cx, cy, cz, radius, shape);
    const auto sdf_init = build_sdf_init_kernel(dense_, sdf_, cx, cy, cz, radius, shape);
    pop_init_a.launch(stream_);
    pop_init_b.launch(stream_);
    cells_init.launch(stream_);
    scratch_init.launch(stream_);
    sdf_init.launch(stream_);
    stream_.wait();
    sdf_halo_(stream_);
    stream_.wait();

    rebuild_kernels();

    carbon::Skeleton precompute(sys_);
    precompute.add(wall_k_);
    precompute.compile();
    precompute.run();
    precompute.wait();
    carbon::Transfer<int32_t>(sys_, cells_scratch_.gpu_buffers(), cells_.gpu_buffers())(stream_);
    stream_.wait();
  }

  [[nodiscard]] auto
  system() noexcept -> carbon::System& {
    return sys_;
  }

  [[nodiscard]] auto
  system() const noexcept -> const carbon::System& {
    return sys_;
  }

  [[nodiscard]] auto
  config() const noexcept -> const LiveSimConfig& {
    return config_;
  }

  [[nodiscard]] auto
  stream() const noexcept -> const carbon::Stream& {
    return stream_;
  }

  auto
  prepare_for_render() -> void {
    stream_.wait();
    vel_halo_(stream_);
    sdf_halo_(stream_);
    stream_.wait();
  }

  auto
  prepare_cp_render() -> void {
    prepare_for_render();
    rho_halo_(stream_);
    stream_.wait();
  }

  [[nodiscard]] auto
  dense() noexcept -> carbon::Dense& {
    return dense_;
  }

  [[nodiscard]] auto
  dense() const noexcept -> const carbon::Dense& {
    return dense_;
  }

  [[nodiscard]] auto
  pop_a() noexcept -> carbon::Dense::Field<float,
                                           D3Q27::Q>& {
    return pop_a_;
  }

  [[nodiscard]] auto
  pop_b() noexcept -> carbon::Dense::Field<float,
                                           D3Q27::Q>& {
    return pop_b_;
  }

  [[nodiscard]] auto
  cells() noexcept -> carbon::Dense::Field<int32_t,
                                           1>& {
    return cells_;
  }

  [[nodiscard]] auto
  velocity() noexcept -> carbon::Dense::Field<float,
                                              3>& {
    return velocity_;
  }

  [[nodiscard]] auto
  density() noexcept -> carbon::Dense::Field<float,
                                             1>& {
    return density_;
  }

  [[nodiscard]] auto
  qfield() noexcept -> carbon::Dense::Field<float,
                                            1>& {
    return qfield_;
  }

  [[nodiscard]] auto
  qpos() noexcept -> carbon::Dense::Field<float,
                                          1>& {
    return qpos_;
  }

  [[nodiscard]] auto
  sdf() noexcept -> carbon::Dense::Field<float,
                                         1>& {
    return sdf_;
  }

  [[nodiscard]] auto
  params() const noexcept -> LbmParameters {
    return params_;
  }

  [[nodiscard]] auto
  re() const noexcept -> float {
    return re_;
  }

  [[nodiscard]] auto
  rho_inf() const noexcept -> float {
    return rho_inf_;
  }

  [[nodiscard]] auto
  k_cp_scale() const noexcept -> float {
    return k_cp_scale_;
  }

  [[nodiscard]] auto
  q_wake_max() const noexcept -> float {
    return q_wake_max_;
  }

  [[nodiscard]] auto
  iteration() const noexcept -> int64_t {
    return iteration_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] auto
  fresh_in_a() const noexcept -> bool {
    return fresh_in_a_;
  }

private:
  template <typename T,
            int Components>
  auto
  zero_storage_(carbon::Dense::Field<T,
                                     Components>& field) -> void {
    const auto& spans = dense_.spans();
    for (size_t g = 0; g < spans.size(); ++g) {
      const size_t bytes = static_cast<size_t>(spans[g].linear_storage_elements()) *
                           static_cast<size_t>(Components) * sizeof(T);
      sys_.async_memset(field.gpu_buffers()[g], 0, bytes, stream_);
    }
  }

  carbon::System& sys_;
  LiveSimConfig config_;
  float re_;
  LbmParameters params_;
  carbon::Stream stream_;
  carbon::Dense dense_;
  carbon::Dense::Field<float, D3Q27::Q> pop_a_;
  carbon::Dense::Field<float, D3Q27::Q> pop_b_;
  carbon::Dense::Field<int32_t, 1> cells_;
  carbon::Dense::Field<int32_t, 1> cells_scratch_;
  carbon::Dense::Field<float, 3> velocity_;
  carbon::Dense::Field<float, 1> density_;
  carbon::Dense::Field<float, 1> qfield_;
  carbon::Dense::Field<float, 1> qpos_;
  carbon::Dense::Field<float, 1> sdf_;
  carbon::Skeleton sk_ab_;
  carbon::Skeleton sk_ba_;
  carbon::Kernel wall_k_;
  carbon::Kernel macro_a_;
  carbon::Kernel macro_b_;
  carbon::Kernel qpos_k_;
  carbon::Halo<float, 3> vel_halo_;
  carbon::Halo<float, 1> sdf_halo_;
  carbon::Halo<float, 1> q_halo_;
  carbon::Halo<float, 1> qpos_halo_;
  carbon::Halo<float, 1> rho_halo_;
  static constexpr float k_q_ceiling_decay = 0.92F;

  bool fresh_in_a_{true};
  std::atomic<int64_t> iteration_{0};
  float rho_inf_{1.0F};
  float k_cp_scale_{0.0F};
  float q_wake_max_{0.0F};
};

namespace detail {

inline constexpr float k_sdf_edge_width = 2.0F;
inline constexpr float k_cp_probe_radius = 2.5F;
inline constexpr float k_cp_min = -2.5F;
inline constexpr float k_cp_max = 1.0F;
inline constexpr float k_q_gain = 1.0F;

[[nodiscard]] inline auto
ray_march_max_steps_for_clip(const carbon::float32_3d& clip_min,
                             const carbon::float32_3d& clip_max,
                             float step_size,
                             int margin = 64) -> int {
  const float dx = clip_max.x - clip_min.x;
  const float dy = clip_max.y - clip_min.y;
  const float dz = clip_max.z - clip_min.z;
  const float diag = std::sqrt(dx * dx + dy * dy + dz * dz);
  const float step = (step_size > 0.0F) ? step_size : 0.5F;
  const int steps = static_cast<int>(std::ceil(diag / step)) + margin;
  return std::clamp(steps, 512, 4096);
}

[[nodiscard]] inline auto
build_vel_opts(const carbon::float32_3d& clip_min,
               const carbon::float32_3d& clip_max) -> carbon::RayMarchOptions {
  carbon::RayMarchOptions opts{};
  opts.step_size = 0.0F;
  opts.max_steps = ray_march_max_steps_for_clip(clip_min, clip_max, opts.step_size);
  opts.shading = carbon::RayMarchShading::ScalarField;
  opts.colormap = carbon::Colormap::Jet;
  opts.value_threshold = 0.0F;
  opts.value_min = 0.0F;
  opts.value_max = 1.5F;
  opts.opacity_scale = 1.8F;
  opts.use_clip_bounds = true;
  opts.clip_bounds_min = clip_min;
  opts.clip_bounds_max = clip_max;
  return opts;
}

[[nodiscard]] inline auto
build_q_opts(const LiveSim& sim,
             const carbon::float32_3d& clip_min,
             const carbon::float32_3d& clip_max) -> carbon::RayMarchOptions {
  carbon::RayMarchOptions opts{};
  opts.max_steps = 1024;
  const float q_ceil = (sim.q_wake_max() > 0.0F) ? (sim.q_wake_max() * k_q_gain * 0.5F) : 1.0F;
  opts.shading = carbon::RayMarchShading::ScalarField;
  opts.colormap = carbon::Colormap::Viridis;
  opts.value_threshold = 0.0F;
  opts.value_min = 0.0F;
  opts.value_max = q_ceil;
  opts.opacity_scale = 200.0F;
  opts.step_size = 0.15F;
  opts.use_clip_bounds = true;
  const float q_x0 = std::max(clip_min.x, sim.config().cx() - 5.0F * sim.config().radius());
  const float q_x1 = std::min(clip_max.x, sim.config().cx() + 5.0F * sim.config().radius());
  opts.clip_bounds_min = {q_x0, clip_min.y, clip_min.z};
  opts.clip_bounds_max = {q_x1, clip_max.y, clip_max.z};
  return opts;
}

[[nodiscard]] inline auto
build_q_tint_opts(const carbon::RayMarchOptions& q_opts) -> carbon::RayMarchOptions {
  carbon::RayMarchOptions opts = q_opts;
  opts.shading = carbon::RayMarchShading::ScalarFieldAttribute;
  opts.colormap = carbon::Colormap::Jet;
  opts.value_min = 0.0F;
  opts.value_max = 1.5F;
  return opts;
}

[[nodiscard]] inline auto
build_wake_opts(const carbon::float32_3d& clip_min,
                const carbon::float32_3d& clip_max) -> carbon::RayMarchOptions {
  carbon::RayMarchOptions opts{};
  opts.step_size = 0.0F;
  opts.max_steps = ray_march_max_steps_for_clip(clip_min, clip_max, opts.step_size);
  opts.shading = carbon::RayMarchShading::ScalarFieldAttribute;
  opts.colormap = carbon::Colormap::CoolWarm;
  opts.value_threshold = 0.05F;
  opts.value_min = -0.8F;
  opts.value_max = 0.8F;
  opts.opacity_scale = 3.0F;
  opts.use_clip_bounds = true;
  opts.clip_bounds_min = clip_min;
  opts.clip_bounds_max = clip_max;
  return opts;
}

[[nodiscard]] inline auto
build_sphere_opts(const carbon::float32_3d& clip_min,
                  const carbon::float32_3d& clip_max) -> carbon::RayMarchOptions {
  carbon::RayMarchOptions opts{};
  opts.step_size = 0.0F;
  opts.absorption_scale = 0.55F;
  opts.max_steps = ray_march_max_steps_for_clip(clip_min, clip_max, opts.step_size);
  opts.shading = carbon::RayMarchShading::SurfaceBlinn;
  opts.light_dir = {0.35F, -0.55F, 0.75F};
  opts.ambient = 0.22F;
  opts.diffuse = 0.62F;
  opts.specular = 0.18F;
  opts.shininess = 48.0F;
  opts.isosurface_width = k_sdf_edge_width;
  opts.material_color = {0.78F, 0.80F, 0.84F};
  opts.use_clip_bounds = true;
  opts.clip_bounds_min = clip_min;
  opts.clip_bounds_max = clip_max;
  return opts;
}

[[nodiscard]] inline auto
render_mode_names() -> std::vector<std::string> {
  return {"Speed", "Q-criterion", "Wake (u - u_inf)", "Q-criterion (speed-tinted)", "Surface Cp"};
}

[[nodiscard]] inline auto
style_layers(int mode) -> Layers {
  switch (mode) {
    case 1:
      return Layers::Q;
    case 2:
      return Layers::Wake;
    case 3:
      return Layers::QTinted;
    default:
      return Layers::Speed;
  }
}

}  // namespace detail

[[nodiscard]] inline auto
view_frame_bounds(const LiveSimConfig& cfg) -> std::pair<carbon::float32_3d,
                                                         carbon::float32_3d> {
  const float cx = cfg.cx();
  const float cy = cfg.cy();
  const float cz = cfg.cz();
  const float r = cfg.radius();
  const float z0 = std::max(0.0F, cz - 1.5F * r);
  const float z1 = std::min(static_cast<float>(cfg.nz), cz + 10.0F * r);
  const float y0 = std::max(0.0F, cy - 5.0F * r);
  const float y1 = std::min(static_cast<float>(cfg.ny), cy + 5.0F * r);
  const float x0 = std::max(0.0F, cx - 5.0F * r);
  const float x1 = std::min(static_cast<float>(cfg.nx), cx + 5.0F * r);
  return {{x0, y0, z0}, {x1, y1, z1}};
}

[[nodiscard]] inline auto
dense_world_bounds(const carbon::Dense& dense) -> std::pair<carbon::float32_3d,
                                                            carbon::float32_3d> {
  const carbon::int64_3d& global = dense.global_extent();
  const carbon::float32_3d origin = dense.spans()[0].origin();
  const float spacing = dense.spans()[0].spacing();
  return {origin,
          {origin.x + spacing * static_cast<float>(global.x),
           origin.y + spacing * static_cast<float>(global.y),
           origin.z + spacing * static_cast<float>(global.z)}};
}

[[nodiscard]] inline auto
gpu_slab_boxes(const carbon::Dense& dense) -> std::vector<std::pair<carbon::float32_3d,
                                                                    carbon::float32_3d>> {
  std::vector<std::pair<carbon::float32_3d, carbon::float32_3d>> boxes;
  const size_t n = dense.gpu_count();
  if (n == 0) {
    return boxes;
  }
  const carbon::int64_3d& global = dense.global_extent();
  const carbon::float32_3d origin = dense.spans()[0].origin();
  const float spacing = dense.spans()[0].spacing();
  boxes.reserve(n);
  for (size_t g = 0; g < n; ++g) {
    const int64_t z0 = dense.z_offset_global(g);
    const int64_t lz = dense.local_z(g);
    boxes.emplace_back(
        carbon::float32_3d{origin.x, origin.y, origin.z + spacing * static_cast<float>(z0)},
        carbon::float32_3d{origin.x + spacing * static_cast<float>(global.x),
                           origin.y + spacing * static_cast<float>(global.y),
                           origin.z + spacing * static_cast<float>(z0 + lz)});
  }
  return boxes;
}

struct WindTunnelRenderContext {
  carbon::float32_3d bounds_min{};
  carbon::float32_3d bounds_max{};
  carbon::float32_3d clip_min{};
  carbon::float32_3d clip_max{};
  carbon::float32_3d background{1.0F, 1.0F, 1.0F};
  carbon::float32_3d free_stream{};

  carbon::RayMarchOptions vel_opts{};
  carbon::RayMarchOptions q_opts{};
  carbon::RayMarchOptions q_tint_opts{};
  carbon::RayMarchOptions wake_opts{};
  carbon::RayMarchOptions sphere_opts{};

  std::vector<std::string> render_mode_names;

  [[nodiscard]] static auto
  build(LiveSim& sim) -> WindTunnelRenderContext {
    WindTunnelRenderContext ctx;
    ctx.sim_ = &sim;
    const LiveSimConfig& cfg = sim.config();
    ctx.cx_ = cfg.cx();
    ctx.cy_ = cfg.cy();
    ctx.radius_ = cfg.radius();
    ctx.shape_ = cfg.shape;
    ctx.cz_ = cfg.cz();
    ctx.inv_ulb_ = 1.0F / sim.params().ulb;

    const auto grid_bounds = dense_world_bounds(sim.dense());
    ctx.bounds_min = grid_bounds.first;
    ctx.bounds_max = grid_bounds.second;
    ctx.clip_min = grid_bounds.first;
    ctx.clip_max = grid_bounds.second;
    ctx.free_stream = {0.0F, 0.0F, sim.params().ulb};

    ctx.vel_opts = detail::build_vel_opts(ctx.clip_min, ctx.clip_max);
    ctx.q_opts = detail::build_q_opts(sim, ctx.clip_min, ctx.clip_max);
    ctx.q_tint_opts = detail::build_q_tint_opts(ctx.q_opts);
    ctx.wake_opts = detail::build_wake_opts(ctx.clip_min, ctx.clip_max);
    ctx.sphere_opts = detail::build_sphere_opts(ctx.clip_min, ctx.clip_max);
    ctx.render_mode_names = detail::render_mode_names();

    return ctx;
  }

private:
  LiveSim* sim_{nullptr};
  float inv_ulb_{1.0F};
  float cx_{0.0F};
  float cy_{0.0F};
  float radius_{0.0F};
  ObstacleShape shape_{ObstacleShape::Cylinder};
  float cz_{0.0F};

public:
  auto
  render_cp_for_camera(const carbon::Camera& camera,
                       carbon::SurfelImage& cp_out) -> void {
    LiveSim& sim = *sim_;
    sim.prepare_cp_render();
    carbon::Stream stream = sim.stream();
    const LiveSimConfig& local_cfg = sim.config();
    const float local_cx = local_cfg.cx();
    const float local_cy = local_cfg.cy();
    const float local_radius = local_cfg.radius();
    const float local_cz = local_cfg.cz();
    const ObstacleShape local_shape = local_cfg.shape;
    cp_out.clear(stream);
    const auto cp_splat = carbon::make_surfel_kernel(
        sim.dense(), cp_out, camera,
        [&sim, local_cx, local_cy, local_cz, local_radius,
         local_shape](carbon::KernelLoader& loader) {
          const auto rho = loader.read(sim.density());
          const float cp_scale = sim.k_cp_scale();
          const float rho_inf = sim.rho_inf();
          return
              [rho, local_cx, local_cy, local_cz, local_radius, local_shape, cp_scale, rho_inf]
              CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> carbon::SurfelSample {
                const carbon::int64_3d gc = span.global_location(idx);
                const float px = static_cast<float>(gc.x) + 0.5F;
                const float py = static_cast<float>(gc.y) + 0.5F;
                const float pz = static_cast<float>(gc.z) + 0.5F;
                carbon::SurfelSample sample{};
                sample.sdf = obstacle_sdf(px, py, pz, local_shape, local_cx, local_cy, local_cz,
                                          local_radius);
                sample.normal =
                    obstacle_surface_normal(px, py, pz, local_shape, local_cx, local_cy, local_cz);
                const float probe_r = local_radius + detail::k_cp_probe_radius;
                float ox = sample.normal.x * probe_r;
                float oy = sample.normal.y * probe_r;
                float oz = sample.normal.z * probe_r;
                const float olen = std::sqrt(ox * ox + oy * oy + oz * oz);
                constexpr float k_halo_probe = 1.0F;
                if (olen > k_halo_probe) {
                  const float scale = k_halo_probe / olen;
                  ox *= scale;
                  oy *= scale;
                  oz *= scale;
                }
                const carbon::int8_3d probe{
                    static_cast<int8_t>(std::lround(px + ox - 0.5F) - gc.x),
                    static_cast<int8_t>(std::lround(py + oy - 0.5F) - gc.y),
                    static_cast<int8_t>(std::lround(pz + oz - 0.5F) - gc.z),
                };
                const carbon::NeighbourValue<float> probed = rho.neighbour(span, idx, probe, 0);
                const float rho_local = rho.read(span, idx, 0);
                const float rho_val = probed.valid ? probed.value : rho_local;
                sample.attribute = (rho_val - rho_inf) * cp_scale;
                return sample;
              };
        },
        /*band_width=*/1.5F, /*splat_radius_px=*/8.0F);
    cp_splat.launch(stream);
    cp_out.combine(stream);
    stream.wait();
  }

  auto
  render_for_camera(const carbon::Camera& camera,
                    carbon::RayMarchImage& vel_out,
                    carbon::RayMarchImage& q_out,
                    carbon::RayMarchImage& sphere_out,
                    Layers layers) -> void {
    LiveSim& sim = *sim_;
    sim.prepare_for_render();
    carbon::Stream stream = sim.stream();
    carbon::Dense& dense = sim.dense();
    auto& velocity = sim.velocity();
    auto& qpos = sim.qpos();
    auto& sdf = sim.sdf();
    const float inv_ulb = inv_ulb_;
    const float cx = cx_;
    const float cy = cy_;
    const float radius = radius_;
    const float cz = cz_;
    const ObstacleShape shape = shape_;
    const carbon::float32_3d free_stream_fs = free_stream;
    constexpr float q_gain = detail::k_q_gain;

    const bool want_speed = (layers == Layers::Speed);
    const bool want_wake = (layers == Layers::Wake);
    const bool want_q = (layers == Layers::Q);
    const bool want_q_tint = (layers == Layers::QTinted);

    if (want_speed || want_wake) {
      vel_out.clear(stream);
    }
    if (want_q || want_q_tint) {
      q_out.clear(stream);
    }
    sphere_out.clear(stream);

    if (want_speed) {
      const auto vel_march = carbon::make_ray_march_kernel(
          dense, vel_out, camera, vel_opts, [&velocity, inv_ulb](carbon::KernelLoader& loader) {
            const auto field = loader.read(velocity);
            return [field, inv_ulb]
                   CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                     const float vx = field.read(span, idx, 0);
                     const float vy = field.read(span, idx, 1);
                     const float vz = field.read(span, idx, 2);
                     return std::sqrt(vx * vx + vy * vy + vz * vz) * inv_ulb;
                   };
          });
      vel_march.launch(stream);
      vel_out.combine(stream);
      vel_out.dist_combine();
    }

    if (want_wake) {
      const auto wake_march = carbon::make_ray_march_kernel(
          dense, vel_out, camera, wake_opts,
          [&velocity, inv_ulb, free_stream_fs](carbon::KernelLoader& loader) {
            const auto field = loader.read(velocity);
            return [field, inv_ulb, free_stream_fs]
                   CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                     const carbon::float32_3d u{field.read(span, idx, 0), field.read(span, idx, 1),
                                                field.read(span, idx, 2)};
                     return carbon::render::speed_deviation(u, free_stream_fs, inv_ulb);
                   };
          },
          [&velocity, inv_ulb, free_stream_fs](carbon::KernelLoader& loader) {
            const auto field = loader.read(velocity);
            return [field, inv_ulb, free_stream_fs]
                   CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                     return (field.read(span, idx, 2) - free_stream_fs.z) * inv_ulb;
                   };
          });
      wake_march.launch(stream);
      vel_out.combine(stream);
      vel_out.dist_combine();
    }

    if (want_q_tint) {
      const auto q_tint_march = carbon::make_ray_march_kernel(
          dense, q_out, camera, q_tint_opts,
          [&qpos, cx, cy, cz, radius, shape, q_gain](carbon::KernelLoader& loader) {
            const auto field = loader.read(qpos);
            return [field, cx, cy, cz, radius, shape, q_gain]
                   CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                     const carbon::int64_3d gc = span.global_location(idx);
                     if (inside_obstacle(gc, shape, cx, cy, cz, radius)) {
                       return 0.0F;
                     }
                     const float q = field.read(span, idx, 0);
                     return (q > 0.0F) ? (q * q_gain) : 0.0F;
                   };
          },
          [&velocity, inv_ulb](carbon::KernelLoader& loader) {
            const auto field = loader.read(velocity);
            return [field, inv_ulb]
                   CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                     const float vx = field.read(span, idx, 0);
                     const float vy = field.read(span, idx, 1);
                     const float vz = field.read(span, idx, 2);
                     return std::sqrt(vx * vx + vy * vy + vz * vz) * inv_ulb;
                   };
          });
      q_tint_march.launch(stream);
      q_out.combine(stream);
      q_out.dist_combine();
    }

    if (want_q) {
      q_opts = detail::build_q_opts(sim, clip_min, clip_max);
      const auto q_march = carbon::make_ray_march_kernel(
          dense, q_out, camera, q_opts,
          [&qpos, cx, cy, cz, radius, shape, q_gain](carbon::KernelLoader& loader) {
            const auto field = loader.read(qpos);
            return [field, cx, cy, cz, radius, shape, q_gain]
                   CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                     const carbon::int64_3d gc = span.global_location(idx);
                     if (inside_obstacle(gc, shape, cx, cy, cz, radius)) {
                       return 0.0F;
                     }
                     const float q = field.read(span, idx, 0);
                     if (q <= 0.0F) {
                       return 0.0F;
                     }
                     return q * q_gain;
                   };
          });
      q_march.launch(stream);
      q_out.combine(stream);
      q_out.dist_combine();
    }

    const auto sphere_march = carbon::make_ray_march_kernel(
        dense, sphere_out, camera, sphere_opts, [&sdf](carbon::KernelLoader& loader) {
          const auto field = loader.read(sdf);
          return [field] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
            return field.read(span, idx, 0);
          };
        });
    sphere_march.launch(stream);
    sphere_out.combine(stream);
    sphere_out.dist_combine();
    stream.wait();
  }
};

}  // namespace carbon::examples::wind_tunnel
