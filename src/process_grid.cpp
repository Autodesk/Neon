#include <carbon/process_grid.h>

#include <carbon/logging.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace carbon {

namespace {

constexpr std::array<int, 3> kAxisOrderZYX{2, 1, 0};

[[nodiscard]] auto
prime_factors_descending(int32_t n) -> std::vector<int32_t> {
  std::vector<int32_t> factors;
  for (int32_t p = 2; static_cast<int64_t>(p) * p <= n; ++p) {
    while (n % p == 0) {
      factors.push_back(p);
      n /= p;
    }
  }
  if (n > 1) {
    factors.push_back(n);
  }
  std::sort(factors.begin(), factors.end(), [](int32_t a, int32_t b) {
    return a > b;
  });
  return factors;
}

[[nodiscard]] auto
factorized_dims(int32_t world_size,
                DecompositionMode mode) -> int32_3d {
  std::array<int32_t, 3> dims{1, 1, 1};
  std::array<bool, 3> free_axis{false, false, false};

  switch (mode) {
    case DecompositionMode::Slab:
      free_axis[2] = true;
      break;
    case DecompositionMode::Pencil:
      free_axis[1] = true;
      free_axis[2] = true;
      break;
    case DecompositionMode::Brick:
      free_axis[0] = true;
      free_axis[1] = true;
      free_axis[2] = true;
      break;
    case DecompositionMode::Custom:
      log::throw_with_log<std::invalid_argument>(
          "carbon::ProcessGrid::make: Custom mode requires ProcessGrid::custom()");
      return {1, 1, 1};
  }

  for (const int32_t factor : prime_factors_descending(world_size)) {
    int best_axis = -1;
    for (const int axis : kAxisOrderZYX) {
      if (!free_axis[axis]) {
        continue;
      }
      if (best_axis < 0 || dims[axis] < dims[best_axis]) {
        best_axis = axis;
      }
    }
    if (best_axis < 0) {
      log::throw_with_log<std::invalid_argument>(
          "carbon::ProcessGrid::make: decomposition mode {} has no free axis",
          static_cast<int>(mode));
      return {1, 1, 1};
    }
    dims[best_axis] *= factor;
  }

  return {dims[0], dims[1], dims[2]};
}

}  // namespace

auto
ProcessGrid::make(int32_t world_size,
                  DecompositionMode mode) -> ProcessGrid {
  if (world_size <= 0) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::ProcessGrid::make: world_size must be > 0 (got {})", world_size);
    return ProcessGrid{};
  }
  if (mode == DecompositionMode::Custom) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::ProcessGrid::make: Custom mode requires ProcessGrid::custom()");
    return ProcessGrid{};
  }
  return ProcessGrid{factorized_dims(world_size, mode), mode};
}

auto
ProcessGrid::custom(int32_t world_size,
                    int32_t px,
                    int32_t py,
                    int32_t pz) -> ProcessGrid {
  if (world_size <= 0) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::ProcessGrid::custom: world_size must be > 0 (got {})", world_size);
    return ProcessGrid{};
  }
  if (px <= 0 || py <= 0 || pz <= 0) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::ProcessGrid::custom: all dimensions must be > 0 (got px={}, py={}, pz={})",
        px,
        py,
        pz);
    return ProcessGrid{};
  }

  const int64_t xy = static_cast<int64_t>(px) * py;
  if (xy > world_size) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::ProcessGrid::custom: px*py ({}) exceeds world_size ({})", xy, world_size);
    return ProcessGrid{};
  }
  const int64_t xyz = xy * pz;
  if (xyz != world_size) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::ProcessGrid::custom: px*py*pz ({}) != world_size ({})", xyz, world_size);
    return ProcessGrid{};
  }

  return ProcessGrid{int32_3d{px, py, pz}, DecompositionMode::Custom};
}

auto
ProcessGrid::coords_of(int32_t rank) const -> int32_3d {
  if (rank < 0 || rank >= world_size()) {
    log::throw_with_log<std::out_of_range>(
        "carbon::ProcessGrid::coords_of: rank {} out of range [0, {})", rank, world_size());
    return {-1, -1, -1};
  }

  const int32_t pz = rank % dims_.z;
  const int32_t py = (rank / dims_.z) % dims_.y;
  const int32_t px = rank / (dims_.y * dims_.z);
  return {px, py, pz};
}

auto
ProcessGrid::rank_of(int32_3d coords) const -> int32_t {
  if (coords.x < 0 || coords.x >= dims_.x || coords.y < 0 || coords.y >= dims_.y ||
      coords.z < 0 || coords.z >= dims_.z) {
    log::throw_with_log<std::out_of_range>(
        "carbon::ProcessGrid::rank_of: coords ({}, {}, {}) outside grid ({}, {}, {})",
        coords.x,
        coords.y,
        coords.z,
        dims_.x,
        dims_.y,
        dims_.z);
    return -1;
  }

  return static_cast<int32_t>((static_cast<int64_t>(coords.x) * dims_.y + coords.y) * dims_.z +
                              coords.z);
}

}  // namespace carbon