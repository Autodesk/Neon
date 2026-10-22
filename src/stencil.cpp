#include <carbon/logging.h>
#include <carbon/stencil.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace carbon {

namespace {

[[nodiscard]] auto
fits_int8(std::int32_t v) noexcept -> bool {
  return v >= static_cast<std::int32_t>(std::numeric_limits<std::int8_t>::min()) &&
         v <= static_cast<std::int32_t>(std::numeric_limits<std::int8_t>::max());
}

}  // namespace

Stencil::Stencil(const std::vector<int32_3d>& points,
                 bool remove_center) {
  points_.reserve(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    const int32_3d& p = points[i];
    if (!fits_int8(p.x) || !fits_int8(p.y) || !fits_int8(p.z)) {
      log::throw_with_log<std::out_of_range>(
          "Stencil: offset {} = ({}, {}, {}) does not fit int8_t [-128, 127]; carbon stencils "
          "are stored as int8_3d (Chebyshev radius up to 127)",
          i, p.x, p.y, p.z);
    }
    points_.push_back(int8_3d{static_cast<std::int8_t>(p.x), static_cast<std::int8_t>(p.y),
                              static_cast<std::int8_t>(p.z)});
  }
  if (remove_center) {
    points_.erase(std::remove_if(points_.begin(), points_.end(),
                                 [](const int8_3d& p) noexcept {
                                   return p.x == 0 && p.y == 0 && p.z == 0;
                                 }),
                  points_.end());
  }
  radius_ = compute_radius_(points_);
}

auto
Stencil::points() const noexcept -> const std::vector<int8_3d>& {
  return points_;
}

auto
Stencil::size() const noexcept -> size_t {
  return points_.size();
}

auto
Stencil::empty() const noexcept -> bool {
  return points_.empty();
}

auto
Stencil::radius() const noexcept -> std::int32_t {
  return radius_;
}

auto
Stencil::lattice_profile() const noexcept -> LatticeProfile {
  LatticeProfile p{};
  bool in_pos_run = false;
  bool in_neg_run = false;
  int cur_pos_run = 0;
  int cur_neg_run = 0;
  for (const int8_3d& s : points_) {
    const int z = static_cast<int>(s.z);
    if (z > 0) {
      ++p.positive_z;
      if (!in_pos_run) {
        ++p.run_count_positive;
        cur_pos_run = 0;
      }
      ++cur_pos_run;
      p.max_run_positive = std::max(p.max_run_positive, cur_pos_run);
      in_pos_run = true;
      in_neg_run = false;
    } else if (z < 0) {
      ++p.negative_z;
      if (!in_neg_run) {
        ++p.run_count_negative;
        cur_neg_run = 0;
      }
      ++cur_neg_run;
      p.max_run_negative = std::max(p.max_run_negative, cur_neg_run);
      in_neg_run = true;
      in_pos_run = false;
    } else {
      ++p.zero_z;
      in_pos_run = false;
      in_neg_run = false;
    }
  }
  p.fully_groupable = (p.positive_z == 0 || p.run_count_positive == 1) &&
                      (p.negative_z == 0 || p.run_count_negative == 1);
  return p;
}

auto
Stencil::reorder_for_lattice_fusion(const std::vector<int32_3d>& offsets)
    -> Stencil::LatticeReorderResult {
  LatticeReorderResult out{};
  out.reordered.reserve(offsets.size());
  out.permutation.reserve(offsets.size());
  // Bucket the input by sign(c.z), preserving the relative order within each bucket
  // (stable sort behaviour). We do three linear passes -- the data is small and the
  // resulting code is easier to read than std::stable_sort with a comparator that
  // also has to deal with std::iota'd indices.
  for (size_t i = 0; i < offsets.size(); ++i) {
    if (offsets[i].z < 0) {
      out.reordered.push_back(offsets[i]);
      out.permutation.push_back(i);
    }
  }
  for (size_t i = 0; i < offsets.size(); ++i) {
    if (offsets[i].z == 0) {
      out.reordered.push_back(offsets[i]);
      out.permutation.push_back(i);
    }
  }
  for (size_t i = 0; i < offsets.size(); ++i) {
    if (offsets[i].z > 0) {
      out.reordered.push_back(offsets[i]);
      out.permutation.push_back(i);
    }
  }
  return out;
}

auto
to_string(StencilSemantic semantic) noexcept -> std::string_view {
  switch (semantic) {
    case StencilSemantic::Standard:
      return "standard";
    case StencilSemantic::Lattice:
      return "lattice";
  }
  return "unknown";
}

auto
Stencil::compute_radius_(const std::vector<int8_3d>& points) noexcept -> std::int32_t {
  std::int32_t r = 0;
  for (const int8_3d& p : points) {
    r = std::max(r, std::abs(static_cast<std::int32_t>(p.x)));
    r = std::max(r, std::abs(static_cast<std::int32_t>(p.y)));
    r = std::max(r, std::abs(static_cast<std::int32_t>(p.z)));
  }
  return r;
}

}  // namespace carbon
