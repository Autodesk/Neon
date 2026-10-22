#include <carbon/carbon.h>
#include <carbon/stencil.h>

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

int
main() {
  using carbon::int32_3d;
  using carbon::int8_3d;
  using carbon::Stencil;

  {
    const Stencil empty;
    assert(empty.empty());
    assert(empty.size() == 0);
    assert(empty.radius() == 0);
    assert(empty.points().empty());
  }

  {
    const std::vector<int32_3d> pts{
        {0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
    };
    // Stencil stores int8_3d; compare element-by-element via the int32 input.
    const Stencil s(pts);
    assert(!s.empty());
    assert(s.size() == pts.size());
    assert(s.points().size() == pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
      assert(s.points()[i].x == static_cast<std::int8_t>(pts[i].x));
      assert(s.points()[i].y == static_cast<std::int8_t>(pts[i].y));
      assert(s.points()[i].z == static_cast<std::int8_t>(pts[i].z));
    }
    assert(s.radius() == 1);
  }

  {
    const std::vector<int32_3d> pts{
        {0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
    };

    const Stencil s(pts, true);
    assert(s.size() == 6);
    for (const int8_3d& p : s.points()) {
      assert(!(p.x == 0 && p.y == 0 && p.z == 0));
    }
    assert(s.radius() == 1);
  }

  {
    std::vector<int32_3d> pts;
    for (std::int32_t dz = -1; dz <= 1; ++dz) {
      for (std::int32_t dy = -1; dy <= 1; ++dy) {
        for (std::int32_t dx = -1; dx <= 1; ++dx) {
          pts.push_back({dx, dy, dz});
        }
      }
    }
    const Stencil full(pts);
    assert(full.size() == 27);
    assert(full.radius() == 1);

    const Stencil no_center(pts, true);
    assert(no_center.size() == 26);
    assert(no_center.radius() == 1);
  }

  {
    const std::vector<int32_3d> pts{
        {-3, 0, 0}, {3, 0, 0}, {0, -2, 0}, {0, 2, 0}, {0, 0, -5}, {0, 0, 5},
    };
    const Stencil s(pts);
    assert(s.radius() == 5);
  }

  {
    const std::vector<int32_3d> pts{
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
    };
    const Stencil center_only(pts);
    assert(center_only.size() == 3);
    assert(center_only.radius() == 0);

    const Stencil emptied(pts, true);
    assert(emptied.empty());
    assert(emptied.radius() == 0);
  }

  {
    const std::vector<int32_3d> pts{
        {-4, 0, 0}, {0, 0, 0}, {0, 3, 0}, {0, 0, 0}, {0, 0, 2}, {0, 0, 0},
    };
    const Stencil s(pts, true);
    assert(s.size() == 3);
    assert((s.points()[0] == int8_3d{-4, 0, 0}));
    assert((s.points()[1] == int8_3d{0, 3, 0}));
    assert((s.points()[2] == int8_3d{0, 0, 2}));
    assert(s.radius() == 4);
  }

  // Reject offsets that don't fit int8_t. The check runs on the raw input before
  // remove_center, so a too-large offset throws even when remove_center is requested.
  {
    const std::vector<int32_3d> pts{{128, 0, 0}};
    bool threw = false;
    try {
      Stencil s(pts);
      (void)s;
    } catch (const std::out_of_range&) {
      threw = true;
    }
    assert(threw);
  }
  {
    const std::vector<int32_3d> pts{{0, 0, -129}};
    bool threw = false;
    try {
      Stencil s(pts, /*remove_center=*/true);
      (void)s;
    } catch (const std::out_of_range&) {
      threw = true;
    }
    assert(threw);
  }
  // Boundary values: ±127 fit, ±128 does not.
  {
    const std::vector<int32_3d> pts{{127, -127, 127}};
    const Stencil s(pts);
    assert(s.radius() == 127);
  }

  // Stencil::reorder_for_lattice_fusion: D3Q19 in a "natural" non-grouped order ->
  // grouped output, run-length statistics confirm fully_groupable.
  {
    // 19 velocities laid out so c.z signs are interleaved (worst case for fusion).
    const std::vector<int32_3d> natural{
        {0, 0, 0},    // c.z = 0
        {1, 0, 0},    // c.z = 0
        {0, 0, 1},    // c.z = +1
        {-1, 0, 0},   // c.z = 0
        {0, 0, -1},   // c.z = -1
        {0, 1, 0},    // c.z = 0
        {1, 0, 1},    // c.z = +1
        {0, -1, 0},   // c.z = 0
        {-1, 0, -1},  // c.z = -1
        {1, 1, 0},    // c.z = 0
        {0, 1, 1},    // c.z = +1
        {-1, -1, 0},  // c.z = 0
        {0, -1, -1},  // c.z = -1
        {1, -1, 0},   // c.z = 0
        {-1, 1, 0},   // c.z = 0
        {-1, 0, 1},   // c.z = +1
        {1, 0, -1},   // c.z = -1
        {0, -1, 1},   // c.z = +1
        {0, 1, -1},   // c.z = -1
    };
    const auto r = Stencil::reorder_for_lattice_fusion(natural);
    assert(r.reordered.size() == natural.size());
    assert(r.permutation.size() == natural.size());
    // Permutation must be a bijection over [0, N).
    {
      std::vector<int> seen(natural.size(), 0);
      for (size_t i = 0; i < r.permutation.size(); ++i) {
        assert(r.permutation[i] < natural.size());
        ++seen[r.permutation[i]];
      }
      for (size_t i = 0; i < seen.size(); ++i) {
        assert(seen[i] == 1);
      }
    }
    // reordered[i] must equal natural[permutation[i]].
    for (size_t i = 0; i < r.reordered.size(); ++i) {
      const auto& a = r.reordered[i];
      const auto& b = natural[r.permutation[i]];
      assert(a.x == b.x && a.y == b.y && a.z == b.z);
    }
    // Layout: all c.z<0 first, then c.z==0, then c.z>0.
    int phase = 0;  // 0: looking for c.z<0; 1: c.z==0; 2: c.z>0
    for (const int32_3d& p : r.reordered) {
      if (phase == 0) {
        if (p.z == 0) {
          phase = 1;
        } else if (p.z > 0) {
          phase = 2;
        } else {
          assert(p.z < 0);
        }
      }
      if (phase == 1) {
        if (p.z > 0) {
          phase = 2;
        } else {
          assert(p.z == 0);
        }
      }
      if (phase == 2) {
        assert(p.z > 0);
      }
    }
    // Stable within each bucket: walk the original list and check that the relative
    // order of items within the same z-bucket is preserved.
    auto bucket = [](int z) -> int {
      if (z < 0)
        return -1;
      if (z > 0)
        return +1;
      return 0;
    };
    for (int target : {-1, 0, +1}) {
      std::vector<size_t> from_natural;
      for (size_t i = 0; i < natural.size(); ++i) {
        if (bucket(natural[i].z) == target) {
          from_natural.push_back(i);
        }
      }
      std::vector<size_t> from_reordered;
      for (size_t i = 0; i < r.reordered.size(); ++i) {
        if (bucket(r.reordered[i].z) == target) {
          from_reordered.push_back(r.permutation[i]);
        }
      }
      assert(from_natural == from_reordered);
    }
    // The Stencil built from the reordered offsets must report
    // fully_groupable == true (single contiguous run for c.z<0 and c.z>0).
    const Stencil s(r.reordered);
    const auto profile = s.lattice_profile();
    assert(profile.fully_groupable);
    assert(profile.run_count_negative == 1);
    assert(profile.run_count_positive == 1);
  }

  // Empty input -> empty output (no exceptions, no surprises).
  {
    const std::vector<int32_3d> empty_input;
    const auto r = Stencil::reorder_for_lattice_fusion(empty_input);
    assert(r.reordered.empty());
    assert(r.permutation.empty());
  }

  // All-zero input (all c.z == 0): everything ends up in the middle bucket and the
  // permutation is identity. A weight table remap through `permutation` is the
  // identity in this case.
  {
    const std::vector<int32_3d> only_zeros{
        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 0},
    };
    const auto r = Stencil::reorder_for_lattice_fusion(only_zeros);
    assert(r.reordered.size() == only_zeros.size());
    for (size_t i = 0; i < r.reordered.size(); ++i) {
      assert(r.permutation[i] == i);
      const auto& a = r.reordered[i];
      const auto& b = only_zeros[i];
      assert(a.x == b.x && a.y == b.y && a.z == b.z);
    }
  }

  return 0;
}
