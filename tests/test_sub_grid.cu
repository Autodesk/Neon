// SubGrid construction test (milestone 1: types, lane tables, fields).
//
// Builds SubGrids over a multi-GPU Dense with a radius-1 z stencil (so every slab has an
// Internal and a Boundary region) and checks, against a brute-force host selection:
//
// - Coverage: per GPU, the lanes are exactly the owned cells whose label bit is in the mask,
//   each once; parent-Internal cells first, then parent-Boundary cells, each run in parent
//   iteration order (strictly increasing storage offset).
// - Determinism: building the same SubGrid twice gives identical lane tables.
// - The general predicate constructor, an empty selection, and multi-bit masks.
// - Fields: fill_owned_host + device round trip, and host-side reads through S partitions
//   and, via the derived Span / Idx, through a parent-field partition.
// - Type rules: S::Span / S::Idx derive from the parent's, S::Partition does not, and a
//   parent Idx does not convert to an S Idx.

#include <carbon/carbon.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using S = carbon::SubGrid<carbon::Dense>;

static_assert(std::is_base_of_v<carbon::Dense::Span,
                                S::Span>);
static_assert(std::is_base_of_v<carbon::Dense::Idx,
                                S::Idx>);
static_assert(!std::is_base_of_v<carbon::Dense::Partition<float,
                                                          1>,
                                 S::Partition<float,
                                              1>>);
static_assert(std::is_convertible_v<const S::Idx&,
                                    const carbon::Dense::Idx&>);
static_assert(std::is_convertible_v<const S::Span&,
                                    const carbon::Dense::Span&>);
static_assert(!std::is_convertible_v<carbon::Dense::Idx,
                                     S::Idx>);
static_assert(!std::is_convertible_v<carbon::Dense::Span,
                                     S::Span>);
static_assert(std::is_trivially_copyable_v<S::Span>);
static_assert(std::is_trivially_copyable_v<S::Idx>);

constexpr std::uint8_t kFaceLabel = 1;
constexpr std::uint8_t kBallLabel = 2;

// 1 on the domain faces, 2 inside a small ball, 0 elsewhere.
[[nodiscard]] auto
label_of(carbon::int64_3d g,
         carbon::int64_3d ext) -> std::uint8_t {
  if (g.x == 0 || g.y == 0 || g.z == 0 || g.x == ext.x - 1 || g.y == ext.y - 1 ||
      g.z == ext.z - 1) {
    return kFaceLabel;
  }
  const int64_t dx = g.x - ext.x / 2;
  const int64_t dy = g.y - ext.y / 2;
  const int64_t dz = g.z - ext.z / 2;
  if (dx * dx + dy * dy + dz * dz <= 9) {
    return kBallLabel;
  }
  return 0;
}

int g_failures = 0;

auto
expect(bool ok,
       const std::string& what) -> void {
  if (!ok) {
    ++g_failures;
    if (g_failures <= 20) {
      CARBON_LOG_ERROR("test_sub_grid: {}", what);
    }
  }
}

struct Expected {
  std::vector<carbon::int64_3d> internal;
  std::vector<carbon::int64_3d> boundary;
};

// Brute force: owned cells of GPU g in storage order, split by owned-local z band.
template <typename Pred>
[[nodiscard]] auto
expected_cells(const carbon::Dense& dense,
               size_t g,
               Pred&& selected) -> Expected {
  const auto& span = dense.spans()[g];
  const int64_t h = span.halo_z();
  const int64_t lz = span.logic_extent().z;
  const auto ext = span.haloed_extent();
  Expected out;
  for (int64_t bz = h; bz < h + lz; ++bz) {
    const int64_t local_z = bz - h;
    const bool boundary = h > 0 && (local_z < h || local_z >= lz - h);
    for (int64_t y = 0; y < ext.y; ++y) {
      for (int64_t x = 0; x < ext.x; ++x) {
        const carbon::int64_3d cell{x, y, bz};
        if (selected(span.global_location(carbon::Dense::Idx(cell)))) {
          (boundary ? out.boundary : out.internal).push_back(cell);
        }
      }
    }
  }
  return out;
}

template <typename Pred>
auto
check_lanes(const carbon::Dense& dense,
            const S& sub,
            Pred&& selected,
            const std::string& tag) -> void {
  expect(sub.gpu_count() == dense.gpu_count(), tag + ": gpu_count");
  for (size_t g = 0; g < sub.gpu_count(); ++g) {
    const Expected want = expected_cells(dense, g, selected);
    const auto ni = static_cast<int64_t>(want.internal.size());
    const auto nb = static_cast<int64_t>(want.boundary.size());
    expect(sub.internal_count(g) == ni, tag + ": internal_count gpu " + std::to_string(g) +
                                            " got " + std::to_string(sub.internal_count(g)) +
                                            " want " + std::to_string(ni));
    expect(sub.boundary_count(g) == nb, tag + ": boundary_count gpu " + std::to_string(g) +
                                            " got " + std::to_string(sub.boundary_count(g)) +
                                            " want " + std::to_string(nb));
    if (sub.internal_count(g) != ni || sub.boundary_count(g) != nb) {
      continue;
    }
    const auto& lanes = sub.host_spans()[g].sub();
    expect(lanes.count() == ni + nb, tag + ": span count");
    expect(lanes.lane_offset(carbon::DataView::Internal) == 0, tag + ": internal offset");
    expect(lanes.lane_count(carbon::DataView::Internal) == ni, tag + ": internal lane_count");
    expect(lanes.lane_offset(carbon::DataView::Boundary) == ni, tag + ": boundary offset");
    expect(lanes.lane_count(carbon::DataView::Boundary) == nb, tag + ": boundary lane_count");
    expect(lanes.lane_count(carbon::DataView::Standard) == ni + nb, tag + ": standard lane_count");
    expect(sub.spans()[g].sub().count() == ni + nb, tag + ": device span count");
    for (int64_t lane = 0; lane < ni + nb; ++lane) {
      const carbon::int64_3d want_cell = lane < ni ? want.internal[static_cast<size_t>(lane)]
                                                   : want.boundary[static_cast<size_t>(lane - ni)];
      expect(sub.parent_idx(g, lane).cell() == want_cell,
             tag + ": gpu " + std::to_string(g) + " lane " + std::to_string(lane) + " cell");
    }
  }
}

auto
same_tables(const S& a,
            const S& b) -> bool {
  if (a.gpu_count() != b.gpu_count()) {
    return false;
  }
  for (size_t g = 0; g < a.gpu_count(); ++g) {
    if (a.count(g) != b.count(g) || a.internal_count(g) != b.internal_count(g)) {
      return false;
    }
    for (int64_t lane = 0; lane < a.count(g); ++lane) {
      if (a.parent_idx(g, lane).cell() != b.parent_idx(g, lane).cell()) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] auto
encode(carbon::int64_3d g,
       int c) -> float {
  return static_cast<float>(c * 1000000 + g.z * 10000 + g.y * 100 + g.x);
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_WARN("test_sub_grid: no devices, skipping");
    return 0;
  }

  try {
    carbon::System sys(carbon::get_all_devices());
    const size_t n_gpus = sys.gpu_group().size();
    carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                            /*remove_center=*/true);
    const carbon::int64_3d ext{12, 10, 8 * static_cast<int64_t>(n_gpus)};
    carbon::Dense dense(sys, ext, std::move(stencil));
    expect(dense.spans()[0].halo_z() == 1, "expected halo_z == 1");

    auto labels = dense.newField<std::uint8_t, 1>(carbon::MemorySide::HostDevice, "labels");
    labels.fill_owned_host([&](carbon::int64_3d g, int) {
      return label_of(g, ext);
    });
    carbon::Stream stream = sys.stream();
    carbon::Transfer<std::uint8_t> upload(sys, labels.cpu_buffers(), labels.gpu_buffers());
    upload(stream);
    stream.wait();

    // Faces only.
    const S faces = dense.new_sub_grid(labels, carbon::RegionMask{kFaceLabel}, "faces");
    check_lanes(
        dense, faces,
        [&](carbon::int64_3d g) {
          return label_of(g, ext) == kFaceLabel;
        },
        "faces");
    expect(faces.name() == "faces", "faces name");
    expect(faces.mask() == carbon::RegionMask{kFaceLabel}, "faces mask");

    // Determinism.
    const S faces_again = dense.new_sub_grid(labels, carbon::RegionMask{kFaceLabel});
    expect(same_tables(faces, faces_again), "rebuilding the same SubGrid changed its lane table");
    expect(faces.uid() != faces_again.uid(), "two SubGrids share a uid");

    // Multi-bit mask.
    const S both = dense.new_sub_grid(labels, carbon::RegionMask{kFaceLabel, kBallLabel});
    check_lanes(
        dense, both,
        [&](carbon::int64_3d g) {
          return label_of(g, ext) != 0;
        },
        "both");

    // General predicate constructor.
    const S every_third = dense.new_sub_grid([](carbon::KernelLoader&) {
      return [] CARBON_HD(const carbon::Dense::Span& s, const carbon::Dense::Idx& i) {
        return s.global_location(i).x % 3 == 0;
      };
    });
    check_lanes(
        dense, every_third,
        [](carbon::int64_3d g) {
          return g.x % 3 == 0;
        },
        "every_third");
    expect(every_third.mask().is_all(), "predicate SubGrid default mask is all");

    // Empty selection.
    const S empty = dense.new_sub_grid(labels, carbon::RegionMask{7});
    expect(empty.total_count() == 0, "empty selection total_count");
    auto empty_field = empty.newField<float, 1>(carbon::MemorySide::HostDevice, "empty");
    empty_field.fill_owned_host([](carbon::int64_3d, int) {
      return 1.0F;
    });
    expect(empty_field.partitions().size() == n_gpus, "empty field partitions");

    // Fields: fill on host, round trip through the device, read back via partitions.
    auto f = faces.newField<float, 2>(carbon::MemorySide::HostDevice, "f");
    expect(f.name() == "f", "field name");
    expect(f.partitions().size() == n_gpus && f.host_partitions().size() == n_gpus,
           "field partition counts");
    for (size_t g = 0; g < n_gpus; ++g) {
      expect(f.cpu_buffers()[g].size() == static_cast<size_t>(faces.count(g)) * 2U,
             "field buffer size gpu " + std::to_string(g));
      expect(f.partitions()[g].owner() == faces.uid(), "partition owner");
    }
    f.fill_owned_host([](carbon::int64_3d g, int c) {
      return encode(g, c);
    });
    carbon::Transfer<float> up(sys, f.cpu_buffers(), f.gpu_buffers());
    up(stream);
    for (size_t g = 0; g < n_gpus; ++g) {
      float* h = f.cpu_buffers()[g].data();
      for (size_t k = 0; k < f.cpu_buffers()[g].size(); ++k) {
        h[k] = -1.0F;
      }
    }
    carbon::Transfer<float> down(sys, f.gpu_buffers(), f.cpu_buffers());
    down(stream);
    stream.wait();

    for (size_t g = 0; g < n_gpus; ++g) {
      const S::Span& s = faces.host_spans()[g];
      const auto& fp = f.host_partitions()[g];
      const auto& lp = labels.host_partitions()[g];
      for (int64_t lane = 0; lane < faces.count(g); ++lane) {
        const S::Idx i(faces.parent_idx(g, lane), lane);
        const carbon::int64_3d global = s.global_location(i);
        expect(global == faces.global_location(g, lane), "global_location");
        for (int c = 0; c < 2; ++c) {
          expect(fp.read(s, i, c) == encode(global, c),
                 "field round trip gpu " + std::to_string(g) + " lane " + std::to_string(lane));
        }
        expect(lp.read(s, i, 0) == kFaceLabel, "parent-field read through S span / idx");
      }
    }
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_sub_grid: exception: {}", e.what());
    return 1;
  }

  if (g_failures > 0) {
    CARBON_LOG_ERROR("test_sub_grid: {} failure(s)", g_failures);
    return 1;
  }
  CARBON_LOG_INFO("test_sub_grid: OK");
  return 0;
}
