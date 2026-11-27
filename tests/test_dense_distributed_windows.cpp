// Host-only verification of the DataView -> ViewWindow translation of DenseDistributed's Span.
// No GPU, no MPI.
//
// The CUDA/HIP compute kernel no longer takes a DataView template parameter: the launcher asks
// the Span for the box(es) of a view (Span::view_windows), stamps each into the span
// (Span::with_window) and the kernel iterates window_origin() + (ix, iy, iz) over window_extent().
// This test pins that translation to the legacy path it replaces:
//
//  * for every view the union of the windows is exactly the set of buffer cells that
//    iter_extent(view) + set_and_validate(view, ...) produce, and the windows are pairwise
//    disjoint (every cell visited exactly once),
//  * Internal + Boundary tile Standard, Standard is the span's default window,
//  * a slab (halo.x == halo.y == 0) yields at most two Boundary windows, both along Z,
//  * with_window changes the window and nothing else.
//
// What it does NOT cover: real kernel launches (see the dense_distributed_dist test).

#include <carbon/boundary_mode_distributed.h>
#include <carbon/data_view.h>
#include <carbon/detail/dense_distributed/span.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using carbon::BoundaryModeDistributed;
using carbon::DataView;
using carbon::int64_3d;
namespace dd = carbon::detail::dense_distributed;

std::string g_ctx;

#define CHECK(cond)                                                                       \
  do {                                                                                    \
    if (!(cond)) {                                                                        \
      std::fprintf(stderr, "CHECK failed: %s  (%s:%d)\n  context: %s\n", #cond, __FILE__, \
                   __LINE__, g_ctx.c_str());                                              \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

auto
eq3(const int64_3d& a, const int64_3d& b) -> bool {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

auto
describe(const int64_3d& owned, const int64_3d& halo, bool legacy) -> std::string {
  char buf[160];
  std::snprintf(buf, sizeof(buf), "owned=(%lld,%lld,%lld) halo=(%lld,%lld,%lld) legacy_slab=%d",
                static_cast<long long>(owned.x), static_cast<long long>(owned.y),
                static_cast<long long>(owned.z), static_cast<long long>(halo.x),
                static_cast<long long>(halo.y), static_cast<long long>(halo.z), legacy ? 1 : 0);
  return buf;
}

struct Visits {
  int64_3d haloed;
  std::vector<int> count;

  explicit Visits(const int64_3d& h)
      : haloed(h),
        count(static_cast<size_t>(h.x * h.y * h.z), 0) {}

  auto
  at(int64_t x, int64_t y, int64_t z) -> int& {
    CHECK(x >= 0 && x < haloed.x && y >= 0 && y < haloed.y && z >= 0 && z < haloed.z);
    return count[static_cast<size_t>((z * haloed.y + y) * haloed.x + x)];
  }
};

// Buffer cells visited by the legacy path: every valid thread of iter_extent(view).
auto
legacy_cells(const dd::Span& span, DataView view) -> Visits {
  Visits v(span.haloed_extent());
  const int64_3d e = span.iter_extent(view);
  for (int64_t iz = 0; iz < e.z; ++iz) {
    for (int64_t iy = 0; iy < e.y; ++iy) {
      for (int64_t ix = 0; ix < e.x; ++ix) {
        const auto r = span.set_and_validate(view, ix, iy, iz);
        if (r.valid) {
          v.at(r.idx.cell().x, r.idx.cell().y, r.idx.cell().z) += 1;
        }
      }
    }
  }
  return v;
}

// Buffer cells visited by the new path: window_origin() + (ix, iy, iz) over window_extent(),
// exactly what dense_grid_kernel does, for every window of the view.
auto
window_cells(const dd::Span& span, DataView view) -> Visits {
  Visits v(span.haloed_extent());
  const dd::ViewWindowSet set = span.view_windows(view);
  CHECK(set.count >= 0 && set.count <= dd::ViewWindowSet::kMaxWindows);
  int64_t volume = 0;
  for (int i = 0; i < set.count; ++i) {
    const dd::ViewWindow& w = set.windows[i];
    CHECK(!w.empty());  // empty boxes must have been dropped
    volume += w.volume();
    const dd::Span s = span.with_window(w);
    const int64_3d e = s.window_extent();
    const int64_3d o = s.window_origin();
    for (int64_t iz = 0; iz < e.z; ++iz) {
      for (int64_t iy = 0; iy < e.y; ++iy) {
        for (int64_t ix = 0; ix < e.x; ++ix) {
          v.at(o.x + ix, o.y + iy, o.z + iz) += 1;
        }
      }
    }
  }
  CHECK(volume == set.total_volume());
  return v;
}

void
check_span(const int64_3d& owned, const int64_3d& halo, bool legacy_slab) {
  g_ctx = describe(owned, halo, legacy_slab);
  const int64_3d haloed{owned.x + 2 * halo.x, owned.y + 2 * halo.y, owned.z + 2 * halo.z};
  const int64_3d global{owned.x * 2, owned.y * 2, owned.z * 2};
  const dd::Span span(haloed, global, haloed.x * haloed.y, int64_3d{0, 0, 0}, halo,
                      BoundaryModeDistributed::Open, legacy_slab);

  // Default window = Standard window = all owned cells.
  CHECK(eq3(span.window_origin(), halo));
  CHECK(eq3(span.window_extent(), span.logic_extent()));
  CHECK(eq3(span.standard_window().origin, halo));
  CHECK(eq3(span.standard_window().extent, span.logic_extent()));

  // with_window changes only the window.
  {
    const dd::ViewWindow w{int64_3d{1, 2, 3}, int64_3d{4, 5, 6}};
    const dd::Span t = span.with_window(w);
    CHECK(eq3(t.window_origin(), w.origin) && eq3(t.window_extent(), w.extent));
    CHECK(eq3(t.haloed_extent(), span.haloed_extent()) && eq3(t.halo(), span.halo()));
    CHECK(t.xy_slab_elements() == span.xy_slab_elements() && t.y_stride() == span.y_stride());
    CHECK(t.linear_storage_elements() == span.linear_storage_elements() &&
          t.lead_pad() == span.lead_pad());
    CHECK(t.storage_kind() == span.storage_kind());
  }

  // A legacy Z slab thinner than 2 * halo.z has overlapping Boundary bands by construction (the
  // old path visits cells twice); that degenerate shape is skipped for Boundary, as in
  // test_dense_distributed_host.
  const bool degenerate = legacy_slab && owned.z < 2 * halo.z;

  for (const DataView view : {DataView::Standard, DataView::Internal, DataView::Boundary}) {
    if (view == DataView::Boundary && degenerate) {
      continue;
    }
    Visits legacy = legacy_cells(span, view);
    Visits fresh = window_cells(span, view);
    for (size_t i = 0; i < legacy.count.size(); ++i) {
      CHECK(fresh.count[i] == legacy.count[i]);  // same cells ...
      CHECK(fresh.count[i] <= 1);                // ... each visited at most once
    }
  }

  // Internal + Boundary tile Standard.
  if (!degenerate) {
    const int64_t total = span.view_windows(DataView::Internal).total_volume() +
                          span.view_windows(DataView::Boundary).total_volume();
    CHECK(total == span.view_windows(DataView::Standard).total_volume());
  }

  // Slab: halo only along Z -> Boundary is the two Z boxes and nothing else.
  if (halo.x == 0 && halo.y == 0) {
    const dd::ViewWindowSet b = span.boundary_windows();
    CHECK(b.count <= 2);
    for (int i = 0; i < b.count; ++i) {
      CHECK(b.windows[i].extent.x == owned.x && b.windows[i].extent.y == owned.y);
      CHECK(b.windows[i].origin.x == halo.x && b.windows[i].origin.y == halo.y);
    }
  }
}

}  // namespace

int
main() {
  const int64_3d halos[] = {
      {0, 0, 0},  // no stencil
      {0, 0, 1},  // slab, 1-deep halo
      {0, 0, 2},  // slab, 2-deep halo
      {0, 1, 1},  // pencil
      {1, 0, 1},  // pencil along the other axis
      {1, 1, 1},  // brick
      {2, 1, 2},  // anisotropic brick
      {2, 2, 2},  // wide brick
  };
  // Includes chunks thinner than 2 * halo on some / all axes and a 1-cell-thick chunk.
  const int64_t sizes[] = {1, 2, 3, 4, 5, 8};
  int64_t spans_checked = 0;
  for (const int64_3d& halo : halos) {
    for (const int64_t nx : sizes) {
      for (const int64_t ny : sizes) {
        for (const int64_t nz : sizes) {
          for (const bool legacy : {false, true}) {
            // `legacy_slab` is only meaningful for a Z slab.
            if (legacy && !(halo.x == 0 && halo.y == 0)) {
              continue;
            }
            check_span(int64_3d{nx, ny, nz}, halo, legacy);
            ++spans_checked;
          }
        }
      }
    }
  }
  std::printf("dense_distributed_windows: OK (%lld spans)\n", static_cast<long long>(spans_checked));
  return 0;
}
