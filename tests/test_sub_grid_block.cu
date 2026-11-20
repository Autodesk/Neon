// SubGrid over a Block (block-sparse) parent.
//
// A multi-GPU Block over a ball (so many bricks are partially active) and a SubGrid of a few
// hundred scattered "marker" voxels: a hash selection plus the two voxel rows on either side
// of the 2-GPU slab cut. The selection is checked to reach brick faces, inactive neighbours
// and (on several GPUs) the slab cut. Every result is compared exactly against a host
// reference built from the Block's host bitmasks and origins:
//
// - Construction: label and predicate constructors give the same lanes, in the Block's
//   storage order (Internal bricks, then Boundary bricks, voxels z-y-x).
// - Device kernel: parent read / write, SubGrid field read / write, and x/y neighbour reads
//   of a parent field that cross brick faces.
// - Host kernel: global_location (origins) and the same x/y neighbour reads (connectivity and
//   bitmask) through the host topology tables.
// - Reductions: Sum / Max / count, Internal + Boundary == Standard, BlockAtomic, host and
//   universal, an empty SubGrid, tokens and the ownership throw.
// - Skeleton: a Block writer followed by a SubGrid compute and a SubGrid reduction that
//   stencil-read the parent in all six directions (across the slab cut through HaloBlock),
//   under every OverlapMode, with the OCC split present exactly when overlap is on.

#include <carbon/carbon.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using B = carbon::Block<SB>;
using S = carbon::SubGrid<B>;
using carbon::DataView;
using carbon::int64_3d;

constexpr std::uint8_t kMark = 1;

[[nodiscard]] CARBON_HD constexpr auto
id_of(int64_3d g) -> int64_t {
  return (g.z * 128 + g.y) * 128 + g.x;
}

/** Scattered markers plus two sparse rows on either side of the 2-GPU slab cut. */
[[nodiscard]] CARBON_HD constexpr auto
marked(int64_3d g,
       int64_t n) -> bool {
  const auto h = (static_cast<std::uint64_t>(g.x) * 73856093ULL) ^
                 (static_cast<std::uint64_t>(g.y) * 19349663ULL) ^
                 (static_cast<std::uint64_t>(g.z) * 83492791ULL);
  if (h % 211ULL == 0ULL) {
    return true;
  }
  const int64_t cut = n / 2;
  return (g.z == cut || g.z == cut - 1) && (g.x * 3 + g.y) % 29 == 0;
}

[[nodiscard]] auto
in_ball(int64_3d g,
        int64_t n) -> bool {
  const double c = static_cast<double>(n - 1) * 0.5;
  const double r = static_cast<double>(n) * 0.45;
  const double dx = static_cast<double>(g.x) - c;
  const double dy = static_cast<double>(g.y) - c;
  const double dz = static_cast<double>(g.z) - c;
  return dx * dx + dy * dy + dz * dz <= r * r;
}

[[nodiscard]] auto
active(int64_3d g,
       int64_t n) -> bool {
  return g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < n && g.y < n && g.z < n && in_ball(g, n);
}

template <typename P,
          typename SpanT,
          typename IdxT>
[[nodiscard]] CARBON_HD auto
xy_sum(const P& p,
       const SpanT& s,
       const IdxT& i) -> int64_t {
  const auto xp = p.neighbour(s, i, carbon::int8_3d{1, 0, 0}, 0);
  const auto xm = p.neighbour(s, i, carbon::int8_3d{-1, 0, 0}, 0);
  const auto yp = p.neighbour(s, i, carbon::int8_3d{0, 1, 0}, 0);
  const auto ym = p.neighbour(s, i, carbon::int8_3d{0, -1, 0}, 0);
  return (xp.valid ? xp.value : 0) + (xm.valid ? xm.value : 0) + (yp.valid ? yp.value : 0) +
         (ym.valid ? ym.value : 0);
}

template <typename P,
          typename SpanT,
          typename IdxT>
[[nodiscard]] CARBON_HD auto
six_sum(const P& p,
        const SpanT& s,
        const IdxT& i) -> int64_t {
  const auto zp = p.neighbour(s, i, carbon::int8_3d{0, 0, 1}, 0);
  const auto zm = p.neighbour(s, i, carbon::int8_3d{0, 0, -1}, 0);
  return xy_sum(p, s, i) + (zp.valid ? zp.value : 0) + (zm.valid ? zm.value : 0);
}

/** Host reference: sum of @p offset + id over the active neighbours of @p g in @p dirs . */
[[nodiscard]] auto
ref_sum(int64_3d g,
        int64_t n,
        bool with_z,
        int64_t offset) -> int64_t {
  std::vector<int64_3d> dirs{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
  if (with_z) {
    dirs.push_back({0, 0, 1});
    dirs.push_back({0, 0, -1});
  }
  int64_t acc = 0;
  for (const int64_3d d : dirs) {
    const int64_3d h{g.x + d.x, g.y + d.y, g.z + d.z};
    if (active(h, n)) {
      acc += id_of(h) + offset;
    }
  }
  return acc;
}

int g_failures = 0;

auto
expect(bool ok,
       const std::string& what) -> void {
  if (!ok) {
    ++g_failures;
    if (g_failures <= 20) {
      CARBON_LOG_ERROR("test_sub_grid_block: {}", what);
    }
  }
}

template <typename Fn>
auto
expect_throws(Fn&& fn,
              const std::string& what) -> void {
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return;
  }
  expect(false, what + ": expected std::invalid_argument");
}

template <typename T>
[[nodiscard]] auto
read_back(carbon::ReductionResult<T>& r,
          carbon::Stream& stream,
          DataView view = DataView::Standard) -> T {
  r.sync(stream);
  stream.wait();
  return r.host_value(view);
}

template <typename Field>
auto
to_device(carbon::System& sys,
          Field& f,
          carbon::Stream& stream) -> void {
  carbon::Transfer<typename Field::value_type>(sys, f.cpu_buffers(), f.gpu_buffers())(stream);
  stream.wait();
}

template <typename Field>
auto
to_host(carbon::System& sys,
        Field& f,
        carbon::Stream& stream) -> void {
  carbon::Transfer<typename Field::value_type>(sys, f.gpu_buffers(), f.cpu_buffers())(stream);
  stream.wait();
}

/** A selected voxel as the host reference sees it. */
struct Cell {
  int64_3d global{};
  B::Idx idx{};
};

/** Zeroes every storage slot (owned and ghost) of a Block field on the device. */
template <typename T>
auto
zero_block_field(carbon::System& sys,
                 const B& block,
                 B::Field<T,
                          1>& f,
                 carbon::Stream& stream) -> void {
  for (size_t g = 0; g < block.gpu_count(); ++g) {
    const size_t n = static_cast<size_t>(block.storage_block_count(g)) * SB::VoxelCount;
    if (n != 0U) {
      sys.async_memset(f.gpu_buffers()[g], 0, n * sizeof(T), stream);
    }
  }
  stream.wait();
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_WARN("test_sub_grid_block: no devices, skipping");
    return 0;
  }

  try {
    carbon::System sys(carbon::get_all_devices());
    const size_t n_gpus = sys.gpu_group().size();
    const int64_t n = std::max<int64_t>(64, SB::MemZ * static_cast<int64_t>(n_gpus));
    const B block(
        sys, int64_3d{n, n, n},
        [n](const int64_3d& g) {
          return in_ball(g, n);
        },
        "ball");
    carbon::Stream stream = sys.stream();

    // ---- Host reference: selected voxels per GPU in Block storage order -------------------
    std::vector<std::vector<Cell>> want(n_gpus);
    std::vector<int64_t> want_internal(n_gpus, 0);
    int64_t brick_face_lanes = 0;
    int64_t inactive_neighbour_lanes = 0;
    int64_t slab_cut_lanes = 0;
    for (size_t g = 0; g < n_gpus; ++g) {
      const uint32_t n_internal = block.n_internal_blocks(g);
      for (uint32_t b = 0; b < block.block_count(g); ++b) {
        const auto& bm = block.host_bitmasks(g)[b];
        const carbon::int32_3d o = block.host_origins(g)[b];
        for (int z = 0; z < SB::MemZ; ++z) {
          for (int y = 0; y < SB::MemY; ++y) {
            for (int x = 0; x < SB::MemX; ++x) {
              const int64_3d gc{o.x + x, o.y + y, o.z + z};
              if (!bm.is_active(x, y, z) || !marked(gc, n)) {
                continue;
              }
              want[g].push_back({gc, B::Idx{b, static_cast<int8_t>(x), static_cast<int8_t>(y),
                                            static_cast<int8_t>(z)}});
              want_internal[g] += b < n_internal ? 1 : 0;
              brick_face_lanes += (x == 0 || y == 0 || z == 0 || x == SB::MemX - 1 ||
                                   y == SB::MemY - 1 || z == SB::MemZ - 1)
                                      ? 1
                                      : 0;
              inactive_neighbour_lanes +=
                  (!active({gc.x + 1, gc.y, gc.z}, n) || !active({gc.x - 1, gc.y, gc.z}, n) ||
                   !active({gc.x, gc.y + 1, gc.z}, n) || !active({gc.x, gc.y - 1, gc.z}, n))
                      ? 1
                      : 0;
              slab_cut_lanes += (gc.z == n / 2 || gc.z == n / 2 - 1) ? 1 : 0;
            }
          }
        }
      }
    }
    int64_t want_total = 0;
    for (const auto& w : want) {
      want_total += static_cast<int64_t>(w.size());
    }
    CARBON_LOG_INFO(
        "test_sub_grid_block: {}^3 ball on {} GPU(s), {} blocks, {} active voxels; "
        "{} selected ({} on brick faces, {} next to an inactive voxel, {} on the "
        "slab-cut rows)",
        n, n_gpus, block.block_count(), block.active_voxel_count(), want_total, brick_face_lanes,
        inactive_neighbour_lanes, slab_cut_lanes);
    expect(brick_face_lanes > 0, "selection reaches brick faces");
    expect(inactive_neighbour_lanes > 0, "selection reaches inactive neighbours");
    expect(slab_cut_lanes > 0, "selection reaches the slab cut rows");

    // ---- Construction -----------------------------------------------------------------------
    auto labels = block.newField<std::uint8_t, 1>(carbon::MemorySide::Device, "labels");
    carbon::Kernel::compute(
        block,
        [labels, n](carbon::KernelLoader& L) {
          auto out = L.write(labels);
          return [=] CARBON_HD(const B::Span& s, const B::Idx& i) {
            out.write(s, i, 0, marked(s.global_location(i), n) ? kMark : std::uint8_t{0});
          };
        },
        "label")
        .launch(stream);
    stream.wait();

    const S marks = block.new_sub_grid(labels, carbon::RegionMask{kMark}, "marks");
    const S marks_pred = block.new_sub_grid(
        [n](carbon::KernelLoader&) {
          return [n] CARBON_HD(const B::Span& s, const B::Idx& i) -> bool {
            return marked(s.global_location(i), n);
          };
        },
        carbon::RegionMask{kMark}, "marks_pred");

    expect(marks.gpu_count() == n_gpus, "gpu_count");
    expect(marks.total_count() > 0, "non-empty selection");
    for (size_t g = 0; g < n_gpus; ++g) {
      expect(marks.count(g) == static_cast<int64_t>(want[g].size()),
             "count on gpu " + std::to_string(g));
      expect(marks.internal_count(g) == want_internal[g],
             "internal_count on gpu " + std::to_string(g));
      expect(marks_pred.count(g) == marks.count(g), "predicate constructor count");
      for (int64_t lane = 0; lane < std::min<int64_t>(marks.count(g), want[g].size()); ++lane) {
        const Cell& c = want[g][static_cast<size_t>(lane)];
        const B::Idx& p = marks.parent_idx(g, lane);
        expect(p.block_idx() == c.idx.block_idx() && p.intra() == c.idx.intra(),
               "lane order on gpu " + std::to_string(g));
        expect(marks.global_location(g, lane) == c.global, "global_location");
        if (lane < marks_pred.count(g)) {
          const B::Idx& q = marks_pred.parent_idx(g, lane);
          expect(q.block_idx() == p.block_idx() && q.intra() == p.intra(),
                 "label and predicate constructors agree");
        }
      }
    }
    expect(n_gpus == 1 || marks.boundary_count(0) > 0, "boundary lanes on a multi-GPU slab");
    expect(marks.internal_count(0) > 0, "internal lanes");

    // ---- Device kernel ----------------------------------------------------------------------
    auto u = block.newField<int64_t, 1>(carbon::MemorySide::HostDevice, "u");
    zero_block_field(sys, block, u, stream);
    carbon::Kernel::compute(
        block,
        [u](carbon::KernelLoader& L) {
          auto p = L.write(u);
          return [=] CARBON_HD(const B::Span& s, const B::Idx& i) {
            p.write(s, i, 0, id_of(s.global_location(i)));
          };
        },
        "init_u")
        .launch(stream);
    to_host(sys, u, stream);

    auto out = block.newField<int64_t, 1>(carbon::MemorySide::Device, "out");
    zero_block_field(sys, block, out, stream);
    auto q = marks.newField<int64_t, 1>(carbon::MemorySide::HostDevice, "q");
    q.fill_owned_host([](int64_3d g, int) {
      return g.x + 2 * g.y;
    });
    to_device(sys, q, stream);
    auto nxy = marks.newField<int64_t, 1>(carbon::MemorySide::HostDevice, "nxy");

    const carbon::Kernel k_dev = marks.make_kernel(
        [u, out, q, nxy](carbon::KernelLoader& L) {
          auto uG = L.read_stencil(u);
          auto oG = L.write(out);
          auto qS = L.read_write(q);
          auto nS = L.write(nxy);
          return [=] CARBON_HD(const S::Span& s, const S::Idx& i) {
            oG.write(s, i, 0, 2 * uG.read(s, i, 0) + qS.read(s, i, 0));
            qS.write(s, i, 0, qS.read(s, i, 0) + 1);
            nS.write(s, i, 0, xy_sum(uG, s, i));
          };
        },
        "s_device");
    k_dev.launch(stream);
    stream.wait();
    auto out_host = block.download_to_host(out, stream);
    stream.wait();
    to_host(sys, q, stream);
    to_host(sys, nxy, stream);
    for (size_t g = 0; g < n_gpus; ++g) {
      const B::Span& span = block.span(g);
      int64_t touched = 0;
      for (uint32_t b = 0; b < block.block_count(g); ++b) {
        const auto& bm = block.host_bitmasks(g)[b];
        const carbon::int32_3d o = block.host_origins(g)[b];
        for (int z = 0; z < SB::MemZ; ++z) {
          for (int y = 0; y < SB::MemY; ++y) {
            for (int x = 0; x < SB::MemX; ++x) {
              if (!bm.is_active(x, y, z)) {
                continue;
              }
              const int64_3d gc{o.x + x, o.y + y, o.z + z};
              const B::Idx idx{b, static_cast<int8_t>(x), static_cast<int8_t>(y),
                               static_cast<int8_t>(z)};
              const int64_t got = out_host[g].data()[span.linear_offset(idx, 1, 0)];
              const int64_t want_out = marked(gc, n) ? 2 * id_of(gc) + gc.x + 2 * gc.y : 0;
              touched += got != 0 ? 1 : 0;
              expect(got == want_out, "device kernel parent write at " + std::to_string(gc.x) +
                                          "," + std::to_string(gc.y) + "," + std::to_string(gc.z));
            }
          }
        }
      }
      expect(touched == marks.count(g), "device kernel writes exactly the selected voxels");
      for (int64_t lane = 0; lane < marks.count(g); ++lane) {
        const int64_3d gc = want[g][static_cast<size_t>(lane)].global;
        expect(q.cpu_buffers()[g].data()[lane] == gc.x + 2 * gc.y + 1, "SubGrid field write");
        expect(nxy.cpu_buffers()[g].data()[lane] == ref_sum(gc, n, false, 0),
               "device x/y neighbour sum across bricks");
      }
    }

    // ---- Host kernel ------------------------------------------------------------------------
    auto hloc = marks.newField<int64_t, 1>(carbon::MemorySide::HostDevice, "hloc");
    auto hnxy = marks.newField<int64_t, 1>(carbon::MemorySide::HostDevice, "hnxy");
    const carbon::Kernel k_host = marks.make_host_kernel(
        [u, hloc, hnxy](carbon::KernelLoader& L) {
          auto uG = L.read_stencil(u);
          auto lS = L.write(hloc);
          auto nS = L.write(hnxy);
          return [=](const S::Span& s, const S::Idx& i) {
            lS.write(s, i, 0, id_of(s.global_location(i)));
            nS.write(s, i, 0, xy_sum(uG, s, i));
          };
        },
        "s_host");
    expect(k_host.kind() == carbon::KernelKind::ComputeHost, "host kernel kind");
    k_host.launch(stream);
    for (size_t g = 0; g < n_gpus; ++g) {
      for (int64_t lane = 0; lane < marks.count(g); ++lane) {
        const int64_3d gc = want[g][static_cast<size_t>(lane)].global;
        expect(hloc.cpu_buffers()[g].data()[lane] == id_of(gc), "host global_location");
        expect(hnxy.cpu_buffers()[g].data()[lane] == ref_sum(gc, n, false, 0),
               "host x/y neighbour sum across bricks");
      }
    }

    // ---- Reductions -------------------------------------------------------------------------
    int64_t want_sum = 0;
    int64_t want_internal_sum = 0;
    int64_t want_max = std::numeric_limits<int64_t>::lowest();
    for (size_t g = 0; g < n_gpus; ++g) {
      for (int64_t lane = 0; lane < marks.count(g); ++lane) {
        const int64_3d gc = want[g][static_cast<size_t>(lane)].global;
        want_sum += id_of(gc);
        want_internal_sum += lane < want_internal[g] ? id_of(gc) : 0;
        want_max = std::max(want_max, gc.x + 2 * gc.y + 1);
      }
    }
    auto sum_u = [u](carbon::KernelLoader& L) {
      auto p = L.read(u);
      return [=] CARBON_HD(const S::Span& s, const S::Idx& i) -> int64_t {
        return p.read(s, i, 0);
      };
    };
    auto sum_u_host = [u](carbon::KernelLoader& L) {
      auto p = L.read(u);
      return [=](const S::Span& s, const S::Idx& i) -> int64_t {
        return p.read(s, i, 0);
      };
    };

    carbon::ReductionResult<int64_t> r_sum(sys);
    const carbon::Kernel k_sum = marks.make_reduce_kernel<int64_t>(sum_u, r_sum);
    k_sum.launch(stream);
    expect(read_back(r_sum, stream) == want_sum, "device Sum of a parent field");
    k_sum.launch(stream, DataView::Internal);
    k_sum.launch(stream, DataView::Boundary);
    const int64_t got_internal = read_back(r_sum, stream, DataView::Internal);
    const int64_t got_boundary = r_sum.host_value(DataView::Boundary);
    expect(got_internal == want_internal_sum, "Internal-view sum");
    expect(got_internal + got_boundary == want_sum, "Internal + Boundary == Standard");

    carbon::ReductionResult<int64_t> r_max(sys);
    marks
        .make_reduce_kernel<int64_t>(
            [q](carbon::KernelLoader& L) {
              auto p = L.read(q);
              return [=] CARBON_HD(const S::Span& s, const S::Idx& i) -> int64_t {
                return p.read(s, i, 0);
              };
            },
            r_max, carbon::reduce::Max<int64_t>{})
        .launch(stream);
    expect(read_back(r_max, stream) == want_max, "device Max of a SubGrid field");

    carbon::ReductionResult<int64_t> r_count(sys);
    marks
        .make_reduce_kernel<int64_t>(
            [](carbon::KernelLoader&) {
              return [] CARBON_HD(const S::Span&, const S::Idx&) -> int64_t {
                return 1;
              };
            },
            r_count)
        .launch(stream);
    expect(read_back(r_count, stream) == marks.total_count(), "lane count");

    carbon::ReductionResult<int64_t> r_atomic(sys);
    carbon::reduce::ReduceOptions atomic;
    atomic.finalize = carbon::reduce::ReduceFinalize::BlockAtomic;
    marks.make_reduce_kernel<int64_t>(sum_u, r_atomic, {}, 0U, atomic).launch(stream);
    expect(read_back(r_atomic, stream) == want_sum, "BlockAtomic Sum");

    carbon::ReductionResult<int64_t> r_host(sys);
    marks.make_host_reduce_kernel<int64_t>(sum_u_host, r_host).launch(stream);
    expect(read_back(r_host, stream) == want_sum, "host Sum");

    carbon::ReductionResult<int64_t> r_uni_dev(sys);
    carbon::ReductionResult<int64_t> r_uni_host(sys);
    marks.make_universal_reduce_kernel<int64_t>(sum_u, r_uni_dev).launch(stream);
    marks.make_universal_reduce_kernel<int64_t>(sum_u, r_uni_host).launch_on_host(stream);
    expect(read_back(r_uni_dev, stream) == want_sum, "universal reduce, device launch");
    expect(read_back(r_uni_host, stream) == want_sum, "universal reduce, host launch");

    for (const carbon::AccessToken& t : k_sum.impl()->tokens()) {
      if (t.resource == carbon::ResourceKind::Field) {
        expect(t.mask == marks.mask(), "reduce field token carries the SubGrid mask");
      }
    }

    const S empty = block.new_sub_grid(labels, carbon::RegionMask{9}, "empty");
    expect(empty.total_count() == 0, "empty selection");
    carbon::ReductionResult<int64_t> r_empty_sum(sys);
    carbon::ReductionResult<int64_t> r_empty_max(sys);
    empty.make_reduce_kernel<int64_t>(sum_u, r_empty_sum).launch(stream);
    empty.make_reduce_kernel<int64_t>(sum_u, r_empty_max, carbon::reduce::Max<int64_t>{})
        .launch(stream);
    expect(read_back(r_empty_sum, stream) == 0, "empty SubGrid Sum is 0");
    expect(read_back(r_empty_max, stream) == std::numeric_limits<int64_t>::lowest(),
           "empty SubGrid Max is the identity");

    auto pred_field = marks_pred.newField<int64_t, 1>(carbon::MemorySide::Device, "pred_field");
    expect_throws(
        [&]() {
          (void)marks.make_kernel([pred_field](carbon::KernelLoader& L) {
            auto p = L.write(pred_field);
            return [=] CARBON_HD(const S::Span& s, const S::Idx& i) {
              p.write(s, i, 0, 1);
            };
          });
        },
        "loading another SubGrid's field");

    // ---- Skeleton: stencil reads of the parent across the slab cut --------------------------
    constexpr int kIters = 3;
    int64_t want_six_total = 0;
    for (size_t g = 0; g < n_gpus; ++g) {
      for (const Cell& c : want[g]) {
        want_six_total += ref_sum(c.global, n, true, kIters);
      }
    }
    for (const carbon::OverlapMode mode :
         {carbon::OverlapMode::None, carbon::OverlapMode::Standard, carbon::OverlapMode::Extended,
          carbon::OverlapMode::TwoWayExtended}) {
      const std::string tag = "skeleton overlap=" + std::to_string(static_cast<int>(mode));
      auto v = block.newField<int64_t, 1>(carbon::MemorySide::Device, "v");
      zero_block_field(sys, block, v, stream);
      carbon::Kernel::compute(
          block,
          [v](carbon::KernelLoader& L) {
            auto p = L.write(v);
            return [=] CARBON_HD(const B::Span& s, const B::Idx& i) {
              p.write(s, i, 0, id_of(s.global_location(i)));
            };
          },
          "init_v")
          .launch(stream);
      stream.wait();
      auto six = marks.newField<int64_t, 1>(carbon::MemorySide::HostDevice, "six");
      carbon::ReductionResult<int64_t> r_six(sys);

      carbon::Skeleton sk(sys, carbon::Skeleton::Options(mode));
      sk.add(carbon::Kernel::compute(
          block,
          [v](carbon::KernelLoader& L) {
            auto p = L.read_write(v);
            return [=] CARBON_HD(const B::Span& s, const B::Idx& i) {
              p.write(s, i, 0, p.read(s, i, 0) + 1);
            };
          },
          "bump"));
      sk.add(carbon::Kernel::compute(
          marks,
          [v, six](carbon::KernelLoader& L) {
            auto p = L.read_stencil(v);
            auto o = L.write(six);
            return [=] CARBON_HD(const S::Span& s, const S::Idx& i) {
              o.write(s, i, 0, six_sum(p, s, i));
            };
          },
          "s_nsum"));
      sk.add(carbon::Kernel::reduce<int64_t>(
          marks,
          [v](carbon::KernelLoader& L) {
            auto p = L.read_stencil(v);
            return [=] CARBON_HD(const S::Span& s, const S::Idx& i) -> int64_t {
              return six_sum(p, s, i);
            };
          },
          r_six, carbon::reduce::Sum<int64_t>{}, "s_rsum"));
      sk.compile();
      const std::string order = sk.export_launch_order();
      for (int it = 0; it < kIters; ++it) {
        sk.run();
      }
      sk.wait();

      expect(read_back(r_six, stream) == want_six_total, tag + ": reduction value");
      to_host(sys, six, stream);
      int64_t bad = 0;
      for (size_t g = 0; g < n_gpus; ++g) {
        for (int64_t lane = 0; lane < marks.count(g); ++lane) {
          const int64_3d gc = want[g][static_cast<size_t>(lane)].global;
          bad += six.cpu_buffers()[g].data()[lane] != ref_sum(gc, n, true, kIters) ? 1 : 0;
        }
      }
      expect(bad == 0, tag + ": " + std::to_string(bad) + " lanes with a wrong 6-neighbour sum");
      const auto has = [&order](const char* what) {
        return order.find(what) != std::string::npos;
      };
      const bool split = has("s_nsum [compute/internal]") && has("s_nsum [compute/boundary]") &&
                         has("s_rsum [reduction/internal]") && has("s_rsum [reduction/boundary]");
      expect(split == (mode != carbon::OverlapMode::None),
             tag + ": OCC split of the SubGrid kernels\n" + order);
    }
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_sub_grid_block: exception: {}", e.what());
    return 1;
  }

  if (g_failures > 0) {
    CARBON_LOG_ERROR("test_sub_grid_block: {} failure(s)", g_failures);
    return 1;
  }
  CARBON_LOG_INFO("test_sub_grid_block: OK");
  return 0;
}
