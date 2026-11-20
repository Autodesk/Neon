// A Jacobi step followed by an AXPY, with and without a SubGrid, on Dense and Block parents.
//
// Every iteration computes, in int64 so any scheduling or indexing error is an exact mismatch,
//
//   u_out = (sum of the 6 neighbours of u_in) / 6 + src                at interior voxels
//   u_out = (u_in + sum of its valid neighbours) / (1 + n_valid) + src  at "special" voxels
//   y     = 3 u_out + y                                                 everywhere
//
// where a special voxel is missing at least one face neighbour (domain face, or an inactive
// voxel of the Block ball) and src = (x + 2y + 3z) % 5 keeps the field from flattening.
// u ping-pongs between u_a and u_b, and y accumulates every iteration's u, so a wrong value in
// any of the kIters iterations reaches the final check.
//
// Each parent grid runs the pipeline in these variants, kIters iterations each, then compares
// u_a and y with a host reference on every active voxel:
//
//   no_s       one parent Jacobi kernel that branches on neighbour validity (None, Standard)
//   s          a parent kernel for the interior (region kInterior, skips special voxels) plus
//              a SubGrid kernel over the special voxels (None)
//   s + OCC    the same kernels under Standard, Extended and TwoWayExtended
//
// With a SubGrid, the interior and SubGrid kernels write disjoint regions of u_out, so the
// Skeleton must not order them; with overlap on, both are OCC-split.

#include <carbon/carbon.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using carbon::Dense;
using carbon::int64_3d;
using carbon::OverlapMode;
using carbon::RegionMask;
using SB = carbon::detail::block::DefaultStaticBlock;
using B = carbon::Block<SB>;

constexpr int kIters = 100;
static_assert(kIters % 2 == 0,
              "kIters must be even so the result ends in u_a");

constexpr std::uint8_t kSpecial = 1;
constexpr std::uint8_t kInterior = 2;
constexpr int64_t kAxpyA = 3;
constexpr int64_t kInactive = std::numeric_limits<int64_t>::min();

/** Domain box, optionally restricted to a ball (integer test, identical on host and device). */
struct Geometry {
  int64_3d ext{};
  bool ball = false;

  [[nodiscard]] CARBON_HD auto
  active(int64_3d g) const -> bool {
    if (g.x < 0 || g.y < 0 || g.z < 0 || g.x >= ext.x || g.y >= ext.y || g.z >= ext.z) {
      return false;
    }
    if (!ball) {
      return true;
    }
    // |2g - (n - 1)|^2 <= (0.9 n)^2, i.e. radius 0.45 n around the centre.
    const int64_t dx = 2 * g.x - (ext.x - 1);
    const int64_t dy = 2 * g.y - (ext.y - 1);
    const int64_t dz = 2 * g.z - (ext.z - 1);
    return 100 * (dx * dx + dy * dy + dz * dz) <= 81 * ext.x * ext.x;
  }

  /** True when at least one face neighbour of @p g is missing. */
  [[nodiscard]] CARBON_HD auto
  special(int64_3d g) const -> bool {
    return !active({g.x + 1, g.y, g.z}) || !active({g.x - 1, g.y, g.z}) ||
           !active({g.x, g.y + 1, g.z}) || !active({g.x, g.y - 1, g.z}) ||
           !active({g.x, g.y, g.z + 1}) || !active({g.x, g.y, g.z - 1});
  }
};

[[nodiscard]] CARBON_HD constexpr auto
init_u(int64_3d g) -> int64_t {
  return (7 * g.x + 3 * g.y + 5 * g.z) % 97 + 1;
}

[[nodiscard]] CARBON_HD constexpr auto
init_y(int64_3d g) -> int64_t {
  return g.x + g.y + g.z;
}

[[nodiscard]] CARBON_HD constexpr auto
src(int64_3d g) -> int64_t {
  return (g.x + 2 * g.y + 3 * g.z) % 5;
}

struct NeighbourSum {
  int64_t sum = 0;
  int64_t count = 0;
};

template <typename P,
          typename SpanT,
          typename IdxT>
[[nodiscard]] CARBON_HD auto
neighbour_sum(const P& p,
              const SpanT& s,
              const IdxT& i) -> NeighbourSum {
  NeighbourSum r{};
  const auto add = [&r](const carbon::NeighbourValue<int64_t>& n) {
    if (n.valid) {
      r.sum += n.value;
      ++r.count;
    }
  };
  add(p.neighbour(s, i, carbon::int8_3d{1, 0, 0}, 0));
  add(p.neighbour(s, i, carbon::int8_3d{-1, 0, 0}, 0));
  add(p.neighbour(s, i, carbon::int8_3d{0, 1, 0}, 0));
  add(p.neighbour(s, i, carbon::int8_3d{0, -1, 0}, 0));
  add(p.neighbour(s, i, carbon::int8_3d{0, 0, 1}, 0));
  add(p.neighbour(s, i, carbon::int8_3d{0, 0, -1}, 0));
  return r;
}

[[nodiscard]] CARBON_HD constexpr auto
interior_rule(NeighbourSum n) -> int64_t {
  return n.sum / 6;
}

[[nodiscard]] CARBON_HD constexpr auto
special_rule(int64_t self,
             NeighbourSum n) -> int64_t {
  return (self + n.sum) / (1 + n.count);
}

int g_failures = 0;

auto
expect(bool ok,
       const std::string& what) -> void {
  if (!ok) {
    ++g_failures;
    if (g_failures <= 20) {
      CARBON_LOG_ERROR("test_sub_grid_jacobi_axpy: {}", what);
    }
  }
}

// ---- Parent-specific pieces: field side, storage zeroing, owned-voxel walk, readback ----------

[[nodiscard]] auto
field_side(const Dense&) -> carbon::MemorySide {
  return carbon::MemorySide::HostDevice;
}

[[nodiscard]] auto
field_side(const B&) -> carbon::MemorySide {
  return carbon::MemorySide::Device;
}

auto
zero_storage(carbon::System&,
             const Dense&,
             Dense::Field<int64_t,
                          1>&,
             carbon::Stream&) -> void {}

auto
zero_storage(carbon::System& sys,
             const B& block,
             B::Field<int64_t,
                      1>& f,
             carbon::Stream& stream) -> void {
  for (size_t g = 0; g < block.gpu_count(); ++g) {
    const size_t n = static_cast<size_t>(block.storage_block_count(g)) * SB::VoxelCount;
    if (n != 0U) {
      sys.async_memset(f.gpu_buffers()[g], 0, n * sizeof(int64_t), stream);
    }
  }
  stream.wait();
}

/** Calls @p fn(global, offset) for every owned voxel of GPU @p g ; offset indexes host_values. */
template <typename Fn>
auto
for_each_owned(const Dense& dense,
               size_t g,
               Fn&& fn) -> void {
  const Dense::Span& span = dense.spans()[g];
  const int64_t h = span.halo_z();
  const int64_3d ext = span.haloed_extent();
  for (int64_t z = h; z < h + span.logic_extent().z; ++z) {
    for (int64_t y = 0; y < ext.y; ++y) {
      for (int64_t x = 0; x < ext.x; ++x) {
        const Dense::Idx idx(int64_3d{x, y, z});
        fn(span.global_location(idx), static_cast<size_t>(span.linear_offset(idx)));
      }
    }
  }
}

template <typename Fn>
auto
for_each_owned(const B& block,
               size_t g,
               Fn&& fn) -> void {
  const B::Span& span = block.span(g);
  for (uint32_t b = 0; b < block.block_count(g); ++b) {
    const auto& bm = block.host_bitmasks(g)[b];
    const carbon::int32_3d o = block.host_origins(g)[b];
    for (int z = 0; z < SB::MemZ; ++z) {
      for (int y = 0; y < SB::MemY; ++y) {
        for (int x = 0; x < SB::MemX; ++x) {
          if (!bm.is_active(x, y, z)) {
            continue;
          }
          const B::Idx idx{b, static_cast<int8_t>(x), static_cast<int8_t>(y),
                           static_cast<int8_t>(z)};
          fn(int64_3d{o.x + x, o.y + y, o.z + z},
             static_cast<size_t>(span.linear_offset(idx, 1, 0)));
        }
      }
    }
  }
}

[[nodiscard]] auto
host_values(carbon::System& sys,
            const Dense&,
            Dense::Field<int64_t,
                         1>& f,
            carbon::Stream& stream) -> std::vector<std::vector<int64_t>> {
  carbon::Transfer<int64_t>(sys, f.gpu_buffers(), f.cpu_buffers())(stream);
  stream.wait();
  std::vector<std::vector<int64_t>> out;
  for (size_t g = 0; g < f.cpu_buffers().size(); ++g) {
    const auto& buf = f.cpu_buffers()[g];
    out.emplace_back(buf.data(), buf.data() + buf.size());
  }
  return out;
}

[[nodiscard]] auto
host_values(carbon::System&,
            const B& block,
            B::Field<int64_t,
                     1>& f,
            carbon::Stream& stream) -> std::vector<std::vector<int64_t>> {
  const auto bufs = block.download_to_host(f, stream);
  stream.wait();
  std::vector<std::vector<int64_t>> out;
  for (const auto& buf : bufs) {
    out.emplace_back(buf.data(), buf.data() + buf.size());
  }
  return out;
}

// ---- Host reference ----------------------------------------------------------------------------

struct Reference {
  std::vector<int64_t> u;
  std::vector<int64_t> y;
};

[[nodiscard]] auto
linear(const Geometry& geo,
       int64_3d g) -> size_t {
  return static_cast<size_t>((g.z * geo.ext.y + g.y) * geo.ext.x + g.x);
}

[[nodiscard]] auto
reference(const Geometry& geo) -> Reference {
  const size_t total = static_cast<size_t>(geo.ext.x * geo.ext.y * geo.ext.z);
  std::vector<int64_t> u(total, kInactive);
  std::vector<int64_t> next(total, kInactive);
  std::vector<int64_t> y(total, kInactive);
  std::vector<int64_3d> cells;
  for (int64_t z = 0; z < geo.ext.z; ++z) {
    for (int64_t yy = 0; yy < geo.ext.y; ++yy) {
      for (int64_t x = 0; x < geo.ext.x; ++x) {
        const int64_3d g{x, yy, z};
        if (geo.active(g)) {
          cells.push_back(g);
          u[linear(geo, g)] = init_u(g);
          y[linear(geo, g)] = init_y(g);
        }
      }
    }
  }
  const int64_3d dirs[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  for (int it = 0; it < kIters; ++it) {
    for (const int64_3d g : cells) {
      NeighbourSum n{};
      for (const int64_3d d : dirs) {
        const int64_3d h{g.x + d.x, g.y + d.y, g.z + d.z};
        if (geo.active(h)) {
          n.sum += u[linear(geo, h)];
          ++n.count;
        }
      }
      const int64_t self = u[linear(geo, g)];
      next[linear(geo, g)] = (n.count == 6 ? interior_rule(n) : special_rule(self, n)) + src(g);
    }
    for (const int64_3d g : cells) {
      y[linear(geo, g)] += kAxpyA * next[linear(geo, g)];
    }
    std::swap(u, next);
  }
  return {std::move(u), std::move(y)};
}

// ---- Kernels -----------------------------------------------------------------------------------

template <typename G>
using F = typename G::template Field<int64_t, 1>;

template <typename G>
struct Fields {
  F<G> u_a;
  F<G> u_b;
  F<G> y;
};

template <typename G>
[[nodiscard]] auto
make_fields(carbon::System& sys,
            const G& grid,
            carbon::Stream& stream) -> Fields<G> {
  Fields<G> f{grid.template newField<int64_t, 1>(field_side(grid), "u_a"),
              grid.template newField<int64_t, 1>(field_side(grid), "u_b"),
              grid.template newField<int64_t, 1>(field_side(grid), "y")};
  zero_storage(sys, grid, f.u_a, stream);
  zero_storage(sys, grid, f.u_b, stream);
  zero_storage(sys, grid, f.y, stream);
  carbon::Kernel::compute(
      grid,
      [u_a = f.u_a, u_b = f.u_b, y = f.y](carbon::KernelLoader& L) {
        auto pa = L.write(u_a);
        auto pb = L.write(u_b);
        auto py = L.write(y);
        return [=] CARBON_HD(const typename G::Span& s, const typename G::Idx& i) {
          const int64_3d g = s.global_location(i);
          pa.write(s, i, 0, init_u(g));
          pb.write(s, i, 0, int64_t{0});
          py.write(s, i, 0, init_y(g));
        };
      },
      "init")
      .launch(stream);
  stream.wait();
  return f;
}

// Without a SubGrid: one kernel over every active voxel, branching on neighbour validity.
template <typename G>
[[nodiscard]] auto
jacobi_full(const G& grid,
            const F<G>& in,
            const F<G>& out,
            const std::string& name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      grid,
      [in, out](carbon::KernelLoader& L) {
        auto pin = L.read_stencil(in);
        auto pout = L.write(out);
        return [=] CARBON_HD(const typename G::Span& s, const typename G::Idx& i) {
          const NeighbourSum n = neighbour_sum(pin, s, i);
          const int64_t v = n.count == 6 ? interior_rule(n) : special_rule(pin.read(s, i, 0), n);
          pout.write(s, i, 0, v + src(s.global_location(i)));
        };
      },
      name);
}

// With a SubGrid: the parent kernel declares the interior region and skips special voxels.
template <typename G>
[[nodiscard]] auto
jacobi_interior(const G& grid,
                const F<G>& in,
                const typename G::template Field<std::uint8_t,
                                                 1>& labels,
                const F<G>& out,
                const std::string& name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      grid,
      [in, labels, out](carbon::KernelLoader& L) {
        L.region(RegionMask{kInterior});
        auto lab = L.read(labels, RegionMask::all());
        auto pin = L.read_stencil(in);
        auto pout = L.write(out);
        return [=] CARBON_HD(const typename G::Span& s, const typename G::Idx& i) {
          if (lab.read(s, i, 0) != kInterior) {
            return;
          }
          pout.write(s, i, 0, interior_rule(neighbour_sum(pin, s, i)) + src(s.global_location(i)));
        };
      },
      name);
}

// The special voxels as their own grid: one thread per voxel, no filter.
template <typename G>
[[nodiscard]] auto
jacobi_special(const carbon::SubGrid<G>& sub,
               const F<G>& in,
               const F<G>& out,
               const std::string& name) -> carbon::Kernel {
  using S = carbon::SubGrid<G>;
  return carbon::Kernel::compute(
      sub,
      [in, out](carbon::KernelLoader& L) {
        auto pin = L.read_stencil(in);
        auto pout = L.write(out);
        return [=] CARBON_HD(const typename S::Span& s, const typename S::Idx& i) {
          const int64_t v = special_rule(pin.read(s, i, 0), neighbour_sum(pin, s, i));
          pout.write(s, i, 0, v + src(s.global_location(i)));
        };
      },
      name);
}

template <typename G>
[[nodiscard]] auto
axpy(const G& grid,
     const F<G>& u,
     const F<G>& y,
     const std::string& name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      grid,
      [u, y](carbon::KernelLoader& L) {
        auto pu = L.read(u);
        auto py = L.read_write(y);
        return [=] CARBON_HD(const typename G::Span& s, const typename G::Idx& i) {
          py.write(s, i, 0, kAxpyA * pu.read(s, i, 0) + py.read(s, i, 0));
        };
      },
      name);
}

// ---- Driver ------------------------------------------------------------------------------------

[[nodiscard]] auto
contains(const std::string& text,
         const std::string& needle) -> bool {
  return text.find(needle) != std::string::npos;
}

template <typename G>
auto
compare(carbon::System& sys,
        const G& grid,
        const Geometry& geo,
        F<G>& field,
        const std::vector<int64_t>& want,
        carbon::Stream& stream,
        const std::string& tag) -> void {
  const auto got = host_values(sys, grid, field, stream);
  int64_t bad = 0;
  int64_t checked = 0;
  std::string first;
  for (size_t g = 0; g < grid.gpu_count(); ++g) {
    for_each_owned(grid, g, [&](int64_3d gc, size_t off) {
      const int64_t w = want[linear(geo, gc)];
      const int64_t v = got[g][off];
      ++checked;
      if (v != w) {
        if (bad == 0) {
          first = " (first at " + std::to_string(gc.x) + "," + std::to_string(gc.y) + "," +
                  std::to_string(gc.z) + ": got " + std::to_string(v) + ", want " +
                  std::to_string(w) + ")";
        }
        ++bad;
      }
    });
  }
  expect(checked > 0, tag + ": no voxels checked");
  expect(bad == 0, tag + ": field '" + field.name() + "' differs in " + std::to_string(bad) +
                       " of " + std::to_string(checked) + " voxels" + first);
}

template <typename G>
auto
run_grid(carbon::System& sys,
         const G& grid,
         const Geometry& geo,
         const std::string& grid_tag) -> void {
  using S = carbon::SubGrid<G>;
  carbon::Stream stream = sys.stream();

  // Host side: the reference and the special voxels per GPU.
  const Reference ref = reference(geo);
  std::set<int64_t> distinct;
  for (const int64_t v : ref.u) {
    if (v != kInactive) {
      distinct.insert(v);
    }
  }
  expect(distinct.size() > 10, grid_tag + ": the reference u flattened to " +
                                   std::to_string(distinct.size()) + " distinct values");

  auto labels = grid.template newField<std::uint8_t, 1>(carbon::MemorySide::Device, "labels");
  carbon::Kernel::compute(
      grid,
      [labels, geo](carbon::KernelLoader& L) {
        auto out = L.write(labels);
        return [=] CARBON_HD(const typename G::Span& s, const typename G::Idx& i) {
          out.write(s, i, 0, geo.special(s.global_location(i)) ? kSpecial : kInterior);
        };
      },
      "labels")
      .launch(stream);
  stream.wait();
  const S special = grid.new_sub_grid(labels, RegionMask{kSpecial}, "special");

  int64_t owned_total = 0;
  for (size_t g = 0; g < grid.gpu_count(); ++g) {
    int64_t want = 0;
    for_each_owned(grid, g, [&](int64_3d gc, size_t) {
      ++owned_total;
      int64_t valid = 0;
      for (const int64_3d d : {int64_3d{1, 0, 0}, int64_3d{-1, 0, 0}, int64_3d{0, 1, 0},
                               int64_3d{0, -1, 0}, int64_3d{0, 0, 1}, int64_3d{0, 0, -1}}) {
        valid += geo.active({gc.x + d.x, gc.y + d.y, gc.z + d.z}) ? 1 : 0;
      }
      want += valid < 6 ? 1 : 0;
    });
    expect(special.count(g) == want, grid_tag + ": SubGrid count on gpu " + std::to_string(g) +
                                         " is " + std::to_string(special.count(g)) + ", want " +
                                         std::to_string(want));
    expect(special.internal_count(g) <= special.count(g), grid_tag + ": internal_count");
  }
  expect(special.total_count() > 0, grid_tag + ": empty SubGrid");
  CARBON_LOG_INFO(
      "test_sub_grid_jacobi_axpy: {} on {} GPU(s): {} active voxels, {} special "
      "({} distinct reference values after {} iterations)",
      grid_tag, grid.gpu_count(), owned_total, special.total_count(), distinct.size(), kIters);

  struct Variant {
    std::string name;
    bool sub_grid;
    OverlapMode mode;
  };
  const std::vector<Variant> variants{
      {"no_s", false, OverlapMode::None},
      {"no_s/occ=standard", false, OverlapMode::Standard},
      {"s", true, OverlapMode::None},
      {"s/occ=standard", true, OverlapMode::Standard},
      {"s/occ=extended", true, OverlapMode::Extended},
      {"s/occ=two_way_extended", true, OverlapMode::TwoWayExtended},
  };

  for (const Variant& var : variants) {
    const std::string tag = grid_tag + " " + var.name;
    Fields<G> f = make_fields(sys, grid, stream);
    std::string order;
    try {
      carbon::Skeleton sk(sys, carbon::Skeleton::Options(var.mode));
      if (var.sub_grid) {
        sk.add(jacobi_interior(grid, f.u_a, labels, f.u_b, "int_ab"));
        sk.add(jacobi_special(special, f.u_a, f.u_b, "s_ab"));
        sk.add(axpy(grid, f.u_b, f.y, "axpy_b"));
        sk.add(jacobi_interior(grid, f.u_b, labels, f.u_a, "int_ba"));
        sk.add(jacobi_special(special, f.u_b, f.u_a, "s_ba"));
        sk.add(axpy(grid, f.u_a, f.y, "axpy_a"));
      } else {
        sk.add(jacobi_full(grid, f.u_a, f.u_b, "jacobi_ab"));
        sk.add(axpy(grid, f.u_b, f.y, "axpy_b"));
        sk.add(jacobi_full(grid, f.u_b, f.u_a, "jacobi_ba"));
        sk.add(axpy(grid, f.u_a, f.y, "axpy_a"));
      }
      sk.compile();
      order = sk.export_launch_order();

      if (var.sub_grid) {
        // The interior and SubGrid writers of u_b cover disjoint regions: no edge between them.
        const auto& to_user = sk.effective_to_user();
        size_t e_int = carbon::Skeleton::kNoKernel;
        size_t e_s = carbon::Skeleton::kNoKernel;
        for (size_t e = 0; e < to_user.size(); ++e) {
          e_int = to_user[e] == 0U ? e : e_int;
          e_s = to_user[e] == 1U ? e : e_s;
        }
        const bool found = e_int != carbon::Skeleton::kNoKernel &&
                           e_s != carbon::Skeleton::kNoKernel && e_int < sk.edges().size();
        expect(found, tag + ": int_ab / s_ab not found in the effective kernel list");
        if (found) {
          const auto& succ = sk.edges()[e_int];
          expect(std::find(succ.begin(), succ.end(), e_s) == succ.end(),
                 tag + ": int_ab and s_ab write disjoint regions but are ordered");
        }
      }

      for (int cycle = 0; cycle < kIters / 2; ++cycle) {
        sk.run();
      }
      sk.wait();
    } catch (const std::exception& e) {
      expect(false, tag + ": threw: " + e.what());
      continue;
    }

    compare(sys, grid, geo, f.u_a, ref.u, stream, tag);
    compare(sys, grid, geo, f.y, ref.y, stream, tag);

    const auto split = [&order](const char* kernel) {
      return contains(order, std::string(kernel) + " [compute/internal]") &&
             contains(order, std::string(kernel) + " [compute/boundary]");
    };
    const std::vector<const char*> stencil_kernels =
        var.sub_grid ? std::vector<const char*>{"int_ab", "s_ab"}
                     : std::vector<const char*>{"jacobi_ab"};
    for (const char* k : stencil_kernels) {
      expect(split(k) == (var.mode != OverlapMode::None),
             tag + ": OCC split of " + k + " should be " +
                 (var.mode != OverlapMode::None ? "present" : "absent") + "\n" + order);
    }
  }
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_WARN("test_sub_grid_jacobi_axpy: no devices, skipping");
    return 0;
  }

  try {
    carbon::System sys(carbon::get_all_devices());
    const auto n_gpus = static_cast<int64_t>(sys.gpu_group().size());

    {
      const Geometry geo{int64_3d{24, 20, 8 * n_gpus}, false};
      carbon::Stencil stencil(
          std::vector<carbon::int32_3d>{
              {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
          /*remove_center=*/true);
      const Dense dense(sys, geo.ext, std::move(stencil));
      run_grid(sys, dense, geo, "dense");
    }
    {
      const int64_t n = std::max<int64_t>(64, SB::MemZ * n_gpus);
      const Geometry geo{int64_3d{n, n, n}, true};
      const B block(
          sys, geo.ext,
          [geo](const int64_3d& g) {
            return geo.active(g);
          },
          "ball");
      run_grid(sys, block, geo, "block");
    }
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_sub_grid_jacobi_axpy: exception: {}", e.what());
    return 1;
  }

  if (g_failures > 0) {
    CARBON_LOG_ERROR("test_sub_grid_jacobi_axpy: {} failure(s)", g_failures);
    return 1;
  }
  CARBON_LOG_INFO("test_sub_grid_jacobi_axpy: OK");
  return 0;
}
