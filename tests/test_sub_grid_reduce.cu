// SubGrid reductions.
//
// Over a multi-GPU Dense with a radius-1 z stencil and a "domain faces" SubGrid, every
// result is compared exactly (integer values) against a host reference over the selection:
//
// - Sum of a parent field, Max of a SubGrid field, and a lane count, on the device.
// - Views: the Internal and Boundary results add up to the Standard one.
// - ReduceFinalize::BlockAtomic, the host reduction, and the universal reduction (device and
//   host launches) agree with the default device reduction.
// - Tokens: Map reads carry the SubGrid mask.
// - Errors: loading another SubGrid's field and ReduceOptions::stage1_threads throw.
// - An empty SubGrid reduces to the operator's identity.
// - Skeleton: a reduction that stencil-reads a parent field after a Dense writer is
//   OCC-split (Internal / Boundary partials + merge) and gives the reference value under
//   every OverlapMode.

#include <carbon/carbon.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using S = carbon::SubGrid<carbon::Dense>;
using carbon::DataView;
using carbon::Dense;

constexpr std::uint8_t kFace = 1;
constexpr std::uint8_t kBall = 2;

[[nodiscard]] auto
label_of(carbon::int64_3d g,
         carbon::int64_3d ext) -> std::uint8_t {
  if (g.x == 0 || g.y == 0 || g.z == 0 || g.x == ext.x - 1 || g.y == ext.y - 1 ||
      g.z == ext.z - 1) {
    return kFace;
  }
  const int64_t dx = g.x - ext.x / 2;
  const int64_t dy = g.y - ext.y / 2;
  const int64_t dz = g.z - ext.z / 2;
  return dx * dx + dy * dy + dz * dz <= 9 ? kBall : std::uint8_t{0};
}

[[nodiscard]] auto
id_of(carbon::int64_3d g) -> int64_t {
  return g.z * 10000 + g.y * 100 + g.x;
}

int g_failures = 0;

auto
expect(bool ok,
       const std::string& what) -> void {
  if (!ok) {
    ++g_failures;
    if (g_failures <= 20) {
      CARBON_LOG_ERROR("test_sub_grid_reduce: {}", what);
    }
  }
}

template <typename Field>
auto
upload(carbon::System& sys,
       Field& f,
       carbon::Stream& stream) -> void {
  carbon::Transfer<typename Field::value_type>(sys, f.cpu_buffers(), f.gpu_buffers())(stream);
  stream.wait();
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

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_WARN("test_sub_grid_reduce: no devices, skipping");
    return 0;
  }

  try {
    carbon::System sys(carbon::get_all_devices());
    const size_t n_gpus = sys.gpu_group().size();
    carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                            /*remove_center=*/true);
    const carbon::int64_3d ext{12, 10, 8 * static_cast<int64_t>(n_gpus)};
    Dense dense(sys, ext, std::move(stencil));
    carbon::Stream stream = sys.stream();

    auto labels = dense.newField<std::uint8_t, 1>(carbon::MemorySide::HostDevice, "labels");
    labels.fill_owned_host([&](carbon::int64_3d g, int) {
      return label_of(g, ext);
    });
    upload(sys, labels, stream);
    auto ids = dense.newField<int64_t, 1>(carbon::MemorySide::HostDevice, "ids");
    ids.fill_owned_host([](carbon::int64_3d g, int) {
      return id_of(g);
    });
    upload(sys, ids, stream);

    const S faces = dense.new_sub_grid(labels, carbon::RegionMask{kFace}, "faces");
    auto weight = faces.newField<int64_t, 1>(carbon::MemorySide::HostDevice, "weight");
    weight.fill_owned_host([](carbon::int64_3d g, int) {
      return (g.x * 7 + g.y * 13) % 101;
    });
    upload(sys, weight, stream);

    // Host references, total and per view.
    int64_t want_sum = 0;
    int64_t want_max = std::numeric_limits<int64_t>::lowest();
    int64_t want_internal_sum = 0;
    for (size_t g = 0; g < n_gpus; ++g) {
      for (int64_t lane = 0; lane < faces.count(g); ++lane) {
        const carbon::int64_3d c = faces.global_location(g, lane);
        want_sum += id_of(c);
        want_max = std::max(want_max, (c.x * 7 + c.y * 13) % 101);
        if (lane < faces.internal_count(g)) {
          want_internal_sum += id_of(c);
        }
      }
    }

    auto sum_ids = [ids](carbon::KernelLoader& L) {
      auto p = L.read(ids);
      return [=] CARBON_HD(const S::Span& s, const S::Idx& i) -> int64_t {
        return p.read(s, i, 0);
      };
    };
    auto sum_ids_host = [ids](carbon::KernelLoader& L) {
      auto p = L.read(ids);
      return [=](const S::Span& s, const S::Idx& i) -> int64_t {
        return p.read(s, i, 0);
      };
    };

    // ---- Device Sum, Max, count -------------------------------------------------------
    carbon::ReductionResult<int64_t> r_sum(sys);
    const carbon::Kernel k_sum = faces.make_reduce_kernel<int64_t>(sum_ids, r_sum);
    expect(k_sum.kind() == carbon::KernelKind::Reduction, "reduce kernel kind");
    k_sum.launch(stream);
    expect(read_back(r_sum, stream) == want_sum, "device Sum of a parent field");

    carbon::ReductionResult<int64_t> r_max(sys);
    carbon::Kernel::reduce<int64_t>(
        faces,
        [weight](carbon::KernelLoader& L) {
          auto w = L.read(weight);
          return [=] CARBON_HD(const S::Span& s, const S::Idx& i) -> int64_t {
            return w.read(s, i, 0);
          };
        },
        r_max, carbon::reduce::Max<int64_t>{}, "max_weight")
        .launch(stream);
    expect(read_back(r_max, stream) == want_max, "device Max of a SubGrid field");

    carbon::ReductionResult<int64_t> r_count(sys);
    faces
        .make_reduce_kernel<int64_t>(
            [](carbon::KernelLoader&) {
              return [] CARBON_HD(const S::Span&, const S::Idx&) -> int64_t {
                return 1;
              };
            },
            r_count)
        .launch(stream);
    expect(read_back(r_count, stream) == faces.total_count(), "lane count");

    // ---- Views ----------------------------------------------------------------------------
    k_sum.launch(stream, DataView::Internal);
    k_sum.launch(stream, DataView::Boundary);
    const int64_t got_internal = read_back(r_sum, stream, DataView::Internal);
    const int64_t got_boundary = r_sum.host_value(DataView::Boundary);
    expect(faces.boundary_count(0) > 0, "test geometry has boundary lanes");
    expect(got_internal == want_internal_sum, "Internal-view sum");
    expect(got_internal + got_boundary == want_sum, "Internal + Boundary == Standard");

    // ---- BlockAtomic, host, universal -----------------------------------------------------
    carbon::ReductionResult<int64_t> r_atomic(sys);
    carbon::reduce::ReduceOptions atomic;
    atomic.finalize = carbon::reduce::ReduceFinalize::BlockAtomic;
    faces.make_reduce_kernel<int64_t>(sum_ids, r_atomic, {}, 0U, atomic).launch(stream);
    expect(read_back(r_atomic, stream) == want_sum, "BlockAtomic Sum");

    carbon::ReductionResult<int64_t> r_host(sys);
    const carbon::Kernel k_host = faces.make_host_reduce_kernel<int64_t>(sum_ids_host, r_host);
    expect(k_host.kind() == carbon::KernelKind::ReductionHost, "host reduce kind");
    k_host.launch(stream);
    expect(read_back(r_host, stream) == want_sum, "host Sum");

    carbon::ReductionResult<int64_t> r_uni_dev(sys);
    carbon::ReductionResult<int64_t> r_uni_host(sys);
    auto sum_ids_universal = [ids](carbon::KernelLoader& L) {
      auto p = L.read(ids);
      return [=] CARBON_HD(const S::Span& s, const S::Idx& i) -> int64_t {
        return p.read(s, i, 0);
      };
    };
    faces.make_universal_reduce_kernel<int64_t>(sum_ids_universal, r_uni_dev).launch(stream);
    faces.make_universal_reduce_kernel<int64_t>(sum_ids_universal, r_uni_host)
        .launch_on_host(stream);
    expect(read_back(r_uni_dev, stream) == want_sum, "universal reduce, device launch");
    expect(read_back(r_uni_host, stream) == want_sum, "universal reduce, host launch");

    // ---- Tokens -----------------------------------------------------------------------------
    for (const carbon::AccessToken& t : k_sum.impl()->tokens()) {
      if (t.resource == carbon::ResourceKind::Field) {
        expect(t.mask == faces.mask(), "reduce field token carries the SubGrid mask");
      }
    }

    // ---- Errors -----------------------------------------------------------------------------
    const S ball = dense.new_sub_grid(labels, carbon::RegionMask{kBall}, "ball");
    auto ball_field = ball.newField<int64_t, 1>(carbon::MemorySide::Device, "ball_field");
    carbon::ReductionResult<int64_t> r_err(sys);
    expect_throws(
        [&]() {
          (void)faces.make_reduce_kernel<int64_t>(
              [ball_field](carbon::KernelLoader& L) {
                auto b = L.read(ball_field);
                return [=] CARBON_HD(const S::Span& s, const S::Idx& i) -> int64_t {
                  return b.read(s, i, 0);
                };
              },
              r_err);
        },
        "reducing another SubGrid's field");
    carbon::reduce::ReduceOptions threads;
    threads.stage1_threads = 128U;
    expect_throws(
        [&]() {
          (void)faces.make_reduce_kernel<int64_t>(sum_ids, r_err, {}, 0U, threads);
        },
        "ReduceOptions::stage1_threads");

    // ---- Empty SubGrid ----------------------------------------------------------------------
    const S empty = dense.new_sub_grid(labels, carbon::RegionMask{9});
    carbon::ReductionResult<int64_t> r_empty_sum(sys);
    carbon::ReductionResult<int64_t> r_empty_max(sys);
    empty.make_reduce_kernel<int64_t>(sum_ids, r_empty_sum).launch(stream);
    empty
        .make_reduce_kernel<int64_t>(
            [ids](carbon::KernelLoader& L) {
              auto p = L.read(ids);
              return [=] CARBON_HD(const S::Span& s, const S::Idx& i) -> int64_t {
                return p.read(s, i, 0);
              };
            },
            r_empty_max, carbon::reduce::Max<int64_t>{})
        .launch(stream);
    expect(read_back(r_empty_sum, stream) == 0, "empty SubGrid Sum is 0");
    expect(read_back(r_empty_max, stream) == std::numeric_limits<int64_t>::lowest(),
           "empty SubGrid Max is the identity");

    // ---- Skeleton: OCC-split reduction over a stencil read ----------------------------------
    constexpr int kIters = 3;
    // After kIters bumps every cell holds id + kIters; the reduction sums z neighbours.
    int64_t want_zsum = 0;
    for (size_t g = 0; g < n_gpus; ++g) {
      for (int64_t lane = 0; lane < faces.count(g); ++lane) {
        const carbon::int64_3d c = faces.global_location(g, lane);
        if (c.z + 1 < ext.z) {
          want_zsum += id_of({c.x, c.y, c.z + 1}) + kIters;
        }
        if (c.z > 0) {
          want_zsum += id_of({c.x, c.y, c.z - 1}) + kIters;
        }
      }
    }
    for (const carbon::OverlapMode mode :
         {carbon::OverlapMode::None, carbon::OverlapMode::Standard, carbon::OverlapMode::Extended,
          carbon::OverlapMode::TwoWayExtended}) {
      const std::string tag = "skeleton overlap=" + std::to_string(static_cast<int>(mode));
      auto u = dense.newField<int64_t, 1>(carbon::MemorySide::HostDevice, "u");
      u.fill_owned_host([](carbon::int64_3d g, int) {
        return id_of(g);
      });
      upload(sys, u, stream);
      carbon::ReductionResult<int64_t> r_z(sys);
      carbon::Skeleton sk(sys, carbon::Skeleton::Options(mode));
      sk.add(carbon::Kernel::compute(
          dense,
          [u](carbon::KernelLoader& L) {
            auto p = L.read_write(u);
            return [=] CARBON_HD(const Dense::Span& s, const Dense::Idx& i) {
              p.write(s, i, 0, p.read(s, i, 0) + 1);
            };
          },
          "bump"));
      sk.add(carbon::Kernel::reduce<int64_t>(
          faces,
          [u](carbon::KernelLoader& L) {
            auto p = L.read_stencil(u);
            return [=] CARBON_HD(const S::Span& s, const S::Idx& i) -> int64_t {
              const auto up = p.neighbour(s, i, carbon::int8_3d{0, 0, 1}, 0);
              const auto dn = p.neighbour(s, i, carbon::int8_3d{0, 0, -1}, 0);
              return (up.valid ? up.value : 0) + (dn.valid ? dn.value : 0);
            };
          },
          r_z, carbon::reduce::Sum<int64_t>{}, "s_zsum"));
      sk.compile();
      const std::string order = sk.export_launch_order();
      for (int it = 0; it < kIters; ++it) {
        sk.run();
      }
      sk.wait();
      expect(read_back(r_z, stream) == want_zsum, tag + ": reduction value");
      const bool split = order.find("s_zsum [reduction/internal]") != std::string::npos &&
                         order.find("s_zsum [reduction/boundary]") != std::string::npos;
      expect(split == (mode != carbon::OverlapMode::None),
             tag + ": OCC split of the SubGrid reduction\n" + order);
    }
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_sub_grid_reduce: exception: {}", e.what());
    return 1;
  }

  if (g_failures > 0) {
    CARBON_LOG_ERROR("test_sub_grid_reduce: {} failure(s)", g_failures);
    return 1;
  }
  CARBON_LOG_INFO("test_sub_grid_reduce: OK");
  return 0;
}
