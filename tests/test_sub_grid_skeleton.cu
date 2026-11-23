// SubGrid kernels under the Skeleton (milestone 3).
//
// Over a multi-GPU Dense with a radius-1 z stencil and a "domain faces" SubGrid:
//
// A. A pipeline mixing parent and SubGrid kernels -
//      bump      (Dense, Map)      u = 0.5 u + 1
//      s_smooth  (S, stencil on u) acc += u(z-1) + u(z+1)        [S field]
//      s_scatter (S, Map)          v = 0.25 acc at the S cells   [parent field]
//      d_stencil (Dense, stencil)  w = v + v(z+1) - v(z-1)
//      back      (Dense, Map)      u += 0.001 w
//    is run for a few iterations under every OverlapMode, with a CUDA graph, and with
//    region masks ignored, and every field must match a hand-sequenced single-stream run
//    with explicit halos bit for bit. Standard OCC must split s_smooth (an S kernel that
//    stencil-reads a parent field after its halo); Extended OCC must also split s_scatter,
//    the S kernel producing d_stencil's stencil input.
// B. Two SubGrids with disjoint masks writing the same parent field get no edge and run on
//    different streams; with masks ignored they are ordered. Both write their cells.

#include <carbon/carbon.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

namespace {

using S = carbon::SubGrid<carbon::Dense>;
using carbon::Dense;

constexpr std::uint8_t kFace = 1;
constexpr std::uint8_t kBall = 2;
constexpr int kIters = 3;

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

int g_failures = 0;

auto
expect(bool ok,
       const std::string& what) -> void {
  if (!ok) {
    ++g_failures;
    if (g_failures <= 20) {
      CARBON_LOG_ERROR("test_sub_grid_skeleton: {}", what);
    }
  }
}

template <typename Field>
auto
transfer(carbon::System& sys,
         Field& f,
         carbon::Stream& stream,
         bool to_device) -> void {
  using T = typename Field::value_type;
  if (to_device) {
    carbon::Transfer<T>(sys, f.cpu_buffers(), f.gpu_buffers())(stream);
  } else {
    carbon::Transfer<T>(sys, f.gpu_buffers(), f.cpu_buffers())(stream);
  }
  stream.wait();
}

struct Fields {
  Dense::Field<float, 1> u;
  Dense::Field<float, 1> v;
  Dense::Field<float, 1> w;
  S::Field<float, 1> acc;
};

[[nodiscard]] auto
make_fields(carbon::System& sys,
            const Dense& dense,
            const S& sub,
            carbon::Stream& stream) -> Fields {
  Fields f{dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u"),
           dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v"),
           dense.newField<float, 1>(carbon::MemorySide::HostDevice, "w"),
           sub.newField<float, 1>(carbon::MemorySide::HostDevice, "acc")};
  f.u.fill_owned_host([](carbon::int64_3d g, int) {
    return static_cast<float>((g.x * 7 + g.y * 3 + g.z * 5) % 17) * 0.125F;
  });
  f.v.fill_owned_host([](carbon::int64_3d g, int) {
    return static_cast<float>(g.z) * 0.5F;
  });
  f.w.fill_owned_host([](carbon::int64_3d, int) {
    return 0.0F;
  });
  f.acc.fill_owned_host([](carbon::int64_3d g, int) {
    return static_cast<float>(g.x) * 0.25F;
  });
  transfer(sys, f.u, stream, true);
  transfer(sys, f.v, stream, true);
  transfer(sys, f.w, stream, true);
  transfer(sys, f.acc, stream, true);
  return f;
}

[[nodiscard]] CARBON_HD inline auto
or_zero(carbon::NeighbourValue<float> n) -> float {
  return n.valid ? n.value : 0.0F;
}

// User kernels of pipeline A, in program order.
[[nodiscard]] auto
pipeline(const Dense& dense,
         const S& sub,
         const Fields& f) -> std::vector<carbon::Kernel> {
  std::vector<carbon::Kernel> ks;
  ks.push_back(carbon::Kernel::compute(
      dense,
      [u = f.u](carbon::KernelLoader& L) {
        auto pu = L.read_write(u);
        return [=] CARBON_HD(const Dense::Span& s, const Dense::Idx& i) {
          pu.write(s, i, 0, 0.5F * pu.read(s, i, 0) + 1.0F);
        };
      },
      "bump"));
  ks.push_back(carbon::Kernel::compute(
      sub,
      [u = f.u, acc = f.acc](carbon::KernelLoader& L) {
        auto pu = L.read_stencil(u);
        auto pa = L.read_write(acc);
        return [=] CARBON_HD(const S::Span& s, const S::Idx& i) {
          const float up = or_zero(pu.neighbour(s, i, carbon::int8_3d{0, 0, 1}, 0));
          const float dn = or_zero(pu.neighbour(s, i, carbon::int8_3d{0, 0, -1}, 0));
          pa.write(s, i, 0, pa.read(s, i, 0) + up + dn);
        };
      },
      "s_smooth"));
  ks.push_back(sub.make_kernel(
      [acc = f.acc, v = f.v](carbon::KernelLoader& L) {
        auto pa = L.read(acc);
        auto pv = L.write(v);
        return [=] CARBON_HD(const S::Span& s, const S::Idx& i) {
          pv.write(s, i, 0, 0.25F * pa.read(s, i, 0));
        };
      },
      "s_scatter"));
  ks.push_back(carbon::Kernel::compute(
      dense,
      [v = f.v, w = f.w](carbon::KernelLoader& L) {
        auto pv = L.read_stencil(v);
        auto pw = L.write(w);
        return [=] CARBON_HD(const Dense::Span& s, const Dense::Idx& i) {
          const float up = or_zero(pv.neighbour(s, i, carbon::int8_3d{0, 0, 1}, 0));
          const float dn = or_zero(pv.neighbour(s, i, carbon::int8_3d{0, 0, -1}, 0));
          pw.write(s, i, 0, pv.read(s, i, 0) + up - dn);
        };
      },
      "d_stencil"));
  ks.push_back(carbon::Kernel::compute(
      dense,
      [u = f.u, w = f.w](carbon::KernelLoader& L) {
        auto pw = L.read(w);
        auto pu = L.read_write(u);
        return [=] CARBON_HD(const Dense::Span& s, const Dense::Idx& i) {
          pu.write(s, i, 0, pu.read(s, i, 0) + 0.001F * pw.read(s, i, 0));
        };
      },
      "back"));
  return ks;
}

// Single stream, program order, halos placed by hand.
auto
run_reference(const Dense& dense,
              const S& sub,
              Fields& f,
              carbon::Stream& stream) -> void {
  const std::vector<carbon::Kernel> ks = pipeline(dense, sub, f);
  const carbon::Kernel halo_u = f.u.make_halo_kernel("halo_u");
  const carbon::Kernel halo_v = f.v.make_halo_kernel("halo_v");
  for (int it = 0; it < kIters; ++it) {
    ks[0].launch(stream);
    halo_u.launch(stream);
    ks[1].launch(stream);
    ks[2].launch(stream);
    halo_v.launch(stream);
    ks[3].launch(stream);
    ks[4].launch(stream);
  }
  stream.wait();
}

auto
compare_owned(const Dense& dense,
              const Dense::Field<float,
                                 1>& a,
              const Dense::Field<float,
                                 1>& b,
              const std::string& tag) -> void {
  for (size_t g = 0; g < dense.gpu_count(); ++g) {
    const auto& span = dense.spans()[g];
    const auto& pa = a.host_partitions()[g];
    const auto& pb = b.host_partitions()[g];
    const int64_t h = span.halo_z();
    const carbon::int64_3d ext = span.haloed_extent();
    int64_t bad = 0;
    for (int64_t z = h; z < h + span.logic_extent().z; ++z) {
      for (int64_t y = 0; y < ext.y; ++y) {
        for (int64_t x = 0; x < ext.x; ++x) {
          const Dense::Idx idx(carbon::int64_3d{x, y, z});
          bad += pa.read(span, idx, 0) != pb.read(span, idx, 0) ? 1 : 0;
        }
      }
    }
    expect(bad == 0, tag + ": field '" + a.name() + "' differs on gpu " + std::to_string(g) +
                         " in " + std::to_string(bad) + " cell(s)");
  }
}

auto
compare_sub(const S::Field<float,
                           1>& a,
            const S::Field<float,
                           1>& b,
            const std::string& tag) -> void {
  for (size_t g = 0; g < a.cpu_buffers().size(); ++g) {
    int64_t bad = 0;
    for (size_t k = 0; k < a.cpu_buffers()[g].size(); ++k) {
      bad += a.cpu_buffers()[g].data()[k] != b.cpu_buffers()[g].data()[k] ? 1 : 0;
    }
    expect(bad == 0, tag + ": SubGrid field differs on gpu " + std::to_string(g) + " in " +
                         std::to_string(bad) + " lane(s)");
  }
}

[[nodiscard]] auto
contains(const std::string& text,
         const std::string& needle) -> bool {
  return text.find(needle) != std::string::npos;
}

auto
subtest_pipeline(carbon::System& sys,
                 const Dense& dense,
                 const S& sub) -> void {
  carbon::Stream stream = sys.stream();
  Fields ref = make_fields(sys, dense, sub, stream);
  run_reference(dense, sub, ref, stream);
  transfer(sys, ref.u, stream, false);
  transfer(sys, ref.v, stream, false);
  transfer(sys, ref.w, stream, false);
  transfer(sys, ref.acc, stream, false);

  struct Case {
    std::string tag;
    carbon::Skeleton::Options opts;
  };
  std::vector<Case> cases;
  for (const carbon::OverlapMode mode :
       {carbon::OverlapMode::None, carbon::OverlapMode::Standard, carbon::OverlapMode::Extended,
        carbon::OverlapMode::TwoWayExtended}) {
    cases.push_back(
        {"overlap=" + std::to_string(static_cast<int>(mode)), carbon::Skeleton::Options(mode)});
  }
  cases.push_back({"graph", carbon::Skeleton::Options(carbon::OverlapMode::None, false, 0U, false,
                                                      /*cuda_graph=*/true)});
  carbon::Skeleton::Options ignored(carbon::OverlapMode::Standard);
  ignored.ignore_region_masks = true;
  cases.push_back({"standard/ignore_region_masks", ignored});

  for (const Case& c : cases) {
    Fields f = make_fields(sys, dense, sub, stream);
    std::string order;
    try {
      carbon::Skeleton sk(sys, c.opts);
      for (carbon::Kernel& k : pipeline(dense, sub, f)) {
        sk.add(std::move(k));
      }
      sk.compile();
      order = sk.export_launch_order();
      for (int it = 0; it < kIters; ++it) {
        sk.run();
      }
      sk.wait();
    } catch (const std::exception& e) {
      expect(false, c.tag + ": threw: " + e.what());
      continue;
    }
    transfer(sys, f.u, stream, false);
    transfer(sys, f.v, stream, false);
    transfer(sys, f.w, stream, false);
    transfer(sys, f.acc, stream, false);
    compare_owned(dense, f.u, ref.u, c.tag);
    compare_owned(dense, f.v, ref.v, c.tag);
    compare_owned(dense, f.w, ref.w, c.tag);
    compare_sub(f.acc, ref.acc, c.tag);

    const bool split_smooth = contains(order, "s_smooth [compute/internal]") &&
                              contains(order, "s_smooth [compute/boundary]");
    const bool split_scatter = contains(order, "s_scatter [compute/internal]") &&
                               contains(order, "s_scatter [compute/boundary]");
    const carbon::OverlapMode mode = c.opts.overlap_mode;
    if (mode == carbon::OverlapMode::None) {
      expect(!split_smooth && !split_scatter, c.tag + ": no kernel should be split\n" + order);
    } else {
      expect(split_smooth, c.tag + ": s_smooth should be OCC-split\n" + order);
    }
    if (mode == carbon::OverlapMode::Extended || mode == carbon::OverlapMode::TwoWayExtended) {
      expect(split_scatter, c.tag + ": s_scatter should be split by Extended OCC\n" + order);
    }
  }
}

auto
subtest_disjoint(carbon::System& sys,
                 const Dense& dense,
                 const S& faces,
                 const S& ball) -> void {
  carbon::Stream stream = sys.stream();
  auto v = dense.newField<int, 1>(carbon::MemorySide::HostDevice, "v_marks");

  auto marker = [&v](const S& sub, int value, const char* name) {
    return sub.make_kernel(
        [v, value](carbon::KernelLoader& L) {
          auto pv = L.write(v);
          return [=] CARBON_HD(const S::Span& s, const S::Idx& i) {
            pv.write(s, i, 0, value);
          };
        },
        name);
  };

  for (const bool ignore : {false, true}) {
    v.fill_owned_host([](carbon::int64_3d, int) {
      return -1;
    });
    transfer(sys, v, stream, true);
    carbon::Skeleton::Options opts;
    opts.ignore_region_masks = ignore;
    const std::string tag = ignore ? "disjoint/ignored" : "disjoint";
    try {
      carbon::Skeleton sk(sys, opts);
      sk.add(marker(faces, 1, "faces_mark"));
      sk.add(marker(ball, 2, "ball_mark"));
      sk.compile();
      using Edges = std::vector<std::vector<size_t>>;
      if (ignore) {
        expect(sk.edges() == Edges{{1U}, {}}, tag + ": unmasked writers must be ordered");
      } else {
        expect(sk.edges() == Edges{{}, {}}, tag + ": disjoint SubGrids must not be ordered");
        expect(sk.kernel_streams()[0] != sk.kernel_streams()[1],
               tag + ": disjoint SubGrids should run on different streams");
      }
      sk.run();
      sk.wait();
    } catch (const std::exception& e) {
      expect(false, tag + ": threw: " + e.what());
      continue;
    }
    transfer(sys, v, stream, false);
    const carbon::int64_3d ext = dense.global_extent();
    for (size_t g = 0; g < dense.gpu_count(); ++g) {
      const auto& span = dense.spans()[g];
      const auto& pv = v.host_partitions()[g];
      const int64_t h = span.halo_z();
      int64_t bad = 0;
      for (int64_t z = h; z < h + span.logic_extent().z; ++z) {
        for (int64_t y = 0; y < ext.y; ++y) {
          for (int64_t x = 0; x < ext.x; ++x) {
            const Dense::Idx idx(carbon::int64_3d{x, y, z});
            const std::uint8_t label = label_of(span.global_location(idx), ext);
            const int want = label == kFace ? 1 : (label == kBall ? 2 : -1);
            bad += pv.read(span, idx, 0) != want ? 1 : 0;
          }
        }
      }
      expect(bad == 0, tag + ": wrong marks on gpu " + std::to_string(g));
    }
  }
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_WARN("test_sub_grid_skeleton: no devices, skipping");
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
    transfer(sys, labels, stream, true);

    const S faces = dense.new_sub_grid(labels, carbon::RegionMask{kFace}, "faces");
    const S ball = dense.new_sub_grid(labels, carbon::RegionMask{kBall}, "ball");

    subtest_pipeline(sys, dense, faces);
    subtest_disjoint(sys, dense, faces, ball);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_sub_grid_skeleton: exception: {}", e.what());
    return 1;
  }

  if (g_failures > 0) {
    CARBON_LOG_ERROR("test_sub_grid_skeleton: {} failure(s)", g_failures);
    return 1;
  }
  CARBON_LOG_INFO("test_sub_grid_skeleton: OK");
  return 0;
}
