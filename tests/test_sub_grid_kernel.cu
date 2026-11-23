// SubGrid kernel test (milestone 2: S.make_kernel / S.make_host_kernel).
//
// Over a multi-GPU Dense with a radius-1 z stencil and a "domain faces" SubGrid S:
//
// - Map: an S kernel reads a parent field through the S span / idx and writes an S field.
// - Views: Internal + Boundary launches touch every lane exactly once; Internal alone touches
//   exactly the first internal_count lanes.
// - Scatter: an S kernel writes a parent field only at the selected cells.
// - Stencil on a parent field: after the parent halo, an S kernel reads z neighbours through
//   read_stencil and matches a host reference, including across GPU slabs.
// - Host kernel: make_host_kernel gives the same result as the device kernel.
// - Tokens: Map accesses carry the SubGrid mask, stencil reads stay all ones.
// - Ownership: loading another SubGrid's field, or an S field from a Dense kernel, throws.
// - An empty SubGrid launches as a no-op.

#include <carbon/carbon.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using S = carbon::SubGrid<carbon::Dense>;
using carbon::DataView;

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
value_of(carbon::int64_3d g) -> float {
  return static_cast<float>(g.z * 10000 + g.y * 100 + g.x);
}

int g_failures = 0;

auto
expect(bool ok,
       const std::string& what) -> void {
  if (!ok) {
    ++g_failures;
    if (g_failures <= 20) {
      CARBON_LOG_ERROR("test_sub_grid_kernel: {}", what);
    }
  }
}

template <typename Field>
auto
download(carbon::System& sys,
         Field& f,
         carbon::Stream& stream) -> void {
  carbon::Transfer<typename Field::value_type> d(sys, f.gpu_buffers(), f.cpu_buffers());
  d(stream);
  stream.wait();
}

template <typename Field>
auto
upload(carbon::System& sys,
       Field& f,
       carbon::Stream& stream) -> void {
  carbon::Transfer<typename Field::value_type> u(sys, f.cpu_buffers(), f.gpu_buffers());
  u(stream);
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

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_WARN("test_sub_grid_kernel: no devices, skipping");
    return 0;
  }

  try {
    carbon::System sys(carbon::get_all_devices());
    const size_t n_gpus = sys.gpu_group().size();
    carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                            /*remove_center=*/true);
    const carbon::int64_3d ext{12, 10, 8 * static_cast<int64_t>(n_gpus)};
    carbon::Dense dense(sys, ext, std::move(stencil));
    carbon::Stream stream = sys.stream();

    auto labels = dense.newField<std::uint8_t, 1>(carbon::MemorySide::HostDevice, "labels");
    labels.fill_owned_host([&](carbon::int64_3d g, int) {
      return label_of(g, ext);
    });
    upload(sys, labels, stream);

    auto u = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u");
    u.fill_owned_host([](carbon::int64_3d g, int) {
      return value_of(g);
    });
    upload(sys, u, stream);

    const S faces = dense.new_sub_grid(labels, carbon::RegionMask{kFace}, "faces");

    // ---- Map: parent read -> S write --------------------------------------------------
    auto twice = faces.newField<float, 1>(carbon::MemorySide::HostDevice, "twice");
    const carbon::Kernel map_k = faces.make_kernel(
        [u, twice](carbon::KernelLoader& L) {
          auto uG = L.read(u);
          auto uS = L.write(twice);
          return [=] CARBON_HD(const S::Span& s, const S::Idx& i) {
            uS.write(s, i, 0, 2.0F * uG.read(s, i, 0));
          };
        },
        "map");
    expect(map_k.kind() == carbon::KernelKind::Compute, "map kernel kind");
    expect(map_k.name() == "map", "map kernel name");
    map_k.launch(stream);
    stream.wait();
    download(sys, twice, stream);
    for (size_t g = 0; g < n_gpus; ++g) {
      for (int64_t lane = 0; lane < faces.count(g); ++lane) {
        const float want = 2.0F * value_of(faces.global_location(g, lane));
        expect(twice.cpu_buffers()[g].data()[lane] == want,
               "map gpu " + std::to_string(g) + " lane " + std::to_string(lane));
      }
    }

    // Tokens: Map accesses carry the SubGrid mask.
    for (const carbon::AccessToken& t : map_k.impl()->tokens()) {
      expect(t.mask == faces.mask(), "map token '" + t.field_name + "' mask is the SubGrid mask");
    }

    // ---- Views: Internal + Boundary == Standard, each lane once -------------------------
    auto hits = faces.newField<int, 1>(carbon::MemorySide::HostDevice, "hits");
    const carbon::Kernel inc_k = faces.make_kernel([hits](carbon::KernelLoader& L) {
      auto h = L.read_write(hits);
      return [=] CARBON_HD(const S::Span& s, const S::Idx& i) {
        h.write(s, i, 0, h.read(s, i, 0) + 1);
      };
    });
    auto zero_hits = [&]() {
      hits.fill_owned_host([](carbon::int64_3d, int) {
        return 0;
      });
      upload(sys, hits, stream);
    };
    zero_hits();
    inc_k.launch(stream, DataView::Internal);
    inc_k.launch(stream, DataView::Boundary);
    stream.wait();
    download(sys, hits, stream);
    for (size_t g = 0; g < n_gpus; ++g) {
      for (int64_t lane = 0; lane < faces.count(g); ++lane) {
        expect(hits.cpu_buffers()[g].data()[lane] == 1, "internal+boundary hits lane once");
      }
    }
    zero_hits();
    inc_k.launch(stream, DataView::Internal);
    stream.wait();
    download(sys, hits, stream);
    for (size_t g = 0; g < n_gpus; ++g) {
      expect(faces.internal_count(g) > 0 && faces.boundary_count(g) > 0,
             "test geometry has both internal and boundary lanes");
      for (int64_t lane = 0; lane < faces.count(g); ++lane) {
        const int want = lane < faces.internal_count(g) ? 1 : 0;
        expect(hits.cpu_buffers()[g].data()[lane] == want, "internal-only launch lane set");
      }
    }

    // ---- Scatter: S kernel writes a parent field at the selected cells only -------------
    auto marks = dense.newField<int, 1>(carbon::MemorySide::HostDevice, "marks");
    marks.fill_owned_host([](carbon::int64_3d, int) {
      return 0;
    });
    upload(sys, marks, stream);
    faces
        .make_kernel([marks](carbon::KernelLoader& L) {
          auto m = L.write(marks);
          return [=] CARBON_HD(const S::Span& s, const S::Idx& i) {
            m.write(s, i, 0, 7);
          };
        })
        .launch(stream);
    stream.wait();
    download(sys, marks, stream);
    for (size_t g = 0; g < n_gpus; ++g) {
      const auto& span = dense.spans()[g];
      const auto& mp = marks.host_partitions()[g];
      const int64_t h = span.halo_z();
      for (int64_t z = h; z < h + span.logic_extent().z; ++z) {
        for (int64_t y = 0; y < ext.y; ++y) {
          for (int64_t x = 0; x < ext.x; ++x) {
            const carbon::Dense::Idx idx(carbon::int64_3d{x, y, z});
            const int want = label_of(span.global_location(idx), ext) == kFace ? 7 : 0;
            expect(mp.read(span, idx, 0) == want, "scatter to parent field");
          }
        }
      }
    }

    // ---- Stencil on a parent field (through the parent halo) ----------------------------
    auto zsum = faces.newField<float, 1>(carbon::MemorySide::HostDevice, "zsum");
    u.make_halo_kernel("halo_u").launch(stream);
    const carbon::Kernel stencil_k = faces.make_kernel([u, zsum](carbon::KernelLoader& L) {
      auto uG = L.read_stencil(u);
      auto out = L.write(zsum);
      return [=] CARBON_HD(const S::Span& s, const S::Idx& i) {
        const auto up = uG.neighbour(s, i, carbon::int8_3d{0, 0, 1}, 0);
        const auto dn = uG.neighbour(s, i, carbon::int8_3d{0, 0, -1}, 0);
        out.write(s, i, 0, (up.valid ? up.value : 0.0F) + (dn.valid ? dn.value : 0.0F));
      };
    });
    bool saw_stencil = false;
    for (const carbon::AccessToken& t : stencil_k.impl()->tokens()) {
      if (t.pattern == carbon::AccessPattern::Stencil) {
        saw_stencil = true;
        expect(t.mask.is_all(), "stencil token keeps the all-ones mask");
      }
    }
    expect(saw_stencil, "stencil kernel recorded a stencil token");
    stencil_k.launch(stream);
    stream.wait();
    download(sys, zsum, stream);
    for (size_t g = 0; g < n_gpus; ++g) {
      for (int64_t lane = 0; lane < faces.count(g); ++lane) {
        const carbon::int64_3d c = faces.global_location(g, lane);
        float want = 0.0F;
        if (c.z + 1 < ext.z) {
          want += value_of({c.x, c.y, c.z + 1});
        }
        if (c.z > 0) {
          want += value_of({c.x, c.y, c.z - 1});
        }
        expect(zsum.cpu_buffers()[g].data()[lane] == want,
               "parent stencil gpu " + std::to_string(g) + " lane " + std::to_string(lane));
      }
    }

    // ---- Host kernel ----------------------------------------------------------------------
    auto twice_host = faces.newField<float, 1>(carbon::MemorySide::HostDevice, "twice_host");
    const carbon::Kernel host_k = faces.make_host_kernel([u, twice_host](carbon::KernelLoader& L) {
      auto uG = L.read(u);
      auto uS = L.write(twice_host);
      return [=](const S::Span& s, const S::Idx& i) {
        uS.write(s, i, 0, 2.0F * uG.read(s, i, 0));
      };
    });
    expect(host_k.kind() == carbon::KernelKind::ComputeHost, "host kernel kind");
    host_k.launch(stream);
    for (size_t g = 0; g < n_gpus; ++g) {
      for (int64_t lane = 0; lane < faces.count(g); ++lane) {
        expect(twice_host.cpu_buffers()[g].data()[lane] == twice.cpu_buffers()[g].data()[lane],
               "host kernel matches device kernel");
      }
    }

    // ---- Ownership -----------------------------------------------------------------------
    const S ball = dense.new_sub_grid(labels, carbon::RegionMask{kBall}, "ball");
    auto ball_field = ball.newField<float, 1>(carbon::MemorySide::Device, "ball_field");
    expect_throws(
        [&]() {
          (void)faces.make_kernel([ball_field](carbon::KernelLoader& L) {
            auto b = L.read(ball_field);
            return [=] CARBON_HD(const S::Span& s, const S::Idx& i) {
              (void)b.read(s, i, 0);
            };
          });
        },
        "loading another SubGrid's field");
    expect_throws(
        [&]() {
          (void)dense.make_kernel([ball_field](carbon::KernelLoader& L) {
            (void)L.read(ball_field);
            return [] CARBON_HD(const carbon::Dense::Span&, const carbon::Dense::Idx&) {
            };
          });
        },
        "loading a SubGrid field from a Dense kernel");

    // ---- Empty SubGrid -------------------------------------------------------------------
    const S empty = dense.new_sub_grid(labels, carbon::RegionMask{9});
    auto empty_field = empty.newField<float, 1>(carbon::MemorySide::Device, "empty_field");
    const carbon::Kernel empty_k = empty.make_kernel([empty_field](carbon::KernelLoader& L) {
      auto e = L.write(empty_field);
      return [=] CARBON_HD(const S::Span& s, const S::Idx& i) {
        e.write(s, i, 0, 1.0F);
      };
    });
    for (const DataView v : {DataView::Standard, DataView::Internal, DataView::Boundary}) {
      empty_k.launch(stream, v);
    }
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_sub_grid_kernel: exception: {}", e.what());
    return 1;
  }

  if (g_failures > 0) {
    CARBON_LOG_ERROR("test_sub_grid_kernel: {} failure(s)", g_failures);
    return 1;
  }
  CARBON_LOG_INFO("test_sub_grid_kernel: OK");
  return 0;
}
