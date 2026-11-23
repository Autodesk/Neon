// SubGrid over a block-sparse grid: a few scattered voxels that need special treatment.
//
// WHAT IT SHOWS
//
//   A Block (block-sparse) grid G holds a ball. A small set of "marker" voxels, about one in
//   --every active voxels and scattered so that most of them sit in different bricks, needs
//   an extra update: here, the average of its x/y neighbours in a parent field.
//
//   Without a SubGrid the update is a Block kernel over every active voxel that reads a label
//   and returns unless the voxel is a marker: one thread per active voxel, almost all idle.
//   S = SubGrid(G, F, B) stores, per GPU, the list of marker voxels and launches one thread
//   per marker. Its kernel reads and writes parent fields at the marker (and its neighbours)
//   with the SubGrid's Span and Idx, which convert to the Block's.
//
//   The example builds S from the label field, checks that both kernels write the same
//   values, and times them for --iters launches each.
//
//   The x/y neighbours of a voxel stay in its GPU's slab (slabs split along z), so these
//   standalone launches need no halo. Under a Skeleton, a stencil read in any direction gets
//   its HaloBlock automatically, for SubGrid kernels as for Block kernels.
//
//   carbon_example_sub_grid_block_markers [--n 128] [--every 1000] [--iters 200]
//
// See docs/skeleton/README.md, "SubGrid", and sub_grid_cube.cu for the Dense version.

#include <carbon/carbon.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using B = carbon::Block<>;
using S = carbon::SubGrid<B>;
using carbon::RegionMask;

constexpr std::uint8_t kMarker = 1;

[[nodiscard]] auto
in_ball(carbon::int64_3d g,
        int64_t n) -> bool {
  const double c = static_cast<double>(n - 1) * 0.5;
  const double r = static_cast<double>(n) * 0.45;
  const double dx = static_cast<double>(g.x) - c;
  const double dy = static_cast<double>(g.y) - c;
  const double dz = static_cast<double>(g.z) - c;
  return dx * dx + dy * dy + dz * dz <= r * r;
}

/** Deterministic scatter: about one voxel in @p every . */
[[nodiscard]] CARBON_HD constexpr auto
is_marker(carbon::int64_3d g,
          int64_t every) -> bool {
  const auto h = (static_cast<std::uint64_t>(g.x) * 73856093ULL) ^
                 (static_cast<std::uint64_t>(g.y) * 19349663ULL) ^
                 (static_cast<std::uint64_t>(g.z) * 83492791ULL);
  return h % static_cast<std::uint64_t>(every) == 0ULL;
}

/** Average of the active x/y neighbours of @p i in @p u (0 when there are none). */
template <typename P,
          typename SpanT,
          typename IdxT>
[[nodiscard]] CARBON_HD auto
xy_average(const P& u,
           const SpanT& s,
           const IdxT& i) -> float {
  const auto xp = u.neighbour(s, i, carbon::int8_3d{1, 0, 0}, 0);
  const auto xm = u.neighbour(s, i, carbon::int8_3d{-1, 0, 0}, 0);
  const auto yp = u.neighbour(s, i, carbon::int8_3d{0, 1, 0}, 0);
  const auto ym = u.neighbour(s, i, carbon::int8_3d{0, -1, 0}, 0);
  const int n = (xp.valid ? 1 : 0) + (xm.valid ? 1 : 0) + (yp.valid ? 1 : 0) + (ym.valid ? 1 : 0);
  const float sum = (xp.valid ? xp.value : 0.0F) + (xm.valid ? xm.value : 0.0F) +
                    (yp.valid ? yp.value : 0.0F) + (ym.valid ? ym.value : 0.0F);
  return n > 0 ? sum / static_cast<float>(n) : 0.0F;
}

// The update without a SubGrid: every active voxel, filtered by its label.
[[nodiscard]] auto
build_filtered(const B& block,
               const B::Field<std::uint8_t,
                              1>& labels,
               const B::Field<float,
                              1>& u,
               B::Field<float,
                        1>& out) -> carbon::Kernel {
  return carbon::Kernel::compute(
      block,
      [labels, u, out](carbon::KernelLoader& loader) {
        const auto lab = loader.read(labels);
        const auto in = loader.read_stencil(u);
        const auto dst = loader.write(out);
        return [lab, in, dst] CARBON_HD(const B::Span& s, const B::Idx& i) {
          if (lab.read(s, i, 0) != kMarker) {
            return;
          }
          dst.write(s, i, 0, xy_average(in, s, i));
        };
      },
      "markers_filtered");
}

// The same update as a SubGrid kernel: one thread per marker, no filter.
[[nodiscard]] auto
build_sub_grid(const S& markers,
               const B::Field<float,
                              1>& u,
               B::Field<float,
                        1>& out) -> carbon::Kernel {
  return carbon::Kernel::compute(
      markers,
      [u, out](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(u);
        const auto dst = loader.write(out);
        return [in, dst] CARBON_HD(const S::Span& s, const S::Idx& i) {
          dst.write(s, i, 0, xy_average(in, s, i));
        };
      },
      "markers_sub_grid");
}

struct Args {
  int64_t n = 128;
  int64_t every = 1000;
  int iters = 200;
};

[[nodiscard]] auto
parse_args(int argc,
           char** argv) -> Args {
  Args a{};
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string_view flag = argv[i];
    if (flag == "--n") {
      a.n = std::atoll(argv[i + 1]);
    } else if (flag == "--every") {
      a.every = std::atoll(argv[i + 1]);
    } else if (flag == "--iters") {
      a.iters = std::atoi(argv[i + 1]);
    } else {
      throw std::invalid_argument("unknown flag " + std::string{flag});
    }
  }
  if (a.n < 8 || a.every < 1 || a.iters < 1) {
    throw std::invalid_argument("need --n >= 8, --every >= 1 and --iters >= 1");
  }
  return a;
}

// Mean microseconds per launch of @p k over @p iters launches, after one warm-up launch.
[[nodiscard]] auto
time_launches(carbon::Stream& stream,
              const carbon::Kernel& k,
              int iters) -> double {
  k.launch(stream);
  stream.wait();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    k.launch(stream);
  }
  stream.wait();
  const std::chrono::duration<double, std::micro> dt = std::chrono::steady_clock::now() - t0;
  return dt.count() / iters;
}

/** Zeroes every storage slot (owned and ghost) of a Block field on the device. */
auto
zero(carbon::System& sys,
     const B& block,
     B::Field<float,
              1>& f,
     carbon::Stream& stream) -> void {
  for (size_t g = 0; g < block.gpu_count(); ++g) {
    const size_t n = static_cast<size_t>(block.storage_block_count(g)) * B::VoxelCount;
    if (n != 0U) {
      sys.async_memset(f.gpu_buffers()[g], 0, n * sizeof(float), stream);
    }
  }
}

}  // namespace

int
main(int argc,
     char** argv) {
  try {
    carbon::init();
    const Args args = parse_args(argc, argv);
    const int64_t n = args.n;
    const int64_t every = args.every;

    carbon::System sys(carbon::get_all_devices());
    const B block(
        sys, carbon::int64_3d{n, n, n},
        [n](const carbon::int64_3d& g) {
          return in_ball(g, n);
        },
        "ball");
    carbon::Stream stream = sys.stream();

    // Labels on the device, then S = SubGrid(G, F, B) with F = "label bit in B".
    auto labels = block.newField<std::uint8_t, 1>(carbon::MemorySide::Device, "labels");
    carbon::Kernel::compute(
        block,
        [labels, every](carbon::KernelLoader& loader) {
          const auto out = loader.write(labels);
          return [out, every] CARBON_HD(const B::Span& s, const B::Idx& i) {
            out.write(s, i, 0, is_marker(s.global_location(i), every) ? kMarker : std::uint8_t{0});
          };
        },
        "label")
        .launch(stream);
    stream.wait();
    const S markers = block.new_sub_grid(labels, RegionMask{kMarker}, "markers");

    std::cout << "sub_grid_block_markers: " << n << "^3 ball on " << sys.gpu_group().size()
              << " GPU(s), " << block.block_count() << " bricks, " << block.active_voxel_count()
              << " active voxels; SubGrid '" << markers.name() << "' has " << markers.total_count()
              << " markers\n";
    for (size_t g = 0; g < markers.gpu_count(); ++g) {
      std::cout << "  GPU " << g << ": " << markers.count(g) << " lanes ("
                << markers.internal_count(g) << " in internal bricks + "
                << markers.boundary_count(g) << " in bricks next to another GPU's slab)\n";
    }

    auto u = block.newField<float, 1>(carbon::MemorySide::Device, "u");
    carbon::Kernel::compute(
        block,
        [u](carbon::KernelLoader& loader) {
          const auto out = loader.write(u);
          return [out] CARBON_HD(const B::Span& s, const B::Idx& i) {
            const carbon::int64_3d g = s.global_location(i);
            out.write(s, i, 0, static_cast<float>((g.x * 7 + g.y * 3 + g.z) % 101 + 1));
          };
        },
        "init_u")
        .launch(stream);
    auto out_filtered = block.newField<float, 1>(carbon::MemorySide::Device, "out_filtered");
    auto out_sub = block.newField<float, 1>(carbon::MemorySide::Device, "out_sub");
    zero(sys, block, out_filtered, stream);
    zero(sys, block, out_sub, stream);

    const carbon::Kernel filtered = build_filtered(block, labels, u, out_filtered);
    const carbon::Kernel sub = build_sub_grid(markers, u, out_sub);
    filtered.launch(stream);
    sub.launch(stream);
    const auto host_filtered = block.download_to_host(out_filtered, stream);
    const auto host_sub = block.download_to_host(out_sub, stream);
    stream.wait();
    int64_t mismatches = 0;
    int64_t written = 0;
    for (size_t g = 0; g < block.gpu_count(); ++g) {
      const size_t owned = static_cast<size_t>(block.block_count(g)) * B::VoxelCount;
      for (size_t k = 0; k < owned; ++k) {
        mismatches += host_filtered[g].data()[k] == host_sub[g].data()[k] ? 0 : 1;
        written += host_sub[g].data()[k] != 0.0F ? 1 : 0;
      }
    }
    std::cout << "voxels written by the SubGrid kernel: " << written
              << ", voxels where the two kernels differ: " << mismatches << "\n\n";

    const double t_filtered = time_launches(stream, filtered, args.iters);
    const double t_sub = time_launches(stream, sub, args.iters);
    std::cout << std::fixed << std::setprecision(1) << "marker update, " << args.iters
              << " launches: Block + label filter " << t_filtered << " us, SubGrid " << t_sub
              << " us per launch\n";

    if (mismatches != 0 || written != markers.total_count()) {
      std::cerr << "sub_grid_block_markers: FAILED\n";
      return EXIT_FAILURE;
    }
    std::cout << "sub_grid_block_markers: OK\n";
  } catch (const std::exception& e) {
    std::cerr << "sub_grid_block_markers: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
