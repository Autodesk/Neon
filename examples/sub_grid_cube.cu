// SubGrid, the smallest complete example: the boundary voxels of a cube as their own grid.
//
// WHAT IT SHOWS
//
//   region_mask_cube.cu writes the boundary of a cube with a Dense kernel that declares the
//   boundary region and then has to skip every interior voxel itself: the launch still covers
//   the whole cube. A SubGrid S = SubGrid(G, F, B) instead stores, per GPU, the list of parent
//   voxels selected by F and launches one thread per selected voxel. Here G is the cube, F is
//   "on a face of the cube" and B is the boundary region bit.
//
// WHAT A SUBGRID KERNEL CAN DO
//
//   - read / write parent fields at its voxel: `u.read(s, i, 0)` with the SubGrid's Span and
//     Idx, which convert to the parent's;
//   - read / write its own compact fields (one value per selected voxel, nothing elsewhere);
//   - stencil-read a parent field through the parent's halo (not used here). SubGrid fields
//     have no halo, so they cannot be stencil-read.
//
//   Its parent-field accesses are declared with region B automatically, and the kernel only
//   ever visits B voxels, so the "keep the promise" filter of region_mask_cube is gone.
//
// THE PIPELINE (parent field u, compact SubGrid field half)
//
//     1. fill_boundary    SubGrid  u = 1                 region: boundary (from the SubGrid)
//     2. fill_interior    Dense    u = 2                 region: interior (filters itself)
//     3. scale_interior   Dense    u = 10 * u            region: interior (filters itself)
//     4. sample_boundary  SubGrid  half = 0.5 * u        region: boundary
//     5. count_wrong      Dense    sums wrong voxels of u, whole cube
//
//   Expected dependencies (masks honoured): 1 and 2 write disjoint regions of u, so they run
//   side by side; 3 waits for 2; 4 waits for 1 only; 5 waits for 1 and 3.
//
// It then times the boundary fill both ways, a Dense kernel over the whole cube that skips
// interior voxels versus the SubGrid kernel, for --iters launches each.
//
// The per-GPU lane counts are split into lanes away from and next to a GPU-slab boundary,
// the two DataViews OCC launches separately. This grid declares no stencil, so it has no
// halo band and every lane is "internal".
//
//   carbon_example_sub_grid_cube [--n 128] [--iters 200] [--dot sub_grid_cube]
//
// See docs/skeleton/README.md, "SubGrid", and region_mask_cube.cu.

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
#include <vector>

namespace {

using Span = carbon::Dense::Span;
using Idx = carbon::Dense::Idx;
using Field = carbon::Dense::Field<float, 1>;
using S = carbon::SubGrid<carbon::Dense>;
using carbon::RegionMask;

const RegionMask kInterior = RegionMask::bit(0);
const RegionMask kBoundary = RegionMask::bit(1);

[[nodiscard]] CARBON_HD constexpr auto
on_boundary(carbon::int64_3d g,
            int64_t n) noexcept -> bool {
  return g.x == 0 || g.y == 0 || g.z == 0 || g.x == n - 1 || g.y == n - 1 || g.z == n - 1;
}

constexpr float kBoundaryValue = 1.0F;
constexpr float kInteriorValue = 2.0F;
constexpr float kInteriorScale = 10.0F;
constexpr float kSampleScale = 0.5F;

// ---- SubGrid kernels: no geometric filter, the SubGrid only launches boundary voxels -------

[[nodiscard]] auto
build_fill_boundary(const S& boundary,
                    Field& u) -> carbon::Kernel {
  return carbon::Kernel::compute(
      boundary,
      [&u](carbon::KernelLoader& loader) {
        const auto out = loader.write(u);  // parent field, declared with region kBoundary
        return [out] CARBON_HD(const S::Span& s, const S::Idx& i) {
          out.write(s, i, 0, kBoundaryValue);
        };
      },
      "fill_boundary");
}

[[nodiscard]] auto
build_sample_boundary(const S& boundary,
                      Field& u,
                      S::Field<float,
                               1>& half) -> carbon::Kernel {
  return carbon::Kernel::compute(
      boundary,
      [&u, &half](carbon::KernelLoader& loader) {
        const auto in = loader.read(u);
        const auto out = loader.write(half);  // compact SubGrid field
        return [in, out] CARBON_HD(const S::Span& s, const S::Idx& i) {
          out.write(s, i, 0, kSampleScale * in.read(s, i, 0));
        };
      },
      "sample_boundary");
}

// ---- Dense kernels ------------------------------------------------------------------------

[[nodiscard]] auto
build_fill_interior(carbon::Dense& dense,
                    Field& u,
                    int64_t n) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&u, n](carbon::KernelLoader& loader) {
        loader.region(kInterior);
        const auto out = loader.write(u);
        return [out, n] CARBON_HD(Span s, Idx idx) {
          if (on_boundary(s.global_location(idx), n)) {
            return;
          }
          out.write(s, idx, 0, kInteriorValue);
        };
      },
      "fill_interior");
}

[[nodiscard]] auto
build_scale_interior(carbon::Dense& dense,
                     Field& u,
                     int64_t n) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&u, n](carbon::KernelLoader& loader) {
        loader.region(kInterior);
        const auto io = loader.read_write(u);
        return [io, n] CARBON_HD(Span s, Idx idx) {
          if (on_boundary(s.global_location(idx), n)) {
            return;
          }
          io.write(s, idx, 0, io.read(s, idx, 0) * kInteriorScale);
        };
      },
      "scale_interior");
}

[[nodiscard]] auto
build_count_wrong(carbon::Dense& dense,
                  Field& u,
                  int64_t n,
                  carbon::ReductionResult<int64_t>& wrong) -> carbon::Kernel {
  return carbon::Kernel::reduce<int64_t>(
      dense,
      [&u, n](carbon::KernelLoader& loader) {
        const auto in = loader.read(u);
        return [in, n] CARBON_HD(Span s, Idx idx) -> int64_t {
          const float expected = on_boundary(s.global_location(idx), n)
                                     ? kBoundaryValue
                                     : kInteriorValue * kInteriorScale;
          return in.read(s, idx, 0) == expected ? 0 : 1;
        };
      },
      wrong, carbon::reduce::Sum<int64_t>{}, "count_wrong");
}

// The region_mask_cube way to fill the boundary: launch over the whole cube, skip the rest.
[[nodiscard]] auto
build_fill_boundary_dense(carbon::Dense& dense,
                          Field& u,
                          int64_t n) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&u, n](carbon::KernelLoader& loader) {
        loader.region(kBoundary);
        const auto out = loader.write(u);
        return [out, n] CARBON_HD(Span s, Idx idx) {
          if (!on_boundary(s.global_location(idx), n)) {
            return;
          }
          out.write(s, idx, 0, kBoundaryValue);
        };
      },
      "fill_boundary_dense");
}

// ---- Helpers ------------------------------------------------------------------------------

struct Args {
  int64_t n = 128;
  int iters = 200;
  std::string dot_prefix{};
};

[[nodiscard]] auto
parse_args(int argc,
           char** argv) -> Args {
  Args a{};
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string_view flag = argv[i];
    if (flag == "--n") {
      a.n = std::atoll(argv[i + 1]);
    } else if (flag == "--iters") {
      a.iters = std::atoi(argv[i + 1]);
    } else if (flag == "--dot") {
      a.dot_prefix = argv[i + 1];
    } else {
      throw std::invalid_argument("unknown flag " + std::string{flag});
    }
  }
  if (a.n < 3) {
    throw std::invalid_argument("need --n >= 3 (so the cube has an interior)");
  }
  if (a.iters < 1) {
    throw std::invalid_argument("need --iters >= 1");
  }
  return a;
}

auto
print_schedule(carbon::Skeleton& sk) -> void {
  sk.compile();
  const auto& kernels = sk.effective_kernels();
  const auto& successors = sk.min_edges();
  const auto& streams = sk.kernel_streams();
  std::vector<std::string> waits_for(kernels.size());
  for (size_t i = 0; i < kernels.size(); ++i) {
    for (const size_t j : successors[i]) {
      waits_for[j] += (waits_for[j].empty() ? "" : ", ") + std::string{kernels[i].name()};
    }
  }
  std::cout << "schedule, masks honoured (" << sk.num_streams() << " stream(s))\n";
  for (size_t i = 0; i < kernels.size(); ++i) {
    std::cout << "  " << std::left << std::setw(16) << kernels[i].name() << " stream " << streams[i]
              << "   waits for: " << (waits_for[i].empty() ? "-" : waits_for[i]) << '\n';
  }
  std::cout << '\n';
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

}  // namespace

int
main(int argc,
     char** argv) {
  try {
    carbon::init();
    const Args args = parse_args(argc, argv);
    const int64_t n = args.n;

    carbon::System sys(carbon::get_all_devices());
    carbon::Dense dense(sys, carbon::int64_3d{n, n, n}, carbon::Stencil{});

    // S = SubGrid(G, F, B): F selects the face voxels, B is the region its kernels declare.
    const S boundary = dense.new_sub_grid(
        [n](carbon::KernelLoader&) {
          return [n] CARBON_HD(const Span& s, const Idx& idx) {
            return on_boundary(s.global_location(idx), n);
          };
        },
        kBoundary, "boundary");

    std::cout << "sub_grid_cube: " << n << "^3 cube on " << sys.gpu_group().size()
              << " GPU(s); SubGrid '" << boundary.name() << "' has " << boundary.total_count()
              << " of " << n * n * n << " voxels\n";
    for (size_t g = 0; g < boundary.gpu_count(); ++g) {
      std::cout << "  GPU " << g << ": " << boundary.count(g) << " lanes ("
                << boundary.internal_count(g) << " internal + " << boundary.boundary_count(g)
                << " next to a GPU-slab boundary)\n";
    }
    std::cout << '\n';

    Field u = dense.newField<float, 1>(carbon::MemorySide::Device, "u");
    auto half = boundary.newField<float, 1>(carbon::MemorySide::HostDevice, "half");
    carbon::ReductionResult<int64_t> wrong(sys);

    carbon::Skeleton sk(sys);
    sk.add(build_fill_boundary(boundary, u));
    sk.add(build_fill_interior(dense, u, n));
    sk.add(build_scale_interior(dense, u, n));
    sk.add(build_sample_boundary(boundary, u, half));
    sk.add(build_count_wrong(dense, u, n, wrong));
    print_schedule(sk);

    if (!args.dot_prefix.empty()) {
      const bool ok =
          sk.export_dot_file(args.dot_prefix + ".dot", "sub_grid_cube") &&
          sk.export_launch_matrix_dot_file(args.dot_prefix + "_matrix.dot", "sub_grid_cube");
      std::cout << (ok ? "wrote " : "FAILED to write ") << args.dot_prefix << "{,_matrix}.dot\n\n";
    }

    sk.run();
    sk.wait();
    carbon::Stream stream = sys.stream();
    wrong.sync(stream);
    carbon::Transfer<float>(sys, half.gpu_buffers(), half.cpu_buffers())(stream);
    stream.wait();
    int64_t wrong_half = 0;
    for (size_t g = 0; g < boundary.gpu_count(); ++g) {
      for (int64_t lane = 0; lane < boundary.count(g); ++lane) {
        wrong_half += half.cpu_buffers()[g].data()[lane] == kSampleScale * kBoundaryValue ? 0 : 1;
      }
    }
    std::cout << "wrong voxels of u: " << wrong.host_value()
              << ", wrong lanes of the SubGrid field: " << wrong_half << "\n\n";

    const carbon::Kernel dense_fill = build_fill_boundary_dense(dense, u, n);
    const carbon::Kernel sub_fill = build_fill_boundary(boundary, u);
    const double t_dense = time_launches(stream, dense_fill, args.iters);
    const double t_sub = time_launches(stream, sub_fill, args.iters);
    std::cout << std::fixed << std::setprecision(1) << "boundary fill, " << args.iters
              << " launches: Dense + filter " << t_dense << " us, SubGrid " << t_sub
              << " us per launch\n";

    if (wrong.host_value() != 0 || wrong_half != 0) {
      std::cerr << "sub_grid_cube: FAILED\n";
      return EXIT_FAILURE;
    }
    std::cout << "sub_grid_cube: OK\n";
  } catch (const std::exception& e) {
    std::cerr << "sub_grid_cube: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
