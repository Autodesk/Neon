// Region masks, the smallest complete example: boundary voxels vs. the rest of a cube.
//
// WHAT IT SHOWS
//
//   A Skeleton orders two kernels whenever they touch the same field and at least one of them
//   writes it. That is safe, but too strict when the kernels touch *different cells* of the
//   field. A RegionMask lets each kernel say which part of the domain it touches, and the
//   Skeleton then only orders kernels whose parts overlap.
//
// THE DOMAIN
//
//   An n x n x n cube, split into two regions that together cover every voxel exactly once:
//
//     region bit 0, "interior" : voxels with no coordinate on a face of the cube
//     region bit 1, "boundary" : voxels with at least one coordinate equal to 0 or n - 1
//
//   The bit numbers are ours to choose; Carbon never learns the geometry. It only assumes that
//   different bits name disjoint sets of voxels.
//
// THE PIPELINE (one field u, four kernels)
//
//     1. fill_boundary   writes u on the boundary   u = 1               region: boundary
//     2. fill_interior   writes u on the interior   u = 2               region: interior
//     3. scale_interior  updates u on the interior  u = 10 * u          region: interior
//     4. count_wrong     reads u everywhere         sums wrong voxels   region: whole cube
//
//   Expected dependencies:
//
//     fill_boundary ──────────────────────┐
//                                          ├──> count_wrong
//     fill_interior ──> scale_interior ───┘
//
//   - 1 and 2 both write u, but on disjoint regions: no edge, they may run concurrently.
//   - 3 updates the interior that 2 wrote: same region, so 3 waits for 2 (and only for 2).
//   - 4 reads the whole cube, so it overlaps both regions and waits for 1 and 3.
//
//   Without masks every kernel would wait for the previous one: 1 -> 2 -> 3 -> 4.
//
// HOW A KERNEL USES A MASK (two rules, both needed)
//
//   a) Declare: `loader.region(M)` in the loading lambda tells the Skeleton "every access of
//      this kernel stays inside region M". It only feeds the dependency analysis.
//   b) Keep the promise: the launch still covers the whole grid, so the per-voxel body must
//      skip voxels outside M itself (here with `on_boundary`). A kernel that writes outside its
//      declared region is a silent data race: Carbon trusts the declaration.
//
// The example compiles the pipeline twice, once with masks honoured and once with
// Skeleton::Options::ignore_region_masks (the whole-field schedule), prints both dependency
// tables, runs both, and checks that every voxel holds the expected value. With
// `--dot <prefix>` it also writes, for each schedule (`masked` / `ignored`), the dependency
// graph <prefix>_<schedule>.dot and the launch matrix <prefix>_<schedule>_matrix.dot (one
// column per stream, one row per launch / event record / event wait); render them with
// `dot -Tsvg`.
//
// `--overlap standard` compiles both schedules with OverlapMode::Standard (OCC). OCC splits
// stencil kernels around their halo; this pipeline has no stencil, so the schedules stay the
// same and the flag only shows that masks and OCC compose.
//
//   carbon_example_region_mask_cube [--n 32] [--overlap none|standard] [--dot region_mask_cube]
//
// See docs/skeleton/README.md, "Region masks", for the full contract, and region_mask_bc.cu
// for the same idea inside a heat solver.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/region_mask.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

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
using carbon::RegionMask;

// ---- The two regions -------------------------------------------------------------------------

constexpr unsigned kInteriorBit = 0;
constexpr unsigned kBoundaryBit = 1;

const RegionMask kInterior = RegionMask::bit(kInteriorBit);
const RegionMask kBoundary = RegionMask::bit(kBoundaryBit);

// The geometric test behind kBoundary. Every kernel that declares a region filters itself with
// it, so declaration and behaviour cannot drift apart.
[[nodiscard]] CARBON_HD constexpr auto
on_boundary(carbon::int64_3d g,
            int64_t n) noexcept -> bool {
  return g.x == 0 || g.y == 0 || g.z == 0 || g.x == n - 1 || g.y == n - 1 || g.z == n - 1;
}

// ---- The values the pipeline writes -----------------------------------------------------------

constexpr float kBoundaryValue = 1.0F;
constexpr float kInteriorValue = 2.0F;
constexpr float kInteriorScale = 10.0F;

// ---- The four kernels -------------------------------------------------------------------------

// 1. u = 1 on the boundary.
[[nodiscard]] auto
build_fill_boundary(carbon::Dense& dense,
                    Field& u,
                    int64_t n) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&u, n](carbon::KernelLoader& loader) {
        loader.region(kBoundary);  // (a) declare: this kernel only touches boundary voxels
        const auto out = loader.write(u);
        return [out, n] CARBON_HD(Span s, Idx idx) {
          if (!on_boundary(s.global_location(idx), n)) {
            return;  // (b) keep the promise: leave interior voxels alone
          }
          out.write(s, idx, 0, kBoundaryValue);
        };
      },
      "fill_boundary");
}

// 2. u = 2 on the interior.
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

// 3. u = 10 * u on the interior. Reads and writes the same voxel only (a cell-local update),
//    which is what lets a single region describe the whole access.
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

// 4. Count the voxels that do not hold their expected value. No region is declared, so the read
//    covers the whole cube and this kernel waits for every writer of u.
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

// ---- Helpers ----------------------------------------------------------------------------------

struct Args {
  int64_t n = 32;
  carbon::OverlapMode overlap = carbon::OverlapMode::None;
  std::string dot_prefix{};
};

[[nodiscard]] auto
parse_overlap(std::string_view s) -> carbon::OverlapMode {
  if (s == "none") {
    return carbon::OverlapMode::None;
  }
  if (s == "standard") {
    return carbon::OverlapMode::Standard;
  }
  throw std::invalid_argument("--overlap must be none or standard, got " + std::string{s});
}

[[nodiscard]] auto
parse_args(int argc,
           char** argv) -> Args {
  Args a{};
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string_view flag = argv[i];
    if (flag == "--n") {
      a.n = std::atoll(argv[i + 1]);
    } else if (flag == "--overlap") {
      a.overlap = parse_overlap(argv[i + 1]);
    } else if (flag == "--dot") {
      a.dot_prefix = argv[i + 1];
    } else {
      throw std::invalid_argument("unknown flag " + std::string{flag});
    }
  }
  if (a.n < 3) {
    throw std::invalid_argument("need --n >= 3 (so the cube has an interior)");
  }
  return a;
}

auto
add_pipeline(carbon::Skeleton& sk,
             carbon::Dense& dense,
             Field& u,
             int64_t n,
             carbon::ReductionResult<int64_t>& wrong) -> void {
  sk.add(build_fill_boundary(dense, u, n));
  sk.add(build_fill_interior(dense, u, n));
  sk.add(build_scale_interior(dense, u, n));
  sk.add(build_count_wrong(dense, u, n, wrong));
}

// One line per kernel: the stream it was given and the kernels it waits for. `min_edges()` is
// the dependency graph after removing edges implied by longer paths.
auto
print_schedule(carbon::Skeleton& sk,
               std::string_view title) -> void {
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

  std::cout << title << " (" << sk.num_streams() << " stream(s))\n";
  for (size_t i = 0; i < kernels.size(); ++i) {
    std::cout << "  " << std::left << std::setw(16) << kernels[i].name() << " stream " << streams[i]
              << "   waits for: " << (waits_for[i].empty() ? "-" : waits_for[i]) << '\n';
  }
  std::cout << '\n';
}

// Runs the pipeline once and returns how many voxels ended up with the wrong value.
[[nodiscard]] auto
run_and_count_wrong(carbon::System& sys,
                    carbon::Skeleton& sk,
                    carbon::ReductionResult<int64_t>& wrong) -> int64_t {
  sk.run();
  sk.wait();
  carbon::Stream stream = sys.stream();
  wrong.sync(stream);
  stream.wait();
  return wrong.host_value();
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

    const int64_t interior_voxels = (n - 2) * (n - 2) * (n - 2);
    std::cout << "region_mask_cube: " << n << "^3 cube on " << sys.gpu_group().size()
              << " GPU(s): " << n * n * n - interior_voxels << " boundary + " << interior_voxels
              << " interior voxels\n"
              << "  interior region = " << carbon::to_string(kInterior)
              << ", boundary region = " << carbon::to_string(kBoundary) << "\n\n";

    // Same pipeline twice, each on its own field so the two runs cannot interfere.
    Field u_masked = dense.newField<float, 1>(carbon::MemorySide::Device, "u_masked");
    Field u_whole = dense.newField<float, 1>(carbon::MemorySide::Device, "u_whole");
    carbon::ReductionResult<int64_t> wrong_masked(sys);
    carbon::ReductionResult<int64_t> wrong_whole(sys);

    const carbon::Skeleton::Options honour_masks(args.overlap);
    carbon::Skeleton masked(sys, honour_masks);
    add_pipeline(masked, dense, u_masked, n, wrong_masked);

    carbon::Skeleton::Options ignore_masks(args.overlap);
    ignore_masks.ignore_region_masks = true;
    carbon::Skeleton whole(sys, ignore_masks);
    add_pipeline(whole, dense, u_whole, n, wrong_whole);

    print_schedule(masked, "masks honoured");
    print_schedule(whole, "masks ignored, every access covers the whole cube");

    if (!args.dot_prefix.empty()) {
      const bool ok =
          masked.export_dot_file(args.dot_prefix + "_masked.dot", "region_mask_cube_masked") &&
          masked.export_launch_matrix_dot_file(args.dot_prefix + "_masked_matrix.dot",
                                               "region_mask_cube_masked") &&
          whole.export_dot_file(args.dot_prefix + "_ignored.dot", "region_mask_cube_ignored") &&
          whole.export_launch_matrix_dot_file(args.dot_prefix + "_ignored_matrix.dot",
                                              "region_mask_cube_ignored");
      std::cout << (ok ? "wrote " : "FAILED to write ") << args.dot_prefix
                << "_{masked,ignored}{,_matrix}.dot\n\n";
    }

    const int64_t bad_masked = run_and_count_wrong(sys, masked, wrong_masked);
    const int64_t bad_whole = run_and_count_wrong(sys, whole, wrong_whole);
    std::cout << "wrong voxels: masks honoured = " << bad_masked
              << ", masks ignored = " << bad_whole << '\n';
    if (bad_masked != 0 || bad_whole != 0) {
      std::cerr << "region_mask_cube: FAILED\n";
      return EXIT_FAILURE;
    }
    std::cout << "region_mask_cube: OK (boundary = " << kBoundaryValue
              << ", interior = " << kInteriorValue * kInteriorScale << " everywhere)\n";
  } catch (const std::exception& e) {
    std::cerr << "region_mask_cube: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
