// Region masks: run a boundary-condition kernel beside the interior stencil.
//
// The heat problem of jacobi_unified (hot z- face, cold z+ face, zero x/y faces) split the
// way production solvers are written: one kernel updates the interior with a 6-point stencil,
// another applies the Dirichlet values on the boundary voxels. Both write the same field, so
// without more information the Skeleton must serialise them. A per-cell label field plus a
// RegionMask per kernel tells it the two kernels write disjoint cells:
//
//   label 0      : interior          -> kInterior = RegionMask::from_label(0)
//   labels 1..3  : hot / cold / side -> kBoundary = ~kInterior (the labels are a full
//                                       partition, so the complement is "every other cell")
//
// Each kernel declares its footprint with L.region(...) and filters itself on the label; the
// label field is read before the filter, i.e. everywhere, so that read is declared all-ones
// (harmless: nothing writes labels inside the pipeline). Stencil reads stay whole-field, so
// the next half step's stencil still waits on both kernels (through the auto-halo).
//
// The example runs the same pipeline twice — masks honoured and Options::ignore_region_masks
// — prints both schedules, checks the fields are bit-identical, and with --dot <prefix>
// writes <prefix>_schedule.dot / <prefix>_kernels.dot / <prefix>_matrix.dot (the launch
// matrix: one column per stream) plus <prefix>_matrix_ignored.dot for the whole-field
// schedule (render with `dot -Tsvg`).
//
// `--overlap standard` compiles both schedules with OverlapMode::Standard: each interior
// stencil is split into an Internal half, which runs while its halo is in flight, and a
// Boundary half after it. The masks still let the boundary-condition kernels run on their own
// streams.
//
//   carbon_example_region_mask_bc [--n 64] [--iters 200] [--overlap none|standard]
//                                 [--dot region_mask_bc]

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/memory_side.h>
#include <carbon/neighbour_value.h>
#include <carbon/region_mask.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Span = carbon::Dense::Span;
using Idx = carbon::Dense::Idx;
using Field = carbon::Dense::Field<float, 1>;
using Labels = carbon::Dense::Field<int32_t, 1>;
using carbon::RegionMask;

constexpr int32_t kInteriorLabel = 0;
constexpr int32_t kHotLabel = 1;
constexpr int32_t kColdLabel = 2;
constexpr int32_t kSideLabel = 3;

const RegionMask kInterior = RegionMask::from_label(kInteriorLabel);
const RegionMask kBoundary = ~kInterior;

struct Args {
  int64_t n = 64;
  int iters = 200;
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
    } else if (flag == "--iters") {
      a.iters = std::atoi(argv[i + 1]);
    } else if (flag == "--overlap") {
      a.overlap = parse_overlap(argv[i + 1]);
    } else if (flag == "--dot") {
      a.dot_prefix = argv[i + 1];
    } else {
      throw std::invalid_argument("unknown flag " + std::string{flag});
    }
  }
  if (a.n < 4 || a.iters < 1) {
    throw std::invalid_argument("need --n >= 4 and --iters >= 1");
  }
  return a;
}

[[nodiscard]] auto
build_labels(carbon::Dense& dense,
             Labels& labels,
             carbon::int64_3d global) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&labels, global](carbon::KernelLoader& loader) {
        const auto out = loader.write(labels);
        return [out, global] CARBON_HD(Span s, Idx idx) {
          const carbon::int64_3d g = s.global_location(idx);
          int32_t label = kInteriorLabel;
          if (g.z == 0) {
            label = kHotLabel;
          } else if (g.z == global.z - 1) {
            label = kColdLabel;
          } else if (g.x == 0 || g.y == 0 || g.x == global.x - 1 || g.y == global.y - 1) {
            label = kSideLabel;
          }
          out.write(s, idx, 0, label);
        };
      },
      "init_labels");
}

[[nodiscard]] auto
build_zero(carbon::Dense& dense,
           Field& f,
           std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&f](carbon::KernelLoader& loader) {
        const auto out = loader.write(f);
        return [out] CARBON_HD(Span s, Idx idx) {
          out.write(s, idx, 0, 0.0F);
        };
      },
      std::move(name));
}

// Interior cells: 6-point average of @p fin.
[[nodiscard]] auto
build_interior(carbon::Dense& dense,
               Labels& labels,
               Field& fin,
               Field& fout,
               std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&labels, &fin, &fout](carbon::KernelLoader& loader) {
        loader.region(kInterior);
        const auto lab = loader.read(labels, RegionMask::all());
        const auto in = loader.read_stencil(fin);
        const auto out = loader.write(fout);
        return [lab, in, out] CARBON_HD(Span s, Idx idx) {
          if (lab.read(s, idx, 0) != kInteriorLabel) {
            return;
          }
          const float sum = in.neighbour(s, idx, carbon::int8_3d{1, 0, 0}, 0).value +
                            in.neighbour(s, idx, carbon::int8_3d{-1, 0, 0}, 0).value +
                            in.neighbour(s, idx, carbon::int8_3d{0, 1, 0}, 0).value +
                            in.neighbour(s, idx, carbon::int8_3d{0, -1, 0}, 0).value +
                            in.neighbour(s, idx, carbon::int8_3d{0, 0, 1}, 0).value +
                            in.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0).value;
          out.write(s, idx, 0, sum * (1.0F / 6.0F));
        };
      },
      std::move(name));
}

// Boundary cells: Dirichlet values by label.
[[nodiscard]] auto
build_boundary(carbon::Dense& dense,
               Labels& labels,
               Field& fout,
               std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&labels, &fout](carbon::KernelLoader& loader) {
        loader.region(kBoundary);
        const auto lab = loader.read(labels, RegionMask::all());
        const auto out = loader.write(fout);
        return [lab, out] CARBON_HD(Span s, Idx idx) {
          const int32_t label = lab.read(s, idx, 0);
          if (label == kInteriorLabel) {
            return;
          }
          out.write(s, idx, 0, label == kHotLabel ? 1.0F : (label == kColdLabel ? -1.0F : 0.0F));
        };
      },
      std::move(name));
}

auto
add_pipeline(carbon::Skeleton& sk,
             carbon::Dense& dense,
             Labels& labels,
             Field& u,
             Field& v) -> void {
  sk.add(build_interior(dense, labels, u, v, "interior_uv"));
  sk.add(build_boundary(dense, labels, v, "boundary_v"));
  sk.add(build_interior(dense, labels, v, u, "interior_vu"));
  sk.add(build_boundary(dense, labels, u, "boundary_u"));
}

[[nodiscard]] auto
time_runs(carbon::Skeleton& sk,
          int iters) -> double {
  const auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; ++it) {
    sk.run();
  }
  sk.wait();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int
main(int argc,
     char** argv) {
  try {
    carbon::init();
    const Args args = parse_args(argc, argv);

    carbon::System sys(carbon::get_all_devices());
    const size_t n_gpus = sys.gpu_group().size();
    const carbon::int64_3d global{args.n, args.n, args.n};
    const carbon::Stencil stencil(
        std::vector<carbon::int32_3d>{
            {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
        /*remove_center=*/true);
    carbon::Dense dense(sys, global, stencil);

    auto labels = dense.newField<int32_t, 1>(carbon::MemorySide::Device, "labels");
    auto u = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u");
    auto v = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v");
    auto u_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_ref");
    auto v_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v_ref");

    carbon::Skeleton init(sys);
    init.add(build_labels(dense, labels, global));
    init.add(build_zero(dense, u, "zero_u"));
    init.add(build_zero(dense, v, "zero_v"));
    init.add(build_zero(dense, u_ref, "zero_u_ref"));
    init.add(build_zero(dense, v_ref, "zero_v_ref"));
    init.run();
    init.wait();

    const carbon::Skeleton::Options honour_masks(args.overlap);
    carbon::Skeleton masked(sys, honour_masks);
    add_pipeline(masked, dense, labels, u, v);
    carbon::Skeleton::Options whole_field(args.overlap);
    whole_field.ignore_region_masks = true;
    carbon::Skeleton reference(sys, whole_field);
    add_pipeline(reference, dense, labels, u_ref, v_ref);
    masked.compile();
    reference.compile();

    std::cout << "region_mask_bc: n=" << args.n << " iters=" << args.iters << " gpus=" << n_gpus
              << " overlap="
              << (args.overlap == carbon::OverlapMode::Standard ? "standard" : "none")
              << "\n  interior mask = " << carbon::to_string(kInterior)
              << ", boundary mask = " << carbon::to_string(kBoundary) << "\n\n";
    std::cout << "masks honoured: " << masked.num_streams() << " stream(s)\n"
              << masked.export_launch_order() << '\n';
    std::cout << "masks ignored: " << reference.num_streams() << " stream(s)\n"
              << reference.export_launch_order() << '\n';

    if (!args.dot_prefix.empty()) {
      const bool ok =
          masked.export_dot_file(args.dot_prefix + "_schedule.dot", "region_mask_bc") &&
          masked.export_kernel_list_dot_file(args.dot_prefix + "_kernels.dot", "region_mask_bc") &&
          masked.export_launch_matrix_dot_file(args.dot_prefix + "_matrix.dot", "region_mask_bc") &&
          reference.export_launch_matrix_dot_file(args.dot_prefix + "_matrix_ignored.dot",
                                                  "region_mask_bc_ignored");
      std::cout << (ok ? "wrote " : "FAILED to write ") << args.dot_prefix
                << "_{schedule,kernels,matrix,matrix_ignored}.dot\n\n";
    }

    const double ms_masked = time_runs(masked, args.iters);
    const double ms_reference = time_runs(reference, args.iters);
    std::cout << "masks honoured: " << ms_masked * 1000.0 / args.iters << " us/iter\n"
              << "masks ignored:  " << ms_reference * 1000.0 / args.iters << " us/iter\n";

    carbon::Stream s = sys.stream();
    carbon::Transfer<float>(sys, u.gpu_buffers(), u.cpu_buffers())(s);
    carbon::Transfer<float>(sys, u_ref.gpu_buffers(), u_ref.cpu_buffers())(s);
    s.wait();
    size_t mismatches = 0;
    for (size_t g = 0; g < n_gpus; ++g) {
      const auto& span = dense.spans()[g];
      const int64_t begin = span.halo_z() * span.xy_slab_elements();
      const int64_t end = begin + span.logic_extent().z * span.xy_slab_elements();
      const float* a = u.cpu_buffers()[g].data();
      const float* b = u_ref.cpu_buffers()[g].data();
      for (int64_t i = begin; i < end; ++i) {
        mismatches += a[i] != b[i] ? 1U : 0U;
      }
    }
    if (mismatches != 0U) {
      std::cerr << "region_mask_bc: " << mismatches
                << " cell(s) differ between the masked and whole-field schedules\n";
      return EXIT_FAILURE;
    }
    std::cout << "masked and whole-field results are bit-identical\n";
  } catch (const std::exception& e) {
    std::cerr << "region_mask_bc: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
