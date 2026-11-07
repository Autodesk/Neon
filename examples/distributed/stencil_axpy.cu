// stencil_axpy.cu
//
// A very small distributed example: one stencil + one axpy on carbon::DenseDistributed, with the
// 3D domain decomposition chosen on the command line (slab | pencil | brick).
//
//   1. stencil:  s(c) = sum of x over the 6 in-domain face neighbours of cell c
//   2. axpy:     y(c) = y(c) + alpha * s(c)
//
// Both kernels are added to one carbon::Skeleton; the Skeleton inserts the halo exchange that the
// stencil needs, so the example works unchanged for any decomposition and any number of ranks.
//
// Verification (identical on every rank, independent of the decomposition):
//   x is a position-dependent integer field, so a missing or misplaced halo changes the result.
//   SUM(s) and SUM(y) are compared against a serial host reference over the whole global domain.
//
// Built for CPU (SYCL backend, AdaptiveCpp `omp` target, CARBON_ENABLE_MPI=ON) this is a plain
// `mpirun` program: every rank is one OpenMP "device".
//
//   mpirun -np 4 ./carbon_example_distributed_stencil_axpy pencil --n 128 --alpha 0.5 --iters 10
//
// Exit code: 0 = PASS, 1 = FAIL (the same on every rank).

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense_distributed.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/process_grid.h>
#include <carbon/reduce.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>
#ifdef CARBON_HAVE_MPI_NCCL
#  include <carbon/distributed.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using Grid = carbon::DenseDistributed;
using Span = Grid::Span;
using Idx = Grid::Idx;

// Position-dependent integer-valued seed. Sums of these stay exact in double.
[[nodiscard]] auto
seed(carbon::int64_3d g) -> double {
  return static_cast<double>((g.x + 2 * g.y + 3 * g.z) % 5 + 1);
}

// Serial reference: SUM over all cells of the sum of seed() over the in-domain face neighbours.
[[nodiscard]] auto
reference_stencil_sum(int64_t n) -> double {
  static constexpr int kDirs[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                      {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
  double total = 0.0;
  for (int64_t z = 0; z < n; ++z) {
    for (int64_t y = 0; y < n; ++y) {
      for (int64_t x = 0; x < n; ++x) {
        for (const auto& d : kDirs) {
          const int64_t nx = x + d[0];
          const int64_t ny = y + d[1];
          const int64_t nz = z + d[2];
          if (nx >= 0 && nx < n && ny >= 0 && ny < n && nz >= 0 && nz < n) {
            total += seed({nx, ny, nz});
          }
        }
      }
    }
  }
  return total;
}

[[nodiscard]] auto
decomposition_mode(const std::string& s) -> carbon::DecompositionMode {
  if (s == "slab") {
    return carbon::DecompositionMode::Slab;
  }
  if (s == "pencil") {
    return carbon::DecompositionMode::Pencil;
  }
  if (s == "brick") {
    return carbon::DecompositionMode::Brick;
  }
  std::cerr << "error: decomposition must be slab, pencil or brick (got '" << s << "')\n";
  std::exit(EXIT_FAILURE);
}

// Global (all ranks) sum of component 0 of a field. The cross-rank allreduce is done by the
// reduction itself (MPI in a host build, NCCL in a CUDA build).
template <typename FieldT>
[[nodiscard]] auto
global_sum(carbon::System& sys,
           const Grid& grid,
           const FieldT& field,
           const char* name) -> double {
  carbon::ReductionResult<double> result(sys);
  carbon::Skeleton sk(sys);
  sk.add(carbon::Kernel::reduce<double>(
      grid,
      [&field](carbon::KernelLoader& loader) {
        const auto p = loader.read(field);
        return [p] CARBON_HOST_DEVICE(Span s, Idx idx) -> double { return p.read(s, idx, 0); };
      },
      result, carbon::reduce::Sum<double>{}, name, /*shared_mem=*/0U));
  sk.compile();
  sk.run();
  sk.wait();
  result.sync(sys.stream());
  sys.stream().wait();
  return result.host_value();
}

[[nodiscard]] auto
close_enough(double got,
             double expected) -> bool {
  return std::abs(got - expected) <= 1e-9 * std::max(1.0, std::abs(expected));
}

}  // namespace

int main(int argc,  char** argv) {
  // ---- CLI:  <slab|pencil|brick>  [--n N]  [--alpha A]  [--iters K] ------------------------
  std::string dec = "slab";
  int64_t n = 64;
  double alpha = 0.5;  // power of two => the checksums below are exact
  int iters = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::cout << "usage: " << argv[0]
                << " <slab|pencil|brick> [--n N] [--alpha A] [--iters K]\n";
      return EXIT_SUCCESS;
    }
    if (a == "--n" && i + 1 < argc) {
      n = std::stoll(argv[++i]);
    } else if (a == "--alpha" && i + 1 < argc) {
      alpha = std::stod(argv[++i]);
    } else if (a == "--iters" && i + 1 < argc) {
      iters = std::stoi(argv[++i]);
    } else if (!a.empty() && a[0] != '-') {
      dec = a;
    } else {
      std::cerr << "error: unknown argument '" << a << "' (try --help)\n";
      return EXIT_FAILURE;
    }
  }
  if (n < 4 || iters < 1) {
    std::cerr << "error: need --n >= 4 and --iters >= 1\n";
    return EXIT_FAILURE;
  }
  const carbon::DecompositionMode dec_mode = decomposition_mode(dec);

  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("stencil_axpy: no compute device visible");
    return EXIT_FAILURE;
  }

  // ---- System: one rank = one device (an OpenMP "device" in a CPU build) --------------------
#ifdef CARBON_HAVE_MPI_NCCL
  auto distributed = std::make_shared<const carbon::Distributed>();
  const bool use_mpi = distributed->world_size() > 1;
  carbon::System sys =
      use_mpi ? carbon::System(distributed) : carbon::System(carbon::get_all_devices());
  const int world_rank = use_mpi ? distributed->world_rank() : 0;
  const int world_size =
      use_mpi ? distributed->world_size() : static_cast<int>(sys.gpu_group().size());
#else
  carbon::System sys(carbon::get_all_devices());
  const int world_rank = 0;
  const int world_size = static_cast<int>(sys.gpu_group().size());
#endif

  // ---- Grid: the 6-neighbour stencil sets the halo width of the decomposition ---------------
  const carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1},
                                    {0, 0, -1}},
      /*remove_center=*/true);
  const carbon::ProcessGrid pg = carbon::ProcessGrid::make(world_size, dec_mode);
  Grid grid(sys, carbon::int64_3d{n, n, n}, pg, stencil);

  if (world_rank == 0) {
    const auto d = pg.dims();
    std::cout << "decomposition=" << dec << "  process_grid=" << d.x << "x" << d.y << "x" << d.z
              << "  ranks=" << world_size << "  N=" << n << "  iters=" << iters
              << "  alpha=" << alpha << '\n';
  }

  auto x = grid.newField<double, 1>(carbon::MemorySide::HostDevice, "x");
  auto s = grid.newField<double, 1>(carbon::MemorySide::HostDevice, "s");
  auto y = grid.newField<double, 1>(carbon::MemorySide::HostDevice, "y");

  x.fill_owned_host([](carbon::int64_3d g, int) -> double { return seed(g); });
  y.fill_owned_host([](carbon::int64_3d, int) -> double { return 1.0; });
  {
    carbon::Stream st = sys.stream();
    carbon::Transfer<double>(sys, x.cpu_buffers(), x.gpu_buffers())(st);
    carbon::Transfer<double>(sys, y.cpu_buffers(), y.gpu_buffers())(st);
    st.wait();
  }

  // ---- The two kernels, in one Skeleton (halo exchange for x is inserted automatically) ------
  carbon::Skeleton sk(sys);
  sk.add(carbon::Kernel::compute(
      grid,
      [&x, &s](carbon::KernelLoader& loader) {
        const auto px = loader.read_stencil(x);
        const auto ps = loader.write(s);
        return [px, ps] CARBON_HOST_DEVICE(Span sp, Idx idx) {
          double acc = 0.0;
          acc += px.neighbour(sp, idx, carbon::int8_3d{1, 0, 0}, 0).value_or(0.0);
          acc += px.neighbour(sp, idx, carbon::int8_3d{-1, 0, 0}, 0).value_or(0.0);
          acc += px.neighbour(sp, idx, carbon::int8_3d{0, 1, 0}, 0).value_or(0.0);
          acc += px.neighbour(sp, idx, carbon::int8_3d{0, -1, 0}, 0).value_or(0.0);
          acc += px.neighbour(sp, idx, carbon::int8_3d{0, 0, 1}, 0).value_or(0.0);
          acc += px.neighbour(sp, idx, carbon::int8_3d{0, 0, -1}, 0).value_or(0.0);
          ps.write(sp, idx, 0, acc);
        };
      },
      "stencil_neighbour_sum"));
  sk.add(carbon::Kernel::compute(
      grid,
      [&s, &y, alpha](carbon::KernelLoader& loader) {
        const auto ps = loader.read(s);
        auto py = loader.read_write(y);
        return [ps, py, alpha] CARBON_HOST_DEVICE(Span sp, Idx idx) {
          py.write(sp, idx, 0, py.read(sp, idx, 0) + alpha * ps.read(sp, idx, 0));
        };
      },
      "axpy"));

  double seconds = 0.0;
  try {
    sk.compile();
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
      sk.run();
    }
    sk.wait();
    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("stencil_axpy: kernel failed: {}", e.what());
    return EXIT_FAILURE;
  }

  // ---- Verification: global sums vs. a serial host reference --------------------------------
  double sum_s = 0.0;
  double sum_y = 0.0;
  try {
    sum_s = global_sum(sys, grid, s, "sum_s");
    sum_y = global_sum(sys, grid, y, "sum_y");
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("stencil_axpy: checksum failed: {}", e.what());
    return EXIT_FAILURE;
  }

  const double ref_s = reference_stencil_sum(n);
  const double ref_y = static_cast<double>(n) * n * n + iters * alpha * ref_s;
  const bool ok = close_enough(sum_s, ref_s) && close_enough(sum_y, ref_y);

  if (world_rank == 0) {
    std::cout.precision(15);
    std::cout << "SUM(s) = " << sum_s << "   expected " << ref_s << '\n'
              << "SUM(y) = " << sum_y << "   expected " << ref_y << '\n'
              << "time   = " << seconds << " s for " << iters << " iteration(s)\n"
              << (ok ? "PASS" : "FAIL") << " (decomposition=" << dec << ", ranks=" << world_size
              << ")\n";
  }
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
