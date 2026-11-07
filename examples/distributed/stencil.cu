// scalar_mul_distributed.cu
//
// Minimal correctness test for the 3D domain decomposition of
// carbon::DenseDistributed (slab | pencil | brick).
//
//   1. takes the decomposition type as input:  slab | pencil | brick
//   2. builds a carbon::DenseDistributed grid on a 7-point stencil
//      (the stencil is what drives the per-axis halo the decomposition honours)
//   3. runs one trivial kernel through carbon::Skeleton:   y = a * x
//   4. verifies with a global Sum reduction (rank-invariant):
//         SUM(y) == a * N^3
//      which can only hold if the owned cells tile the global domain exactly
//      once and the kernel touched every owned cell (no overlap, no gaps).
//
// Usage:
//   ./scalar_mul_distributed slab                     # single process, local GPUs
//   mpirun -np 4 ./scalar_mul_distributed pencil      # MPI, 4 ranks
//   mpirun -np 8 ./scalar_mul_distributed brick --n 128 --scalar 0.25
//
// Exit code: 0 = PASS, 1 = FAIL.

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
#ifdef CARBON_HAVE_MPI_NCCL
#  include <carbon/distributed.h>
#endif

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace {

[[nodiscard]] auto
decomposition_mode(std::string_view s) -> carbon::DecompositionMode {
  if (s == "slab") {
    return carbon::DecompositionMode::Slab;
  }
  if (s == "pencil") {
    return carbon::DecompositionMode::Pencil;
  }
  if (s == "brick") {
    return carbon::DecompositionMode::Brick;
  }
  std::cerr << "error: decomposition must be slab, pencil, or brick (got '" << s << "')\n";
  std::exit(EXIT_FAILURE);
}

} // namespace

int
main(int argc,
char** argv) {
  // ---- CLI:  <slab|pencil|brick>  [--n 64]  [--scalar 0.5] ------------------------------
  std::string dec = "slab";
  int64_t n = 64;
  float scalar = 0.5f;  // power of two => the checksum below is exact even in float
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::cout << "usage: " << argv[0] << " <slab|pencil|brick> [--n N] [--scalar A]\n";
      return EXIT_SUCCESS;
    }
    if ((a == "--n" || a == "--scalar") && i + 1 < argc) {
      if (a == "--n") {
        n = std::stoll(argv[++i]);
      } else {
        scalar = std::stof(argv[++i]);
      }
    } else if (a[0] != '-') {
      dec = a;
    }
  }
  if (n < 4) {
    std::cerr << "error: --n must be >= 4 (got " << n << ")\n";
    return EXIT_FAILURE;
  }
  const auto dec_mode = decomposition_mode(dec);

  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("scalar_mul_distributed: no CUDA devices visible");
    return EXIT_FAILURE;
  }

  // ---- System bootstrap (same pattern as the LBM benchmark) ------------------------------
#ifdef CARBON_HAVE_MPI_NCCL
  auto distributed = std::make_shared<const carbon::Distributed>();
  const bool use_mpi = distributed->world_size() > 1;
  carbon::System sys =
      use_mpi ? carbon::System(distributed) : carbon::System(carbon::get_all_devices());
  const int world_rank = use_mpi ? distributed->world_rank() : 0;
  const int world_size = use_mpi ? distributed->world_size()
                                 : static_cast<int>(sys.gpu_group().size());
#else
  carbon::System sys(carbon::get_all_devices());
  const int world_rank = 0;
  const int world_size = static_cast<int>(sys.gpu_group().size());
  const bool use_mpi = false;
#endif

  // ---- 7-point stencil -> per-axis halo for the chosen decomposition ---------------------
  const carbon::Stencil stencil(std::vector<carbon::int32_3d>{
                                     {1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                                     {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
                                /*remove_center=*/true);
  const carbon::int64_3d global{n, n, n};

  // Auto-factorised process grid:
  //   slab:   1 x 1  x R        pencil: 1 x sqrt x sqrt       brick: cbrt x cbrt x cbrt
  const carbon::ProcessGrid pg = carbon::ProcessGrid::make(world_size, dec_mode);
  if (world_rank == 0) {
    const auto d = pg.dims();
    CARBON_LOG_INFO("scalar_mul_distributed: N={}  decomposition={}  process_grid={}x{}x{}  "
                    "world_size={} {}",
                    n, dec, d.x, d.y, d.z, world_size, use_mpi ? "(mpi+nccl)" : "(in-process)");
  }

  carbon::DenseDistributed grid(sys, global, pg, stencil);
  using Span = typename carbon::DenseDistributed::Span;
  using Idx = typename carbon::DenseDistributed::Idx;

  auto fIn = grid.template newField<float, 1>(carbon::MemorySide::HostDevice, "x");
  auto fOut = grid.template newField<float, 1>(carbon::MemorySide::HostDevice, "y");

  // Seed every owned cell with 1.0 (halo cells stay zero-initialised).
  fIn.fill_owned_host([](carbon::int64_3d, int) -> float { return 1.0f; });
  {
    carbon::Stream s = sys.stream();
    carbon::Transfer<float>(sys, fIn.cpu_buffers(), fIn.gpu_buffers())(s);
    s.wait();
  }

  // ---- the kernel:  y = a * x  on every owned cell ----------------------------------------
  // The *read* side is the exact pattern from the LBM benchmark's checksum kernel
  // (loader.read + .read(s, idx, comp)); the write side is its mirror.
  carbon::Skeleton sk(sys);
  sk.add(carbon::Kernel::map(  // (1) builder name TBC against carbon/kernel.h
      grid,
      [&](carbon::KernelLoader& loader) {
        const auto xin = loader.read(fIn);
        const auto yout = loader.write(fOut);  // (2) TBC
        return [xin, yout, scalar] CARBON_HOST_DEVICE(Span s, Idx idx) {
          yout.write(s, idx, 0, scalar * xin.read(s, idx, 0));  // (3) TBC
        };
      },
      "scalar_mul");

  try {
    sk.compile();
    sk.run();
    sk.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("scalar_mul_distributed: kernel failed: {}", e.what());
    return EXIT_FAILURE;
  }

  // ---- verification: global checksum (rank-invariant) -------------------------------------
  // Every owned cell contributes exactly `scalar`, so SUM(y) == a * N^3 iff the owned cells
  // tile the global domain exactly once and the kernel ran on all of them.
  carbon::ReductionResult<double> sum_y(sys);
  carbon::Skeleton chk(sys);
  chk.add(carbon::Kernel::reduce<double>(
      grid,
      [&](carbon::KernelLoader& loader) {
        const auto yin = loader.read(fOut);
        return [yin] CARBON_HOST_DEVICE(Span s, Idx idx) -> double {
          return static_cast<double>(yin.read(s, idx, 0));
        };
      },
      sum_y, carbon::reduce::Sum<double>{}, "checksum_y", /*shared_mem=*/0U));
  try {
    chk.compile();
    chk.run();
    chk.wait();
    sum_y.sync(sys.stream());
    sys.stream().wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("scalar_mul_distributed: checksum failed: {}", e.what());
    return EXIT_FAILURE;
  }

  const double got = sum_y.host_value();
  const double expected = static_cast<double>(scalar) * static_cast<double>(n) * n * n;
  const double rel = std::abs(got - expected) / std::max(1.0, std::abs(expected));
  if (world_rank == 0) {
    std::cout << "decomposition=" << dec << "  SUM(y)=" << got << "  expected=" << expected << '\n';
  }
  if (rel > 1e-9) {
    CARBON_LOG_ERROR("scalar_mul_distributed: FAIL (decomposition={}) got={} expected={}", dec,
                     got, expected);
    return EXIT_FAILURE;
  }
  if (world_rank == 0) {
    std::cout << "PASS (decomposition=" << dec << "  N=" << n << "  ranks=" << world_size << ")\n";
  }
  return EXIT_SUCCESS;
}