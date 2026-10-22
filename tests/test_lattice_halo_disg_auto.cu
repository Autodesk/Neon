// Plumbing test for lattice-semantic auto-halo on carbon::DenseDisg via
// KernelLoader::read_stencil(field, stencil, StencilSemantic::Lattice) and the
// Skeleton auto-insertion path (TypedLatticeHaloFactory ->
// DenseDisg::Field::make_halo_kernel(name, stencil, semantic) -> HaloDisg).
//
// The numerical correctness of the disg lattice halo itself is covered by
// test_lattice_halo_disg.cu (which drives HaloDisg directly). This test only
// asserts that the DenseDisg field exposes the 3-arg make_halo_kernel so the
// Skeleton auto-inserts a HaloDisg-backed halo kernel ahead of a lattice
// stencil read (parity with test_lattice_halo_dense_auto.cu for the aggregated
// layout), compiles, and runs without error.
//
// Skipped (returns 0) when fewer than 2 visible CUDA devices are available.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense_disg.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <vector>

namespace {

constexpr int kComponents = 19;

[[nodiscard]] auto
d3q19_natural() -> std::vector<carbon::int32_3d> {
  return {
      {-1, 0, 0},  {0, -1, 0}, {0, 0, -1}, {-1, -1, 0}, {-1, 1, 0}, {-1, 0, -1}, {-1, 0, 1},
      {0, -1, -1}, {0, -1, 1}, {0, 0, 0},  {1, 0, 0},   {0, 1, 0},  {0, 0, 1},   {1, 1, 0},
      {1, -1, 0},  {1, 0, 1},  {1, 0, -1}, {0, 1, 1},   {0, 1, -1},
  };
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 2) {
    CARBON_LOG_WARN(
        "test_lattice_halo_disg_auto: needs >=2 visible CUDA devices for a meaningful exchange; "
        "have {} -- skipping",
        carbon::device_count());
    return 0;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();
  const carbon::int64_3d global{4, 5, 8 * static_cast<int64_t>(n_gpus)};

  const std::vector<carbon::int32_3d> velocities = d3q19_natural();
  carbon::Stencil lattice_stencil(velocities);
  carbon::DenseDisg dense(sys, global, /*halo_z=*/1);
  auto field = dense.newField<int32_t, kComponents>(carbon::MemorySide::HostDevice, "pop");

  carbon::Skeleton sk(sys);
  sk.add(carbon::Kernel::compute(
      dense,
      [&field, lattice_stencil](carbon::KernelLoader& loader) {
        const auto pop =
            loader.read_stencil(field, lattice_stencil, carbon::StencilSemantic::Lattice);
        return [pop] CARBON_HOST_DEVICE(carbon::DenseDisg::Span s, carbon::DenseDisg::Idx idx) {
          (void)pop.read(s, idx, 0);
        };
      },
      "touch_pop_disg"));

  try {
    sk.compile();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_lattice_halo_disg_auto: skeleton compile failed: {}", e.what());
    return 1;
  }

  const size_t expected_kernels = 2U;
  if (sk.effective_kernels().size() != expected_kernels) {
    CARBON_LOG_ERROR(
        "test_lattice_halo_disg_auto: expected {} effective kernels (auto HaloDisg + compute), "
        "got {}",
        expected_kernels, sk.effective_kernels().size());
    return 1;
  }

  try {
    sk.run();
    sk.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_lattice_halo_disg_auto: skeleton run failed: {}", e.what());
    return 1;
  }

  CARBON_LOG_INFO("test_lattice_halo_disg_auto: ok ({} slabs, auto HaloDisg + compute)", n_gpus);
  return 0;
}
