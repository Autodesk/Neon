// Single-node "hello, multi-GPU" example.
//
// Shows the canonical bootstrap:
//   1. carbon::init()                       -- one-time logger setup.
//   2. carbon::get_all_devices()            -- enumerate every CUDA device visible to this process.
//   3. carbon::System(get_all_devices())    -- one `System` over every visible GPU on this node.
//   4. carbon::Stencil(...)                 -- 7-point face-neighbour stencil (radius 1).
//   5. carbon::Dense(system, extent, stnc)  -- z-slab Cartesian grid distributed across those GPUs,
//                                              with a 1-thick z halo on either side of every slab.
//
// Then prints the slab geometry chosen by `Dense` so you can confirm the work was actually split
// across devices: each row reports the GPU's owned z range (z_offset_global .. +local_z) and
// owned cell count, plus the haloed storage extent (owned z + 2 * halo_z) used to size buffers.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

int
main() {
  carbon::init();

  const int device_count = carbon::device_count();
  if (device_count <= 0) {
    CARBON_LOG_ERROR(
        "init_and_dense: no CUDA devices visible (cudaGetDeviceCount); "
        "set CUDA_VISIBLE_DEVICES or run on a host with a GPU.");
    return EXIT_FAILURE;
  }
  CARBON_LOG_INFO("init_and_dense: {} CUDA device(s) visible on this node", device_count);

  carbon::System system(carbon::get_all_devices());
  const size_t n_gpu = system.gpu_group().size();
  CARBON_LOG_INFO("init_and_dense: System bound to {} GPU(s)", n_gpu);

  // Classic 7-point stencil: the cell itself plus its six face neighbours. Chebyshev radius is 1,
  // which makes Dense reserve a 1-thick z halo on each side of every slab.
  const carbon::Stencil stencil(std::vector<carbon::int32_3d>{
      {0, 0, 0},
      {-1, 0, 0},
      {1, 0, 0},
      {0, -1, 0},
      {0, 1, 0},
      {0, 0, -1},
      {0, 0, 1},
  });

  // Pick a z extent that doesn't divide evenly by the GPU count when possible, so the output
  // visibly shows Dense giving the leading (z % N) GPUs one extra layer.
  const carbon::int64_3d global_extent{64, 64, 67};
  const carbon::Dense dense(system, global_extent, stencil);

  CARBON_LOG_INFO("init_and_dense: global extent = ({}, {}, {})", global_extent.x, global_extent.y,
                  global_extent.z);
  CARBON_LOG_INFO("init_and_dense: stencil = {} points, radius = {}", stencil.size(),
                  stencil.radius());

  for (size_t gpu = 0; gpu < dense.gpu_count(); ++gpu) {
    const auto local = dense.local_extent(gpu);
    const auto& span = dense.spans()[gpu];
    CARBON_LOG_INFO(
        "  gpu[{}]: owned z in [{}, {}) (local_z = {}), local extent = ({}, {}, {}), "
        "owned cells = {}, storage extent = ({}, {}, {}), storage cells = {}",
        gpu, dense.z_offset_global(gpu), dense.z_offset_global(gpu) + dense.local_z(gpu),
        dense.local_z(gpu), local.x, local.y, local.z, dense.linear_elements(gpu),
        span.haloed_extent().x, span.haloed_extent().y, span.haloed_extent().z,
        span.linear_storage_elements());
  }

  return EXIT_SUCCESS;
}
