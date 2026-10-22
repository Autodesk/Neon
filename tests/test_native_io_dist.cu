// carbon::io::write_native / read_native distributed (one-shard-per-rank) round trip.
//
// Launched under `mpiexec -n 2`. Each rank constructs a distributed Dense (one slab per
// rank), allocates a HostDevice scalar field, fills it on the host with encode(x, y,
// global_z), and exercises the full write -> read -> verify cycle with the per-rank shard
// suffix that carbon::io appends automatically. Each rank only ever opens its own shard;
// no cross-rank coordination is needed for native binary by design.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/distributed.h>
#include <carbon/io.h>
#include <carbon/memory_side.h>
#include <carbon/stencil.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kSentinel = -7777;

[[nodiscard]] auto
encode(int64_t x,
       int64_t y,
       int64_t global_z) noexcept -> int {
  return static_cast<int>(global_z * 10000 + y * 100 + x);
}

}  // namespace

int
main() {
  carbon::init();
  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_native_io_dist: world_size={} < 2, skipping", distributed->world_size());
    return 0;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_native_io_dist: no CUDA devices, skipping");
    return 0;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                          /*remove_center=*/true);

  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(world_size)};
  carbon::Dense dense(sys, global, std::move(stencil));

  auto field = dense.newField<int, 1>(carbon::MemorySide::Host);

  // Pre-fill halos with a sentinel; fill_owned_host then writes only owned cells.
  for (size_t g = 0; g < field.cpu_buffers().size(); ++g) {
    int* h = field.cpu_buffers()[g].data();
    const size_t n = field.cpu_buffers()[g].size();
    for (size_t j = 0; j < n; ++j) {
      h[j] = kSentinel;
    }
  }
  field.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
    return encode(gc.x, gc.y, gc.z);
  });

  const std::string path = "carbon_native_io_dist_test.bin";
  // Each rank cleans up its own shard before / after to keep reruns idempotent.
  const std::string my_shard = path + ".rank" + std::to_string(rank);
  std::remove(my_shard.c_str());

  try {
    carbon::io::write_native(path, dense, field);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_native_io_dist: rank {} write_native failed: {}", rank, e.what());
    return 1;
  }

  // Read back into a fresh field (halo sentinel-filled so missed reads stay catchable).
  auto field2 = dense.newField<int, 1>(carbon::MemorySide::Host);
  for (size_t g = 0; g < field2.cpu_buffers().size(); ++g) {
    int* h = field2.cpu_buffers()[g].data();
    const size_t n = field2.cpu_buffers()[g].size();
    for (size_t j = 0; j < n; ++j) {
      h[j] = kSentinel;
    }
  }
  try {
    carbon::io::read_native(path, dense, field2);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_native_io_dist: rank {} read_native failed: {}", rank, e.what());
    return 1;
  }

  // Verify: owned cells match the encoder, halos remained at the sentinel.
  size_t mismatches = 0;
  {
    const auto& span = dense.spans()[0];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const int* h = field2.cpu_buffers()[0].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      const bool is_halo = (z < halo_z) || (z >= halo_z + lim_z);
      const int64_t global_z = z_origin + z;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          const int got = h[off];
          const int want = is_halo ? kSentinel : encode(x, y, global_z);
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_native_io_dist: rank {} (x,y,z)=({},{},{}) gz={} got={} want={} "
                  "is_halo={}",
                  rank, x, y, z, global_z, got, want, is_halo);
            }
            ++mismatches;
          }
        }
      }
    }
  }
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_native_io_dist: rank {} {} mismatches", rank, mismatches);
    return 1;
  }

  std::remove(my_shard.c_str());
  CARBON_LOG_INFO("test_native_io_dist: rank {} of {} ok", rank, world_size);
  return 0;
}
