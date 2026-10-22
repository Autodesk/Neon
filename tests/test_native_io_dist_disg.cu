// carbon::io::write_native / read_native distributed (one-shard-per-rank) round trip
// for the DenseDisg layout.
//
// Launched under `mpiexec -n 2`. Each rank constructs a distributed DenseDisg (one slab
// per rank, carbon's one-rank-per-GPU model), allocates a HostDevice scalar field, fills
// the owned region via the disg_addr offset formula, and exercises the full
// write -> read -> verify cycle with the per-rank shard suffix that carbon::io appends
// automatically. Each rank only ever opens its own shard; native binary needs no
// cross-rank coordination by design.

#include <carbon/carbon.h>
#include <carbon/dense_disg.h>
#include <carbon/distributed.h>
#include <carbon/io.h>
#include <carbon/memory_side.h>

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

// disg_addr: per-cell offset into a DenseDisg field's host mirror — bit-equal with the
// I/O path's row_offset overload, lifted to per-cell granularity for verification.
// Cloned from test_dense_disg_host.cu so the layout knowledge stays in one place.
[[nodiscard]] auto
disg_addr(const carbon::DenseDisg::Span& span,
          int64_t bz,
          int64_t x,
          int64_t y,
          int c,
          int components) -> int64_t {
  const int64_t lz = span.logic_extent().z;
  const int64_t xdim = span.haloed_extent().x;
  const int64_t xy = span.xy_slab_elements();
  int64_t region_first_z = 0;
  int64_t region_z_dim = 1;
  int64_t off = x + y * xdim;
  if (bz <= 1 || bz >= lz) {
    region_first_z = bz;
    region_z_dim = 1;
  } else {
    region_first_z = 2;
    region_z_dim = lz - 2;
    off += (bz - 2) * xy;
  }
  return off + xy * (region_first_z * static_cast<int64_t>(components) +
                     static_cast<int64_t>(c) * region_z_dim);
}

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
    CARBON_LOG_INFO("test_native_io_dist_disg: world_size={} < 2, skipping",
                    distributed->world_size());
    return 0;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_native_io_dist_disg: no CUDA devices, skipping");
    return 0;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);

  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(world_size)};
  carbon::DenseDisg disg(sys, global, /*halo_z=*/1);

  auto field = disg.newField<int, 1>(carbon::MemorySide::Host);

  // Pre-fill owned + halo with the sentinel; overwrite owned cells via disg_addr.
  for (size_t g = 0; g < field.cpu_buffers().size(); ++g) {
    int* h = field.cpu_buffers()[g].data();
    const size_t n = field.cpu_buffers()[g].size();
    for (size_t j = 0; j < n; ++j) {
      h[j] = kSentinel;
    }
  }
  {
    const auto& span = disg.spans()[0];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t lim_z = span.logic_extent().z;
    const int64_t z_origin = span.z_offset();
    int* h = field.cpu_buffers()[0].data();
    for (int64_t bz = halo_z; bz < halo_z + lim_z; ++bz) {
      const int64_t global_z = z_origin + bz;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          h[disg_addr(span, bz, x, y, 0, /*components=*/1)] = encode(x, y, global_z);
        }
      }
    }
  }

  const std::string path = "carbon_native_io_dist_disg_test.bin";
  const std::string my_shard = path + ".rank" + std::to_string(rank);
  std::remove(my_shard.c_str());

  try {
    carbon::io::write_native(path, disg, field);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_native_io_dist_disg: rank {} write_native failed: {}", rank, e.what());
    return 1;
  }

  // Read back into a fresh field with halo cards sentinel-filled.
  auto field2 = disg.newField<int, 1>(carbon::MemorySide::Host);
  for (size_t g = 0; g < field2.cpu_buffers().size(); ++g) {
    int* h = field2.cpu_buffers()[g].data();
    const size_t n = field2.cpu_buffers()[g].size();
    for (size_t j = 0; j < n; ++j) {
      h[j] = kSentinel;
    }
  }
  try {
    carbon::io::read_native(path, disg, field2);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_native_io_dist_disg: rank {} read_native failed: {}", rank, e.what());
    return 1;
  }

  size_t mismatches = 0;
  {
    const auto& span = disg.spans()[0];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const int* h = field2.cpu_buffers()[0].data();
    for (int64_t bz = 0; bz < haloed.z; ++bz) {
      const bool is_halo = (bz < halo_z) || (bz >= halo_z + lim_z);
      const int64_t global_z = z_origin + bz;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = disg_addr(span, bz, x, y, 0, /*components=*/1);
          const int got = h[off];
          const int want = is_halo ? kSentinel : encode(x, y, global_z);
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_native_io_dist_disg: rank {} (bz,y,x)=({},{},{}) gz={} got={} want={} "
                  "is_halo={}",
                  rank, bz, y, x, global_z, got, want, is_halo);
            }
            ++mismatches;
          }
        }
      }
    }
  }
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_native_io_dist_disg: rank {} {} mismatches", rank, mismatches);
    return 1;
  }

  std::remove(my_shard.c_str());
  CARBON_LOG_INFO("test_native_io_dist_disg: rank {} of {} ok", rank, world_size);
  return 0;
}
