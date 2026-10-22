// DataView split correctness: Standard == Internal + Boundary.
//
// Carbon splits a slab's owned z extent into three views (see Span::iter_extent /
// Span::set_and_validate):
//
// - Standard: every owned cell, one launch.
// - Internal: the inner (Lz - 2 * halo_z) layers — owned cells that do not depend on halos.
// - Boundary: the lower halo_z layers and the upper halo_z layers — owned cells that depend
//   on neighbouring slabs' data through the halo exchange.
//
// Internal + Boundary together must cover exactly the same set of owned cells as Standard
// (no overlap, no gaps), and the per-cell write coordinates must match. This test pins both
// of those down on a single GPU (sufficient — the per-slab geometry is what's under test;
// multi-GPU reuses the same Span code path) by:
//
// 1. Filling u_in with a deterministic per-global-cell value, on owned cells AND halo cells,
//    so a wrong z mapping in set_and_validate would read the wrong source value.
// 2. Allocating two output fields u_std and u_split, both initialized to a sentinel value on
//    every storage cell (owned + halo).
// 3. Running a single kernel that writes a coordinate-dependent value into u_out for each
//    visited cell. The value is a function of the cell's global (x, y, z), so any z mismatch
//    between the iteration index and the buffer position lands in the wrong cell and the
//    bit-for-bit comparison below fails.
// 4. Launch sequence A: kernel(u_in, u_std), DataView::Standard.
// 5. Launch sequence B: kernel(u_in, u_split), DataView::Internal then DataView::Boundary.
// 6. Verify on every owned cell: u_std[c] == u_split[c]. Halo cells in u_split must still
//    be the sentinel — neither Internal nor Boundary may write outside the owned region.
//
// We also independently sanity-check the iteration extents the Dense computed for the three
// views, so a regression in iter_extent (e.g. drop the 2*halo_z from Internal) is caught
// even before the kernel runs.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/stencil.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

namespace {

constexpr float kSentinel = -7777.0F;

// Deterministic per-global-cell value. Distinct for every (x, y, z) triple in the test
// extents (and for several layers of halo on either side of z), so a misrouted write or
// read shows up as a mismatch.
[[nodiscard]] CARBON_HD auto
encode(int64_t x,
       int64_t y,
       int64_t global_z) -> float {
  return static_cast<float>(global_z * 100000 + y * 1000 + x);
}

[[nodiscard]] auto
sync(const carbon::Stream& stream,
     const char* tag) -> bool {
  try {
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_data_view_split: sync after {} failed: {}", tag, e.what());
    return false;
  }
  return true;
}

// Fill every storage cell (lower halo + owned + upper halo) of @p f's host mirror with
// @p fn(global_x, global_y, global_z). Used both for the input field (so neighbour reads
// see real data outside the slab too) and for prefilling outputs with a sentinel.
template <typename Fn>
auto
fill_storage_host(const carbon::Dense& dense,
                  carbon::Dense::Field<float,
                                       1>& f,
                  Fn fn) -> void {
  for (size_t g = 0; g < dense.gpu_count(); ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t z_origin = span.z_offset();
    float* h = f.cpu_buffers()[g].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      const int64_t gz = z_origin + z;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          h[off] = fn(x, y, gz);
        }
      }
    }
  }
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_data_view_split: no CUDA devices visible");
    return 1;
  }

  // Force a single-GPU System: the iter_extent / set_and_validate split is per-Span, so
  // single-GPU is the right granularity for this test. Multi-GPU correctness piggybacks on
  // the existing dense_dist / halo tests once they get DataView-aware variants.
  std::vector<carbon::detail::Gpu> all = carbon::get_all_devices();
  std::vector<carbon::detail::Gpu> single;
  single.push_back(std::move(all.front()));
  carbon::System sys(std::move(single));

  // Radius-1 stencil so halo_z == 1; the slab is thick enough (Lz = 8) that 2 * halo_z = 2
  // is well within Lz, so the Internal / Boundary split has a non-empty inner band.
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  if (stencil.radius() != 1) {
    CARBON_LOG_ERROR("test_data_view_split: unexpected stencil radius {}", stencil.radius());
    return 1;
  }

  const carbon::int64_3d global{6, 5, 8};
  const carbon::Dense dense(sys, global, stencil);
  const auto& span = dense.spans()[0];
  if (span.halo_z() != 1) {
    CARBON_LOG_ERROR("test_data_view_split: expected halo_z=1, got {}", span.halo_z());
    return 1;
  }
  const int64_t lz = span.logic_extent().z;
  if (lz != global.z) {
    CARBON_LOG_ERROR("test_data_view_split: expected single-slab Lz={}, got {}", global.z, lz);
    return 1;
  }

  // Iteration-extent sanity. Standard == owned, Internal == owned with z reduced by 2*halo_z,
  // Boundary == 2*halo_z layers (folded lower + upper bands).
  {
    const auto std_ext = span.iter_extent(carbon::DataView::Standard);
    const auto int_ext = span.iter_extent(carbon::DataView::Internal);
    const auto bnd_ext = span.iter_extent(carbon::DataView::Boundary);
    if (std_ext.x != global.x || std_ext.y != global.y || std_ext.z != lz) {
      CARBON_LOG_ERROR(
          "test_data_view_split: Standard iter_extent ({}, {}, {}) != owned ({}, {}, {})",
          std_ext.x, std_ext.y, std_ext.z, global.x, global.y, lz);
      return 1;
    }
    if (int_ext.z != lz - 2 * span.halo_z()) {
      CARBON_LOG_ERROR("test_data_view_split: Internal z extent {} != Lz - 2*halo_z {}", int_ext.z,
                       lz - 2 * span.halo_z());
      return 1;
    }
    if (bnd_ext.z != 2 * span.halo_z()) {
      CARBON_LOG_ERROR("test_data_view_split: Boundary z extent {} != 2*halo_z {}", bnd_ext.z,
                       2 * span.halo_z());
      return 1;
    }
  }

  auto u_in = dense.newField<float, 1>(carbon::MemorySide::HostDevice);
  auto u_std = dense.newField<float, 1>(carbon::MemorySide::HostDevice);
  auto u_split = dense.newField<float, 1>(carbon::MemorySide::HostDevice);

  fill_storage_host(dense, u_in, [](int64_t x, int64_t y, int64_t gz) {
    return encode(x, y, gz);
  });
  fill_storage_host(dense, u_std, [](int64_t, int64_t, int64_t) {
    return kSentinel;
  });
  fill_storage_host(dense, u_split, [](int64_t, int64_t, int64_t) {
    return kSentinel;
  });

  carbon::Stream stream = sys.stream();

  try {
    carbon::Transfer<float>(sys, u_in.cpu_buffers(), u_in.gpu_buffers())(stream);
    carbon::Transfer<float>(sys, u_std.cpu_buffers(), u_std.gpu_buffers())(stream);
    carbon::Transfer<float>(sys, u_split.cpu_buffers(), u_split.gpu_buffers())(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_data_view_split: upload failed: {}", e.what());
    return 1;
  }
  if (!sync(stream, "upload")) {
    return 1;
  }

  // The kernel writes (10 * in[idx]) + 1 at each visited cell. The factor of 10 multiplies a
  // value that already encodes (x, y, gz), so a wrong z mapping produces a wrong product and
  // the bit-for-bit comparison below fails. Reading via Partition::read also exercises the
  // SoA offset math against the haloed Span, which must agree with set_and_validate's z.
  auto build_step = [&](carbon::Dense::Field<float, 1>& f_in,
                        carbon::Dense::Field<float, 1>& f_out) {
    return dense.make_kernel(
        [&f_in, &f_out](carbon::KernelLoader& loader) {
          const auto in = loader(f_in);
          const auto out = loader(f_out);
          return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
            const float v = in.read(s, idx, 0);
            out.write(s, idx, 0, 10.0F * v + 1.0F);
          };
        },
        /*shared_mem=*/0U);
  };

  auto step_std = build_step(u_in, u_std);
  auto step_split = build_step(u_in, u_split);

  // Launch A: Standard once over u_std.
  try {
    step_std.launch(stream, carbon::DataView::Standard);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_data_view_split: Standard launch failed: {}", e.what());
    return 1;
  }
  if (!sync(stream, "Standard launch")) {
    return 1;
  }

  // Launch B: Internal then Boundary over u_split. Order shouldn't matter (the two
  // iteration domains don't overlap) but we pick Internal first to mirror the typical
  // overlap pattern (Internal runs while the halo exchange is in flight; Boundary runs
  // once the exchange is complete).
  try {
    step_split.launch(stream, carbon::DataView::Internal);
    step_split.launch(stream, carbon::DataView::Boundary);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_data_view_split: Internal/Boundary launch failed: {}", e.what());
    return 1;
  }
  if (!sync(stream, "Internal/Boundary launches")) {
    return 1;
  }

  try {
    carbon::Transfer<float>(sys, u_std.gpu_buffers(), u_std.cpu_buffers())(stream);
    carbon::Transfer<float>(sys, u_split.gpu_buffers(), u_split.cpu_buffers())(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_data_view_split: download failed: {}", e.what());
    return 1;
  }
  if (!sync(stream, "download")) {
    return 1;
  }

  // Verify u_std[c] == u_split[c] on every owned cell, and u_split[halo] == sentinel.
  size_t owned_checked = 0;
  size_t halo_checked = 0;
  for (size_t g = 0; g < dense.gpu_count(); ++g) {
    const auto& s = dense.spans()[g];
    const auto haloed = s.haloed_extent();
    const int64_t pitch = s.xy_slab_elements();
    const int64_t halo_z = s.halo_z();
    const float* hs = u_std.cpu_buffers()[g].data();
    const float* hp = u_split.cpu_buffers()[g].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      const bool owned = (z >= halo_z) && (z < halo_z + s.logic_extent().z);
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          const float vs = hs[off];
          const float vp = hp[off];
          if (owned) {
            if (vs != vp) {
              CARBON_LOG_ERROR(
                  "test_data_view_split: owned cell mismatch at gpu={} (x={}, y={}, bz={}): "
                  "u_std={} u_split={}",
                  g, x, y, z, vs, vp);
              return 1;
            }
            if (vs == kSentinel) {
              CARBON_LOG_ERROR(
                  "test_data_view_split: Standard left an owned cell as sentinel at gpu={} "
                  "(x={}, y={}, bz={})",
                  g, x, y, z);
              return 1;
            }
            ++owned_checked;
          } else {
            if (vp != kSentinel) {
              CARBON_LOG_ERROR(
                  "test_data_view_split: Internal/Boundary touched a halo cell at gpu={} "
                  "(x={}, y={}, bz={}): expected sentinel, got {}",
                  g, x, y, z, vp);
              return 1;
            }
            ++halo_checked;
          }
        }
      }
    }
  }

  // Every owned cell on the slab must have been validated; a regression that, say, made
  // Internal cover only half its band would leave the matching u_std value distinct from
  // u_split's sentinel and the comparison above would already fire. The counts here are a
  // belt-and-braces check that the loop didn't silently skip the slab.
  const int64_t expected_owned = global.x * global.y * lz;
  if (static_cast<int64_t>(owned_checked) != expected_owned) {
    CARBON_LOG_ERROR("test_data_view_split: checked {} owned cells, expected {}", owned_checked,
                     expected_owned);
    return 1;
  }
  if (halo_checked == 0U) {
    CARBON_LOG_ERROR(
        "test_data_view_split: no halo cells checked — test cannot have verified that "
        "Internal / Boundary stayed inside the owned region");
    return 1;
  }

  CARBON_LOG_INFO(
      "test_data_view_split: OK — {} owned cells match between Standard and Internal+Boundary; "
      "{} halo cells stayed at sentinel",
      owned_checked, halo_checked);
  return 0;
}
