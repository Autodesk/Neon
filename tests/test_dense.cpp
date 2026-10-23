#include <carbon/carbon.h>
#include <carbon/data_view.h>
#include <carbon/dense.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>

#include "expect_throw.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

struct Host_dense_kernel_fn_smoke {
  void
  operator()(carbon::Dense::Span s,
             carbon::Dense::Idx i) const {
    (void)s;
    (void)i;
  }
};
// dense_kernel_fn is the internal concept driving Dense::launch_dense_grid; not aliased onto
// Dense because user kernels never need to refer to it. Keep the detail-namespace name here.
static_assert(carbon::detail::dense::dense_kernel_fn<Host_dense_kernel_fn_smoke>);

int
main() {
  constexpr carbon::int64_3d e{2, 3, 4};
  static_assert(e.x == 2 && e.y == 3 && e.z == 4);
  static_assert(e.volume() == 24);
  static_assert(e.with_z(99).z == 99);
  static_assert(e.with_z(99).x == 2);
  assert(e.as_array()[0] == 2 && e.as_array()[1] == 3 && e.as_array()[2] == 4);

  {
    using Idx = carbon::Dense::Idx;
    using Span = carbon::Dense::Span;
    constexpr carbon::int64_3d slab_extent{10, 8, 3};
    constexpr carbon::int64_3d global_extent{10, 8, 64};
    constexpr Span slab(slab_extent, global_extent, 96, 40);
    static_assert(slab.haloed_extent().x == 10);
    static_assert(slab.logic_extent().x == 10);
    static_assert(slab.halo_z() == 0);
    static_assert(slab.xy_slab_elements() == 96);
    static_assert(slab.z_offset() == 40);
    static_assert(slab.global_extent().z == 64);

    static_assert(slab.global_location(slab.make_idx(1, 2, 3)).z == 43);
    static_assert(slab.make_idx(carbon::int64_3d{1, 2, 3}).cell().x == 1);

    constexpr Idx id(5, 6, 7);
    static_assert(id.cell().x == 5 && id.cell().y == 6 && id.cell().z == 7);
    constexpr Span slab_with_offset(slab_extent, global_extent, 96, 100);
    static_assert(slab_with_offset.global_location(id).z == 107);
    static_assert(slab_with_offset.global_location(id).x == 5 &&
                  slab_with_offset.global_location(id).y == 6 &&
                  slab_with_offset.global_location(id).z == 107);
  }

  {
    using carbon::DataView;
    using PartInt = carbon::Dense::Partition<int, 1>;

    constexpr PartInt default_part;
    static_assert(default_part.ptr() == nullptr);
    static_assert(default_part.data_view() == DataView::Standard);

    int storage = 42;
    const PartInt standard_part(&storage);
    assert(standard_part.ptr() == &storage);
    assert(standard_part.data_view() == DataView::Standard);

    const PartInt boundary_part(&storage, DataView::Boundary);
    assert(boundary_part.data_view() == DataView::Boundary);

    PartInt mutable_part(&storage, DataView::Internal);
    assert(mutable_part.data_view() == DataView::Internal);
    mutable_part.reset();
    assert(mutable_part.ptr() == nullptr);
    assert(mutable_part.data_view() == DataView::Standard);
    mutable_part.reset(&storage, DataView::Boundary);
    assert(mutable_part.data_view() == DataView::Boundary);
    assert(mutable_part.origin().x == 0.0F && mutable_part.origin().y == 0.0F &&
           mutable_part.origin().z == 0.0F);
    assert(mutable_part.spacing() == 1.0F);
  }

  {
    using Idx = carbon::Dense::Idx;
    using Span = carbon::Dense::Span;
    using PartInt = carbon::Dense::Partition<int, 3>;

    constexpr carbon::int64_3d local{4, 3, 2};
    const int64_t plane = local.x * local.y;
    const Span span(local, local, plane, 0);
    const int64_t per_component = span.linear_storage_elements();
    const int cardinality = PartInt::static_value;
    std::vector<int> storage(static_cast<size_t>(per_component * cardinality));
    for (int64_t c = 0; c < cardinality; ++c) {
      for (int64_t z = 0; z < local.z; ++z) {
        for (int64_t y = 0; y < local.y; ++y) {
          for (int64_t x = 0; x < local.x; ++x) {
            const int64_t off = c * per_component + z * plane + y * local.x + x;
            storage[static_cast<size_t>(off)] = static_cast<int>(1000 * c + 100 * z + 10 * y + x);
          }
        }
      }
    }
    const PartInt part(storage.data());
    for (int64_t c = 0; c < cardinality; ++c) {
      for (int64_t z = 0; z < local.z; ++z) {
        for (int64_t y = 0; y < local.y; ++y) {
          for (int64_t x = 0; x < local.x; ++x) {
            const Idx i = span.make_idx(x, y, z);
            const int expected = static_cast<int>(1000 * c + 100 * z + 10 * y + x);
            assert(part.read(span, i, static_cast<int>(c)) == expected);
          }
        }
      }
    }
  }

  {
    using Idx = carbon::Dense::Idx;
    using Span = carbon::Dense::Span;
    using PartInt = carbon::Dense::Partition<int, 3>;

    constexpr carbon::int64_3d local{4, 3, 2};
    const int64_t plane = local.x * local.y;
    const Span span(local, local, plane, 0);
    const int64_t per_component = span.linear_storage_elements();
    const int cardinality = PartInt::static_value;
    std::vector<int> storage(static_cast<size_t>(per_component * cardinality));
    for (int64_t c = 0; c < cardinality; ++c) {
      for (int64_t z = 0; z < local.z; ++z) {
        for (int64_t y = 0; y < local.y; ++y) {
          for (int64_t x = 0; x < local.x; ++x) {
            const int64_t off = c * per_component + z * plane + y * local.x + x;
            storage[static_cast<size_t>(off)] = static_cast<int>(1000 * c + 100 * z + 10 * y + x);
          }
        }
      }
    }
    const PartInt part(storage.data());
    for (int64_t c = 0; c < cardinality; ++c) {
      for (int64_t z = 0; z < local.z; ++z) {
        for (int64_t y = 0; y < local.y; ++y) {
          for (int64_t x = 0; x < local.x; ++x) {
            const Idx i = span.make_idx(x, y, z);
            const int expected = static_cast<int>(1000 * c + 100 * z + 10 * y + x);
            assert(part.read(span, i, static_cast<int>(c)) == expected);
          }
        }
      }
    }
  }

  // Empty-system policy: every grid ctor rejects a System with zero GPUs at
  // construction time (matches Block / DenseDisg). Locks the diagnostic point
  // -- a misconfigured System fails at the grid ctor rather than at the first
  // newField call.
  {
    std::vector<carbon::detail::Gpu> empty_devices;
    carbon::System sys(carbon::detail::GpuGroup(std::move(empty_devices)));
    CARBON_EXPECT_THROW(carbon::Dense(sys, carbon::int64_3d{10, 10, 7}), std::invalid_argument);

    const std::vector<carbon::int32_3d> pts{
        {0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -2}, {0, 0, 2},
    };
    carbon::Stencil s(pts, true);
    CARBON_EXPECT_THROW(carbon::Dense(sys, carbon::int64_3d{8, 8, 8}, s), std::invalid_argument);

    // make_vector still works on the zero-GPU System (it's a helper independent
    // of any grid); the empty-system veto only applies at grid construction.
    const auto empty_vec = sys.make_vector<int>();
    assert(empty_vec.empty());
    assert(empty_vec.capacity() == 0U);

    const auto filled = sys.make_vector([](size_t) {
      return size_t{42};
    });
    assert(filled.empty());
  }

  carbon::init();
  const int device_count = carbon::device_count();
  if (device_count == 0) {
    CARBON_LOG_WARN("test_dense: no CUDA devices; skipping multi-GPU Dense checks");
    return 0;
  }
  carbon::System sys(carbon::get_all_devices());
  const size_t n = sys.gpu_group().size();
  assert(n == static_cast<size_t>(device_count));

  const carbon::int64_3d global{12, 8, 100};
  carbon::Dense dense(sys, global);

  assert(dense.origin().x == 0.0F && dense.origin().y == 0.0F && dense.origin().z == 0.0F);
  assert(dense.spacing() == 1.0F);
  for (size_t i = 0; i < n; ++i) {
    assert(dense.spans()[i].origin().x == dense.origin().x);
    assert(dense.spans()[i].spacing() == dense.spacing());
  }

  assert(dense.global_extent().x == 12);
  assert(dense.global_extent().y == 8);
  assert(dense.global_extent().z == 100);
  assert(dense.gpu_count() == n);
  assert(dense.spans().size() == n);

  int64_t sum_z = 0;
  int64_t prev_end = 0;
  for (size_t i = 0; i < n; ++i) {
    const int64_t lz = dense.local_z(i);
    const int64_t off = dense.z_offset_global(i);
    assert(off == prev_end);
    sum_z += lz;
    prev_end = off + lz;

    const auto loc = dense.local_extent(i);
    assert(loc.x == global.x && loc.y == global.y && loc.z == lz);
    assert(dense.linear_elements(i) == static_cast<size_t>(global.x * global.y * lz));

    const carbon::Dense::Span& sp = dense.spans()[i];
    const carbon::int64_3d owned = sp.logic_extent();
    assert(owned.x == loc.x && owned.y == loc.y && owned.z == loc.z);
    assert(sp.halo_z() == 0);
    assert(sp.haloed_extent().x == loc.x);
    assert(sp.haloed_extent().y == loc.y);
    assert(sp.haloed_extent().z == loc.z);
    assert(sp.z_offset() == off);
    assert(sp.xy_slab_elements() == loc.x * loc.y);
  }
  assert(sum_z == global.z);

  const std::vector<size_t> counts = dense.linear_counts_per_gpu();
  assert(counts.size() == n);
  assert(counts == dense.linear_counts_per_gpu());
  assert(std::accumulate(counts.begin(), counts.end(), size_t{0}) ==
         static_cast<size_t>(global.volume()));

  {
    constexpr carbon::float32_3d o{1.0F, -2.0F, 3.5F};
    constexpr float dx = 0.25F;
    carbon::Dense g(sys, global, carbon::Stencil{}, carbon::BoundaryMode::Open, o, dx);
    assert(g.origin().x == o.x && g.origin().y == o.y && g.origin().z == o.z);
    assert(g.spacing() == dx);
    assert(g.spans()[0].spacing() == dx);
    auto f = g.newField<float, 1>();
    assert(f.partitions()[0].origin().x == o.x);
    assert(f.partitions()[0].spacing() == dx);
  }

  CARBON_EXPECT_THROW(carbon::Dense(sys, global, carbon::Stencil{}, carbon::BoundaryMode::Open,
                                    carbon::float32_3d{}, 0.0F),
                      std::invalid_argument);

  {
    const std::vector<carbon::int32_3d> pts{
        {-2, 0, 0}, {2, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -2}, {0, 0, 2},
    };
    carbon::Stencil stencil(pts);
    const std::int32_t r = stencil.radius();
    assert(r == 2);

    carbon::Dense haloed(sys, global, stencil);
    assert(haloed.stencil().radius() == r);

    auto field = haloed.newField<int, 1>();
    const auto& field_buffers = field.gpu_buffers();
    const auto& field_parts = field.partitions();
    assert(field_buffers.size() == n);
    assert(field_parts.size() == n);

    for (size_t i = 0; i < n; ++i) {
      const int64_t lz = haloed.local_z(i);
      const int64_t haloed_z = lz + 2 * static_cast<int64_t>(r);
      const size_t expected = static_cast<size_t>(global.x) * static_cast<size_t>(global.y) *
                              static_cast<size_t>(haloed_z);
      assert(field_buffers[i].size() == expected);
      // Slab z-origin now lives only on the Span; verify the haloed offset there.
      const auto& span = haloed.spans()[i];
      assert(span.z_offset() == haloed.z_offset_global(i) - static_cast<int64_t>(r));
    }
  }

  // MemorySide threading: Device (default), Host, HostDevice should each populate the right
  // side(s) of the field.
  {
    carbon::Dense plain(sys, global);
    auto host_only = plain.newField<int, 1>(carbon::MemorySide::Host);
    assert(host_only.gpu_buffers().empty());
    assert(host_only.cpu_buffers().size() == n);
    assert(host_only.partitions().size() == n);
    for (size_t i = 0; i < n; ++i) {
      assert(host_only.cpu_buffers()[i].size() ==
             static_cast<size_t>(plain.spans()[i].linear_storage_elements()));
      // Host-only partitions are bound to the host pointer.
      assert(host_only.partitions()[i].ptr() == host_only.cpu_buffers()[i].data());
    }

    auto device_only = plain.newField<int, 1>(carbon::MemorySide::Device);
    assert(device_only.gpu_buffers().size() == n);
    assert(device_only.cpu_buffers().empty());

    auto both = plain.newField<int, 1>(carbon::MemorySide::HostDevice);
    assert(both.gpu_buffers().size() == n);
    assert(both.cpu_buffers().size() == n);
    for (size_t i = 0; i < n; ++i) {
      assert(both.gpu_buffers()[i].size() == both.cpu_buffers()[i].size());
      // Partitions stay on the device side when both sides are present.
      assert(both.partitions()[i].ptr() == both.gpu_buffers()[i].data());
    }
  }

  {
    const carbon::LaunchShape custom{32U, 4U, 2U};
    carbon::Dense custom_block(sys, global, carbon::Stencil{}, carbon::BoundaryMode::Open,
                               carbon::float32_3d{}, 1.0F, custom);
    const auto& blocks = custom_block.gpu_block();
    assert(blocks.size() == n);
    for (size_t i = 0; i < n; ++i) {
      assert(blocks[i] == custom);
    }
    const auto& grids = custom_block.gpu_grid();
    assert(grids.size() == n);
    assert(grids[0].x >= 1U);
  }

  CARBON_EXPECT_THROW(carbon::Dense(sys, global, carbon::Stencil{}, carbon::BoundaryMode::Open,
                                    carbon::float32_3d{}, 1.0F, carbon::LaunchShape{2048U, 1U, 1U}),
                      std::runtime_error);

  return 0;
}
