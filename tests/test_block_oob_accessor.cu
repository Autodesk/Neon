// Verifies that Block's per-GPU vector accessors throw std::out_of_range on
// out-of-range gpu_id / adj_idx instead of silently returning a static empty
// sentinel. Anchors the contract change introduced when P2.α (audit third
// pass) closed the silent-empty-accessor footgun: previously, a stale loop
// bound on the user side surfaced as "this field has zero blocks" downstream
// rather than as a clear precondition failure.
//
// Six accessors are exercised:
//   - Block::span(gpu_id)
//   - Block::host_origins(gpu_id)
//   - Block::host_bitmasks(gpu_id)
//   - Block::host_connectivity(gpu_id)
//   - Block::halo_pairs_upward(adj_idx)
//   - Block::halo_pairs_downward(adj_idx)
//
// Each is called twice: once with a valid in-range index (must not throw) and
// once with the first OOB index (must throw std::out_of_range). The test runs
// on both CUDA and SYCL — its body uses no direct backend APIs.
//
// Bool-flag pattern (rather than CARBON_EXPECT_THROW) is used deliberately:
// the existing macro relies on assert, which is a no-op under -DNDEBUG, so
// failures in a Release build would be silent. Returning non-zero from main
// is the only check that survives every build type.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/devices.h>
#include <carbon/integer3d.h>
#include <carbon/system.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;

[[nodiscard]] auto
sphere_predicate(int64_t domain_n,
                 const carbon::int64_3d& gc) -> bool {
  const double cx = (domain_n - 1) * 0.5;
  const double cy = (domain_n - 1) * 0.5;
  const double cz = (domain_n - 1) * 0.5;
  const double r = domain_n * 0.45;
  const double dx = static_cast<double>(gc.x) - cx;
  const double dy = static_cast<double>(gc.y) - cy;
  const double dz = static_cast<double>(gc.z) - cz;
  return (dx * dx + dy * dy + dz * dz) <= (r * r);
}

template <class Fn>
[[nodiscard]] auto
expect_throws_out_of_range(const char* label,
                           Fn&& fn) -> bool {
  try {
    (void)fn();
  } catch (const std::out_of_range&) {
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_block_oob_accessor: %s threw %s instead of std::out_of_range: %s\n",
                 label, typeid(e).name(), e.what());
    return false;
  } catch (...) {
    std::fprintf(stderr,
                 "test_block_oob_accessor: %s threw a non-std exception instead of "
                 "std::out_of_range\n",
                 label);
    return false;
  }
  std::fprintf(stderr, "test_block_oob_accessor: %s did not throw — expected std::out_of_range\n",
               label);
  return false;
}

template <class Fn>
[[nodiscard]] auto
expect_no_throw(const char* label,
                Fn&& fn) -> bool {
  try {
    (void)fn();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_block_oob_accessor: %s threw unexpectedly: %s (what=%s)\n", label,
                 typeid(e).name(), e.what());
    return false;
  } catch (...) {
    std::fprintf(stderr, "test_block_oob_accessor: %s threw a non-std exception unexpectedly\n",
                 label);
    return false;
  }
  return true;
}

[[nodiscard]] auto
run_test() -> int {
  carbon::init();
  auto devices = carbon::get_all_devices();
  if (devices.empty()) {
    std::printf("test_block_oob_accessor: no GPU devices visible — skipping\n");
    return 0;
  }
  // Stay single-GPU: the OOB contract is identical regardless of how many
  // GPUs are visible, and we want the test to pass on every CI lane shape.
  if (devices.size() > 1U) {
    devices.erase(devices.begin() + 1, devices.end());
  }
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  constexpr int64_t domain_n = 16;
  const carbon::int64_3d domain_extent{domain_n, domain_n, domain_n};
  const auto predicate = [domain_n](const carbon::int64_3d& gc) -> bool {
    return sphere_predicate(domain_n, gc);
  };
  BlockGrid block(system, domain_extent, predicate, "test_block_oob_accessor");
  const size_t gpu_count = block.gpu_count();
  if (gpu_count == 0U) {
    std::fprintf(stderr, "test_block_oob_accessor: gpu_count == 0\n");
    return 1;
  }

  bool ok = true;

  // In-range gpu_id must not throw.
  ok &= expect_no_throw("span(0)", [&] {
    return &block.span(0);
  });
  ok &= expect_no_throw("host_origins(0)", [&] {
    return &block.host_origins(0);
  });
  ok &= expect_no_throw("host_bitmasks(0)", [&] {
    return &block.host_bitmasks(0);
  });
  ok &= expect_no_throw("host_connectivity(0)", [&] {
    return &block.host_connectivity(0);
  });

  // OOB gpu_id must throw std::out_of_range. (gpu_count is the first OOB
  // index by construction.)
  ok &= expect_throws_out_of_range("span(gpu_count)", [&] {
    return &block.span(gpu_count);
  });
  ok &= expect_throws_out_of_range("host_origins(gpu_count)", [&] {
    return &block.host_origins(gpu_count);
  });
  ok &= expect_throws_out_of_range("host_bitmasks(gpu_count)", [&] {
    return &block.host_bitmasks(gpu_count);
  });
  ok &= expect_throws_out_of_range("host_connectivity(gpu_count)", [&] {
    return &block.host_connectivity(gpu_count);
  });

  // halo_pairs_upward / halo_pairs_downward are indexed by adjacency:
  // valid range is [0, gpu_count - 1). With one GPU there are zero
  // adjacencies, so 0 is itself OOB. With more than one GPU, 0 is in-range
  // and gpu_count - 1 is the first OOB.
  if (gpu_count >= 2U) {
    ok &= expect_no_throw("halo_pairs_upward(0)", [&] {
      return &block.halo_pairs_upward(0);
    });
    ok &= expect_no_throw("halo_pairs_downward(0)", [&] {
      return &block.halo_pairs_downward(0);
    });
  }
  ok &= expect_throws_out_of_range("halo_pairs_upward(gpu_count - 1)", [&] {
    return &block.halo_pairs_upward(gpu_count - 1U);
  });
  ok &= expect_throws_out_of_range("halo_pairs_downward(gpu_count - 1)", [&] {
    return &block.halo_pairs_downward(gpu_count - 1U);
  });
  // Far-OOB also throws.
  ok &= expect_throws_out_of_range("halo_pairs_upward(gpu_count + 7)", [&] {
    return &block.halo_pairs_upward(gpu_count + 7U);
  });
  ok &= expect_throws_out_of_range("halo_pairs_downward(gpu_count + 7)", [&] {
    return &block.halo_pairs_downward(gpu_count + 7U);
  });

  if (!ok) {
    return 1;
  }
  std::printf("test_block_oob_accessor: OK (gpu_count=%zu)\n", gpu_count);
  return 0;
}

}  // namespace

int
main() {
  return run_test();
}
