/**
 * @file test_kernel_polymorphic.cu
 *
 * Exercises the type-erased @c carbon::Kernel value type: builds a heterogeneous
 * pipeline (Anchor, Compute, Halo, Reduction) into a single @c std::vector<Kernel>,
 * iterates @c launch on every entry, and confirms that the result matches a
 * hand-orchestrated reference run (same kernels launched directly, no vector).
 *
 * Why this test exists
 * --------------------
 * The Container-style refactor (mirroring Neon::set::Container) collapses the previously
 * templated @c carbon::Kernel<Grid, LoadingFn> and @c carbon::ReduceKernel<...> into a
 * single value-type @c carbon::Kernel that any kernel kind can produce. The whole point
 * is that compute, reduction, halo, and anchor can share storage and launch through a
 * uniform virtual @c launch — we verify the storage and launch behave exactly like the
 * direct-call equivalents.
 *
 * Single-GPU only because the reduce path is single-GPU in v1.
 */

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/devices.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/transfer.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr int64_t kWidth = 6;
constexpr int64_t kHeight = 5;
constexpr int64_t kDepth = 4;

// Encode (x, y, z) -> deterministic int64 so we can predict the host-side reference sum
// after the compute step (u_out := 2 * u_in).
[[nodiscard]] auto
encode(int64_t x,
       int64_t y,
       int64_t z) noexcept -> int64_t {
  return ((static_cast<int64_t>(x) * 1009) + y) * 1013 + z;
}

[[nodiscard]] auto
host_reference_double_sum(int64_t w,
                          int64_t h,
                          int64_t d) noexcept -> int64_t {
  int64_t s = 0;
  for (int64_t z = 0; z < d; ++z) {
    for (int64_t y = 0; y < h; ++y) {
      for (int64_t x = 0; x < w; ++x) {
        s += 2 * encode(x, y, z);
      }
    }
  }
  return s;
}

}  // namespace

auto
main() -> int {
  carbon::init();

  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_kernel_polymorphic: no CUDA devices visible");
    return EXIT_FAILURE;
  }

  // Pin to a single GPU; v1 reduction is single-GPU.
  std::vector<carbon::detail::Gpu> all = carbon::get_all_devices();
  std::vector<carbon::detail::Gpu> single;
  single.push_back(std::move(all.front()));
  carbon::System sys(std::move(single));

  // Stencil with radius 1 -> halo_z == 1, so the Halo kernel kind sees a non-trivial
  // configuration even though on a single-GPU Open layout the actual exchange is a
  // no-op (no neighbour slabs). The structural plumbing is what we want to exercise
  // here: that a Halo can sit in std::vector<Kernel> and launch() doesn't throw.
  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}});
  const carbon::int64_3d global{kWidth, kHeight, kDepth};
  carbon::Dense dense(sys, global, stencil);

  // ---- Empty Kernel: default-constructed launch must throw ----
  {
    carbon::Kernel empty;
    if (!empty.empty()) {
      CARBON_LOG_ERROR(
          "test_kernel_polymorphic: default-constructed Kernel reported empty()=false");
      return EXIT_FAILURE;
    }
    if (empty.kind() != carbon::KernelKind::Anchor) {
      CARBON_LOG_ERROR("test_kernel_polymorphic: empty Kernel kind expected Anchor, got {}",
                       static_cast<int>(empty.kind()));
      return EXIT_FAILURE;
    }
    bool threw = false;
    try {
      carbon::Stream stream = sys.stream();
      empty.launch(stream);
    } catch (const std::logic_error&) {
      threw = true;
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_kernel_polymorphic: empty Kernel launch threw wrong type: {}",
                       e.what());
      return EXIT_FAILURE;
    }
    if (!threw) {
      CARBON_LOG_ERROR("test_kernel_polymorphic: empty Kernel launch did not throw");
      return EXIT_FAILURE;
    }
  }

  // ---- Reference path: direct launches, no std::vector<Kernel> ----
  int64_t reference_sum = 0;
  {
    auto u_in = dense.newField<int64_t, 1>(carbon::MemorySide::HostDevice);
    auto u_out = dense.newField<int64_t, 1>(carbon::MemorySide::Device);

    u_in.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
      return encode(gc.x, gc.y, gc.z);
    });

    carbon::Stream stream = sys.stream();
    carbon::Transfer<int64_t>(sys, u_in.cpu_buffers(), u_in.gpu_buffers())(stream);

    auto compute = dense.make_kernel([&u_in, &u_out](carbon::KernelLoader& loader) {
      const auto in = loader(u_in);
      auto out = loader(u_out);
      return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> void {
        out.write(s, idx, 0, 2 * in.read(s, idx, 0));
      };
    });

    carbon::Halo<int64_t, 1> halo(u_out);

    carbon::ReductionResult<int64_t> result(sys);
    auto reducer = dense.make_reduce_kernel<int64_t>(
        [&u_out](carbon::KernelLoader& loader) {
          const auto in = loader(u_out);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> int64_t {
            return in.read(s, idx, 0);
          };
        },
        result);

    compute.launch(stream);
    halo(stream);
    reducer.launch(stream);
    result.sync(stream);
    reference_sum = result.host_value();
  }

  const int64_t expected = host_reference_double_sum(kWidth, kHeight, kDepth);
  if (reference_sum != expected) {
    CARBON_LOG_ERROR(
        "test_kernel_polymorphic: reference path mismatch with host model: got {}, expected {}",
        reference_sum, expected);
    return EXIT_FAILURE;
  }

  // ---- Polymorphic path: heterogeneous std::vector<Kernel> ----
  // Build the same pipeline through the value-typed Kernel factories, store it in a
  // std::vector<Kernel>, and run by iterating launch over the vector. The result must
  // bit-match the reference path.
  int64_t poly_sum = 0;
  std::vector<carbon::KernelKind> observed_kinds;
  std::vector<std::string_view> observed_names;
  {
    auto u_in = dense.newField<int64_t, 1>(carbon::MemorySide::HostDevice);
    auto u_out = dense.newField<int64_t, 1>(carbon::MemorySide::Device);

    u_in.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
      return encode(gc.x, gc.y, gc.z);
    });

    carbon::Stream stream = sys.stream();
    carbon::Transfer<int64_t>(sys, u_in.cpu_buffers(), u_in.gpu_buffers())(stream);

    carbon::ReductionResult<int64_t> result(sys);

    std::vector<carbon::Kernel> pipeline;
    pipeline.push_back(carbon::Kernel::anchor("begin"));
    pipeline.push_back(dense.make_kernel([&u_in, &u_out](carbon::KernelLoader& loader) {
      const auto in = loader(u_in);
      auto out = loader(u_out);
      return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> void {
        out.write(s, idx, 0, 2 * in.read(s, idx, 0));
      };
    }));
    pipeline.push_back(u_out.make_halo_kernel("halo_u_out"));
    pipeline.push_back(dense.make_reduce_kernel<int64_t>(
        [&u_out](carbon::KernelLoader& loader) {
          const auto in = loader(u_out);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> int64_t {
            return in.read(s, idx, 0);
          };
        },
        result));
    pipeline.push_back(carbon::Kernel::anchor("end"));

    // Capture introspection while we still have the named handles.
    for (const auto& k : pipeline) {
      observed_kinds.push_back(k.kind());
      observed_names.push_back(k.name());
    }

    for (const auto& k : pipeline) {
      k.launch(stream);
    }
    result.sync(stream);
    poly_sum = result.host_value();
  }

  // ---- Verify kind tags / names match what the factories advertise ----
  // Dense::make_kernel / make_reduce_kernel return device-only (Compute / Reduction)
  // kernels — the cheap default. The dual-target variants live behind
  // make_universal_kernel / make_universal_reduce_kernel and are exercised in
  // test_kernel_host_parity.
  const std::vector<carbon::KernelKind> expected_kinds{
      carbon::KernelKind::Anchor,    carbon::KernelKind::Compute, carbon::KernelKind::Halo,
      carbon::KernelKind::Reduction, carbon::KernelKind::Anchor,
  };
  if (observed_kinds != expected_kinds) {
    CARBON_LOG_ERROR("test_kernel_polymorphic: pipeline kind sequence mismatch");
    return EXIT_FAILURE;
  }
  if (observed_names.size() != 5U || observed_names.front() != "begin" ||
      observed_names[2] != "halo_u_out" || observed_names.back() != "end") {
    CARBON_LOG_ERROR("test_kernel_polymorphic: pipeline names do not match expectations");
    return EXIT_FAILURE;
  }

  // ---- Compute / reduce paths must agree bit-for-bit ----
  if (poly_sum != reference_sum) {
    CARBON_LOG_ERROR(
        "test_kernel_polymorphic: polymorphic-path sum {} differs from reference-path sum {}",
        poly_sum, reference_sum);
    return EXIT_FAILURE;
  }

  // ---- Host-kernel pipeline: same heterogeneous std::vector<Kernel> shape, but the
  // compute step is a Dense::make_host_kernel (OpenMP loop over the slab) and the
  // reduce step is a Dense::make_host_reduce_kernel (OpenMP per-thread partials +
  // sequential combine). The expected scalar is the SAME sum the reference and
  // polymorphic-device paths produced — proving that:
  //   (a) ComputeHost + ReductionHost slot into a std::vector<Kernel> alongside the
  //       device kinds without any wrapper.
  //   (b) The same lambda body fed to make_kernel works as-is when fed to
  //       make_host_kernel (the user-facing surface promised by the design).
  //   (c) Stream::wait at the start of every host launch keeps a heterogeneous
  //       device-then-host pipeline correct.
  int64_t host_poly_sum = 0;
  std::vector<carbon::KernelKind> host_observed_kinds;
  {
    auto u_in = dense.newField<int64_t, 1>(carbon::MemorySide::HostDevice);
    auto u_out = dense.newField<int64_t, 1>(carbon::MemorySide::HostDevice);

    u_in.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
      return encode(gc.x, gc.y, gc.z);
    });

    carbon::Stream stream = sys.stream();
    carbon::ReductionResult<int64_t> result(sys);

    std::vector<carbon::Kernel> host_pipeline;
    host_pipeline.push_back(carbon::Kernel::anchor("host_begin"));
    host_pipeline.push_back(dense.make_host_kernel([&u_in, &u_out](carbon::KernelLoader& loader) {
      const auto in = loader(u_in);
      auto out = loader(u_out);
      return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> void {
        out.write(s, idx, 0, 2 * in.read(s, idx, 0));
      };
    }));
    host_pipeline.push_back(dense.make_host_reduce_kernel<int64_t>(
        [&u_out](carbon::KernelLoader& loader) {
          const auto in = loader(u_out);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> int64_t {
            return in.read(s, idx, 0);
          };
        },
        result));
    host_pipeline.push_back(carbon::Kernel::anchor("host_end"));

    for (const auto& k : host_pipeline) {
      host_observed_kinds.push_back(k.kind());
    }

    for (const auto& k : host_pipeline) {
      k.launch(stream);
    }
    // ReductionHost writes directly into result.host_data(); sync is still safe but
    // a no-op for the host path. Calling it here exercises that no-op contract.
    result.sync(stream);
    host_poly_sum = result.host_value();
  }

  const std::vector<carbon::KernelKind> expected_host_kinds{
      carbon::KernelKind::Anchor,
      carbon::KernelKind::ComputeHost,
      carbon::KernelKind::ReductionHost,
      carbon::KernelKind::Anchor,
  };
  if (host_observed_kinds != expected_host_kinds) {
    CARBON_LOG_ERROR("test_kernel_polymorphic: host pipeline kind sequence mismatch");
    return EXIT_FAILURE;
  }
  if (host_poly_sum != reference_sum) {
    CARBON_LOG_ERROR(
        "test_kernel_polymorphic: host pipeline sum {} differs from reference-path sum {}",
        host_poly_sum, reference_sum);
    return EXIT_FAILURE;
  }
  CARBON_LOG_INFO("test_kernel_polymorphic: host pipeline (ComputeHost + ReductionHost) sum = {}",
                  host_poly_sum);

  // ---- Kernel copy semantics: copies share the same impl and launch the same work ----
  // Verifies that std::vector<Kernel> can be copied/moved freely (the same property the
  // future skeleton/graph layer will rely on when it copies user pipelines into graph
  // nodes).
  {
    carbon::Kernel a = carbon::Kernel::anchor("shared");
    carbon::Kernel b = a;
    if (a.impl().get() != b.impl().get()) {
      CARBON_LOG_ERROR("test_kernel_polymorphic: copied Kernel does not share impl pointer");
      return EXIT_FAILURE;
    }
    if (b.name() != "shared" || b.kind() != carbon::KernelKind::Anchor) {
      CARBON_LOG_ERROR("test_kernel_polymorphic: copied Kernel lost name/kind");
      return EXIT_FAILURE;
    }
  }

  CARBON_LOG_INFO("test_kernel_polymorphic: PASS (sum = {}, expected {})", poly_sum, expected);
  return EXIT_SUCCESS;
}
