// Skeleton::Options::halo_stream_priority: under an OCC overlap mode the stream carrying the
// auto-inserted halo gets the greatest driver priority when High is requested, and the anchor
// stream keeps the default priority. Under OverlapMode::None no halo stream exists.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/neighbour_value.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr carbon::Skeleton::Options kPairOptions{carbon::OverlapMode::Standard,
                                                 carbon::StreamPriority::High};
static_assert(kPairOptions.overlap_mode == carbon::OverlapMode::Standard);
static_assert(kPairOptions.halo_stream_priority == carbon::StreamPriority::High);
static_assert(!kPairOptions.strict_halo && kPairOptions.max_streams == 0U &&
              !kPairOptions.log_dot_on_compile && !kPairOptions.use_cuda_graph);

constexpr carbon::Skeleton::Options kPositionalOptions{
    carbon::OverlapMode::Extended, true, 3U, false, false, carbon::StreamPriority::High};
static_assert(kPositionalOptions.halo_stream_priority == carbon::StreamPriority::High);
static_assert(kPositionalOptions.strict_halo && kPositionalOptions.max_streams == 3U);

constexpr carbon::Skeleton::Options kLegacyOptions{carbon::OverlapMode::Standard, false, 0U, false,
                                                   true};
static_assert(kLegacyOptions.halo_stream_priority == carbon::StreamPriority::Normal);
static_assert(kLegacyOptions.use_cuda_graph);

[[nodiscard]] auto
build_jacobi(carbon::Dense& dense,
             carbon::Dense::Field<float,
                                  1>& fin,
             carbon::Dense::Field<float,
                                  1>& fout,
             std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&fin, &fout](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(fin);
        const auto out = loader.write(fout);
        return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          const carbon::NeighbourValue<float> zp = in.neighbour(s, idx, {0, 0, 1}, 0);
          const carbon::NeighbourValue<float> zm = in.neighbour(s, idx, {0, 0, -1}, 0);
          const float self = in.read(s, idx, 0);
          out.write(s, idx, 0, (zp.valid && zm.valid) ? 0.5F * (zp.value + zm.value) : self);
        };
      },
      std::move(name),
      /*shared_mem=*/0U);
}

constexpr auto kNoStream = static_cast<std::size_t>(-1);

[[nodiscard]] auto
halo_stream_index(const carbon::Skeleton& sk) -> std::size_t {
  const auto& kernels = sk.effective_kernels();
  for (std::size_t i = 0; i < kernels.size(); ++i) {
    if (kernels[i].kind() == carbon::KernelKind::Halo) {
      return sk.kernel_streams()[i];
    }
  }
  return kNoStream;
}

#if CARBON_BACKEND_IS_CUDALIKE
[[nodiscard]] auto
native_priority(const carbon::Stream& stream) -> int {
  int priority = 0;
  const carbon::detail::gpu_error_t err =
      carbon::detail::gpu_stream_get_priority(stream.native_at(0), &priority);
  if (err != carbon::detail::gpu_success) {
    CARBON_LOG_ERROR("test_skeleton_halo_priority: gpu_stream_get_priority failed: {}",
                     carbon::detail::gpu_error_string(err));
    std::exit(EXIT_FAILURE);
  }
  return priority;
}
#endif

[[nodiscard]] auto
check_overlap(carbon::System& sys,
              carbon::Dense& dense,
              carbon::Dense::Field<float,
                                   1>& u,
              carbon::Dense::Field<float,
                                   1>& v,
              carbon::OverlapMode mode,
              carbon::StreamPriority priority,
              const char* label) -> bool {
  carbon::Skeleton sk(sys, carbon::Skeleton::Options{mode, priority});
  sk.add(build_jacobi(dense, u, v, std::string{"step_"} + label));
  sk.compile();
  const std::size_t halo = halo_stream_index(sk);
  if (halo == kNoStream || halo == 0U || halo >= sk.num_streams()) {
    CARBON_LOG_ERROR(
        "test_skeleton_halo_priority[{}]: halo must run on a secondary stream "
        "(got {} of {})",
        label, halo, sk.num_streams());
    return false;
  }
#if CARBON_BACKEND_IS_CUDALIKE
  int least = 0;
  int greatest = 0;
  (void)carbon::detail::gpu_stream_priority_range(&least, &greatest);
  if (greatest == least) {
    CARBON_LOG_INFO(
        "test_skeleton_halo_priority: device exposes a single stream priority; "
        "skipping native priority checks");
    return true;
  }
  const int anchor_priority = native_priority(sk.streams()[0]);
  const int halo_priority = native_priority(sk.streams()[halo]);
  const bool want_high = priority == carbon::StreamPriority::High;
  if (anchor_priority == greatest) {
    CARBON_LOG_ERROR("test_skeleton_halo_priority[{}]: anchor stream must keep default priority",
                     label);
    return false;
  }
  if ((halo_priority == greatest) != want_high) {
    CARBON_LOG_ERROR(
        "test_skeleton_halo_priority[{}]: halo stream priority {} (range least={} "
        "greatest={}), expected {}",
        label, halo_priority, least, greatest, want_high ? "greatest" : "default");
    return false;
  }
#endif
  return true;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_skeleton_halo_priority: no devices visible");
    return EXIT_FAILURE;
  }

  std::vector<carbon::detail::Gpu> one = carbon::get_all_devices();
  one.erase(one.begin() + 1, one.end());
  carbon::System sys(one);
  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, 1}, {0, 0, -1}},
                          /*remove_center=*/true);
  carbon::Dense dense(sys, {8, 6, 12}, stencil);
  auto u = dense.newField<float, 1>(carbon::MemorySide::Device, "u");
  auto v = dense.newField<float, 1>(carbon::MemorySide::Device, "v");

  if (!check_overlap(sys, dense, u, v, carbon::OverlapMode::Standard, carbon::StreamPriority::High,
                     "standard_high") ||
      !check_overlap(sys, dense, u, v, carbon::OverlapMode::Standard,
                     carbon::StreamPriority::Normal, "standard_normal") ||
      !check_overlap(sys, dense, u, v, carbon::OverlapMode::Extended, carbon::StreamPriority::High,
                     "extended_high")) {
    return EXIT_FAILURE;
  }

  // Ping-pong pair on one System, as in the LBM benchmark: both halo streams are high priority.
  {
    const carbon::Skeleton::Options opts{carbon::OverlapMode::Standard,
                                         carbon::StreamPriority::High};
    carbon::Skeleton sk_ab(sys, opts);
    sk_ab.add(build_jacobi(dense, u, v, "step_ab"));
    carbon::Skeleton sk_ba(sys, opts);
    sk_ba.add(build_jacobi(dense, v, u, "step_ba"));
    sk_ab.compile();
    sk_ba.compile();
    for (int it = 0; it < 4; ++it) {
      ((it % 2 == 0) ? sk_ab : sk_ba).run();
    }
    sk_ab.wait();
    sk_ba.wait();
#if CARBON_BACKEND_IS_CUDALIKE
    int least = 0;
    int greatest = 0;
    (void)carbon::detail::gpu_stream_priority_range(&least, &greatest);
    if (greatest != least) {
      for (const carbon::Skeleton* sk : {&sk_ab, &sk_ba}) {
        if (native_priority(sk->streams()[halo_stream_index(*sk)]) != greatest) {
          CARBON_LOG_ERROR("test_skeleton_halo_priority: ping-pong halo stream not high priority");
          return EXIT_FAILURE;
        }
      }
    }
#endif
  }

  // Recompile after add(): streams_ survives the recompile, so the halo reservation must not
  // settle on a reused normal-priority stream.
  {
    carbon::Skeleton sk(sys, carbon::Skeleton::Options{carbon::OverlapMode::Standard,
                                                       carbon::StreamPriority::High});
    sk.add(build_jacobi(dense, u, v, "step_re0"));
    sk.compile();
    sk.add(build_jacobi(dense, v, u, "step_re1"));
    sk.compile();
    sk.run();
    sk.wait();
#if CARBON_BACKEND_IS_CUDALIKE
    int least = 0;
    int greatest = 0;
    (void)carbon::detail::gpu_stream_priority_range(&least, &greatest);
    const auto& kernels = sk.effective_kernels();
    std::size_t n_halos = 0U;
    for (std::size_t i = 0; i < kernels.size(); ++i) {
      if (kernels[i].kind() != carbon::KernelKind::Halo) {
        continue;
      }
      ++n_halos;
      const std::size_t s = sk.kernel_streams()[i];
      if (s == 0U || (greatest != least && native_priority(sk.streams()[s]) != greatest)) {
        CARBON_LOG_ERROR(
            "test_skeleton_halo_priority[recompile]: halo {} on stream {} is not "
            "a high-priority secondary stream",
            i, s);
        return EXIT_FAILURE;
      }
    }
    if (n_halos < 2U) {
      CARBON_LOG_ERROR("test_skeleton_halo_priority[recompile]: expected two halos, got {}",
                       n_halos);
      return EXIT_FAILURE;
    }
#endif
  }

  // OverlapMode::None: the halo stays on the anchor stream, so the priority has nothing to act on.
  {
    carbon::Skeleton sk(
        sys, carbon::Skeleton::Options{carbon::OverlapMode::None, carbon::StreamPriority::High});
    sk.add(build_jacobi(dense, u, v, "step_none"));
    sk.compile();
    if (sk.num_streams() != 1U || halo_stream_index(sk) != 0U) {
      CARBON_LOG_ERROR("test_skeleton_halo_priority: OverlapMode::None must stay single-stream");
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
