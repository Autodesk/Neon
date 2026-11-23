// carbon::Skeleton region-mask dependencies (AccessToken::mask / KernelLoader::region).
//
// The domain is labelled by a full partition of three atoms:
//
//   kWall  : the two x side walls (x == 0 || x == nx - 1)   — boundary-condition cells
//   kLower : interior cells with global z <  nz / 2
//   kUpper : interior cells with global z >= nz / 2
//
// Kernels declare their footprint with L.region(mask) and filter themselves in the body.
// The subtests lock:
//
//   A. Structure: disjoint masked writers lose their edge and land on two streams;
//      overlapping masks, all-ones vs masked, and masked write vs overlapping read keep it;
//      R/R never conflicts.
//   B. Stencil reads stay whole-field: a stencil reader (and its auto-halo) waits on every
//      masked writer of the field; a masked Map reader still waits on the halo.
//   C. Loader API: explicit per-call masks win over region() in either order, stencil
//      tokens stay all-ones, region() is readable in the launch phase.
//   D. Empty masks conflict with nothing; the max_streams cap still holds.
//   E. Options::ignore_region_masks and CARBON_IGNORE_REGION_MASKS reproduce the unmasked
//      schedule exactly (same edges, same launch order); schedules are deterministic.
//   F. Numerics on every visible GPU: masked pipelines under every OverlapMode (including
//      two masked stencils sharing one halo and a masked reduction racing a disjoint
//      writer) match the same pipeline run sequentially with masks ignored.
//
// Every non-default schedule is compared bit-for-bit against that sequential reference, so
// a mask that wrongly drops a true dependency shows up as a numeric mismatch.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/region_mask.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace {

using Field = carbon::Dense::Field<float, 1>;
using carbon::RegionMask;

constexpr std::uint8_t kWall = 0;
constexpr std::uint8_t kLower = 1;
constexpr std::uint8_t kUpper = 2;

const RegionMask kWallMask = RegionMask::from_label(kWall);
const RegionMask kLowerMask = RegionMask::from_label(kLower);
const RegionMask kUpperMask = RegionMask::from_label(kUpper);
const RegionMask kInteriorMask = kLowerMask | kUpperMask;

struct Geo {
  int64_t nx;
  int64_t nz;

  [[nodiscard]] CARBON_HD auto
  label(const carbon::int64_3d& g) const -> std::uint8_t {
    if (g.x == 0 || g.x == nx - 1) {
      return kWall;
    }
    return g.z < nz / 2 ? kLower : kUpper;
  }
};

auto
fail(const std::string& msg) -> bool {
  CARBON_LOG_ERROR("test_skeleton_region_mask: {}", msg);
  return false;
}

// out = in * a + b on the cells whose label is in @p mask; every other cell is untouched.
[[nodiscard]] auto
build_affine(carbon::Dense& dense,
             Field& in,
             Field& out,
             RegionMask mask,
             float a,
             float b,
             Geo geo,
             std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&in, &out, mask, a, b, geo](carbon::KernelLoader& loader) {
        loader.region(mask);
        const auto src = loader.read(in);
        const auto dst = loader.write(out);
        return [src, dst, mask, a, b, geo] CARBON_HOST_DEVICE(carbon::Dense::Span s,
                                                              carbon::Dense::Idx idx) {
          if (!mask.test(geo.label(s.global_location(idx)))) {
            return;
          }
          dst.write(s, idx, 0, src.read(s, idx, 0) * a + b);
        };
      },
      std::move(name),
      /*shared_mem=*/0U);
}

// 6-point average of @p in (whole-field stencil read) written to @p out on @p mask cells.
[[nodiscard]] auto
build_stencil(carbon::Dense& dense,
              Field& in,
              Field& out,
              RegionMask mask,
              Geo geo,
              std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&in, &out, mask, geo](carbon::KernelLoader& loader) {
        loader.region(mask);
        const auto src = loader.read_stencil(in);
        const auto dst = loader.write(out);
        return [src, dst, mask, geo] CARBON_HOST_DEVICE(carbon::Dense::Span s,
                                                        carbon::Dense::Idx idx) {
          if (!mask.test(geo.label(s.global_location(idx)))) {
            return;
          }
          const carbon::NeighbourValue<float> xp = src.neighbour(s, idx, {1, 0, 0}, 0);
          const carbon::NeighbourValue<float> xm = src.neighbour(s, idx, {-1, 0, 0}, 0);
          const carbon::NeighbourValue<float> yp = src.neighbour(s, idx, {0, 1, 0}, 0);
          const carbon::NeighbourValue<float> ym = src.neighbour(s, idx, {0, -1, 0}, 0);
          const carbon::NeighbourValue<float> zp = src.neighbour(s, idx, {0, 0, 1}, 0);
          const carbon::NeighbourValue<float> zm = src.neighbour(s, idx, {0, 0, -1}, 0);
          float sum = 0.0F;
          float n = 0.0F;
          for (const carbon::NeighbourValue<float>& nb : {xp, xm, yp, ym, zp, zm}) {
            if (nb.valid) {
              sum += nb.value;
              n += 1.0F;
            }
          }
          dst.write(s, idx, 0, n > 0.0F ? sum / n : src.read(s, idx, 0));
        };
      },
      std::move(name),
      /*shared_mem=*/0U);
}

// Sum of @p in over the cells whose label is in @p mask.
[[nodiscard]] auto
build_masked_sum(carbon::Dense& dense,
                 Field& in,
                 RegionMask mask,
                 Geo geo,
                 carbon::ReductionResult<float>& result) -> carbon::Kernel {
  return dense.make_reduce_kernel<float>(
      [&in, mask, geo](carbon::KernelLoader& loader) {
        loader.region(mask);
        const auto src = loader.read(in);
        return [src, mask, geo] CARBON_HOST_DEVICE(carbon::Dense::Span s,
                                                   carbon::Dense::Idx idx) -> float {
          return mask.test(geo.label(s.global_location(idx))) ? src.read(s, idx, 0) : 0.0F;
        };
      },
      result, carbon::reduce::Sum<float>{});
}

struct Fields {
  Field u;
  Field v;
  Field src;
  Field dst;
  Field out;
};

[[nodiscard]] auto
make_fields(carbon::Dense& dense) -> Fields {
  return Fields{dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u"),
                dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v"),
                dense.newField<float, 1>(carbon::MemorySide::HostDevice, "src"),
                dense.newField<float, 1>(carbon::MemorySide::HostDevice, "dst"),
                dense.newField<float, 1>(carbon::MemorySide::HostDevice, "out")};
}

template <typename Fn>
auto
for_each_field(Fields& f,
               Fn fn) -> void {
  fn(f.u, 0);
  fn(f.v, 1);
  fn(f.src, 2);
  fn(f.dst, 3);
  fn(f.out, 4);
}

// Fills the whole storage (owned + halo) of every field with a distinct deterministic value.
auto
init_fields(const carbon::Dense& dense,
            Fields& f) -> void {
  for_each_field(f, [&dense](Field& field, int seed) {
    for (size_t g = 0; g < dense.gpu_count(); ++g) {
      const auto& span = dense.spans()[g];
      const auto haloed = span.haloed_extent();
      const int64_t pitch = span.xy_slab_elements();
      const int64_t z_origin = span.z_offset();
      float* h = field.cpu_buffers()[g].data();
      for (int64_t z = 0; z < haloed.z; ++z) {
        for (int64_t y = 0; y < haloed.y; ++y) {
          for (int64_t x = 0; x < haloed.x; ++x) {
            h[z * pitch + y * haloed.x + x] =
                static_cast<float>(seed * 1000 + x + y * 11 + (z_origin + z) * 113);
          }
        }
      }
    }
  });
}

[[nodiscard]] auto
transfer_fields(const carbon::System& sys,
                Fields& f,
                const carbon::Stream& s,
                bool to_device) -> bool {
  try {
    for_each_field(f, [&sys, &s, to_device](Field& field, int) {
      if (to_device) {
        carbon::Transfer<float>(sys, field.cpu_buffers(), field.gpu_buffers())(s);
      } else {
        carbon::Transfer<float>(sys, field.gpu_buffers(), field.cpu_buffers())(s);
      }
    });
    s.wait();
  } catch (const std::exception& e) {
    return fail(std::string("transfer failed: ") + e.what());
  }
  return true;
}

[[nodiscard]] auto
compare_owned(const carbon::Dense& dense,
              const Field& a,
              const Field& b,
              const std::string& tag) -> bool {
  for (size_t g = 0; g < dense.gpu_count(); ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t lim_z = span.logic_extent().z;
    const float* ha = a.cpu_buffers()[g].data();
    const float* hb = b.cpu_buffers()[g].data();
    for (int64_t z = halo_z; z < halo_z + lim_z; ++z) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          if (ha[off] != hb[off]) {
            return fail(tag + ": mismatch on field '" + a.name() + "' at gpu=" + std::to_string(g) +
                        " (x=" + std::to_string(x) + ", y=" + std::to_string(y) +
                        ", z=" + std::to_string(z) + "): test=" + std::to_string(ha[off]) +
                        " ref=" + std::to_string(hb[off]));
          }
        }
      }
    }
  }
  return true;
}

using Builder = std::function<std::vector<carbon::Kernel>(carbon::Dense&, Fields&)>;

[[nodiscard]] auto
make_skeleton(carbon::System& sys,
              carbon::Dense& dense,
              Fields& f,
              const Builder& build,
              carbon::Skeleton::Options opts) -> carbon::Skeleton {
  carbon::Skeleton sk(sys, opts);
  for (carbon::Kernel& k : build(dense, f)) {
    sk.add(std::move(k));
  }
  return sk;
}

// Runs @p build under @p opts for @p iters iterations and compares every field against the
// same pipeline run on one stream with masks ignored (the pre-region-mask schedule).
[[nodiscard]] auto
run_and_compare(carbon::System& sys,
                carbon::Dense& dense,
                const Builder& build,
                carbon::Skeleton::Options opts,
                int iters,
                const std::string& tag) -> bool {
  Fields test = make_fields(dense);
  Fields ref = make_fields(dense);
  init_fields(dense, test);
  init_fields(dense, ref);
  carbon::Stream s = sys.stream();
  if (!transfer_fields(sys, test, s, true) || !transfer_fields(sys, ref, s, true)) {
    return false;
  }

  carbon::Skeleton::Options ref_opts;
  ref_opts.max_streams = 1U;
  ref_opts.ignore_region_masks = true;
  try {
    carbon::Skeleton sk_test = make_skeleton(sys, dense, test, build, opts);
    carbon::Skeleton sk_ref = make_skeleton(sys, dense, ref, build, ref_opts);
    for (int it = 0; it < iters; ++it) {
      sk_test.run();
    }
    sk_test.wait();
    for (int it = 0; it < iters; ++it) {
      sk_ref.run();
    }
    sk_ref.wait();
    if (opts.overlap_mode == carbon::OverlapMode::None && !opts.ignore_region_masks &&
        opts.max_streams != 1U && sk_test.num_streams() < 2U) {
      return fail(tag + ": expected the masked schedule to use more than one stream");
    }
  } catch (const std::exception& e) {
    return fail(tag + ": run threw: " + e.what());
  }

  if (!transfer_fields(sys, test, s, false) || !transfer_fields(sys, ref, s, false)) {
    return false;
  }
  return compare_owned(dense, test.u, ref.u, tag) && compare_owned(dense, test.v, ref.v, tag) &&
         compare_owned(dense, test.src, ref.src, tag) &&
         compare_owned(dense, test.dst, ref.dst, tag) &&
         compare_owned(dense, test.out, ref.out, tag);
}

[[nodiscard]] auto
compiled(carbon::System& sys,
         carbon::Dense& dense,
         Fields& f,
         const Builder& build,
         carbon::Skeleton::Options opts = {}) -> carbon::Skeleton {
  carbon::Skeleton sk = make_skeleton(sys, dense, f, build, opts);
  sk.compile();
  return sk;
}

using Edges = std::vector<std::vector<size_t>>;

// ---------- A. structure ----------

[[nodiscard]] auto
subtest_structure(carbon::System& sys,
                  carbon::Dense& dense,
                  Geo geo) -> bool {
  Fields f = make_fields(dense);

  auto two_writers = [geo](RegionMask m0, RegionMask m1) {
    return [geo, m0, m1](carbon::Dense& d, Fields& fl) {
      std::vector<carbon::Kernel> ks;
      ks.push_back(build_affine(d, fl.src, fl.dst, m0, 2.0F, 1.0F, geo, "w0"));
      ks.push_back(build_affine(d, fl.src, fl.dst, m1, 3.0F, 2.0F, geo, "w1"));
      return ks;
    };
  };

  {
    carbon::Skeleton sk = compiled(sys, dense, f, two_writers(kLowerMask, kUpperMask));
    if (sk.edges() != Edges{{}, {}}) {
      return fail("disjoint writers: expected no edge");
    }
    if (sk.kernel_streams() != std::vector<size_t>{0U, 1U} || sk.num_streams() != 2U) {
      return fail("disjoint writers: expected streams {0, 1}");
    }
  }
  {
    carbon::Skeleton sk = compiled(sys, dense, f, two_writers(kInteriorMask, kUpperMask));
    if (sk.edges() != Edges{{1U}, {}}) {
      return fail("overlapping writers: expected edge 0 -> 1");
    }
  }
  {
    carbon::Skeleton sk = compiled(sys, dense, f, two_writers(RegionMask::all(), kUpperMask));
    if (sk.edges() != Edges{{1U}, {}} || sk.num_streams() != 1U) {
      return fail("all-ones vs masked writer: expected edge 0 -> 1 on one stream");
    }
  }
  {
    // W dst on kLower, then R dst on kUpper / kLower.
    for (const bool overlap : {false, true}) {
      const RegionMask reader = overlap ? kLowerMask : kUpperMask;
      carbon::Skeleton sk = compiled(sys, dense, f, [geo, reader](carbon::Dense& d, Fields& fl) {
        std::vector<carbon::Kernel> ks;
        ks.push_back(build_affine(d, fl.src, fl.dst, kLowerMask, 2.0F, 0.0F, geo, "w"));
        ks.push_back(build_affine(d, fl.dst, fl.out, reader, 1.0F, 1.0F, geo, "r"));
        return ks;
      });
      if (sk.edges() != (overlap ? Edges{{1U}, {}} : Edges{{}, {}})) {
        return fail(std::string("masked write vs ") + (overlap ? "overlapping" : "disjoint") +
                    " read: wrong edges");
      }
    }
  }
  {
    // R/R on the same mask never conflicts; the writes go to different fields.
    carbon::Skeleton sk = compiled(sys, dense, f, [geo](carbon::Dense& d, Fields& fl) {
      std::vector<carbon::Kernel> ks;
      ks.push_back(build_affine(d, fl.src, fl.dst, kLowerMask, 2.0F, 0.0F, geo, "r0"));
      ks.push_back(build_affine(d, fl.src, fl.out, kLowerMask, 3.0F, 0.0F, geo, "r1"));
      return ks;
    });
    if (sk.edges() != Edges{{}, {}}) {
      return fail("R/R on the same mask: expected no edge");
    }
  }
  return true;
}

// ---------- B. stencil reads stay whole-field ----------

auto
writers_then_stencil(Geo geo) -> Builder {
  return [geo](carbon::Dense& d, Fields& fl) {
    std::vector<carbon::Kernel> ks;
    ks.push_back(build_affine(d, fl.src, fl.dst, kLowerMask, 2.0F, 1.0F, geo, "w_lower"));
    ks.push_back(build_affine(d, fl.src, fl.dst, kUpperMask, 3.0F, 2.0F, geo, "w_upper"));
    ks.push_back(build_affine(d, fl.src, fl.dst, kWallMask, 4.0F, 3.0F, geo, "w_wall"));
    ks.push_back(build_stencil(d, fl.dst, fl.out, RegionMask::all(), geo, "smooth"));
    return ks;
  };
}

// dst = f(src) everywhere; interior out = stencil(dst); wall out = map(dst).
auto
stencil_plus_bc(Geo geo) -> Builder {
  return [geo](carbon::Dense& d, Fields& fl) {
    std::vector<carbon::Kernel> ks;
    ks.push_back(build_affine(d, fl.src, fl.dst, RegionMask::all(), 0.5F, 1.0F, geo, "fill"));
    ks.push_back(build_stencil(d, fl.dst, fl.out, kInteriorMask, geo, "interior"));
    ks.push_back(build_affine(d, fl.dst, fl.out, kWallMask, -1.0F, 0.0F, geo, "bc"));
    return ks;
  };
}

[[nodiscard]] auto
subtest_stencil(carbon::System& sys,
                carbon::Dense& dense,
                Geo geo) -> bool {
  Fields f = make_fields(dense);
  {
    // Effective order: w_lower, w_upper, w_wall, auto_halo_dst, smooth.
    carbon::Skeleton sk = compiled(sys, dense, f, writers_then_stencil(geo));
    const auto& eff = sk.effective_kernels();
    if (eff.size() != 5U || eff[3].kind() != carbon::KernelKind::Halo) {
      return fail("writers + stencil: expected one auto-halo before the stencil reader");
    }
    if (sk.min_edges() != Edges{{3U}, {3U}, {3U}, {4U}, {}}) {
      return fail("writers + stencil: the halo must wait on every masked writer");
    }
    if (sk.edges()[0] != std::vector<size_t>{3U, 4U}) {
      return fail("writers + stencil: the stencil read must conflict with each masked writer");
    }
  }
  {
    // Effective order: fill, auto_halo_dst, interior, bc.
    carbon::Skeleton sk = compiled(sys, dense, f, stencil_plus_bc(geo));
    const auto& eff = sk.effective_kernels();
    if (eff.size() != 4U || eff[1].kind() != carbon::KernelKind::Halo) {
      return fail("stencil + bc: unexpected effective kernel list");
    }
    if (sk.min_edges() != Edges{{1U}, {2U, 3U}, {}, {}}) {
      return fail("stencil + bc: bc must wait on the halo and run beside the interior");
    }
    if (sk.kernel_streams()[2] == sk.kernel_streams()[3]) {
      return fail("stencil + bc: interior and bc should be on different streams");
    }
  }
  return true;
}

// ---------- C. loader API ----------

[[nodiscard]] auto
subtest_loader(carbon::System& sys,
               carbon::Dense& dense) -> bool {
  Fields f = make_fields(dense);

  auto token_masks = [](const carbon::Kernel& k) {
    std::vector<RegionMask> masks;
    for (const carbon::AccessToken& t : k.impl()->tokens()) {
      masks.push_back(t.mask);
    }
    return masks;
  };

  // Explicit mask before region(), stencil read after it.
  auto k_before = carbon::Kernel::compute(
      dense,
      [&f](carbon::KernelLoader& loader) {
        const auto a = loader.read(f.u, RegionMask::all());
        loader.region(kLowerMask);
        const auto b = loader.write(f.v);
        const auto c = loader.read_stencil(f.src);
        return [a, b, c] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          b.write(s, idx, 0, a.read(s, idx, 0) + c.read(s, idx, 0));
        };
      },
      "before", 0U);
  if (token_masks(k_before) !=
      std::vector<RegionMask>{RegionMask::all(), kLowerMask, RegionMask::all()}) {
    return fail("explicit mask before region(): wrong token masks");
  }

  // region() first, explicit mask afterwards; a later region() re-stamps implicit tokens only.
  auto k_after = carbon::Kernel::compute(
      dense,
      [&f](carbon::KernelLoader& loader) {
        loader.region(kLowerMask);
        const auto a = loader.read(f.u, kUpperMask);
        const auto b = loader.read_write(f.v);
        loader.region(kInteriorMask);
        return [a, b] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          b.write(s, idx, 0, a.read(s, idx, 0) + b.read(s, idx, 0));
        };
      },
      "after", 0U);
  if (token_masks(k_after) != std::vector<RegionMask>{kUpperMask, kInteriorMask}) {
    return fail("explicit mask after region(): wrong token masks");
  }

  // The launch-phase loader exposes the declared region to the loading lambda.
  RegionMask seen = RegionMask::none();
  auto k_seen = carbon::Kernel::compute(
      dense,
      [&f, &seen](carbon::KernelLoader& loader) {
        loader.region(kUpperMask);
        seen = loader.region();
        const auto a = loader.write(f.out);
        return [a] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          a.write(s, idx, 0, 0.0F);
        };
      },
      "seen", 0U);
  seen = RegionMask::none();
  try {
    carbon::Stream s = sys.stream();
    k_seen.launch(s, carbon::DataView::Standard);
    s.wait();
  } catch (const std::exception& e) {
    return fail(std::string("launch-phase region() kernel threw: ") + e.what());
  }
  if (seen != kUpperMask) {
    return fail("region() not visible in the launch phase: got " + carbon::to_string(seen));
  }
  return true;
}

// ---------- D. empty masks and the stream cap ----------

[[nodiscard]] auto
subtest_empty_and_cap(carbon::System& sys,
                      carbon::Dense& dense,
                      Geo geo) -> bool {
  Fields f = make_fields(dense);
  {
    carbon::Skeleton sk = compiled(sys, dense, f, [geo](carbon::Dense& d, Fields& fl) {
      std::vector<carbon::Kernel> ks;
      ks.push_back(build_affine(d, fl.src, fl.dst, RegionMask::none(), 1.0F, 0.0F, geo, "e"));
      ks.push_back(build_affine(d, fl.src, fl.dst, RegionMask::all(), 2.0F, 0.0F, geo, "a"));
      return ks;
    });
    if (sk.edges() != Edges{{}, {}}) {
      return fail("empty mask: expected it to conflict with nothing");
    }
  }

  // Four writers on atoms no cell carries: pairwise disjoint, so four parallel chains.
  auto four = [geo](carbon::Dense& d, Fields& fl) {
    std::vector<carbon::Kernel> ks;
    for (unsigned b = 10; b < 14; ++b) {
      ks.push_back(build_affine(d, fl.src, fl.dst, RegionMask::bit(b), 1.0F, 0.0F, geo,
                                "w" + std::to_string(b)));
    }
    return ks;
  };
  {
    carbon::Skeleton sk = compiled(sys, dense, f, four);
    if (sk.num_streams() != 4U) {
      return fail("four disjoint writers: expected 4 streams");
    }
  }
  {
    carbon::Skeleton::Options opts;
    opts.max_streams = 2U;
    carbon::Skeleton sk = compiled(sys, dense, f, four, opts);
    if (sk.num_streams() > 2U) {
      return fail("max_streams = 2 not honoured with masked kernels");
    }
  }
  return true;
}

// ---------- E. kill switch and determinism ----------

[[nodiscard]] auto
subtest_ignore_and_determinism(carbon::System& sys,
                               carbon::Dense& dense,
                               Geo geo) -> bool {
  Fields f = make_fields(dense);
  const Builder masked = stencil_plus_bc(geo);
  const Builder unmasked = [geo](carbon::Dense& d, Fields& fl) {
    std::vector<carbon::Kernel> ks;
    ks.push_back(build_affine(d, fl.src, fl.dst, RegionMask::all(), 0.5F, 1.0F, geo, "fill"));
    ks.push_back(build_stencil(d, fl.dst, fl.out, RegionMask::all(), geo, "interior"));
    ks.push_back(build_affine(d, fl.dst, fl.out, RegionMask::all(), -1.0F, 0.0F, geo, "bc"));
    return ks;
  };

  carbon::Skeleton sk_unmasked = compiled(sys, dense, f, unmasked);
  const std::string order_unmasked = sk_unmasked.export_launch_order();

  carbon::Skeleton::Options ignore;
  ignore.ignore_region_masks = true;
  carbon::Skeleton sk_ignored = compiled(sys, dense, f, masked, ignore);
  if (sk_ignored.region_masks_honoured()) {
    return fail("Options::ignore_region_masks did not disable masks");
  }
  if (sk_ignored.edges() != sk_unmasked.edges() ||
      sk_ignored.export_launch_order() != order_unmasked) {
    return fail("ignore_region_masks: schedule differs from the unmasked pipeline");
  }

  if (setenv("CARBON_IGNORE_REGION_MASKS", "1", 1) != 0) {
    return fail("setenv failed");
  }
  carbon::Skeleton sk_env = compiled(sys, dense, f, masked);
  unsetenv("CARBON_IGNORE_REGION_MASKS");
  if (sk_env.region_masks_honoured() || sk_env.export_launch_order() != order_unmasked) {
    return fail("CARBON_IGNORE_REGION_MASKS=1: schedule differs from the unmasked pipeline");
  }

  carbon::Skeleton sk_a = compiled(sys, dense, f, masked);
  carbon::Skeleton sk_b = compiled(sys, dense, f, masked);
  if (!sk_a.region_masks_honoured()) {
    return fail("masks not honoured by default");
  }
  if (sk_a.export_launch_order() != sk_b.export_launch_order()) {
    return fail("masked schedule is not deterministic");
  }
  if (sk_a.export_launch_order() == order_unmasked) {
    return fail("masked schedule should differ from the unmasked one");
  }

  // Exporters annotate non-all masks only, and say when masks are ignored.
  auto has = [](const std::string& text, const char* needle) {
    return text.find(needle) != std::string::npos;
  };
  const std::string list_masked = sk_a.export_kernel_list_dot("masked");
  const std::string list_unmasked = sk_unmasked.export_kernel_list_dot("unmasked");
  const std::string list_ignored = sk_ignored.export_kernel_list_dot("ignored");
  if (!has(list_masked, "region {1-2}") || !has(list_masked, "region {0}")) {
    return fail("export_kernel_list_dot: missing region annotations:\n" + list_masked);
  }
  if (has(list_unmasked, "region {") || has(list_unmasked, "region_masks=ignored")) {
    return fail("export_kernel_list_dot: unmasked pipeline must render without annotations");
  }
  if (has(list_ignored, "region {") || !has(list_ignored, "region_masks=ignored")) {
    return fail("export_kernel_list_dot: ignored masks must be reported, not annotated");
  }
  const std::string dot_masked = sk_a.export_dot("masked");
  if (!has(dot_masked, "region {0}") || has(sk_unmasked.export_dot("unmasked"), "region {")) {
    return fail("export_dot: tooltips must carry non-all masks only");
  }
  return true;
}

// ---------- F. numerics ----------

// Jacobi ping-pong where every half step writes the interior with a stencil and the walls
// with a Map BC, so the two kernels of each half step can run side by side.
auto
jacobi_with_bc(Geo geo) -> Builder {
  return [geo](carbon::Dense& d, Fields& fl) {
    std::vector<carbon::Kernel> ks;
    ks.push_back(build_stencil(d, fl.u, fl.v, kInteriorMask, geo, "step_uv"));
    ks.push_back(build_affine(d, fl.u, fl.v, kWallMask, 0.5F, 1.0F, geo, "bc_uv"));
    ks.push_back(build_stencil(d, fl.v, fl.u, kInteriorMask, geo, "step_vu"));
    ks.push_back(build_affine(d, fl.v, fl.u, kWallMask, 0.5F, 1.0F, geo, "bc_vu"));
    return ks;
  };
}

// Two masked stencils over the same input share one auto-halo.
auto
split_stencils(Geo geo) -> Builder {
  return [geo](carbon::Dense& d, Fields& fl) {
    std::vector<carbon::Kernel> ks;
    ks.push_back(build_affine(d, fl.src, fl.dst, RegionMask::all(), 1.0F, 1.0F, geo, "bump"));
    ks.push_back(build_stencil(d, fl.dst, fl.out, kLowerMask, geo, "smooth_lower"));
    ks.push_back(build_stencil(d, fl.dst, fl.out, kUpperMask, geo, "smooth_upper"));
    ks.push_back(build_affine(d, fl.dst, fl.out, kWallMask, 2.0F, 0.0F, geo, "bc"));
    ks.push_back(build_affine(d, fl.out, fl.src, RegionMask::all(), 0.25F, 0.0F, geo, "back"));
    return ks;
  };
}

[[nodiscard]] auto
subtest_numerics(carbon::System& sys,
                 carbon::Dense& dense,
                 Geo geo) -> bool {
  {
    Fields f = make_fields(dense);
    carbon::Skeleton sk = compiled(sys, dense, f, split_stencils(geo));
    size_t halos = 0;
    for (const carbon::Kernel& k : sk.effective_kernels()) {
      halos += k.kind() == carbon::KernelKind::Halo ? 1U : 0U;
    }
    // Effective order: bump, auto_halo_dst, smooth_lower, smooth_upper, bc, back.
    if (halos != 1U || sk.edges()[2] != std::vector<size_t>{5U}) {
      return fail("split stencils: expected one shared halo and no edge between the stencils");
    }
  }

  constexpr int kIters = 4;
  const carbon::OverlapMode modes[] = {carbon::OverlapMode::None, carbon::OverlapMode::Standard,
                                       carbon::OverlapMode::Extended,
                                       carbon::OverlapMode::TwoWayExtended};
  const std::pair<const char*, Builder> pipelines[] = {
      {"writers_then_stencil", writers_then_stencil(geo)},
      {"stencil_plus_bc", stencil_plus_bc(geo)},
      {"jacobi_with_bc", jacobi_with_bc(geo)},
      {"split_stencils", split_stencils(geo)},
  };
  for (const auto& [name, build] : pipelines) {
    for (const carbon::OverlapMode mode : modes) {
      carbon::Skeleton::Options opts(mode);
      const std::string tag =
          std::string(name) + "/overlap=" + std::to_string(static_cast<int>(mode));
      if (!run_and_compare(sys, dense, build, opts, kIters, tag)) {
        return false;
      }
    }
    carbon::Skeleton::Options capped;
    capped.max_streams = 1U;
    if (!run_and_compare(sys, dense, build, capped, kIters, std::string(name) + "/max1")) {
      return false;
    }
    carbon::Skeleton::Options graph(carbon::OverlapMode::None, false, 0U, false,
                                    /*cuda_graph=*/true);
    if (!run_and_compare(sys, dense, build, graph, kIters, std::string(name) + "/graph")) {
      return false;
    }
  }
  return true;
}

// A masked reduction over kLower runs beside a writer of kUpper and must see none of it.
[[nodiscard]] auto
subtest_masked_reduction(carbon::System& sys,
                         carbon::Dense& dense,
                         Geo geo) -> bool {
  constexpr float kValue = 3.0F;
  Field fld = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "fld");
  Field one = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "one");
  size_t lower_cells = 0;
  for (size_t g = 0; g < dense.gpu_count(); ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t lim_z = span.logic_extent().z;
    for (int64_t z = 0; z < haloed.z; ++z) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          fld.cpu_buffers()[g].data()[off] = kValue;
          one.cpu_buffers()[g].data()[off] = 1.0F;
          const bool owned = z >= halo_z && z < halo_z + lim_z;
          const carbon::int64_3d gl{x, y, span.z_offset() + z};
          lower_cells += owned && geo.label(gl) == kLower ? 1U : 0U;
        }
      }
    }
  }
  carbon::ReductionResult<float> sum(sys);
  carbon::Stream s = sys.stream();
  try {
    carbon::Transfer<float>(sys, fld.cpu_buffers(), fld.gpu_buffers())(s);
    carbon::Transfer<float>(sys, one.cpu_buffers(), one.gpu_buffers())(s);
    s.wait();

    for (const bool overlap : {false, true}) {
      carbon::Skeleton sk(sys, s);
      sk.add(build_affine(dense, one, fld, overlap ? kInteriorMask : kUpperMask, 100.0F, 0.0F, geo,
                          "writer"));
      sk.add(build_masked_sum(dense, fld, kLowerMask, geo, sum));
      sk.compile();
      if (sk.edges() != (overlap ? Edges{{1U}, {}} : Edges{{}, {}})) {
        return fail("masked reduction: wrong edges");
      }
      if (overlap) {
        continue;
      }
      if (sk.num_streams() != 2U) {
        return fail("masked reduction: expected the reduction beside the writer");
      }
      sk.run();
      sum.sync(s);
      s.wait();
    }
  } catch (const std::exception& e) {
    return fail(std::string("masked reduction threw: ") + e.what());
  }
  const float expected = kValue * static_cast<float>(lower_cells);
  if (sum.host_value() != expected) {
    return fail("masked reduction: got " + std::to_string(sum.host_value()) + ", want " +
                std::to_string(expected));
  }
  return true;
}

}  // namespace

auto
main() -> int {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_skeleton_region_mask: no CUDA devices visible");
    return EXIT_FAILURE;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  const carbon::int64_3d global{8, 5, 4 * static_cast<int64_t>(n_gpus)};
  carbon::Dense dense(sys, global, std::move(stencil));
  const Geo geo{global.x, global.z};

  const bool ok = subtest_structure(sys, dense, geo) && subtest_stencil(sys, dense, geo) &&
                  subtest_loader(sys, dense) && subtest_empty_and_cap(sys, dense, geo) &&
                  subtest_ignore_and_determinism(sys, dense, geo) &&
                  subtest_numerics(sys, dense, geo) && subtest_masked_reduction(sys, dense, geo);
  if (!ok) {
    return EXIT_FAILURE;
  }
  CARBON_LOG_INFO("test_skeleton_region_mask: OK on {} GPU(s)", n_gpus);
  return EXIT_SUCCESS;
}
