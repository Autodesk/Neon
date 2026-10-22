// Unified salt-and-pepper denoising filter across every carbon grid + every
// transport. Companion to @c jacobi_unified.cu and @c cg_poisson_unified.cu :
// one source file, one @c Skeleton -driven pipeline, and (most importantly)
// one per-cell device lambda that compiles and runs on:
//
//   - Grids:     carbon::Dense   (dense Cartesian box)
//                carbon::DenseDisg (disaggregated dense slabs, region-partitioned)
//                carbon::Block<DefaultStaticBlock> (sparse 8^3 brick grid)
//   - Backends:  CUDA  (default, nvcc)
//                SYCL  (AdaptiveCpp -DCARBON_BACKEND=Sycl)
//   - Transports: single-GPU, multi-GPU same node, cross-rank NCCL/RCCL/oneCCL
//                 (the latter is enabled by -DCARBON_ENABLE_MPI=ON + mpiexec)
//
// What this example does
// ----------------------
//   1. Loads a grayscale image from a PGM (P5, binary 8-bit) file on every
//      rank. With no CLI arguments the example uses the bundled NASA astronaut
//      photo at @c examples/data/astronaut.pgm (the modern public-domain
//      replacement for the retired "Lenna" test image); pass @c --input <path>
//      to denoise any other PGM you have lying around.
//   2. Corrupts a copy with salt-and-pepper noise: @c --pepper-prob fraction
//      of pixels become 0.0, @c --salt-prob fraction become 1.0. The noise
//      pattern is deterministic via a simple LCG so the example reproduces
//      bit-for-bit run to run for a given @c --seed .
//   3. Uploads both the clean and noisy images to per-GPU @c Buffer<float> s
//      and seeds the per-cell @c clean / @c noisy fields by reading from those
//      buffers via @c Span::global_location . The per-cell init kernel is the
//      same source for every grid family because every grid's @c Span now
//      reports @c global_location(idx) -> int64_3d ( @c Dense ,
//      @c DenseDisg , and @c Block after the @c Span::origins() extension that
//      ships with this example).
//   4. Applies a 3×3 median filter in the x/z plane (8 neighbours + self,
//      all at @c y=0 ). Boundary pixels see fewer than 9 valid neighbours;
//      the filter takes the median of whatever it has — no special-casing
//      needed because @c Partition::neighbour returns @c valid=false at the
//      global edges and any missing brick face, and we just skip those entries
//      when collecting the candidate set on device.
//   5. Computes PSNR (in dB) of the noisy image vs the clean image, and of
//      the denoised image vs the clean image, via two @c Kernel::reduce
//      kernels (sum-of-squared differences). The reductions go through
//      stage-3 @c ncclAllReduce automatically when @c sys.distributed() is
//      set, so PSNR matches across single / multi-GPU / cross-rank runs.
//   6. Optionally writes the noisy and filtered images back out as PGM via
//      @c --noisy-output / @c --output — single-rank only; multi-rank skips
//      the dump because it would need an MPI gather first.
//
// Why y == 1
// ----------
// The dataset is naturally 2D, but the carbon grid is 3D so we collapse one
// axis. Pinning @c y to 1 (rather than @c z ) means @c halo_z is the only
// halo carbon adds, and a multi-GPU run would split the image along its
// height ( @c z ) — ideal for a denoiser whose work is local to a few rows.
// Neighbour reads in the y direction would land at @c by=1 (out of the
// haloed buffer), so @c Partition::neighbour returns @c valid=false and the
// filter transparently degrades to "this neighbour does not exist", same as
// for the global x/z edges. The stencil below only carries x/z offsets so
// the no-op y branch is never queried.
//
// Block predicate
// ---------------
// Block grids retain a circle inscribed in the image's xz-plane (radius
// @c 0.45 * min(width, height) ) so the demo shows the filter working on a
// real sparse domain. With the bundled 512×512 astronaut.pgm this leaves a
// faceless disc centred on the image — every pixel outside the disc is
// absent from the field and contributes nothing to the PSNR. To denoise the
// full image with Block, override the predicate with @c --full-block-domain .
// In every case Block requires the image dimensions to be a multiple of
// the brick size ( @c DefaultStaticBlock::MemX = MemZ = 8); a mismatched
// PGM aborts with a clear error message.
//
// CLI
// ---
//   --grid {dense|disg|block}    Grid family (default: dense)
//   --input         <path>       PGM (P5) file to denoise. Default: bundled.
//   --output        <path>       Write filtered image as PGM (single-rank only).
//   --noisy-output  <path>       Write the noisy input as PGM (single-rank only).
//   --salt-prob     <p>          Fraction of pixels forced to white. Default 0.05.
//   --pepper-prob   <p>          Fraction of pixels forced to black. Default 0.05.
//   --seed          <u32>        LCG seed for the noise pattern. Default 0xC0FFEE.
//   --overlap {none|standard|extended|two-way-extended}
//                                Skeleton::Options::overlap_mode . Default: none.
//   --full-block-domain          For --grid block: skip the circle predicate
//                                and retain every pixel (image dims must still
//                                be a multiple of 8).
//   --dot                        Dump the inferred filter-pipeline DAG as
//                                GraphViz DOT on stdout.
//   --help                       Print this help and exit.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/dense_disg.h>
#include <carbon/detail/backend/select.h>
#include <carbon/devices.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/distributed.h>
#endif

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

#ifndef CARBON_EXAMPLE_DATA_DIR
#define CARBON_EXAMPLE_DATA_DIR "."
#endif

using SB = carbon::detail::block::DefaultStaticBlock;

constexpr std::uint32_t kDefaultSeed = 0xC0FFEEU;
constexpr float kDefaultPepperProb = 0.05F;
constexpr float kDefaultSaltProb = 0.05F;
constexpr int kAsciiPreviewMaxWidth = 96;

struct CliOptions {
  std::string grid_kind = "dense";
  std::string input_path;
  std::string output_path;
  std::string noisy_output_path;
  float salt_prob = kDefaultSaltProb;
  float pepper_prob = kDefaultPepperProb;
  std::uint32_t seed = kDefaultSeed;
  std::string overlap_raw;
  bool full_block_domain = false;
  bool dump_dot = false;
};

class Lcg {
public:
  explicit Lcg(std::uint32_t seed) noexcept
      : state_(seed) {}

  [[nodiscard]] auto
  next_u32() noexcept -> std::uint32_t {
    state_ = state_ * 1664525U + 1013904223U;
    return state_;
  }

  [[nodiscard]] auto
  next_unit() noexcept -> float {
    constexpr float kInv32 = 1.0F / 4294967296.0F;
    return static_cast<float>(next_u32()) * kInv32;
  }

private:
  std::uint32_t state_;
};

void
print_help() {
  std::cout
      << "carbon_example_salt_pepper_filter -- unified 3x3 median filter across grids/transports\n"
         "Usage: carbon_example_salt_pepper_filter [options]\n"
         "\n"
         "Options:\n"
         "  --grid {dense|disg|block}    Grid family (default: dense)\n"
         "  --input         <path>       PGM (P5) file to denoise. Default: bundled.\n"
         "  --output        <path>       Write filtered image as PGM (single-rank only).\n"
         "  --noisy-output  <path>       Write the noisy input as PGM (single-rank only).\n"
         "  --salt-prob     <p>          Fraction of pixels forced to white. Default 0.05.\n"
         "  --pepper-prob   <p>          Fraction of pixels forced to black. Default 0.05.\n"
         "  --seed          <u32>        LCG seed for the noise pattern. Default 0xC0FFEE.\n"
         "  --overlap {none|standard|extended|two-way-extended}\n"
         "                                Skeleton overlap mode (default: none).\n"
         "  --full-block-domain          Block: retain every pixel instead of the inscribed "
         "circle.\n"
         "  --dot                        Dump the filter DAG as GraphViz DOT.\n"
         "  --help                       Print this help and exit.\n";
}

[[nodiscard]] auto
parse_prob(std::string_view text,
           const char* flag) -> float {
  float v = 0.0F;
  const auto* first = text.data();
  const auto* last = text.data() + text.size();
  const auto res = std::from_chars(first, last, v);
  if (res.ec != std::errc{} || res.ptr != last) {
    throw std::invalid_argument(std::string{flag} + ": expected a number, got \"" +
                                std::string{text} + "\"");
  }
  if (!(v >= 0.0F && v <= 1.0F)) {
    throw std::invalid_argument(std::string{flag} + ": expected a probability in [0, 1], got " +
                                std::string{text});
  }
  return v;
}

[[nodiscard]] auto
parse_u32(std::string_view text,
          const char* flag) -> std::uint32_t {
  std::uint32_t v = 0U;
  const auto* first = text.data();
  const auto* last = text.data() + text.size();
  int base = 10;
  if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
    first += 2;
    base = 16;
  }
  const auto res = std::from_chars(first, last, v, base);
  if (res.ec != std::errc{} || res.ptr != last) {
    throw std::invalid_argument(std::string{flag} +
                                ": expected an unsigned 32-bit integer, got \"" +
                                std::string{text} + "\"");
  }
  return v;
}

[[nodiscard]] auto
parse_cli(int argc,
          char** argv) -> CliOptions {
  CliOptions opts;
  opts.input_path = std::string{CARBON_EXAMPLE_DATA_DIR} + "/astronaut.pgm";

  auto need_value = [&](int i, const char* flag) -> std::string_view {
    if (i + 1 >= argc) {
      throw std::invalid_argument(std::string{flag} + ": missing value");
    }
    return std::string_view{argv[i + 1]};
  };
  for (int i = 1; i < argc; ++i) {
    const std::string_view a{argv[i]};
    if (a == "--help" || a == "-h") {
      print_help();
      std::exit(EXIT_SUCCESS);
    } else if (a == "--grid") {
      opts.grid_kind = std::string{need_value(i, "--grid")};
      ++i;
    } else if (a == "--input") {
      opts.input_path = std::string{need_value(i, "--input")};
      ++i;
    } else if (a == "--output") {
      opts.output_path = std::string{need_value(i, "--output")};
      ++i;
    } else if (a == "--noisy-output") {
      opts.noisy_output_path = std::string{need_value(i, "--noisy-output")};
      ++i;
    } else if (a == "--salt-prob") {
      opts.salt_prob = parse_prob(need_value(i, "--salt-prob"), "--salt-prob");
      ++i;
    } else if (a == "--pepper-prob") {
      opts.pepper_prob = parse_prob(need_value(i, "--pepper-prob"), "--pepper-prob");
      ++i;
    } else if (a == "--seed") {
      opts.seed = parse_u32(need_value(i, "--seed"), "--seed");
      ++i;
    } else if (a == "--overlap") {
      opts.overlap_raw = std::string{need_value(i, "--overlap")};
      ++i;
    } else if (a == "--full-block-domain") {
      opts.full_block_domain = true;
    } else if (a == "--dot") {
      opts.dump_dot = true;
    } else {
      throw std::invalid_argument("unrecognised argument: " + std::string{a});
    }
  }
  if (opts.salt_prob + opts.pepper_prob > 1.0F) {
    throw std::invalid_argument("--salt-prob + --pepper-prob exceeds 1.0");
  }
  return opts;
}

[[nodiscard]] auto
parse_overlap_mode(std::string_view s) -> std::optional<carbon::OverlapMode> {
  if (s == "none" || s.empty()) {
    return carbon::OverlapMode::None;
  }
  if (s == "standard") {
    return carbon::OverlapMode::Standard;
  }
  if (s == "extended") {
    return carbon::OverlapMode::Extended;
  }
  if (s == "two-way-extended" || s == "two_way_extended" || s == "twoway") {
    return carbon::OverlapMode::TwoWayExtended;
  }
  return std::nullopt;
}

[[nodiscard]] auto
overlap_mode_name(carbon::OverlapMode m) -> const char* {
  switch (m) {
    case carbon::OverlapMode::None:
      return "none";
    case carbon::OverlapMode::Standard:
      return "standard";
    case carbon::OverlapMode::Extended:
      return "extended";
    case carbon::OverlapMode::TwoWayExtended:
      return "two-way-extended";
  }
  return "?";
}

// ---- PGM I/O ----------------------------------------------------------------

struct GrayImage {
  int64_t width = 0;
  int64_t height = 0;
  std::vector<float> pixels;
};

void
skip_pgm_whitespace(std::ifstream& in) {
  while (in.good()) {
    const int ch = in.peek();
    if (ch == EOF) {
      return;
    }
    if (ch == '#') {
      std::string discarded;
      std::getline(in, discarded);
    } else if (std::isspace(static_cast<unsigned char>(ch))) {
      in.get();
    } else {
      return;
    }
  }
}

[[nodiscard]] auto
load_pgm(const std::string& path) -> GrayImage {
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::file_status st = fs::status(path, ec);
  if (ec) {
    throw std::runtime_error("cannot stat \"" + path + "\": " + ec.message());
  }
  if (!fs::is_regular_file(st)) {
    throw std::runtime_error("\"" + path + "\" is not a regular file");
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open \"" + path + "\" for reading");
  }

  std::string magic;
  in >> magic;
  if (magic != "P5") {
    throw std::runtime_error("\"" + path + "\" is not a binary PGM (P5); got magic = " + magic);
  }
  skip_pgm_whitespace(in);
  int64_t width = 0;
  int64_t height = 0;
  int maxval = 0;
  in >> width;
  skip_pgm_whitespace(in);
  in >> height;
  skip_pgm_whitespace(in);
  in >> maxval;
  if (!in || width <= 0 || height <= 0 || maxval <= 0 || maxval > 65535) {
    throw std::runtime_error("\"" + path + "\": malformed PGM header (W=" + std::to_string(width) +
                             ", H=" + std::to_string(height) +
                             ", maxval=" + std::to_string(maxval) + ")");
  }
  in.get();

  GrayImage img;
  img.width = width;
  img.height = height;
  img.pixels.resize(static_cast<size_t>(width * height));

  const float inv = 1.0F / static_cast<float>(maxval);
  if (maxval < 256) {
    std::vector<unsigned char> raw(static_cast<size_t>(width * height));
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (!in) {
      throw std::runtime_error("\"" + path + "\": short read on 8-bit pixel data");
    }
    for (size_t i = 0; i < raw.size(); ++i) {
      img.pixels[i] = static_cast<float>(raw[i]) * inv;
    }
  } else {
    const size_t n = static_cast<size_t>(width * height);
    std::vector<unsigned char> raw(n * 2);
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (!in) {
      throw std::runtime_error("\"" + path + "\": short read on 16-bit pixel data");
    }
    for (size_t i = 0; i < n; ++i) {
      const auto hi = static_cast<unsigned int>(raw[2 * i]);
      const auto lo = static_cast<unsigned int>(raw[2 * i + 1]);
      img.pixels[i] = static_cast<float>((hi << 8U) | lo) * inv;
    }
  }
  return img;
}

void
save_pgm(const std::string& path,
         int64_t width,
         int64_t height,
         const std::vector<float>& pixels) {
  if (static_cast<size_t>(width * height) != pixels.size()) {
    throw std::runtime_error("save_pgm: pixel count does not match width * height");
  }
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("cannot open \"" + path + "\" for writing");
  }
  out << "P5\n" << width << " " << height << "\n255\n";
  std::vector<unsigned char> raw(pixels.size());
  for (size_t i = 0; i < pixels.size(); ++i) {
    const float v = std::clamp(pixels[i], 0.0F, 1.0F);
    raw[i] = static_cast<unsigned char>(std::lround(v * 255.0F));
  }
  out.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
  if (!out) {
    throw std::runtime_error("short write to \"" + path + "\"");
  }
}

// MSE -> PSNR (peak signal = 1.0 since pixels are normalised to [0, 1]).
[[nodiscard]] auto
psnr_db(double mse) -> double {
  if (mse <= 0.0) {
    return 99.0;
  }
  return 10.0 * std::log10(1.0 / mse);
}

// ---- Buffer helpers ---------------------------------------------------------

// Allocate one host @c Allocation<float> per local GPU, copy the same image
// data into each, and return the resulting @c Buffer<float> . The caller then
// runs a @c Transfer<float> to upload to a matching device buffer.
[[nodiscard]] auto
make_host_image_buffer(const carbon::System& sys,
                       const std::vector<float>& image) -> carbon::Buffer<float> {
  const size_t n_gpus = sys.gpu_group().size();
  std::vector<carbon::detail::Allocation<float>> host_allocs;
  host_allocs.reserve(n_gpus);
  for (size_t g = 0; g < n_gpus; ++g) {
    auto alloc = carbon::Cpu::buffer<float>(image.size());
    std::copy(image.begin(), image.end(), alloc.data());
    host_allocs.emplace_back(std::move(alloc));
  }
  return carbon::Buffer<float>::from_vector(std::move(host_allocs));
}

// ---- Kernel builders (grid-agnostic via Kernel::compute / Kernel::reduce) ----

// Per-cell init that reads from a device-side image buffer (one allocation per
// GPU, captured from @p img_buf at @c relative_gpu_id at @c make_kernel time)
// and writes the matching pixel into @p field . The mapping
// @c global_location -> @c (gz*width + gx) works for every grid family
// because every @c Span exposes @c global_location(idx) -> int64_3d .
template <typename Grid>
[[nodiscard]] auto
build_image_init(const Grid& grid,
                 typename Grid::template Field<float,
                                               1>& field,
                 const carbon::Buffer<float>& img_buf,
                 int64_t width,
                 std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::compute(
      grid,
      [&field, &img_buf, width](carbon::KernelLoader& loader) {
        const size_t g = loader.relative_gpu_id();
        auto out = loader.write(field);
        const float* image_ptr = img_buf[g].data();
        return [out, image_ptr, width] CARBON_HOST_DEVICE(Span s, Idx idx) -> void {
          const carbon::int64_3d gc = s.global_location(idx);
          const int64_t pixel_idx = gc.z * width + gc.x;
          out.write(s, idx, 0, image_ptr[pixel_idx]);
        };
      },
      std::move(name));
}

// 3×3 median filter in the x/z plane. Insertion-sort over the (up to 9)
// valid neighbours and write the middle element. Boundary pixels see fewer
// than 9 candidates; the filter takes the middle of whatever it has -- the
// textbook median for an odd @c N and a safe choice for even @c N .
//
// @c loader.read_stencil(in) annotation triggers an auto-inserted halo before
// this kernel under multi-GPU / multi-rank, exactly the same pattern as
// @c jacobi_unified.cu 's stencil step.
template <typename Grid>
[[nodiscard]] auto
build_median_filter(const Grid& grid,
                    typename Grid::template Field<float,
                                                  1>& in,
                    typename Grid::template Field<float,
                                                  1>& out,
                    std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::compute(
      grid,
      [&in, &out](carbon::KernelLoader& loader) {
        const auto in_p = loader.read_stencil(in);
        auto out_p = loader.write(out);
        return [in_p, out_p] CARBON_HOST_DEVICE(Span s, Idx idx) -> void {
          float buf[9];
          int n = 0;
          buf[n++] = in_p.read(s, idx, 0);
          const carbon::int8_3d offsets[8] = {{1, 0, 0}, {-1, 0, 0}, {0, 0, 1},  {0, 0, -1},
                                              {1, 0, 1}, {1, 0, -1}, {-1, 0, 1}, {-1, 0, -1}};
          for (int k = 0; k < 8; ++k) {
            const auto nv = in_p.neighbour(s, idx, offsets[k], 0);
            if (nv.valid) {
              buf[n++] = nv.value;
            }
          }
          for (int i = 1; i < n; ++i) {
            const float key = buf[i];
            int j = i - 1;
            while (j >= 0 && buf[j] > key) {
              buf[j + 1] = buf[j];
              --j;
            }
            buf[j + 1] = key;
          }
          out_p.write(s, idx, 0, buf[n / 2]);
        };
      },
      std::move(name));
}

// sum_i (a[i] - b[i])^2 . Together with an active-cell-count reduction this
// produces MSE = sum / cells and PSNR = 10 * log10(1 / MSE) . Routed through
// @c Kernel::reduce which gives us ncclAllReduce for free when distributed.
template <typename Grid>
[[nodiscard]] auto
build_reduce_sq_diff(const Grid& grid,
                     typename Grid::template Field<float,
                                                   1>& a,
                     typename Grid::template Field<float,
                                                   1>& b,
                     carbon::ReductionResult<double>& dst,
                     std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::reduce<double>(
      grid,
      [&a, &b](carbon::KernelLoader& loader) {
        const auto a_p = loader.read(a);
        const auto b_p = loader.read(b);
        return [a_p, b_p] CARBON_HOST_DEVICE(Span s, Idx idx) -> double {
          const double d =
              static_cast<double>(a_p.read(s, idx, 0)) - static_cast<double>(b_p.read(s, idx, 0));
          return d * d;
        };
      },
      dst, carbon::reduce::Sum<double>{}, std::move(name), /*shared_mem=*/0U);
}

// Count active cells: every active cell contributes 1.0. For Dense /
// DenseDisg this equals @c nx*ny*nz ; for Block it equals the active-voxel
// count of the predicate (the bitmask filter inside the launch helper makes
// inactive voxels skip the lambda body entirely).
template <typename Grid>
[[nodiscard]] auto
build_reduce_cell_count(const Grid& grid,
                        typename Grid::template Field<float,
                                                      1>& probe,
                        carbon::ReductionResult<double>& dst,
                        std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::reduce<double>(
      grid,
      [&probe](carbon::KernelLoader& loader) {
        const auto p = loader.read(probe);
        return [p] CARBON_HOST_DEVICE(Span s, Idx idx) -> double {
          (void)p.read(s, idx, 0);
          return 1.0;
        };
      },
      dst, carbon::reduce::Sum<double>{}, std::move(name), /*shared_mem=*/0U);
}

// ---- Solver / driver --------------------------------------------------------

template <typename Grid>
auto
run_filter(Grid& grid,
           carbon::System& sys,
           const CliOptions& opts,
           const GrayImage& clean_img,
           const std::vector<float>& noisy_pixels,
           int64_t width,
           int64_t height,
           int world_rank,
           int world_size,
           std::string_view grid_tag) -> int {
  const auto overlap_opt = parse_overlap_mode(opts.overlap_raw);
  if (!overlap_opt) {
    if (world_rank == 0) {
      CARBON_LOG_ERROR(
          "salt_pepper_filter: unknown --overlap '{}' (expected "
          "none|standard|extended|two-way-extended)",
          opts.overlap_raw);
    }
    return EXIT_FAILURE;
  }
  const carbon::OverlapMode overlap_mode = *overlap_opt;

  using Field = typename Grid::template Field<float, 1>;
  Field clean = grid.template newField<float, 1>(carbon::MemorySide::HostDevice, "clean");
  Field noisy = grid.template newField<float, 1>(carbon::MemorySide::Device, "noisy");
  Field out = grid.template newField<float, 1>(carbon::MemorySide::HostDevice, "out");

  // Build per-GPU device buffers for the clean and noisy source images and
  // upload from per-GPU host mirrors. Each GPU on this rank gets its own copy
  // of the full image; the per-cell init kernel then picks the right pixel
  // via @c Span::global_location . Memory-wise this is a few MB per GPU --
  // trivial compared to the field buffers.
  const std::vector<size_t> counts(sys.gpu_group().size(), static_cast<size_t>(width * height));
  auto host_clean_buf = make_host_image_buffer(sys, clean_img.pixels);
  auto host_noisy_buf = make_host_image_buffer(sys, noisy_pixels);
  auto dev_clean_buf = sys.buffer<float>(counts);
  auto dev_noisy_buf = sys.buffer<float>(counts);

  carbon::Stream stream = sys.stream();
  carbon::Transfer<float>(sys, host_clean_buf, dev_clean_buf)(stream);
  carbon::Transfer<float>(sys, host_noisy_buf, dev_noisy_buf)(stream);
  stream.wait();

  carbon::ReductionResult<double> ss_noise(sys);
  carbon::ReductionResult<double> ss_filtered(sys);
  carbon::ReductionResult<double> n_cells(sys);

  // Build the full pipeline as one Skeleton:
  //   init_clean, init_noisy         (writes, parallel-safe)
  //   median_filter(noisy -> out)    (read_stencil(noisy) -> auto-halo)
  //   reduce ss_noise = sum (noisy - clean)^2
  //   reduce ss_filtered = sum (out - clean)^2
  //   reduce n_cells = sum 1
  carbon::Skeleton pipe(sys, carbon::Skeleton::Options{overlap_mode});
  pipe.add(build_image_init(grid, clean, dev_clean_buf, width, "init_clean"));
  pipe.add(build_image_init(grid, noisy, dev_noisy_buf, width, "init_noisy"));
  pipe.add(build_median_filter(grid, noisy, out, "median"));
  pipe.add(build_reduce_sq_diff(grid, noisy, clean, ss_noise, "sq_diff_noisy"));
  pipe.add(build_reduce_sq_diff(grid, out, clean, ss_filtered, "sq_diff_filtered"));
  pipe.add(build_reduce_cell_count(grid, clean, n_cells, "n_cells"));
  pipe.compile();

  if (world_rank == 0) {
    CARBON_LOG_INFO(
        "salt_pepper_filter[{}]: compiled - user_kernels={} effective_kernels={} overlap_mode={}",
        grid_tag, pipe.size(), pipe.effective_kernels().size(), overlap_mode_name(overlap_mode));
    if (opts.dump_dot) {
      CARBON_LOG_INFO("salt_pepper_filter[{}]: --- filter DAG (GraphViz DOT) ---\n{}", grid_tag,
                      pipe.export_dot("salt_pepper_filter"));
    }
  }

  pipe.run();
  pipe.wait();

  ss_noise.sync(stream);
  ss_filtered.sync(stream);
  n_cells.sync(stream);
  stream.wait();

  const double total_cells = n_cells.host_value();
  const double mse_noisy = (total_cells > 0.0) ? ss_noise.host_value() / total_cells : 0.0;
  const double mse_filtered = (total_cells > 0.0) ? ss_filtered.host_value() / total_cells : 0.0;

  if (world_rank == 0) {
    CARBON_LOG_INFO("salt_pepper_filter[{}]: active cells = {:.0f}", grid_tag, total_cells);
    CARBON_LOG_INFO("salt_pepper_filter[{}]: PSNR noisy    vs clean = {:6.2f} dB (MSE = {:.4e})",
                    grid_tag, psnr_db(mse_noisy), mse_noisy);
    CARBON_LOG_INFO("salt_pepper_filter[{}]: PSNR filtered vs clean = {:6.2f} dB (MSE = {:.4e})",
                    grid_tag, psnr_db(mse_filtered), mse_filtered);
  }
  if (mse_filtered >= mse_noisy) {
    if (world_rank == 0) {
      CARBON_LOG_ERROR("salt_pepper_filter[{}]: filter did not reduce MSE ({:.4e} >= {:.4e})",
                       grid_tag, mse_filtered, mse_noisy);
    }
    return EXIT_FAILURE;
  }

  // PGM dump + ASCII preview are Dense-only and single-rank single-GPU.
  // - Multi-rank would need MPI_Gather of the per-rank slabs back to rank 0.
  // - DenseDisg uses a slab layout that doesn't map row-by-row to a flat
  //   (h x w) image without per-region stitching.
  // - Block stores retained voxels in brick order; splatting them back into
  //   a dense (h x w) image is mechanical but adds clutter for a demo.
  // Gated via @c if constexpr so the Dense-specific accessors below
  // ( @c haloed_extent / @c xy_slab_elements / @c halo_z ) are not
  // instantiated for grid families that don't expose them.
  const bool single_proc = (world_size == 1) && (sys.gpu_group().size() == 1U);
  if constexpr (std::is_same_v<Grid, carbon::Dense>) {
    if (single_proc && !opts.noisy_output_path.empty()) {
      save_pgm(opts.noisy_output_path, width, height, noisy_pixels);
      CARBON_LOG_INFO("salt_pepper_filter[{}]: wrote noisy image to {}", grid_tag,
                      opts.noisy_output_path);
    }
    if (single_proc && !opts.output_path.empty()) {
      carbon::Transfer<float>(sys, out.gpu_buffers(), out.cpu_buffers())(stream);
      stream.wait();
      std::vector<float> img(static_cast<size_t>(width * height));
      const auto& span = grid.spans()[0];
      const int64_t pitch = span.xy_slab_elements();
      const int64_t halo_z = span.halo_z();
      const float* h = out.cpu_buffers()[0].data();
      for (int64_t z = 0; z < height; ++z) {
        const int64_t row_off = (halo_z + z) * pitch;
        for (int64_t x = 0; x < width; ++x) {
          img[static_cast<size_t>(z * width + x)] = h[row_off + x];
        }
      }
      save_pgm(opts.output_path, width, height, img);
      CARBON_LOG_INFO("salt_pepper_filter[{}]: wrote filtered image to {}", grid_tag,
                      opts.output_path);
    }
    if (world_rank == 0 && single_proc && width <= kAsciiPreviewMaxWidth) {
      // Centre-row ASCII preview, only when the image fits in the terminal.
      const auto& span = grid.spans()[0];
      const int64_t pitch = span.xy_slab_elements();
      const int64_t halo_z = span.halo_z();
      const int64_t cz = height / 2;
      const int64_t row_off = (halo_z + cz) * pitch;

      auto render = [&](const float* data, const char* tag) {
        std::string line;
        line.reserve(static_cast<size_t>(width));
        for (int64_t x = 0; x < width; ++x) {
          const float v = data[row_off + x];
          const int level = static_cast<int>(std::min(4.0F, std::max(0.0F, std::floor(v * 5.0F))));
          constexpr char ramp[] = " .:*#";
          line.push_back(ramp[level]);
        }
        CARBON_LOG_INFO("salt_pepper_filter[{}]: {} (z = {}): |{}|", grid_tag, tag, cz, line);
      };
      carbon::Transfer<float>(sys, noisy.gpu_buffers(), out.cpu_buffers())(stream);
      stream.wait();
      render(out.cpu_buffers()[0].data(), "noisy   ");
      carbon::Transfer<float>(sys, out.gpu_buffers(), out.cpu_buffers())(stream);
      stream.wait();
      render(out.cpu_buffers()[0].data(), "filtered");
      render(clean.cpu_buffers()[0].data(), "clean   ");
    }
  } else {
    if (world_rank == 0 && (!opts.output_path.empty() || !opts.noisy_output_path.empty()) &&
        single_proc) {
      CARBON_LOG_INFO(
          "salt_pepper_filter[{}]: --output / --noisy-output skipped (PGM dump is implemented only "
          "for --grid dense)",
          grid_tag);
    }
  }

  (void)single_proc;
  (void)clean_img;
  return EXIT_SUCCESS;
}

}  // namespace

auto
main(int argc,
     char** argv) -> int {
  CliOptions opts;
  try {
    opts = parse_cli(argc, argv);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("salt_pepper_filter: {}", e.what());
    print_help();
    return EXIT_FAILURE;
  }

  GrayImage img;
  try {
    img = load_pgm(opts.input_path);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("salt_pepper_filter: {}", e.what());
    return EXIT_FAILURE;
  }

  carbon::init();

#ifdef CARBON_HAVE_MPI_NCCL
  auto distributed = std::make_shared<const carbon::Distributed>();
  const int world_size = distributed->world_size();
  const int world_rank = distributed->world_rank();
  carbon::System sys =
      (world_size > 1) ? carbon::System(distributed) : carbon::System(carbon::get_all_devices());
  const bool is_distributed = (world_size > 1);
#else
  carbon::System sys(carbon::get_all_devices());
  const int world_size = 1;
  const int world_rank = 0;
  const bool is_distributed = false;
#endif

  if (sys.gpu_group().size() == 0U) {
    CARBON_LOG_ERROR("salt_pepper_filter: no devices visible");
    return EXIT_FAILURE;
  }

  // Apply salt + pepper noise on every rank with the same seed so all ranks
  // see identical noisy input. The full image is replicated on each GPU.
  size_t pepper_count = 0;
  size_t salt_count = 0;
  Lcg rng(opts.seed);
  const float pepper_thr = opts.pepper_prob;
  const float salt_thr = opts.pepper_prob + opts.salt_prob;
  std::vector<float> noisy_pixels(img.pixels.size());
  for (size_t i = 0; i < img.pixels.size(); ++i) {
    const float r = rng.next_unit();
    if (r < pepper_thr) {
      ++pepper_count;
      noisy_pixels[i] = 0.0F;
    } else if (r < salt_thr) {
      ++salt_count;
      noisy_pixels[i] = 1.0F;
    } else {
      noisy_pixels[i] = img.pixels[i];
    }
  }

  if (world_rank == 0) {
    CARBON_LOG_INFO("salt_pepper_filter: loaded {} ({} x {} pixels)", opts.input_path, img.width,
                    img.height);
    CARBON_LOG_INFO(
        "salt_pepper_filter: grid={} backend={} gpus_local={} world_size={} distributed={} "
        "overlap={} seed=0x{:08x} pepper={} salt={}",
        opts.grid_kind,
#if CARBON_BACKEND_IS_SYCL
        "SYCL",
#else
        "CUDA",
#endif
        sys.gpu_group().size(), world_size, is_distributed,
        opts.overlap_raw.empty() ? "none" : opts.overlap_raw.c_str(), opts.seed, pepper_count,
        salt_count);
  }

  // Layout: image is (width, 1, height) in (x, y, z). Pinning y to 1 means
  // multi-GPU / multi-rank slabs split along z (rows of the image) -- which
  // is what we want for stencil-locality reasons on a denoiser.
  const carbon::int64_3d global{img.width, 1, img.height};

  // 8-neighbour stencil in the x/z plane (no y offsets -- the image is 2D
  // and any y neighbour is out-of-buffer). halo_z = max |dz| = 1.
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0},
          {-1, 0, 0},
          {0, 0, 1},
          {0, 0, -1},
          {1, 0, 1},
          {1, 0, -1},
          {-1, 0, 1},
          {-1, 0, -1},
      },
      /*remove_center=*/true);

  try {
    if (opts.grid_kind == "dense") {
      carbon::Dense dense(sys, global, stencil);
      return run_filter(dense, sys, opts, img, noisy_pixels, img.width, img.height, world_rank,
                        world_size, "dense");
    }
    if (opts.grid_kind == "disg") {
      carbon::DenseDisg disg(sys, global, /*halo_z=*/1);
      return run_filter(disg, sys, opts, img, noisy_pixels, img.width, img.height, world_rank,
                        world_size, "disg");
    }
    if (opts.grid_kind == "block") {
      // Block requires the image dims to be a multiple of the brick size so
      // every retained block lands on aligned voxel boundaries. The bundled
      // 512x512 PGM satisfies this; arbitrary user PGMs may not.
      if ((img.width % SB::MemX) != 0 || (img.height % SB::MemZ) != 0) {
        CARBON_LOG_ERROR(
            "salt_pepper_filter[block]: image dimensions ({} x {}) must be a multiple of "
            "the brick size (MemX={}, MemZ={})",
            img.width, img.height, SB::MemX, SB::MemZ);
        return EXIT_FAILURE;
      }
      const int64_t nmax = std::max(img.width, img.height);
      const double cx = (static_cast<double>(img.width) - 1.0) * 0.5;
      const double cz = (static_cast<double>(img.height) - 1.0) * 0.5;
      const double radius = static_cast<double>(nmax) * 0.45;
      const bool full = opts.full_block_domain;
      const auto predicate = [full, cx, cz, radius](const carbon::int64_3d& gc) -> bool {
        if (full) {
          return true;
        }
        const double dx = static_cast<double>(gc.x) - cx;
        const double dz = static_cast<double>(gc.z) - cz;
        return (dx * dx + dz * dz) <= (radius * radius);
      };
      using BlockGrid = carbon::Block<SB>;
      BlockGrid block(sys, global, predicate, "salt_pepper_filter_block");
      return run_filter(block, sys, opts, img, noisy_pixels, img.width, img.height, world_rank,
                        world_size, "block");
    }
    CARBON_LOG_ERROR("salt_pepper_filter: unknown --grid '{}' (expected dense|disg|block)",
                     opts.grid_kind);
    return EXIT_FAILURE;
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("salt_pepper_filter: failed: {}", e.what());
    return EXIT_FAILURE;
  }
}
