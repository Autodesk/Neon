// export_launch_order() mirrors Skeleton::run_eager_ operation order.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/neighbour_value.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>

namespace {

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
          const carbon::NeighbourValue<float> xp = in.neighbour(s, idx, {1, 0, 0}, 0);
          const carbon::NeighbourValue<float> xm = in.neighbour(s, idx, {-1, 0, 0}, 0);
          const carbon::NeighbourValue<float> yp = in.neighbour(s, idx, {0, 1, 0}, 0);
          const carbon::NeighbourValue<float> ym = in.neighbour(s, idx, {0, -1, 0}, 0);
          const carbon::NeighbourValue<float> zp = in.neighbour(s, idx, {0, 0, 1}, 0);
          const carbon::NeighbourValue<float> zm = in.neighbour(s, idx, {0, 0, -1}, 0);
          const float self = in.read(s, idx, 0);
          const bool all_valid =
              xp.valid && xm.valid && yp.valid && ym.valid && zp.valid && zm.valid;
          if (!all_valid) {
            out.write(s, idx, 0, self);
            return;
          }
          const float sum = xp.value + xm.value + yp.value + ym.value + zp.value + zm.value;
          out.write(s, idx, 0, sum * (1.0F / 6.0F));
        };
      },
      std::move(name),
      /*shared_mem=*/0U);
}

[[nodiscard]] auto
line_index_of(const std::string& text,
              std::string_view needle) -> std::size_t {
  const std::size_t pos = text.find(needle);
  if (pos == std::string::npos) {
    return static_cast<std::size_t>(-1);
  }
  return static_cast<std::size_t>(
      std::count(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(pos), '\n'));
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_skeleton_launch_order: no CUDA devices visible");
    return EXIT_FAILURE;
  }

  carbon::System sys(carbon::get_all_devices());
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  carbon::Dense dense(sys, {8, 6, 12}, stencil);
  auto u = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u");
  auto v = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v");

  // Single-stream pipeline: no anchor fork/join lines.
  {
    carbon::Skeleton sk_lin(sys, carbon::Skeleton::Options{carbon::OverlapMode::None});
    sk_lin.add(u.make_halo_kernel("halo_u"));
    sk_lin.add(build_jacobi(dense, u, v, "step"));
    sk_lin.compile();
    const std::string order = sk_lin.export_launch_order();
    if (order.find("anchor start") != std::string::npos ||
        order.find("anchor join") != std::string::npos) {
      CARBON_LOG_ERROR(
          "test_skeleton_launch_order: unexpected anchor ops on single-stream schedule");
      return EXIT_FAILURE;
    }
    const std::size_t halo_line = line_index_of(order, "launch  halo_u");
    const std::size_t step_line = line_index_of(order, "launch  step");
    if (halo_line == static_cast<std::size_t>(-1) || step_line == static_cast<std::size_t>(-1) ||
        halo_line >= step_line) {
      CARBON_LOG_ERROR("test_skeleton_launch_order: halo must launch before step\n{}", order);
      return EXIT_FAILURE;
    }
  }

  // Standard overlap: halo on a secondary stream, internal before boundary.
  // Anchor start is load-bearing here — halo's first stream-1 op has no later stream-0
  // data wait to dominate it — so the fork must remain.
  {
    carbon::Skeleton sk_occ(sys, carbon::Skeleton::Options{carbon::OverlapMode::Standard});
    sk_occ.add(build_jacobi(dense, u, v, "step_occ"));
    sk_occ.compile();
    if (sk_occ.num_streams() < 2U) {
      CARBON_LOG_ERROR("test_skeleton_launch_order: expected >=2 streams under Standard overlap");
      return EXIT_FAILURE;
    }
    const std::string order = sk_occ.export_launch_order();
    if (order.find("anchor start") == std::string::npos) {
      CARBON_LOG_ERROR(
          "test_skeleton_launch_order: Standard Jacobi needs anchor start (no dominating "
          "stream-0 wait before halo)\n{}",
          order);
      return EXIT_FAILURE;
    }
    const std::size_t halo_line = line_index_of(order, "[halo/");
    const std::size_t internal_line = line_index_of(order, "[compute/internal]");
    const std::size_t boundary_line = line_index_of(order, "[compute/boundary]");
    if (halo_line == static_cast<std::size_t>(-1) ||
        internal_line == static_cast<std::size_t>(-1) ||
        boundary_line == static_cast<std::size_t>(-1)) {
      CARBON_LOG_ERROR("test_skeleton_launch_order: incomplete overlap listing\n{}", order);
      return EXIT_FAILURE;
    }
    if (!(internal_line < halo_line && halo_line < boundary_line)) {
      CARBON_LOG_ERROR("test_skeleton_launch_order: bad overlap order\n{}", order);
      return EXIT_FAILURE;
    }
    const std::size_t boundary_on_halo_stream =
        line_index_of(order, "stream 1  launch  step_occ [compute/boundary]");
    if (boundary_on_halo_stream == static_cast<std::size_t>(-1)) {
      CARBON_LOG_ERROR("test_skeleton_launch_order: boundary must run on halo stream\n{}", order);
      return EXIT_FAILURE;
    }

    // Launch matrix: one row per launch-order line, one column per stream.
    const std::string matrix = sk_occ.export_launch_matrix_dot("occ_matrix");
    const auto op_count = static_cast<std::size_t>(std::count(order.begin(), order.end(), '\n'));
    const std::string last_row = "r" + std::to_string(op_count) + "_c1";
    const std::string past_last_row = "r" + std::to_string(op_count + 1U) + "_c0";
    if (matrix.find("digraph occ_matrix {") == std::string::npos ||
        matrix.find("<B>stream 1</B>") == std::string::npos ||
        matrix.find(last_row) == std::string::npos ||
        matrix.find(past_last_row) != std::string::npos ||
        matrix.find("<B>step_occ</B>") == std::string::npos ||
        matrix.find("compute &#183; boundary") == std::string::npos ||
        matrix.find("constraint=false") == std::string::npos) {
      CARBON_LOG_ERROR("test_skeleton_launch_order: bad launch matrix\n{}", matrix);
      return EXIT_FAILURE;
    }

    // Kernel list: the stencil kernel plus its auto-inserted halo, with the OCC split
    // launches listed on the stencil kernel's row.
    const std::string list = sk_occ.export_kernel_list_dot("occ_kernels");
    if (list.find("digraph occ_kernels {") == std::string::npos ||
        list.find("<B>step_occ</B>") == std::string::npos ||
        list.find("halo &#183; auto-inserted") == std::string::npos ||
        list.find("stencil") == std::string::npos ||
        list.find("stream 1 &#183; boundary") == std::string::npos ||
        list.find("stream 0 &#183; internal") == std::string::npos) {
      CARBON_LOG_ERROR("test_skeleton_launch_order: bad kernel list\n{}", list);
      return EXIT_FAILURE;
    }
  }

  // Extended: Map producer Boundary→record, then halo waits that event. The later stream-0
  // wait dominates anchor start, so prune must drop the fork.
  {
    auto fill = carbon::Kernel::compute(
        dense,
        [&u](carbon::KernelLoader& loader) {
          const auto out = loader.write(u);
          return [out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
            out.write(s, idx, 0, 1.0F);
          };
        },
        "fill_u", /*shared_mem=*/0U);
    carbon::Skeleton sk_ext(sys, carbon::Skeleton::Options{carbon::OverlapMode::Extended});
    sk_ext.add(std::move(fill));
    sk_ext.add(build_jacobi(dense, u, v, "step_ext"));
    sk_ext.compile();
    if (sk_ext.num_streams() < 2U) {
      CARBON_LOG_ERROR("test_skeleton_launch_order: expected >=2 streams under Extended");
      return EXIT_FAILURE;
    }
    const std::string order = sk_ext.export_launch_order();
    if (order.find("anchor start") != std::string::npos) {
      CARBON_LOG_ERROR(
          "test_skeleton_launch_order: Extended producer→halo wait should prune anchor start\n{}",
          order);
      return EXIT_FAILURE;
    }
    if (order.find("wait") == std::string::npos || order.find("[halo/") == std::string::npos) {
      CARBON_LOG_ERROR(
          "test_skeleton_launch_order: Extended schedule missing halo wait\n{}", order);
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
