#pragma once

#include <carbon/boundary_condition_distributed.h>
#include <carbon/cpu.h>
#include <carbon/dense_distributed.h>
#include <carbon/detail/buffer_group.h>
#include <carbon/detail/dense_distributed/boundary_store.h>
#include <carbon/detail/dense_distributed/boundary_view.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace carbon {

/**
 * Spatially varying boundary conditions for a @ref DenseDistributed grid, held separately from the
 * grid: the condition (kind and value, per component) of every cell of every exterior face.
 *
 * Unlike the per-face constants of @ref BoundaryConditions, the condition can change from cell to
 * cell (a conducting electrode patch on an otherwise insulating face, an inlet on a wall), can
 * differ per component (free-slip wall: Dirichlet normal velocity, Neumann tangential), and belongs
 * to the data rather than the grid, so different fields on one grid can have different conditions.
 *
 * @par Use
 * @code
 *   carbon::BoundaryData<1> bnd(grid, "phi_bc");
 *   bnd.fill([](carbon::Face f, carbon::int64_3d g, int comp) -> carbon::FaceBc {
 *     if (f == carbon::Face::ZHigh && g.x < 8) return carbon::dirichlet(1.0);   // electrode
 *     return carbon::neumann(0.0);                                              // insulating
 *   });
 *   bnd.upload();
 *   // in a kernel loading lambda (no token, no loader change):
 *   const auto bv = bnd.view(loader.relative_gpu_id());
 *   // in the device lambda:
 *   auto n = pin.neighbour_bc(span, bv, idx, carbon::int8_3d{0, 0, 1}, comp);
 * @endcode
 *
 * @par What is stored
 * Each GPU stores only the faces it touches, over its owned tangential range widened by one stencil
 * radius (clipped to the domain), so a diagonal read across a face finds the cell under its target.
 * Faces on a periodic axis are never stored. The stored cells are visited by @ref fill, which must
 * be a deterministic function of global position: cells in the margin are filled independently on
 * each neighbouring GPU and must agree.
 *
 * @par Lifetime and updates
 * Constant while a Skeleton using it runs; the handle must outlive that Skeleton. It needs no
 * access token: Read/Read pairs never conflict in the Skeleton. Do not call @ref fill or
 * @ref upload while such a Skeleton is running.
 *
 * @tparam Components Number of components the conditions are given for. A field read through a view
 *         needs @c component_id < Components.
 */
template <int Components>
class BoundaryData {
  static_assert(Components >= 1,
                "BoundaryData needs at least one component");

public:
  /**
   * Allocate storage for @p grid. All conditions start as @c BcKind::None; call @ref fill and
   * @ref upload.
   */
  BoundaryData(const DenseDistributed& grid,
               std::string name)
      : state_(std::make_shared<State>()) {
    State& s = *state_;
    s.name = std::move(name);
    s.sys = grid.system();
    s.global = grid.global_extent();
    const size_t n = grid.gpu_count();
    if (n == 0U) {
      log::throw_with_log<std::runtime_error>("BoundaryData '{}': grid has no GPUs", s.name);
    }
    const carbon::int32_3d radius3 = grid.stencil().radius3();
    s.stores.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      s.stores.emplace_back(grid.spans()[i], radius3, Components);
    }
    recompute_layout_();
    s.dev_kind = s.sys.gpu_group().template buffer<uint8_t>(s.kind_counts);
    s.dev_value = s.sys.gpu_group().template buffer<double>(s.value_counts);
  }

  [[nodiscard]] auto
  name() const noexcept -> const std::string& {
    return state_->name;
  }

  /**
   * Set the condition of every stored face cell: @p fn(face, global_node, comp) returns a
   * @ref FaceBc. @c global_node is the cell on the face (its own axis at 0 or extent - 1).
   *
   * @throws std::invalid_argument for a non-finite value, or a condition on an axis whose extent
   *         is not larger than the stencil radius.
   */
  template <typename Fn>
  requires std::invocable<Fn&,
                          Face,
                          int64_3d,
                          int> void
  fill(Fn&& fn) {
    for (auto& store : state_->stores) {
      store.fill(state_->global, fn);
    }
  }

  /** Same condition on every cell of each face, taken from the per-face constants @p bc. */
  void
  fill(const BoundaryConditions& bc) {
    fill([&bc](Face f, int64_3d, int) {
      return bc.at(f);
    });
  }

  /** Copy the host arrays to the devices and wait for the copy to finish. */
  void
  upload() {
    State& s = *state_;
    const size_t n = s.stores.size();
    std::vector<detail::Allocation<uint8_t>> hk;
    std::vector<detail::Allocation<double>> hv;
    hk.reserve(n);
    hv.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      auto k = Cpu::buffer<uint8_t>(s.kind_counts[i]);
      auto v = Cpu::buffer<double>(s.value_counts[i]);
      for (int f = 0; f < kFaceCount; ++f) {
        const auto& src_k = s.stores[i].kind_data(static_cast<Face>(f));
        const auto& src_v = s.stores[i].value_data(static_cast<Face>(f));
        if (!src_k.empty()) {
          std::memcpy(k.data() + s.face_offset[i][f], src_k.data(), src_k.size() * sizeof(uint8_t));
          std::memcpy(v.data() + s.face_offset[i][f], src_v.data(), src_v.size() * sizeof(double));
        }
      }
      hk.push_back(std::move(k));
      hv.push_back(std::move(v));
    }
    const auto host_k = detail::BufferGroup<uint8_t>::from_vector(std::move(hk));
    const auto host_v = detail::BufferGroup<double>::from_vector(std::move(hv));
    Stream stream = s.sys.stream();
    Transfer<uint8_t>(s.sys, host_k, s.dev_kind)(stream);
    Transfer<double>(s.sys, host_v, s.dev_value)(stream);
    stream.wait();
  }

  /** Device view for GPU @p gpu (index @c KernelLoader::relative_gpu_id()). */
  [[nodiscard]] auto
  view(size_t gpu) const -> detail::dense_distributed::BoundaryView {
    const State& s = *state_;
    detail::dense_distributed::BoundaryView v;
    for (int f = 0; f < kFaceCount; ++f) {
      const auto& r = s.stores[gpu].range(static_cast<Face>(f));
      if (r.cells() == 0) {
        continue;
      }
      v.faces[f] = detail::dense_distributed::HostBoundaryStore::make_face_view(
          r, s.dev_kind[gpu].data() + s.face_offset[gpu][f],
          s.dev_value[gpu].data() + s.face_offset[gpu][f], Components);
    }
    return v;
  }

  /** Host view for GPU @p gpu, over the host arrays (host kernels and tests). */
  [[nodiscard]] auto
  host_view(size_t gpu) const -> detail::dense_distributed::BoundaryView {
    return state_->stores[gpu].host_view();
  }

private:
  struct State {
    std::string name;
    System sys;
    int64_3d global{};
    std::vector<detail::dense_distributed::HostBoundaryStore> stores;
    std::vector<std::array<size_t, kFaceCount>> face_offset;
    std::vector<size_t> kind_counts;
    std::vector<size_t> value_counts;
    Buffer<uint8_t> dev_kind{};
    Buffer<double> dev_value{};
  };

  void
  recompute_layout_() {
    State& s = *state_;
    const size_t n = s.stores.size();
    s.face_offset.assign(n, {});
    s.kind_counts.assign(n, 0U);
    s.value_counts.assign(n, 0U);
    for (size_t i = 0; i < n; ++i) {
      size_t off = 0;
      for (int f = 0; f < kFaceCount; ++f) {
        s.face_offset[i][f] = off;
        off += static_cast<size_t>(s.stores[i].range(static_cast<Face>(f)).cells()) *
               static_cast<size_t>(Components);
      }
      s.kind_counts[i] = off;
      s.value_counts[i] = off;
    }
  }

  std::shared_ptr<State> state_;
};

}  // namespace carbon
