#pragma once

/**
 * @file sub_grid.h
 * @brief @c carbon::SubGrid — a packed subset of a parent grid's cells with its own compact
 *        fields, built from a parent grid plus a cell selection plus a @ref RegionMask .
 *
 * A SubGrid @c S over a parent grid @c G stores only the selected cells. Each GPU holds a lane
 * table mapping lane @c 0..count-1 to a parent cell; S fields are packed arrays indexed by
 * lane, and S kernels launch one thread per lane instead of one per parent cell.
 *
 * - @b Types. @c S::Span derives from @c G::Span and @c S::Idx from @c G::Idx , so inside an
 *   S kernel @c uG.read(s, i, 0) reads a parent field at the lane's parent cell with no
 *   conversion code, while @c uS.read(s, i, 0) reads the packed S field at the lane.
 * - @b DataView . Lanes are ordered parent-Internal cells first, then parent-Boundary cells,
 *   so every view is one contiguous lane range and S kernels can be OCC-split like parent
 *   kernels (see @ref detail::sub_grid::Lanes ).
 * - @b Region mask. The @ref RegionMask passed at construction is the footprint S kernels
 *   declare on parent fields. It must cover every selected cell (the label constructor
 *   guarantees this by deriving the selection from the mask).
 * - @b No halo. S fields support same-cell access only; stencils belong on parent fields.
 *
 * A SubGrid is immutable: the selection is evaluated once, at construction.
 */

#include <carbon/data_view.h>
#include <carbon/dense.h>
#include <carbon/detail/sub_grid/field.h>
#include <carbon/detail/sub_grid/fwd.h>
#include <carbon/detail/sub_grid/idx.h>
#include <carbon/detail/sub_grid/kernel_impl.h>
#include <carbon/detail/sub_grid/partition.h>
#include <carbon/detail/sub_grid/reduce_traits.h>
#include <carbon/detail/sub_grid/span.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/region_mask.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace carbon {

namespace detail::sub_grid {

struct Builder;

}  // namespace detail::sub_grid

/**
 * Packed subset of a parent grid @p G (v1: @ref Dense ). Build one with
 * @c Dense::new_sub_grid ; see the file comment for the model.
 *
 * Shallow handle: copies share the lane tables, which stay alive as long as any copy or any
 * field built from it does.
 */
template <typename G>
class SubGrid {
  static_assert(!detail::sub_grid::is_sub_grid_v<G>,
                "carbon::SubGrid: a SubGrid of a SubGrid is not supported.");

public:
  using Parent = G;
  using Span = detail::sub_grid::Span<typename G::Span, typename G::Idx>;
  using Idx = detail::sub_grid::Idx<typename G::Idx>;
  using Lanes = detail::sub_grid::Lanes<typename G::Idx>;

  template <typename T, int Components>
  using Partition = detail::sub_grid::Partition<typename G::Span, typename G::Idx, T, Components>;

  template <typename T, int Components>
  using Field = detail::sub_grid::Field<SubGrid, T, Components>;

  /** Empty placeholder: no parent, no GPUs, no lanes. */
  SubGrid() = default;

  [[nodiscard]] auto
  parent() const -> const G& {
    return st_().parent;
  }

  [[nodiscard]] auto
  name() const -> const std::string& {
    return st_().name;
  }

  /** Footprint S kernels declare on parent fields. */
  [[nodiscard]] auto
  mask() const -> const RegionMask& {
    return st_().mask;
  }

  [[nodiscard]] auto
  gpu_count() const noexcept -> size_t {
    return state_ ? state_->count.size() : 0U;
  }

  /** The parent's @ref System (devices, streams, events). */
  [[nodiscard]] auto
  system() const -> const System& {
    return st_().parent.system();
  }

  /** Identity of this SubGrid (shared by its copies); S partitions carry it as their owner. */
  [[nodiscard]] auto
  uid() const noexcept -> std::uint64_t {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(state_.get()));
  }

  /** One span per GPU whose lane table lives in device memory (device kernels). */
  [[nodiscard]] auto
  spans() const -> const std::vector<Span>& {
    return st_().spans;
  }

  /** One span per GPU whose lane table lives in host memory (host kernels, inspection). */
  [[nodiscard]] auto
  host_spans() const -> const std::vector<Span>& {
    return st_().host_spans;
  }

  /** Selected cells on GPU @p gpu . @throws std::out_of_range for a bad @p gpu . */
  [[nodiscard]] auto
  count(size_t gpu) const -> int64_t {
    return st_().count.at(gpu);
  }

  /** Selected cells on GPU @p gpu that lie in the parent's Internal region. */
  [[nodiscard]] auto
  internal_count(size_t gpu) const -> int64_t {
    return st_().internal_count.at(gpu);
  }

  /** Selected cells on GPU @p gpu that lie in the parent's Boundary region. */
  [[nodiscard]] auto
  boundary_count(size_t gpu) const -> int64_t {
    return count(gpu) - internal_count(gpu);
  }

  /** Selected cells summed over the local GPUs. */
  [[nodiscard]] auto
  total_count() const noexcept -> int64_t {
    int64_t total = 0;
    if (state_) {
      for (const int64_t n : state_->count) {
        total += n;
      }
    }
    return total;
  }

  /** Parent cell of @p lane on GPU @p gpu (host lane table). @throws std::out_of_range . */
  [[nodiscard]] auto
  parent_idx(size_t gpu,
             int64_t lane) const -> const typename G::Idx& {
    const int64_t n = count(gpu);
    if (lane < 0 || lane >= n) {
      log::throw_with_log<std::out_of_range>(
          "SubGrid::parent_idx: lane {} out of range [0, {}) on gpu {}", lane, n, gpu);
    }
    return st_().host_spans[gpu].sub().parent_idx(lane);
  }

  /** Global coordinates of the parent cell of @p lane on GPU @p gpu . */
  [[nodiscard]] auto
  global_location(size_t gpu,
                  int64_t lane) const -> int64_3d {
    return st_().host_spans[gpu].global_location(parent_idx(gpu, lane));
  }

  /**
   * @brief Allocates a packed field: per GPU, @c Components planes of @c count(gpu) elements.
   *
   * @p side has the same meaning as for @c Dense::newField , except that a @c HostDevice
   * mirror is allocated eagerly. Partitions bind the device side when present.
   *
   * @throws std::runtime_error when the SubGrid has no GPUs.
   */
  template <typename T,
            int Components>
  [[nodiscard]] auto
  newField(MemorySide side = MemorySide::Device,
           std::string name = {}) const -> Field<T,
                                                 Components> {
    using P = Partition<T, Components>;
    const State& st = st_();
    const size_t n = st.count.size();
    if (n == 0U) {
      log::throw_with_log<std::runtime_error>(
          "SubGrid::newField: gpu_count is zero; cannot allocate field buffers");
    }
    const System& sys = st.parent.system();
    const auto counts = sys.make_vector([&](size_t i) {
      return static_cast<size_t>(st.count[i]) * static_cast<size_t>(Components);
    });
    Buffer<T> gpu_buffers{};
    if (has_device(side)) {
      gpu_buffers = sys.gpu_group().template buffer<T>(counts);
    }
    Buffer<T> cpu_buffers{};
    if (has_host(side)) {
      auto host_allocs = sys.make_vector([&](size_t i) {
        return Cpu::buffer<T>(counts[i]);
      });
      cpu_buffers = detail::BufferGroup<T>::from_vector(std::move(host_allocs));
    }
    const std::uint64_t owner = uid();
    auto parts = sys.make_vector([&](size_t i) {
      T* const ptr = has_device(side) ? gpu_buffers[i].data() : cpu_buffers[i].data();
      return P(ptr, owner);
    });
    std::vector<P> host_parts;
    if (has_host(side)) {
      host_parts = sys.make_vector([&](size_t i) {
        return P(cpu_buffers[i].data(), owner);
      });
    }
    return Field<T, Components>(*this, std::move(gpu_buffers), std::move(cpu_buffers),
                                std::move(parts), std::move(host_parts), side, std::move(name));
  }

  /**
   * @brief Builds a device compute @ref Kernel that runs one thread per lane.
   *
   * Same loading-lambda shape as @c Dense::make_kernel ; the per-cell callable takes
   * @c (SubGrid::Span, SubGrid::Idx) . Inside it:
   * - @c uS.read(s, i, c) / @c uS.write(s, i, c, v) access a field of this SubGrid;
   * - @c uG.read(s, i, c) / @c uG.write(...) access a parent field at the lane's parent
   *   cell (the SubGrid span and index convert to the parent's);
   * - a parent field loaded with @c read_stencil may be read at neighbours of that cell
   *   through the parent's halo. SubGrid fields cannot be stencil-read (compile error).
   *
   * The kernel-wide region is preset to @ref mask , and loading a field of another SubGrid
   * throws here, at construction. @c Kernel::launch(stream, view) runs the lanes of @p view ;
   * the @ref Skeleton OCC-splits the kernel like a Dense one when it stencil-reads a parent
   * field after a halo. Call from a device translation unit.
   */
  template <typename L>
  [[nodiscard]] auto
  make_kernel(L&& loading_fn,
              size_t shared_mem = 0U) const -> Kernel {
    return make_kernel(std::forward<L>(loading_fn), std::string{}, shared_mem);
  }

  /** @ref make_kernel with a diagnostic @p name (logs, Skeleton DOT labels). */
  template <typename L>
  [[nodiscard]] auto
  make_kernel(L&& loading_fn,
              std::string name,
              size_t shared_mem = 0U) const -> Kernel {
    using Impl = detail::sub_grid::DeviceKernelImpl<SubGrid, std::decay_t<L>>;
    return Kernel::from_impl(
        std::make_shared<Impl>(*this, std::forward<L>(loading_fn), shared_mem, std::move(name)));
  }

  /**
   * @brief Host (OpenMP) twin of @ref make_kernel over the host lane tables. Every field
   *        the loader touches needs a host mirror.
   */
  template <typename L>
  [[nodiscard]] auto
  make_host_kernel(L&& loading_fn,
                   std::string name = {}) const -> Kernel {
    using Impl = detail::sub_grid::HostKernelImpl<SubGrid, std::decay_t<L>>;
    return Kernel::from_impl(
        std::make_shared<Impl>(*this, std::forward<L>(loading_fn), std::move(name)));
  }

  /**
   * @brief Device reduction over the lanes: the per-cell callable returns the lane's
   *        contribution, @c CARBON_HD (SubGrid::Span, SubGrid::Idx) -> T .
   *
   * Same pipeline and options as @c Dense::make_reduce_kernel (multi-GPU combine,
   * cross-rank stage, @c ReduceFinalize::BlockAtomic , deferred host copy) except
   * @c ReduceOptions::stage1_threads , which throws. The loader rules of @ref make_kernel
   * apply, and a reduction that stencil-reads a parent field after its halo is OCC-split
   * into Internal / Boundary partials plus a merge. Use @c Kernel::reduce for a name.
   */
  template <typename T,
            typename L,
            typename Op = reduce::Sum<T>>
  [[nodiscard]] auto
  make_reduce_kernel(L&& loading_fn,
                     ReductionResult<T>& dst,
                     Op op = {},
                     size_t shared_mem = 0U,
                     reduce::ReduceOptions options = {}) const -> Kernel {
    return Kernel::reduce<T>(*this, std::forward<L>(loading_fn), dst, std::move(op), {}, shared_mem,
                             options);
  }

  /** Host (OpenMP) twin of @ref make_reduce_kernel ; fields need host mirrors. */
  template <typename T,
            typename L,
            typename Op = reduce::Sum<T>>
  [[nodiscard]] auto
  make_host_reduce_kernel(L&& loading_fn,
                          ReductionResult<T>& dst,
                          Op op = {}) const -> Kernel {
    return Kernel::host_reduce<T>(*this, std::forward<L>(loading_fn), dst, std::move(op), {});
  }

  /** Reduction launchable on the device ( @c launch ) or the host ( @c launch_on_host ). */
  template <typename T,
            typename L,
            typename Op = reduce::Sum<T>>
  [[nodiscard]] auto
  make_universal_reduce_kernel(L&& loading_fn,
                               ReductionResult<T>& dst,
                               Op op = {},
                               size_t shared_mem = 0U) const -> Kernel {
    return Kernel::reduce_universal<T>(*this, std::forward<L>(loading_fn), dst, std::move(op), {},
                                       shared_mem);
  }

private:
  friend struct detail::sub_grid::Builder;

  struct State {
    G parent{};
    RegionMask mask{RegionMask::all()};
    std::string name{};
    Buffer<typename G::Idx> device_tables{};
    Buffer<typename G::Idx> host_tables{};
    std::vector<int64_t> count{};
    std::vector<int64_t> internal_count{};
    std::vector<Span> spans{};
    std::vector<Span> host_spans{};
  };

  explicit SubGrid(std::shared_ptr<const State> state)
      : state_(std::move(state)) {}

  [[nodiscard]] auto
  st_() const -> const State& {
    if (!state_) {
      log::throw_with_log<std::logic_error>("SubGrid: empty (default-constructed) SubGrid");
    }
    return *state_;
  }

  std::shared_ptr<const State> state_{};
};

namespace detail::sub_grid {

/**
 * Builds a @ref SubGrid from per-GPU host selection flags laid out like a parent field
 * (one @c std::uint8_t per storage cell, non-zero = selected).
 *
 * Lane order is the parent's own iteration order, Internal view first and Boundary view
 * second, enumerated through @c Span::iter_extent and @c Span::set_and_validate — so the
 * Internal / Boundary split is the parent's by construction and the table is deterministic.
 */
struct Builder {
  template <typename G>
  [[nodiscard]] static auto
  from_host_flags(const G& parent,
                  const std::vector<const std::uint8_t*>& flags,
                  const RegionMask& mask,
                  std::string name) -> SubGrid<G> {
    using S = SubGrid<G>;
    using PIdx = typename G::Idx;
    using State = typename S::State;

    const auto& parent_spans = parent.spans();
    const size_t n = parent_spans.size();
    if (flags.size() != n) {
      log::throw_with_log<std::invalid_argument>("SubGrid: {} flag buffers for {} GPUs",
                                                 flags.size(), n);
    }

    auto state = std::make_shared<State>();
    state->parent = parent;
    state->mask = mask;
    state->name = std::move(name);

    std::vector<std::vector<PIdx>> selected(n);
    for (size_t g = 0; g < n; ++g) {
      const auto& span = parent_spans[g];
      const std::uint8_t* f = flags[g];
      auto& out = selected[g];
      for (const DataView view : {DataView::Internal, DataView::Boundary}) {
        const int64_3d ext = span.iter_extent(view);
        for (int64_t z = 0; z < ext.z; ++z) {
          for (int64_t y = 0; y < ext.y; ++y) {
            for (int64_t x = 0; x < ext.x; ++x) {
              const auto r = span.set_and_validate(view, x, y, z);
              if (r.valid && f[span.linear_offset(r.idx)] != 0U) {
                out.push_back(r.idx);
              }
            }
          }
        }
        if (view == DataView::Internal) {
          state->internal_count.push_back(static_cast<int64_t>(out.size()));
        }
      }
      state->count.push_back(static_cast<int64_t>(out.size()));
    }

    System sys = parent.system();
    auto host_allocs = sys.make_vector([&](size_t g) {
      auto a = Cpu::buffer<PIdx>(selected[g].size());
      for (size_t k = 0; k < selected[g].size(); ++k) {
        a.data()[k] = selected[g][k];
      }
      return a;
    });
    state->host_tables = BufferGroup<PIdx>::from_vector(std::move(host_allocs));
    const auto counts = sys.make_vector([&](size_t g) {
      return selected[g].size();
    });
    state->device_tables = sys.gpu_group().template buffer<PIdx>(counts);
    Stream stream = sys.stream();
    Transfer<PIdx> upload(sys, state->host_tables, state->device_tables);
    upload(stream);
    stream.wait();

    const auto owner = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(state.get()));
    for (size_t g = 0; g < n; ++g) {
      const int64_t cnt = state->count[g];
      const int64_t internal = state->internal_count[g];
      state->spans.emplace_back(parent_spans[g],
                                Lanes<PIdx>(state->device_tables[g].data(), cnt, internal, owner));
      state->host_spans.emplace_back(
          parent_spans[g], Lanes<PIdx>(state->host_tables[g].data(), cnt, internal, owner));
    }

    S result(std::shared_ptr<const State>(std::move(state)));
    if (n > 0U && result.total_count() == 0) {
      CARBON_LOG_WARN("SubGrid '{}': the selection is empty on every GPU", result.name());
    }
    return result;
  }
};

/** Per-cell body of the selection pass: writes 1 where @p predicate holds, else 0. */
template <typename Predicate, typename Out>
struct SelectFn {
  Predicate predicate;
  Out out;

  template <typename SpanT,
            typename IdxT>
  CARBON_HD auto
  operator()(const SpanT& span,
             const IdxT& idx) const -> void {
    out.write(span, idx, 0, predicate(span, idx) ? std::uint8_t{1} : std::uint8_t{0});
  }
};

/** Selection predicate of the label constructor: the cell's label bit is in @c mask . */
template <typename LabelPartition>
struct LabelPredicate {
  LabelPartition labels;
  RegionMask mask;

  template <typename SpanT,
            typename IdxT>
  [[nodiscard]] CARBON_HD auto
  operator()(const SpanT& span,
             const IdxT& idx) const -> bool {
    return mask.test(labels.read(span, idx, 0));
  }
};

}  // namespace detail::sub_grid

template <std::invocable<KernelLoader&> L>
auto
Dense::new_sub_grid(L&& select,
                    const RegionMask& mask,
                    std::string name) const -> SubGrid<Dense> {
  auto flags = newField<std::uint8_t, 1>(MemorySide::HostDevice, "sub_grid_select");
  const Kernel select_kernel =
      make_kernel([select = std::forward<L>(select), flags](KernelLoader& loader) {
        auto predicate = select(loader);
        auto out = loader.write(flags);
        return detail::sub_grid::SelectFn<decltype(predicate), decltype(out)>{predicate, out};
      });
  System sys = system_;
  Stream stream = sys.stream();
  select_kernel.launch(stream);
  Transfer<std::uint8_t> download(sys, flags.gpu_buffers(), flags.cpu_buffers());
  download(stream);
  stream.wait();

  std::vector<const std::uint8_t*> host_flags;
  host_flags.reserve(gpu_count());
  for (size_t g = 0; g < gpu_count(); ++g) {
    host_flags.push_back(flags.cpu_buffers()[g].data());
  }
  return detail::sub_grid::Builder::from_host_flags(*this, host_flags, mask, std::move(name));
}

template <typename T>
auto
Dense::new_sub_grid(const Field<T,
                                1>& labels,
                    const RegionMask& mask,
                    std::string name) const -> SubGrid<Dense> {
  static_assert(std::is_same_v<T, std::uint8_t>,
                "Dense::new_sub_grid(labels, mask): labels must be a Dense::Field<std::uint8_t, 1> "
                "(one RegionMask bit per label value).");
  return new_sub_grid(
      [labels, mask](KernelLoader& loader) {
        auto label_partition = loader.read(labels);
        return detail::sub_grid::LabelPredicate<decltype(label_partition)>{label_partition, mask};
      },
      mask, std::move(name));
}

}  // namespace carbon
