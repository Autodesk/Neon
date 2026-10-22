#pragma once

#include <carbon/cpu.h>
#include <carbon/dense_disg.h>
#include <carbon/detail/buffer_group.h>
#include <carbon/detail/dense_disg/partition.h>
#include <carbon/detail/dense_disg/span.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace carbon::detail::dense_disg {

/**
 * Per-grid field storage over a @c DenseDisg layout. Mirrors @c carbon::detail::dense::Field
 * but sizes per-slab buffers from the disaggregated total
 * @c span.linear_storage_elements(Components) (= @c (Lz + 2) · C · xy_pitch ) and binds
 * @ref Partition views that read through the disg pitch on @ref Span::linear_offset .
 *
 * Lifetime + side semantics match the aggregated @c Field: device buffers are eager,
 * host mirror is lazy for @c MemorySide::HostDevice and eager for @c MemorySide::Host , and
 * the host-side state is held in a @c shared_ptr so all shallow copies of a field point at
 * the same lazily-allocated mirror.
 *
 * @tparam T          Element type stored per cell.
 * @tparam Components Per-cell component count (1 = scalar, 3 = vector, ...). Same value
 *                    flows into every @ref Partition built from this field.
 */
template <typename T, int Components>
class Field {
  static_assert(Components >= 1,
                "carbon::DenseDisg::Field<T, Components> requires Components >= 1.");

public:
  using value_type = T;
  static constexpr int components = Components;

  Field()
      : host_storage_(std::make_shared<HostStorage>()) {}

  [[nodiscard]] auto
  dense_disg() const noexcept -> const DenseDisg& {
    return dense_;
  }

  [[nodiscard]] auto
  name() const noexcept -> const std::string& {
    return name_;
  }

  auto
  set_name(std::string name) -> Field& {
    name_ = std::move(name);
    return *this;
  }

  [[nodiscard]] auto
  gpu_buffers() noexcept -> ::carbon::Buffer<T>& {
    return gpu_buffers_;
  }

  [[nodiscard]] auto
  gpu_buffers() const noexcept -> const ::carbon::Buffer<T>& {
    return gpu_buffers_;
  }

  [[nodiscard]] auto
  cpu_buffers() -> ::carbon::Buffer<T>& {
    ensure_host_storage_();
    return host_storage_->cpu_buffers;
  }

  [[nodiscard]] auto
  cpu_buffers() const -> const ::carbon::Buffer<T>& {
    ensure_host_storage_();
    return host_storage_->cpu_buffers;
  }

  /**
   * Seed every @b owned cell of the host mirror from @p fn , invoked as
   * @c fn(carbon::int64_3d global, int component) returning @c T . Disaggregated twin of
   * @c dense::Field::fill_owned_host : the (x, y, owned-z) triple is resolved through
   * @c Span::set_and_validate (which populates the five-region pitch fields) and written at
   * @c Span::linear_offset , so the caller only ever thinks in global cell coordinates and
   * never has to know the region layout. Triggers the lazy host allocation for
   * @c MemorySide::HostDevice fields (one-shot, same path as @ref cpu_buffers ).
   *
   * @throws std::invalid_argument when the field has no host mirror (Device-only field or
   *         zero GPUs).
   */
  template <typename Fn>
  void
  fill_owned_host(Fn&& fn) {
    static_assert(std::is_invocable_r_v<T, Fn&, ::carbon::int64_3d, int>,
                  "DenseDisg::Field::fill_owned_host expects a callable invocable as "
                  "fn(carbon::int64_3d global, int component) returning T.");

    const auto& spans = dense_.spans();
    const size_t n_gpus = spans.size();
    if (n_gpus == 0U) {
      return;
    }
    ensure_host_storage_();
    auto& cpu_buffers = host_storage_->cpu_buffers;
    if (cpu_buffers.size() != n_gpus) {
      ::carbon::log::throw_with_log<std::invalid_argument>(
          "DenseDisg::Field::fill_owned_host: host mirror is not allocated "
          "(cpu_buffers().size()={}, "
          "gpu_count={}). Construct the field with MemorySide::Host or MemorySide::HostDevice.",
          cpu_buffers.size(), n_gpus);
    }

    for (size_t g = 0; g < n_gpus; ++g) {
      const auto& span = spans[g];
      const auto haloed = span.haloed_extent();
      const int64_t lz = span.logic_extent().z;
      T* h = cpu_buffers[g].data();
      for (int c = 0; c < Components; ++c) {
        for (int64_t iz = 0; iz < lz; ++iz) {
          for (int64_t y = 0; y < haloed.y; ++y) {
            for (int64_t x = 0; x < haloed.x; ++x) {
              const auto vi = span.set_and_validate(::carbon::DataView::Standard, x, y, iz);
              const int64_3d global = span.global_location(vi.idx);
              const int64_t off = span.linear_offset(vi.idx, Components, c);
              h[off] = static_cast<T>(fn(global, c));
            }
          }
        }
      }
    }
  }

  [[nodiscard]] auto
  partitions() noexcept -> std::vector<Partition<T,
                                                 Components>>& {
    return partitions_;
  }

  [[nodiscard]] auto
  partitions() const noexcept -> const std::vector<Partition<T,
                                                             Components>>& {
    return partitions_;
  }

  [[nodiscard]] auto
  host_partitions() -> std::vector<Partition<T,
                                             Components>>& {
    ensure_host_storage_();
    return host_storage_->host_partitions;
  }

  [[nodiscard]] auto
  host_partitions() const -> const std::vector<Partition<T,
                                                         Components>>& {
    ensure_host_storage_();
    return host_storage_->host_partitions;
  }

  /**
   * Build a halo-exchange @c Kernel for this field. The returned handle wraps a
   * @c HaloDisg<T, Components> constructed in-place; subsequent @c launch calls
   * issue the halo exchange on the user's stream. Single-process layouts with
   * @c gpu_count <= 1 and an empty halo plan still produce a real @c HaloDisg
   * (its @c operator() is a no-op in that shape), keeping the @c Kernel
   * polymorphic API uniform.
   *
   * Used by the @c Skeleton 's auto-halo path (via @c read_stencil ) on
   * stencil-tagged DenseDisg reads, and by user-driven @c OverlappedStencil
   * pipelines that want to overlap the halo exchange with the @c Internal
   * compute.
   */
  [[nodiscard]] auto
  make_halo_kernel(std::string halo_name = {}) const -> ::carbon::Kernel;

  /**
   * @brief Build a halo @c Kernel with explicit @p stencil and @p semantic .
   *
   * Twin of @c Dense::Field::make_halo_kernel(name, stencil, semantic) for the
   * disaggregated layout. Used by @c KernelLoader::read_stencil(field, stencil,
   * semantic) to auto-insert lattice-semantic halos — @c HaloDisg narrows the
   * per-adjacency Send/Recv to the @c c_q.z run-length profile of @p stencil
   * under @c StencilSemantic::Lattice .
   */
  [[nodiscard]] auto
  make_halo_kernel(std::string halo_name,
                   Stencil stencil,
                   StencilSemantic semantic) const -> ::carbon::Kernel;

private:
  // Friended so DenseDisg::newField is the only construction path that produces a
  // populated Field. Mirrors the aggregated dense::Field design.
  Field(DenseDisg dense,
        ::carbon::Buffer<T> gpu_buffers,
        ::carbon::Buffer<T> cpu_buffers,
        std::vector<Partition<T,
                              Components>> partitions,
        std::vector<Partition<T,
                              Components>> host_partitions,
        ::carbon::MemorySide side,
        std::string name = {})
      : dense_(std::move(dense)),
        gpu_buffers_(std::move(gpu_buffers)),
        partitions_(std::move(partitions)),
        side_(side),
        name_(std::move(name)),
        host_storage_(std::make_shared<HostStorage>(std::move(cpu_buffers),
                                                    std::move(host_partitions))) {}

  friend class ::carbon::DenseDisg;

  struct HostStorage {
    ::carbon::Buffer<T> cpu_buffers{};
    std::vector<Partition<T, Components>> host_partitions{};
    std::once_flag init_flag{};

    HostStorage() = default;

    HostStorage(::carbon::Buffer<T> cpu_b,
                std::vector<Partition<T,
                                      Components>> host_p)
        : cpu_buffers(std::move(cpu_b)),
          host_partitions(std::move(host_p)) {}
  };

  auto
  ensure_host_storage_() const -> void {
    if (side_ != ::carbon::MemorySide::HostDevice) {
      return;
    }
    HostStorage& storage = *host_storage_;
    std::call_once(storage.init_flag, [this, &storage]() {
      const auto& spans = dense_.spans();
      const size_t n_gpus = spans.size();
      if (n_gpus == 0U) {
        return;
      }
      auto host_allocs = dense_.system().make_vector([&](size_t i) {
        const size_t total = static_cast<size_t>(spans[i].linear_storage_elements(Components));
        return ::carbon::Cpu::buffer<T>(total);
      });
      storage.cpu_buffers = ::carbon::detail::BufferGroup<T>::from_vector(std::move(host_allocs));
      storage.host_partitions = dense_.system().make_vector([&storage, this](size_t i) {
        const auto& sp = dense_.spans()[i];
        return ::carbon::detail::dense_disg::Partition<T, Components>(
            storage.cpu_buffers[i].data(), ::carbon::DataView::Standard, sp.origin(), sp.spacing());
      });
    });
  }

  DenseDisg dense_;
  ::carbon::Buffer<T> gpu_buffers_{};
  std::vector<Partition<T, Components>> partitions_{};
  ::carbon::MemorySide side_{::carbon::MemorySide::Device};
  std::string name_{};
  mutable std::shared_ptr<HostStorage> host_storage_{};
};

}  // namespace carbon::detail::dense_disg
