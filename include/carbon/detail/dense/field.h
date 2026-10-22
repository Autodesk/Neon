#pragma once

#include <carbon/cpu.h>
#include <carbon/dense.h>
#include <carbon/detail/buffer_group.h>
#include <carbon/detail/dense/partition.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/system.h>  // for carbon::Buffer<T>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace carbon {
class Kernel;  // see <carbon/kernel.h>
}  // namespace carbon

namespace carbon::detail::dense {

/**
 * Per-grid field storage over a @c Dense layout: owns one device @ref Buffer and one host
 * @ref Buffer (either may be empty), plus one @ref Partition per GPU (pointer + slab
 * @c z_offset) aligned with @ref Dense::spans.
 *
 * The owning @c Dense is stored @b by @b value rather than by reference: copying @c Dense is
 * cheap (its @c System is a shallow handle and @c spans_ is a small vector), and storing it by
 * value lets a @c Field outlive the @c Dense it was created from without dangling. Two fields
 * built from the same @c Dense will see identical layouts because the underlying device/stream
 * state is shared via @c System's shared-pointer based group types.
 *
 * A field always carries both a GPU and a CPU @ref Buffer so mirroring code has a stable
 * place to write to; it is fine to leave either side default-constructed (empty) when the
 * corresponding storage has not been allocated yet. @ref Dense::newField today populates the
 * GPU side only and leaves the CPU side empty.
 *
 * @tparam T          Scalar or element type stored per grid cell (same as @ref Buffer).
 * @tparam Components Number of components per cell (compile-time). 1 = scalar field,
 *                    3 = vector field, etc. Same value carried by every @ref Partition built
 *                    from this @c Field.
 */
template <typename T, int Components>
class Field {
  static_assert(Components >= 1,
                "carbon::Dense::Field<T, Components> requires Components >= 1 "
                "(1 = scalar, 3 = vector, ...). Components is the per-cell component count "
                "used to size the SoA storage in Dense::newField.");

public:
  /**
   * Element type stored per cell (component). Exposed so generic code (e.g. @c carbon::io)
   * can recover @c T from a @c Field instance without re-deducing it.
   */
  using value_type = T;

  /**
   * Per-cell component count this field was instantiated with. @c 1 for scalar fields,
   * @c 3 for vector fields, etc. Same role as @c value_type for the second template
   * parameter.
   */
  static constexpr int components = Components;

  /**
   * @brief Default-construct an empty field (no grid, no buffers, no partitions). Useful as a
   *        placeholder before assignment from a real @ref Dense::newField. All buffer / partition
   *        accessors return empty containers in this state.
   */
  Field()
      : host_storage_(std::make_shared<HostStorage>()) {}

  [[nodiscard]] auto
  dense() const noexcept -> const Dense& {
    return dense_;
  }

  /**
   * @brief Optional human-readable name for this field.
   *
   * Free-form, used purely for diagnostics: log messages, the carbon
   * @c Skeleton 's DAG export ( @c Skeleton::export_dot fills node tooltips and
   * edge labels with this string), and any future visualisation / reporting
   * layer. Not interpreted by carbon itself — empty by default; two fields with
   * the same name are not treated as the same field (identity is still the
   * @c BufferGroup uid). Set at @c Dense::newField time or post-hoc via
   * @ref set_name .
   */
  [[nodiscard]] auto
  name() const noexcept -> const std::string& {
    return name_;
  }

  /**
   * @brief Set the field's diagnostic name. Returns @c *this so call sites can
   *        chain ( @c auto u = dense.newField<float, 1>(MemorySide::HostDevice).set_name("u"); ).
   *        Idempotent — overwrites the previous name.
   */
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

  /**
   * @brief Per-GPU host (CPU) mirror buffers.
   *
   * For fields built with @c MemorySide::HostDevice the host mirror is allocated @b
   * lazily on first access (this method, @ref host_partitions , or
   * @ref fill_owned_host ) — Neon-style "NULL_MEM until needed" so a HostDevice
   * field that ends up only ever touched on the device pays no host-allocation cost,
   * and a device pipeline that copies through the mirror via @ref Transfer triggers
   * the host allocation exactly when it first needs it. Allocation is one-shot and
   * idempotent (guarded by an internal @c std::once_flag ); all subsequent calls are
   * O(1).
   *
   * For @c MemorySide::Host the buffers are populated eagerly in
   * @c Dense::newField ; for @c MemorySide::Device this method returns the empty
   * default ( @c size() == 0 ) and never allocates.
   *
   * @par Copy semantics
   * The lazy state is held in a @c shared_ptr<HostStorage> so all shallow copies of a
   * @c Field share the same lazily-allocated host mirror — a @c Halo built from a
   * @c HostDevice field, for instance, sees the same @c cpu_buffers as the field the
   * user fills via @ref fill_owned_host , regardless of which copy triggers the
   * first access. Same handle-sharing model as @c gpu_buffers (carbon's
   * @c Buffer<T> is itself shallow over @c BufferGroup ).
   *
   * @throws std::runtime_error on host-allocation failure (only fires the first time
   *         this is called against a @c HostDevice field).
   */
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

  /**
   * @brief Return per-GPU @ref Partition handles bound to the host (CPU) mirror.
   *
   * Parallel vector to @ref partitions, but every entry's pointer targets
   * @c cpu_buffers()[g].data() instead of the device buffer. Empty when the field was
   * built with @c MemorySide::Device (no CPU mirror was allocated). Used by the
   * carbon::detail::DenseHostKernelImpl / ReduceHostKernelImpl OpenMP launch paths
   * (@c Dense::make_host_kernel, @c Dense::make_host_reduce_kernel); device kernels
   * keep going through @ref partitions as before.
   *
   * @par Lifetime
   * Bound at @c Field construction (in @c Dense::newField); the partitions stay valid
   * for the field's lifetime. Returning by reference avoids per-launch rebinds.
   */
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
   * Fill every owned cell on the host mirror with @p fn applied to the cell's global
   * coordinates and component. Halo cells are not touched (they are filled by @c Halo at
   * exchange time, or are out-of-domain in @c BoundaryMode::Open).
   *
   * @p fn is invoked once per (global cell, component) pair; the iteration order is the
   * standard SoA traversal carbon uses elsewhere (component outermost, then z, y, x), but
   * the function takes no dependency on that order so user code should treat invocations
   * as independent.
   *
   * @tparam Fn  Callable invocable as @c fn(int64_3d global, int component) returning a
   *             value convertible to @c T. Single-component code can simply ignore the
   *             second argument.
   *
   * @throws std::invalid_argument when the host mirror is not allocated
   *         (@c cpu_buffers() is empty for a field built without
   *         @c MemorySide::Host or @c MemorySide::HostDevice).
   *
   * @note This is a host-only loop; after filling, push to device via a
   *       @c Transfer<T>(sys, field.cpu_buffers(), field.gpu_buffers()) on whatever stream
   *       you intend to launch kernels on. We do not auto-upload here so the user retains
   *       control over the stream and synchronisation.
   */
  /**
   * @brief Build a @c carbon::Kernel that runs a halo exchange for this field.
   *
   * Equivalent to @c Kernel::halo(Halo<T, Components>(*this), std::move(name)) — a
   * convenience over the static factory so call sites can stay in field-land:
   *
   * @code
   * auto pipeline = std::vector<carbon::Kernel>{};
   * pipeline.push_back(u.make_halo_kernel("halo_u"));
   * pipeline.push_back(dense.make_kernel(jacobi_loader, 0));  // 0 bytes shared mem
   * @endcode
   *
   * The underlying @ref Halo is constructed in place and moved into the kernel impl, so
   * the returned @c Kernel keeps the field's GPU buffers alive (shallow refcount) for as
   * long as it lives.
   *
   * Definition lives in @c carbon/kernel.h, alongside @ref Dense::make_kernel and
   * @ref Dense::make_reduce_kernel.
   *
   * @param name  Optional human-readable name (used by logs and the future skeleton's
   *              DAG node labels). Defaults to empty.
   *
   * @throws std::invalid_argument when the field has no GPU buffers and the layout
   *         requires inter-slab exchange (propagated from @c Halo's constructor).
   */
  [[nodiscard]] auto
  make_halo_kernel(std::string name = {}) const -> ::carbon::Kernel;

  /**
   * @brief Build a halo @c Kernel with explicit @p stencil and @p semantic .
   *
   * Used by @c KernelLoader::read_stencil(field, stencil, semantic) to auto-insert
   * lattice-semantic halos (e.g. LBM populations filtered by @c c_q.z sign).
   */
  [[nodiscard]] auto
  make_halo_kernel(std::string name,
                   Stencil stencil,
                   StencilSemantic semantic) const -> ::carbon::Kernel;

  template <typename Fn>
  void
  fill_owned_host(Fn&& fn) {
    static_assert(std::is_invocable_r_v<T, Fn&, ::carbon::int64_3d, int>,
                  "Field::fill_owned_host expects a callable invocable as "
                  "fn(carbon::int64_3d global, int component) returning T.");

    const auto& spans = dense_.spans();
    const size_t n_gpus = spans.size();
    if (n_gpus == 0U) {
      return;
    }
    // Trigger lazy host alloc for HostDevice fields — same one-shot path
    // @ref cpu_buffers / @ref host_partitions take. No-op for Host-only (already
    // allocated) and Device-only (no host mirror to allocate).
    ensure_host_storage_();
    auto& cpu_buffers = host_storage_->cpu_buffers;
    if (cpu_buffers.size() != n_gpus) {
      log::throw_with_log<std::invalid_argument>(
          "Field::fill_owned_host: host mirror is not allocated (cpu_buffers().size()={}, "
          "gpu_count={}). Construct the field with MemorySide::Host or MemorySide::HostDevice.",
          cpu_buffers.size(), n_gpus);
    }

    for (size_t g = 0; g < n_gpus; ++g) {
      const auto& span = spans[g];
      const auto haloed = span.haloed_extent();
      const int64_t pitch = span.xy_slab_elements();
      const int64_t halo_z = span.halo_z();
      const int64_t z_origin = span.z_offset();
      const int64_t lim_z = span.logic_extent().z;
      const int64_t per_comp = span.linear_storage_elements();
      T* h = cpu_buffers[g].data();
      for (int c = 0; c < Components; ++c) {
        const int64_t plane = static_cast<int64_t>(c) * per_comp;
        // Owned cells live at buffer-z [halo_z, halo_z + lim_z); their global z is
        // z_origin + z (Span::z_offset is the global z of buffer-local z=0, i.e. the start
        // of the lower halo).
        for (int64_t z = halo_z; z < halo_z + lim_z; ++z) {
          const int64_t global_z = z_origin + z;
          for (int64_t y = 0; y < haloed.y; ++y) {
            for (int64_t x = 0; x < haloed.x; ++x) {
              const int64_t off = plane + z * pitch + y * haloed.x + x;
              h[off] = static_cast<T>(fn(::carbon::int64_3d(x, y, global_z), c));
            }
          }
        }
      }
    }
  }

  /**
   * Like @ref fill_owned_host but also touches the per-slab halo cells (the
   * `halo_z` planes on each side of the owned z range). Useful when a host
   * pipeline wants to seed the whole storage extent — e.g. to deterministically
   * zero halo bytes that the bulk @c Transfer will copy to the device, so the
   * very first stencil read on the device sees defined values even before the
   * first halo exchange.
   *
   * @c fn is invoked as @c fn(global, component) where @c global.z is the
   * caller's global z (= @c span.z_offset() + buffer_z); for halo planes
   * @c global.z is outside the slab's owned range and (under @ref BoundaryMode
   * Open) outside the global grid altogether — the caller is responsible for
   * deciding what value belongs there. Periodic-Z callers should mirror Halo's
   * wrap semantics by hand if they care about the values landing in the halo
   * before the first exchange.
   *
   * @throws std::invalid_argument when the field has no host mirror
   *         (Device-only field, or zero GPUs). HostDevice fields trigger lazy
   *         host alloc on first use, same as @c fill_owned_host.
   */
  template <typename Fn>
  void
  fill_full_host(Fn&& fn) {
    static_assert(std::is_invocable_r_v<T, Fn&, ::carbon::int64_3d, int>,
                  "Field::fill_full_host expects a callable invocable as "
                  "fn(carbon::int64_3d global, int component) returning T.");

    const auto& spans = dense_.spans();
    const size_t n_gpus = spans.size();
    if (n_gpus == 0U) {
      return;
    }
    ensure_host_storage_();
    auto& cpu_buffers = host_storage_->cpu_buffers;
    if (cpu_buffers.size() != n_gpus) {
      log::throw_with_log<std::invalid_argument>(
          "Field::fill_full_host: host mirror is not allocated (cpu_buffers().size()={}, "
          "gpu_count={}). Construct the field with MemorySide::Host or MemorySide::HostDevice.",
          cpu_buffers.size(), n_gpus);
    }

    for (size_t g = 0; g < n_gpus; ++g) {
      const auto& span = spans[g];
      const auto haloed = span.haloed_extent();
      const int64_t pitch = span.xy_slab_elements();
      const int64_t z_origin = span.z_offset();
      const int64_t per_comp = span.linear_storage_elements();
      T* h = cpu_buffers[g].data();
      for (int c = 0; c < Components; ++c) {
        const int64_t plane = static_cast<int64_t>(c) * per_comp;
        // Walk the full storage z range [0, haloed.z); halo cells get their
        // global z from the same `z_origin + z` formula so the callback sees
        // a consistent global-coordinate space.
        for (int64_t z = 0; z < haloed.z; ++z) {
          const int64_t global_z = z_origin + z;
          for (int64_t y = 0; y < haloed.y; ++y) {
            for (int64_t x = 0; x < haloed.x; ++x) {
              const int64_t off = plane + z * pitch + y * haloed.x + x;
              h[off] = static_cast<T>(fn(::carbon::int64_3d(x, y, global_z), c));
            }
          }
        }
      }
    }
  }

private:
  /**
   * The only real construction path. Marked private + friended to @ref ::carbon::Dense so user
   * code cannot bypass @ref Dense::newField (which is responsible for sizing buffers from
   * @c spans()[i].linear_storage_elements() and binding the @ref Partition vector consistently).
   *
   * @p partitions must already be fully bound by the caller: each @ref Partition should carry a
   * pointer into the matching buffer. @ref Dense::newField is currently the only caller.
   *
   * @param dense        Grid layout this field is defined on (moved into the field).
   * @param gpu_buffers  One device buffer per GPU, typically from @ref GpuGroup::buffer. May be
   *                     empty (default-constructed) when no device storage has been allocated.
   * @param cpu_buffers  One host buffer per GPU (same indexing as @p gpu_buffers). May be empty
   *                     (default-constructed) when no host mirror has been allocated.
   * @param partitions       One @ref Partition per GPU, already bound by the caller against
   *                         the side that hosts the data (device when present, otherwise
   *                         host) — drives @ref partitions / @c loader(field) for device
   *                         kernels.
   * @param host_partitions  One @ref Partition per GPU bound to the matching @c cpu_buffers
   *                         entry, OR an empty vector when no host mirror was allocated.
   *                         Drives @ref host_partitions / OpenMP host kernels.
   */
  Field(Dense dense,
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

  friend class ::carbon::Dense;

  /**
   * @brief Shared, lazily-initialisable container for the host (CPU) mirror.
   *
   * Held by @c shared_ptr from @ref Field so all shallow copies of a field point at
   * the same instance — copying a @c HostDevice field into a @c Halo and triggering
   * the host alloc through either copy fills the @b same @c cpu_buffers and
   * @c host_partitions , and a re-trigger from another copy short-circuits through
   * the shared @c init_flag . Mirrors carbon's "shared shallow handle" pattern used
   * by @c BufferGroup and @c GpuGroup .
   *
   * For non- @c HostDevice fields the struct is still allocated (so accessors return
   * a valid reference even on Device-only fields) but stays empty — the once_flag is
   * never tripped because @c ensure_host_storage_ short-circuits on @c side_ .
   */
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

  /**
   * @brief Idempotent on-demand allocation of the host mirror for
   *        @c MemorySide::HostDevice fields.
   *
   * Called from every host-side accessor ( @c cpu_buffers , @c host_partitions ,
   * @c fill_owned_host ) so the user does not have to remember to opt in. Guarded by
   * the shared @c HostStorage::init_flag so concurrent first-accessors race safely
   * and only one wins.
   *
   * No-op for:
   * - @c MemorySide::Device — host mirror is intentionally absent (returned
   *   @c cpu_buffers stays empty).
   * - @c MemorySide::Host — host mirror is allocated eagerly in @c Dense::newField
   *   and the once_flag is never tripped.
   * - Default-constructed fields — @c host_storage_ is null and accessors short-
   *   circuit before reaching @c HostStorage .
   * - Empty layouts ( @c dense_.gpu_count() == 0 ).
   *
   * The allocation mirrors @c Dense::newField 's host-side path verbatim: per-GPU
   * @c Cpu::buffer<T> sized from @c spans()[i].linear_storage_elements() *
   * @c Components , then a parallel @c Partition<T, Components> vector bound to the
   * matching CPU pointer.
   */
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
        const size_t per_component = static_cast<size_t>(spans[i].linear_storage_elements());
        return ::carbon::Cpu::buffer<T>(per_component * static_cast<size_t>(Components));
      });
      storage.cpu_buffers = ::carbon::detail::BufferGroup<T>::from_vector(std::move(host_allocs));
      storage.host_partitions = dense_.system().make_vector([&storage, this](size_t i) {
        const auto& sp = dense_.spans()[i];
        return ::carbon::detail::dense::Partition<T, Components>(
            storage.cpu_buffers[i].data(), ::carbon::DataView::Standard, sp.origin(), sp.spacing());
      });
    });
  }

  Dense dense_;
  ::carbon::Buffer<T> gpu_buffers_{};
  std::vector<Partition<T, Components>> partitions_{};
  ::carbon::MemorySide side_{::carbon::MemorySide::Device};
  // Diagnostic-only label — see @ref name / @ref set_name . Plain by-value string;
  // copies/moves alongside the rest of the field, so a Field passed by value into
  // a Halo or captured into a Kernel takes its name with it.
  std::string name_{};
  // Shared host-mirror state — see @ref HostStorage . Always non-null (allocated by
  // both the default ctor and the friended ctor) so accessors never need a null
  // check; for @c MemorySide::Device it stays default-empty.
  mutable std::shared_ptr<HostStorage> host_storage_{};
};

}  // namespace carbon::detail::dense
