// io_nanovdb.inl — nanoVDB backend, split out of <carbon/io.h>. See
// io_hdf5.inl for the include-context contract; this file is a tail include
// of io.h, not a standalone header.
//
// Gated on CARBON_HAVE_NANOVDB — Carbon must have been built with
// CARBON_ENABLE_NANOVDB=ON for these declarations to be visible.

#ifdef CARBON_HAVE_NANOVDB

namespace detail {

/**
 * Opaque per-file nanoVDB session handle. nanoVDB doesn't natively model "multiple grids
 * with arbitrary per-cell component counts" the way HDF5 does; carbon's mapping is one
 * nanoVDB grid per @c (field, component) pair, all written into a single multi-grid
 * @c .nvdb file (one such file per rank in MPI mode).
 *
 * The PIMPL keeps every nanoVDB header out of @c <carbon/io.h>; the implementation lives
 * in @c src/io_nanovdb.cpp.
 */
class CARBON_API NanoVdbSession {
public:
  NanoVdbSession() noexcept;
  ~NanoVdbSession();
  NanoVdbSession(const NanoVdbSession&) = delete;
  auto
  operator=(const NanoVdbSession&) -> NanoVdbSession& = delete;
  NanoVdbSession(NanoVdbSession&&) noexcept;
  auto
  operator=(NanoVdbSession&&) noexcept -> NanoVdbSession&;

  void
  open_for_write(std::string_view path,
                 const Dense& dense,
                 std::uint32_t num_fields);
  void
  open_for_read(std::string_view path,
                const Dense& dense);

  void
  write_field(std::string_view name,
              const char* dtype_tag,
              size_t element_size,
              int components,
              const std::vector<const void*>& slab_owned_starts,
              const std::vector<int64_t>& slab_owned_z_extents,
              const std::vector<int64_t>& slab_pitches,
              const std::vector<int64_t>& slab_per_comp_strides,
              int64_t slab_global_z_start);
  void
  read_field(std::string_view name,
             const char* dtype_tag,
             size_t element_size,
             int components,
             const std::vector<void*>& slab_owned_starts,
             const std::vector<int64_t>& slab_owned_z_extents,
             const std::vector<int64_t>& slab_pitches,
             const std::vector<int64_t>& slab_per_comp_strides,
             int64_t slab_global_z_start);

  /**
   * Flush the per-field grid handles accumulated by @ref write_field to disk as a single
   * multi-grid @c .nvdb file. Called explicitly from @ref write_nanovdb after the last
   * field has been packed; safe to call exactly once on a write session and a no-op on a
   * read session. Throws on I/O failure; the destructor does not retry.
   */
  void
  close_write();

  [[nodiscard]] auto
  num_fields() const noexcept -> std::uint32_t;

  /**
   * Append @c .rankN to @p base when MPI is active; nanoVDB's on-disk grid container does
   * not have a parallel writer the way HDF5 does, so multi-rank writes use one file per
   * rank rather than collective I/O.
   */
  [[nodiscard]] static auto
  shard_path(std::string_view base,
             const Dense& dense) -> std::string;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace detail

/**
 * Write @p fields to a nanoVDB grid container at @p path. Each field becomes one grid
 * named @c "field_<i>" for scalar fields, or @c Components grids named
 * @c "field_<i>_c<c>" for multi-component fields (nanoVDB has no native arbitrary-N
 * vector grid; component splitting keeps every supported field type round-trippable).
 *
 * Distributed (MPI): one @c .nvdb file per rank, with @c .rank<N> appended to @p path.
 * Unlike HDF5 there is no parallel nanoVDB writer; rank-local containers are the natural
 * unit. A future post-processing step can merge the per-rank files if a single global
 * container is wanted.
 *
 * @note Today this dispatches to a stub backend that throws @c std::runtime_error: the
 *       API and CMake plumbing are committed so consumer code can be ported, but the
 *       grid-build / file-write path requires a full nanoVDB install (the OpenVDB
 *       distribution provides @c nanovdb/tools/CreateNanoGrid.h plus
 *       @c nanovdb/io/IO.h) which we don't bundle. Once a verified nanoVDB-enabled
 *       build environment is available the stub will be replaced; the API surface here
 *       is stable.
 *
 * @throws std::invalid_argument when a field has no host mirror or its element type is
 *         unsupported by nanoVDB.
 * @throws std::runtime_error from the current stub backend (see note above).
 */
template <typename... Fields>
void
write_nanovdb(std::string_view path,
              const Dense& dense,
              const Fields&... fields) {
  detail::NanoVdbSession sess;
  sess.open_for_write(path, dense, static_cast<std::uint32_t>(sizeof...(Fields)));
  size_t i = 0;
  (
      [&] {
        using FieldT = std::remove_cv_t<std::remove_reference_t<decltype(fields)>>;
        using ScalarT = typename FieldT::value_type;
        constexpr int kComponents = FieldT::components;
        std::vector<const void*> starts;
        std::vector<int64_t> z_extents;
        std::vector<int64_t> pitches;
        std::vector<int64_t> per_comps;
        detail::collect_field_views<ScalarT, kComponents>(dense, fields, starts, z_extents, pitches,
                                                          per_comps, "io::write_nanovdb");
        const int64_t z_start =
            dense.spans().empty()
                ? 0
                : (dense.spans().front().z_offset() + dense.spans().front().halo_z());
        sess.write_field("field_" + std::to_string(i), detail::native_dtype_tag<ScalarT>(),
                         sizeof(ScalarT), kComponents, starts, z_extents, pitches, per_comps,
                         z_start);
        ++i;
      }(),
      ...);
  sess.close_write();
}

/**
 * Inverse of @ref write_nanovdb. Field types and order must match the writer's; populates
 * the host mirror, with the caller responsible for any subsequent device upload.
 */
template <typename... Fields>
void
read_nanovdb(std::string_view path,
             const Dense& dense,
             Fields&... fields) {
  detail::NanoVdbSession sess;
  sess.open_for_read(path, dense);
  if (sess.num_fields() != sizeof...(Fields)) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_nanovdb: file '{}' has {} fields, but {} were passed to read", path,
        sess.num_fields(), sizeof...(Fields));
  }
  size_t i = 0;
  (
      [&] {
        using FieldT = std::remove_cv_t<std::remove_reference_t<decltype(fields)>>;
        using ScalarT = typename FieldT::value_type;
        constexpr int kComponents = FieldT::components;
        std::vector<void*> starts;
        std::vector<int64_t> z_extents;
        std::vector<int64_t> pitches;
        std::vector<int64_t> per_comps;
        detail::collect_field_views_mut<ScalarT, kComponents>(
            dense, fields, starts, z_extents, pitches, per_comps, "io::read_nanovdb");
        const int64_t z_start =
            dense.spans().empty()
                ? 0
                : (dense.spans().front().z_offset() + dense.spans().front().halo_z());
        sess.read_field("field_" + std::to_string(i), detail::native_dtype_tag<ScalarT>(),
                        sizeof(ScalarT), kComponents, starts, z_extents, pitches, per_comps,
                        z_start);
        ++i;
      }(),
      ...);
}

#endif  // CARBON_HAVE_NANOVDB
