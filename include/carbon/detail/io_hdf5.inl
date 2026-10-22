// io_hdf5.inl — HDF5 backend, split out of <carbon/io.h>. Tail-included from
// io.h after the shared detail::collect_field_views / native_dtype_tag
// machinery is in scope. Not a standalone header (no include guard, no own
// namespace open); only meaningful in the io.h tail-include context.
//
// Gated on CARBON_HAVE_HDF5 — Carbon must have been built with
// CARBON_ENABLE_HDF5=ON for these declarations to be visible.

#ifdef CARBON_HAVE_HDF5

namespace detail {

/**
 * Opaque per-file HDF5 session handle. The PIMPL keeps @c hid_t out of the public header
 * so consumer TUs never need @c <hdf5.h>; the per-rank file/property-list/dataset state
 * lives in @c src/io_hdf5.cpp.
 */
class CARBON_API Hdf5Session {
public:
  Hdf5Session() noexcept;
  ~Hdf5Session();
  Hdf5Session(const Hdf5Session&) = delete;
  auto
  operator=(const Hdf5Session&) -> Hdf5Session& = delete;
  Hdf5Session(Hdf5Session&&) noexcept;
  auto
  operator=(Hdf5Session&&) noexcept -> Hdf5Session&;

  void
  open_for_write(std::string_view path,
                 const Dense& dense,
                 std::uint32_t num_fields);
  void
  open_for_read(std::string_view path,
                const Dense& dense);

  /**
   * Write one field's payload into a freshly created dataset of shape
   * @c (Components, global.z, global.y, global.x) with type @p dtype_tag, by streaming the
   * owned-cell hyperslabs of every local slab on this rank. Under MPI the writes are
   * collective.
   */
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

  [[nodiscard]] auto
  num_fields() const noexcept -> std::uint32_t;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace detail

/**
 * Write @p fields into a single HDF5 file. Each field becomes a dataset named
 * @c "field_<i>" (zero-based) of rank 4 and shape
 * @c (Components, global.z, global.y, global.x), dtype matched to the field's element
 * type. Root-level attributes record the carbon @c global_extent and a format version.
 *
 * Distributed (MPI + parallel HDF5):
 *   - File is opened with @c H5Pset_fapl_mpio (MPI-IO driver).
 *   - Per field the dataset is created collectively; each rank writes the hyperslab for
 *     its local slabs with a collective transfer property list (@c H5FD_MPIO_COLLECTIVE).
 *   - The user-supplied @p path is written verbatim (one file across all ranks); no
 *     @c .rankN suffix is appended.
 *
 * Distributed (MPI but HDF5 is serial-only): throws @c std::runtime_error explaining the
 * build-time mismatch (rebuild HDF5 with parallel support, e.g. @c libhdf5-mpich-dev).
 *
 * Single-process: file opened with the default file access plist; same per-field shape.
 *
 * @throws std::invalid_argument when a field has no host mirror, when types are
 *         unsupported, or when the file cannot be created.
 */
template <typename... Fields>
void
write_hdf5(std::string_view path,
           const Dense& dense,
           const Fields&... fields) {
  detail::Hdf5Session sess;
  sess.open_for_write(path, dense, static_cast<std::uint32_t>(sizeof...(Fields)));
  size_t i = 0;
  // For each field: pack views, dispatch on its dtype tag.
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
                                                          per_comps, "io::write_hdf5");
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
}

/**
 * Inverse of @ref write_hdf5. Datasets are bound positionally; each field's @c T and
 * @c Components must match the writer's. Populates the host mirror; the caller uploads
 * to device as needed.
 */
template <typename... Fields>
void
read_hdf5(std::string_view path,
          const Dense& dense,
          Fields&... fields) {
  detail::Hdf5Session sess;
  sess.open_for_read(path, dense);
  if (sess.num_fields() != sizeof...(Fields)) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_hdf5: file '{}' has {} fields, but {} were passed to read", path,
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
        detail::collect_field_views_mut<ScalarT, kComponents>(dense, fields, starts, z_extents,
                                                              pitches, per_comps, "io::read_hdf5");
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

#endif  // CARBON_HAVE_HDF5
