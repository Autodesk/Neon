// io_vtk.inl — VTK legacy backend, split out of <carbon/io.h>. See
// io_hdf5.inl for the include-context contract; this file is a tail include
// of io.h, not a standalone header.
//
// VTK legacy is always available (no external dependency, pure stdlib) so
// this file has no #ifdef guard around it.
//
// Writes / reads carbon Dense fields as VTK legacy STRUCTURED_POINTS files (*.vtk),
// BINARY flavour. Format reference: VTK File Formats v3.0 (the legacy / non-XML one).
// One file per rank under MPI (`.rank<N>` suffix), like native binary and nanoVDB.
//
// On-disk shape per file:
//
//   # vtk DataFile Version 3.0
//   carbon dense field shard
//   BINARY
//   DATASET STRUCTURED_POINTS
//   DIMENSIONS <global.x> <global.y> <local_z_total>
//   ORIGIN 0 0 <first_local_global_z>
//   SPACING 1 1 1
//   POINT_DATA <product of DIMENSIONS>
//   SCALARS field_0 <vtk dtype name> <Components>
//   LOOKUP_TABLE default
//   <binary payload, big-endian, component-fastest interleaving>
//   SCALARS field_1 ...
//   ...
//
// The legacy spec mandates network byte order (big-endian) for binary payloads
// regardless of host endianness, so the implementation byte-swaps on LE hosts.
namespace detail {

/**
 * Opaque per-file VTK session handle (PIMPL). Holds the open stream + the cached
 * per-field offsets discovered on @ref open_for_read so @ref read_field can seek
 * to the right SCALARS section. Move-only: matches the carbon convention for the
 * other backend session handles.
 */
class CARBON_API VtkSession {
public:
  VtkSession() noexcept;
  ~VtkSession();
  VtkSession(const VtkSession&) = delete;
  auto
  operator=(const VtkSession&) -> VtkSession& = delete;
  VtkSession(VtkSession&&) noexcept;
  auto
  operator=(VtkSession&&) noexcept -> VtkSession&;

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
              const std::vector<int64_t>& slab_per_comp_strides);
  void
  read_field(std::string_view name,
             const char* dtype_tag,
             size_t element_size,
             int components,
             const std::vector<void*>& slab_owned_starts,
             const std::vector<int64_t>& slab_owned_z_extents,
             const std::vector<int64_t>& slab_pitches,
             const std::vector<int64_t>& slab_per_comp_strides);

  [[nodiscard]] auto
  num_fields() const noexcept -> std::uint32_t;

  [[nodiscard]] static auto
  shard_path(std::string_view base,
             const Dense& dense) -> std::string;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * VTK legacy dtype names (NUL-terminated). Returned as literal pointers; storage
 * is in the binary's read-only data section. Specialized below for every type
 * carbon's @c native_dtype_tag covers.
 */
template <typename T>
[[nodiscard]] constexpr auto
vtk_dtype_name() noexcept -> const char*;

}  // namespace detail

/**
 * Write @p fields to a VTK legacy STRUCTURED_POINTS BINARY file at @p path. Each
 * field becomes one @c SCALARS block with @c Components components under the
 * shared @c POINT_DATA section; binary payload is big-endian per spec, with
 * component-fastest interleaving (the conventional VTK layout).
 *
 * Distributed (MPI): one shard per rank with @c .rank<N> appended to @p path,
 * each shard self-contained (its DIMENSIONS / ORIGIN reflect that rank's local
 * slabs). VTK legacy has no parallel writer.
 */
template <typename... Fields>
void
write_vtk(std::string_view path,
          const Dense& dense,
          const Fields&... fields) {
  detail::VtkSession sess;
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
                                                          per_comps, "io::write_vtk");
        // VTK SCALARS name = the field's user-facing label when set (via
        // Dense::newField(..., name) or Field::set_name), else the legacy
        // "field_<i>" fallback so older callers see no behaviour change. The
        // SCALARS name is what ParaView / vtkmodules surface to the user, so
        // honouring Field::name() is the only sensible default.
        const std::string& field_name = fields.name();
        std::string vtk_name = field_name.empty() ? ("field_" + std::to_string(i)) : field_name;
        sess.write_field(vtk_name, detail::native_dtype_tag<ScalarT>(), sizeof(ScalarT),
                         kComponents, starts, z_extents, pitches, per_comps);
        ++i;
      }(),
      ...);
}

/**
 * Write a VTK multiblock master (@c .vtm XML) at @p base alongside the per-rank
 * @c .rank<N> legacy shards produced by @ref write_vtk. The master references
 * every shard by relative path so opening @c <base>.vtm in ParaView pulls all
 * ranks into one MultiBlockDataSet entry in the pipeline browser — a much
 * better UX than manually opening every @c .rank<N> file. The file is
 * deterministic and tiny (a few hundred bytes).
 *
 * Idempotent across ranks: every rank may call this; only rank 0 actually
 * writes, the others return immediately. Single-process callers (no
 * @c distributed() handle) are a no-op since there is nothing to combine —
 * the legacy @c .vtk file produced by @ref write_vtk already opens directly.
 *
 * @param base   Same path passed to @ref write_vtk. The master file is
 *               written at @c <base>.vtm; per-rank shard names are recorded
 *               relative to that path's parent directory so the @c .vtm and
 *               its referenced @c .rank<N> shards stay portable when moved
 *               together.
 * @param dense  The same @ref Dense passed to @ref write_vtk; provides the
 *               distributed handle this function uses to enumerate world
 *               size on rank 0.
 *
 * Note: ParaView's @c .pvti format would offer spatial merging across
 * pieces but requires the shards to be in the XML @c .vti format which
 * carbon does not currently emit; @c .vtm is the next-best generic
 * "load all pieces in one go" wrapper and works with the legacy shards.
 */
CARBON_API auto
write_vtk_multiblock_index(std::string_view base,
                           const Dense& dense) -> void;

/**
 * Inverse of @ref write_vtk. The reader requires the file's DIMENSIONS / ORIGIN to
 * match @p dense's local owned extent and z origin (per-rank under MPI), and each
 * @c SCALARS block to match the bound field's element type and component count by
 * position; field names are not used by the read path so writers from other tools
 * can be ingested as long as the type / component layout matches.
 */
template <typename... Fields>
void
read_vtk(std::string_view path,
         const Dense& dense,
         Fields&... fields) {
  detail::VtkSession sess;
  sess.open_for_read(path, dense);
  if (sess.num_fields() != sizeof...(Fields)) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: file '{}' has {} fields, but {} were passed to read", path,
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
                                                              pitches, per_comps, "io::read_vtk");
        sess.read_field("field_" + std::to_string(i), detail::native_dtype_tag<ScalarT>(),
                        sizeof(ScalarT), kComponents, starts, z_extents, pitches, per_comps);
        ++i;
      }(),
      ...);
}

namespace detail {

// VTK legacy dtype names per the format spec. Kept inline for compactness; one
// spec per fundamental type carbon's native_dtype_tag handles.
// clang-format off
template <> [[nodiscard]] constexpr auto vtk_dtype_name<float>()         noexcept -> const char* { return "float"; }
template <> [[nodiscard]] constexpr auto vtk_dtype_name<double>()        noexcept -> const char* { return "double"; }
template <> [[nodiscard]] constexpr auto vtk_dtype_name<std::int8_t>()   noexcept -> const char* { return "char"; }
template <> [[nodiscard]] constexpr auto vtk_dtype_name<std::uint8_t>()  noexcept -> const char* { return "unsigned_char"; }
template <> [[nodiscard]] constexpr auto vtk_dtype_name<std::int16_t>()  noexcept -> const char* { return "short"; }
template <> [[nodiscard]] constexpr auto vtk_dtype_name<std::uint16_t>() noexcept -> const char* { return "unsigned_short"; }
template <> [[nodiscard]] constexpr auto vtk_dtype_name<std::int32_t>()  noexcept -> const char* { return "int"; }
template <> [[nodiscard]] constexpr auto vtk_dtype_name<std::uint32_t>() noexcept -> const char* { return "unsigned_int"; }
template <> [[nodiscard]] constexpr auto vtk_dtype_name<int64_t>()  noexcept -> const char* { return "vtktypeint64"; }
template <> [[nodiscard]] constexpr auto vtk_dtype_name<std::uint64_t>() noexcept -> const char* { return "vtktypeuint64"; }
// clang-format on

}  // namespace detail
