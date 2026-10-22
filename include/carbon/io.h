#pragma once

#include <carbon/dense.h>
#include <carbon/dense_disg.h>
#include <carbon/detail/dense/field.h>
#include <carbon/detail/dense_disg/field.h>
#include <carbon/export.h>
#include <carbon/logging.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file io.h
 * @brief Free-function I/O surface for carbon grids + @c Field.
 *
 * Native binary works on @c Dense and @c DenseDisg (Cartesian, z-only partitioned grid
 * families); call shape is identical across the two:
 *
 * @code
 *   carbon::io::write_native(path, grid, field_a, field_b, ...);
 *   carbon::io::read_native (path, grid, field_a, field_b, ...);
 *   // VTK / HDF5 / nanoVDB are Dense-only today.
 * @endcode
 *
 * Both @c Dense and @c DenseDisg share the same on-disk file format: the file header
 * records @c global_extent / @c local_slab_offset / @c local_slab_count / @c
 * local_owned_extent (all family-agnostic; carbon's z-only partitioning gives uniform x/y
 * across slabs), and the payload is component-outer, slab-then-z-then-y row-major over
 * the owned-cell volume. The only family-specific bit is per-row addressing inside the
 * field's host mirror — Dense uses contiguous z-major per slab; DenseDisg uses its
 * three-region (front-card / body / back-card) layout. Both routes go through
 * @c detail::row_ptr / @c detail::row_ptr_mut overloads. As a happy consequence a Dense-
 * written shard can be read into a DenseDisg field of matching @c global_extent and vice
 * versa — useful when porting between layouts.
 *
 * Distributed runs (`CARBON_ENABLE_MPI=ON` and `grid.system().distributed() != nullptr`)
 * use one shard per rank. Native binary appends `.rankN` to the user-supplied path so the
 * same call works across ranks. HDF5 uses a single file via pHDF5 collective writes;
 * nanoVDB uses one file per rank.
 *
 * Today's implementation is synchronous host-side I/O: the field's @c cpu_buffers must
 * already hold the data the user wants to write, and after a read the host mirror is
 * populated. Push to / pull from the device with @c carbon::Transfer<T> outside the I/O
 * call.
 */

namespace carbon::io {

namespace detail {

/**
 * 8-character dtype tag (NUL-padded) identifying a scalar element type in a native binary
 * file. Returned as a literal pointer to avoid TU-local statics; the storage is in the
 * binary's read-only data section.
 *
 * Supported types match the carbon::Field types we expect to round-trip: signed and
 * unsigned 8/16/32/64-bit integers plus 32/64-bit float.
 */
template <typename T>
[[nodiscard]] constexpr auto
native_dtype_tag() noexcept -> const char*;

/**
 * File header of a single-shard native binary file (one shard == one rank's local slabs in
 * MPI mode, or the whole grid in single-process mode). All fields little-endian on disk;
 * @c endianness_check carries a sentinel that the reader inspects to detect a mismatch.
 *
 * @c local_slab_offset / @c local_slab_count describe which global slab range this shard
 * covers; @c local_owned_extent is the @c (x, y, sum-of-owned-z) volume actually written
 * for every field in this shard.
 */
struct NativeFileHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t endianness_check;
  int64_t global_extent[3];
  std::uint64_t local_slab_offset;
  std::uint64_t local_slab_count;
  int64_t local_owned_extent[3];
  std::uint32_t num_fields;
  std::uint32_t reserved;
};
static_assert(sizeof(NativeFileHeader) == 88,
              "NativeFileHeader layout unexpected");

inline constexpr std::uint32_t kNativeVersion = 1U;
inline constexpr std::uint32_t kEndiannessCheck = 0x12345678U;
inline constexpr char kNativeMagic[8] = {'C', 'A', 'R', 'B', 'n', 'B', '0', '1'};

/**
 * Per-field metadata block, immediately following the file header for the first field and
 * then preceding each subsequent field's payload. Payload size is implicit:
 * @c components * x * y * z elements of the per-field dtype, with @c (x, y, z) drawn from
 * the file header's @c local_owned_extent.
 */
struct NativeFieldMeta {
  std::string name;
  char dtype[9];  // 8 + NUL terminator
  std::int32_t components;
};

/**
 * Append @c .rankN to @p base when @c grid.system().distributed() is non-null; otherwise
 * return @p base unchanged. The reader uses the same routine to find shards, so users
 * never see the per-rank suffix. Overloaded per-grid because @c distributed() lives on
 * the grid's @c System and the two grid families don't share a base class — the overload
 * shape is also the dispatch surface keeping the family bookkeeping out of the
 * @c write_native / @c read_native templates below.
 */
[[nodiscard]] CARBON_API auto
shard_path(std::string_view base,
           const Dense& dense) -> std::string;

[[nodiscard]] CARBON_API auto
shard_path(std::string_view base,
           const DenseDisg& dense_disg) -> std::string;

CARBON_API void
write_native_header(std::ofstream& out,
                    const NativeFileHeader& hdr);

[[nodiscard]] CARBON_API auto
read_native_header(std::ifstream& in) -> NativeFileHeader;

CARBON_API void
write_native_field_meta(std::ofstream& out,
                        std::string_view name,
                        const char* dtype_tag,
                        std::int32_t components);

[[nodiscard]] CARBON_API auto
read_native_field_meta(std::ifstream& in) -> NativeFieldMeta;

/**
 * Throw with a clear message if @p got differs from @p want; used by header validation
 * (magic / version / endianness / global extent compatibility on the read path).
 */
CARBON_API void
require(bool cond,
        std::string_view what);

/**
 * Per-row offset into a Dense field's host mirror. The Dense @c Span uses a contiguous
 * z-major layout per slab — every component plane is @c span.linear_storage_elements()
 * elements long, every z layer is @c span.xy_slab_elements() elements long, every y row
 * is @c haloed_extent.x elements long. Used by @ref write_native_field_payload /
 * @ref read_native_field_payload to compute the start of one (c, z_in_buffer, y) row.
 */
template <typename T>
[[nodiscard]] inline auto
row_offset(const Dense::Span& span,
           int c,
           int64_t z_buf,
           int64_t y) noexcept -> int64_t {
  const auto haloed = span.haloed_extent();
  return static_cast<int64_t>(c) * span.linear_storage_elements() +
         z_buf * span.xy_slab_elements() + y * haloed.x;
}

/**
 * Per-row offset into a DenseDisg field's host mirror. Per-slab buffer is split into
 * three z-regions: the front-card (buffer-z 0), front body (buffer-z 1), main body
 * (buffer-z 2..lz-1, packed contiguously) and back card (buffer-z lz, lz+1). Inside each
 * region every component is laid out as a packed plane of @c xy_slab_elements *
 * region_z_dim elements, so the row offset is the standard
 * @c (intra-region-z * xy_slab_elements + y * haloed.x) plus the region's component
 * plane base. The same indexing is used by @c DenseDisg::Span::linear_offset and by the
 * test fixtures (e.g. @c tests/test_dense_disg_overlapped.cu ) — cloning it here keeps
 * the I/O path bit-equal with the kernel-side layout.
 */
template <typename T>
[[nodiscard]] inline auto
row_offset(const DenseDisg::Span& span,
           int c,
           int64_t z_buf,
           int64_t y,
           int components) noexcept -> int64_t {
  const auto haloed = span.haloed_extent();
  const int64_t lz = span.logic_extent().z;
  const int64_t xy = span.xy_slab_elements();
  int64_t region_first_z = 0;
  int64_t region_z_dim = 1;
  int64_t local_no_card = y * haloed.x;
  if (z_buf <= 1 || z_buf >= lz) {
    region_first_z = z_buf;
    region_z_dim = 1;
  } else {
    region_first_z = 2;
    region_z_dim = lz - 2;
    local_no_card += (z_buf - 2) * xy;
  }
  return local_no_card + xy * (region_first_z * static_cast<int64_t>(components) +
                               static_cast<int64_t>(c) * region_z_dim);
}

/**
 * Stream the owned-cell payload of @p field to @p out , SoA per-component layout: for
 * each component @c c in @c [0, Components) , iterate every local slab in order and
 * stream its owned z layers contiguously (z outermost, then y, then x). The per-row
 * pointer math is delegated to the @ref row_offset overload selected by the field's
 * @c Grid 's @c Span — that's the single point where the family's storage layout enters
 * the I/O path. Reads from @p field.cpu_buffers() ; the user must have already
 * downloaded device data.
 *
 * @c FieldT is generic so @c T and @c Components can be deduced from the field directly
 * (@c Grid::Field<T, Components> is a non-deduced context — explicit deduction on a free
 * @c FieldT is the cleaner shape here).
 */
template <typename Grid,
          typename FieldT>
void
write_native_field_payload(std::ofstream& out,
                           const Grid& grid,
                           const FieldT& field) {
  using T = typename FieldT::value_type;
  constexpr int kComponents = FieldT::components;
  const auto& spans = grid.spans();
  const size_t n_gpus = spans.size();
  if (n_gpus == 0U) {
    return;
  }
  if (field.cpu_buffers().size() != n_gpus) {
    log::throw_with_log<std::invalid_argument>(
        "io::write_native: field has no host mirror (cpu_buffers().size()={}, gpu_count={}); "
        "construct the field with MemorySide::Host or HostDevice and download from device "
        "before writing",
        field.cpu_buffers().size(), n_gpus);
  }

  for (int c = 0; c < kComponents; ++c) {
    for (size_t g = 0; g < n_gpus; ++g) {
      const auto& span = spans[g];
      const auto haloed = span.haloed_extent();
      const int64_t halo_z = span.halo_z();
      const int64_t lim_z = span.logic_extent().z;
      const T* h = field.cpu_buffers()[g].data();
      const size_t row_bytes = static_cast<size_t>(haloed.x) * sizeof(T);
      for (int64_t z = halo_z; z < halo_z + lim_z; ++z) {
        for (int64_t y = 0; y < haloed.y; ++y) {
          int64_t off = 0;
          if constexpr (std::is_same_v<Grid, ::carbon::Dense>) {
            off = row_offset<T>(span, c, z, y);
          } else {
            // DenseDisg path threads the per-slab component count through so the helper
            // can compute the region's component plane base without re-deriving
            // Components.
            off = row_offset<T>(span, c, z, y, kComponents);
          }
          out.write(reinterpret_cast<const char*>(h + off),
                    static_cast<std::streamsize>(row_bytes));
        }
      }
    }
  }
}

/**
 * Inverse of @ref write_native_field_payload : read the SoA owned payload from @p in and
 * scatter into @p field.cpu_buffers() . The field must already be allocated on the same
 * @p grid layout (the writer's @c local_owned_extent is checked against the grid's
 * spans by the caller).
 */
template <typename Grid,
          typename FieldT>
void
read_native_field_payload(std::ifstream& in,
                          const Grid& grid,
                          FieldT& field) {
  using T = typename FieldT::value_type;
  constexpr int kComponents = FieldT::components;
  const auto& spans = grid.spans();
  const size_t n_gpus = spans.size();
  if (n_gpus == 0U) {
    return;
  }
  if (field.cpu_buffers().size() != n_gpus) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_native: field has no host mirror (cpu_buffers().size()={}, gpu_count={}); "
        "construct the field with MemorySide::Host or HostDevice before reading",
        field.cpu_buffers().size(), n_gpus);
  }

  for (int c = 0; c < kComponents; ++c) {
    for (size_t g = 0; g < n_gpus; ++g) {
      const auto& span = spans[g];
      const auto haloed = span.haloed_extent();
      const int64_t halo_z = span.halo_z();
      const int64_t lim_z = span.logic_extent().z;
      T* h = field.cpu_buffers()[g].data();
      const size_t row_bytes = static_cast<size_t>(haloed.x) * sizeof(T);
      for (int64_t z = halo_z; z < halo_z + lim_z; ++z) {
        for (int64_t y = 0; y < haloed.y; ++y) {
          int64_t off = 0;
          if constexpr (std::is_same_v<Grid, ::carbon::Dense>) {
            off = row_offset<T>(span, c, z, y);
          } else {
            off = row_offset<T>(span, c, z, y, kComponents);
          }
          in.read(reinterpret_cast<char*>(h + off), static_cast<std::streamsize>(row_bytes));
        }
      }
      // Once a read comes up short, failbit latches and every later read is a no-op that
      // leaves its destination rows untouched. Unchecked, a truncated shard therefore loads
      // as a mixture of file data and whatever the host mirror already held, and the restart
      // proceeds on it. Check per (component, slab) so the diagnostic names the region that
      // ran out rather than just the file.
      if (!in) {
        log::throw_with_log<std::runtime_error>(
            "io::read_native: short read in payload for component {} of {}, slab {} of {} "
            "(expected {} rows of {} bytes); the file is truncated or does not match this "
            "grid's layout",
            c, kComponents, g, n_gpus, lim_z * haloed.y, row_bytes);
      }
    }
  }
}

/**
 * Build a populated @c NativeFileHeader from a Cartesian grid, ready to write to disk.
 * The @c num_fields field must be filled in by the caller (one of the few values not
 * derivable from the grid alone). Overloaded per-family because every field carbon
 * exposes through the I/O surface speaks @c global_extent / @c local_slab_offset /
 * @c gpu_count via concrete grid types rather than a shared base class.
 */
[[nodiscard]] CARBON_API auto
make_native_header(const Dense& dense,
                   std::uint32_t num_fields) -> NativeFileHeader;

[[nodiscard]] CARBON_API auto
make_native_header(const DenseDisg& dense_disg,
                   std::uint32_t num_fields) -> NativeFileHeader;

/**
 * Validate a freshly read header against the in-memory grid the caller intends to load
 * into: magic / version / endianness / global extent / local slab range / local owned
 * extent must all match. Both Cartesian families share the same on-disk format, so the
 * Dense and DenseDisg overloads share the same validation shape and a Dense-written
 * shard can be read into a DenseDisg field of matching @c global_extent (and vice versa).
 */
CARBON_API void
validate_native_header(const NativeFileHeader& hdr,
                       const Dense& dense);

CARBON_API void
validate_native_header(const NativeFileHeader& hdr,
                       const DenseDisg& dense_disg);

}  // namespace detail

/**
 * Write @p fields into a native binary file (or one shard per rank under MPI). Each
 * field's @c cpu_buffers must already hold the data to persist; this is host-side I/O.
 *
 * The @p Grid template parameter accepts @c carbon::Dense or @c carbon::DenseDisg ; both
 * Cartesian families share the same on-disk format (header + per-field meta + SoA owned
 * payload) so a Dense-written shard can be read into a DenseDisg field of matching
 * @c global_extent and vice versa.
 *
 * @param path    Output file base path. In single-process mode the file is written
 *                verbatim; in MPI mode @c .rank<world_rank> is appended.
 * @param grid    Layout the fields live on. Must be identical for every field in the pack.
 * @param fields  One or more @c Grid::Field instances. Each may have its own @c T and
 *                @c Components; types are stored in the per-field metadata so a later
 *                @ref read_native call can validate them.
 *
 * @throws std::invalid_argument when a field is missing its host mirror, when fields have
 *         mismatched layouts, or when the file cannot be opened for writing.
 */
template <typename Grid,
          typename... Fields>
void
write_native(std::string_view path,
             const Grid& grid,
             const Fields&... fields) {
  const std::string shard = detail::shard_path(path, grid);
  std::ofstream out(shard, std::ios::binary);
  if (!out) {
    log::throw_with_log<std::runtime_error>("io::write_native: failed to open '{}' for writing",
                                            shard);
  }

  const detail::NativeFileHeader hdr =
      detail::make_native_header(grid, static_cast<std::uint32_t>(sizeof...(Fields)));
  detail::write_native_header(out, hdr);

  size_t field_index = 0;
  // Fold expression: per field, write meta then payload. The Field type drives the dtype
  // tag and component count, so every field can have its own (T, Components). Template
  // arguments to write_native_field_payload are fully deducible from the (grid, field)
  // pair.
  (
      [&] {
        using FieldT = std::remove_cv_t<std::remove_reference_t<decltype(fields)>>;
        using ScalarT = typename FieldT::value_type;
        constexpr int kComponents = FieldT::components;
        const std::string name = "field_" + std::to_string(field_index++);
        detail::write_native_field_meta(out, name, detail::native_dtype_tag<ScalarT>(),
                                        static_cast<std::int32_t>(kComponents));
        detail::write_native_field_payload(out, grid, fields);
      }(),
      ...);

  // Flush before testing rather than leaving it to the destructor: a full filesystem typically
  // surfaces at the flush of the last buffered block, and an ofstream that fails while being
  // destroyed has no way to report it. Without this, write_native could return successfully
  // having written a partial shard.
  out.flush();
  if (!out.good()) {
    log::throw_with_log<std::runtime_error>("io::write_native: write to '{}' failed", shard);
  }
}

/**
 * Inverse of @ref write_native : load the per-field payloads from @p path into @p fields .
 * Field types and order must match the writer's. The fields' host mirrors are populated;
 * the caller is responsible for any subsequent device upload.
 *
 * @throws std::invalid_argument when the header doesn't match @p grid (magic / version /
 *         extent / slab range), when @c num_fields differs from @c sizeof...(Fields) , or
 *         when a per-field dtype / component count mismatches the bound @c Field type.
 * @throws std::runtime_error when the file cannot be opened or a read short-circuits.
 */
template <typename Grid,
          typename... Fields>
void
read_native(std::string_view path,
            const Grid& grid,
            Fields&... fields) {
  const std::string shard = detail::shard_path(path, grid);
  std::ifstream in(shard, std::ios::binary);
  if (!in) {
    log::throw_with_log<std::runtime_error>("io::read_native: failed to open '{}' for reading",
                                            shard);
  }

  const detail::NativeFileHeader hdr = detail::read_native_header(in);
  detail::validate_native_header(hdr, grid);
  if (hdr.num_fields != sizeof...(Fields)) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_native: file '{}' has {} fields, but {} were passed to read", shard,
        hdr.num_fields, sizeof...(Fields));
  }

  size_t field_index = 0;
  (
      [&] {
        using FieldT = std::remove_cv_t<std::remove_reference_t<decltype(fields)>>;
        using ScalarT = typename FieldT::value_type;
        constexpr int kComponents = FieldT::components;
        const detail::NativeFieldMeta meta = detail::read_native_field_meta(in);
        const char* expected_tag = detail::native_dtype_tag<ScalarT>();
        if (std::string_view(meta.dtype) != std::string_view(expected_tag)) {
          log::throw_with_log<std::invalid_argument>(
              "io::read_native: field {} dtype '{}' does not match bound Field's dtype '{}'",
              field_index, meta.dtype, expected_tag);
        }
        if (meta.components != kComponents) {
          log::throw_with_log<std::invalid_argument>(
              "io::read_native: field {} components {} does not match bound Field's "
              "Components {}",
              field_index, meta.components, kComponents);
        }
        ++field_index;
        detail::read_native_field_payload(in, grid, fields);
      }(),
      ...);

  // The writer emits nothing after the last payload, so anything left means this file was not
  // produced by write_native for this (grid, field-list) pair — a field-order or component-count
  // difference the per-field dtype checks happen not to catch, for instance. Say so rather than
  // returning a field set that was filled from the wrong offsets.
  if (in.peek() != std::ifstream::traits_type::eof()) {
    log::throw_with_log<std::runtime_error>(
        "io::read_native: '{}' has trailing bytes after the {} field payload(s) read; the file "
        "does not match the field list passed to read",
        shard, sizeof...(Fields));
  }
}

namespace detail {

/**
 * Pack the per-slab data pointers / extents from a @p field into the parallel arrays
 * the HDF5 / nanoVDB backends expect. Owned region only: the start pointer skips the
 * lower halo; pitch and per-component stride come straight off the @c Span. The const
 * variant is consumed by writers, the mutable variant by readers.
 */
template <typename T,
          int Components>
void
collect_field_views(const Dense& dense,
                    const Dense::Field<T,
                                       Components>& field,
                    std::vector<const void*>& starts,
                    std::vector<int64_t>& z_extents,
                    std::vector<int64_t>& pitches,
                    std::vector<int64_t>& per_comps,
                    std::string_view callsite) {
  const auto& spans = dense.spans();
  const size_t n_gpus = spans.size();
  if (field.cpu_buffers().size() != n_gpus) {
    log::throw_with_log<std::invalid_argument>(
        "{}: field has no host mirror (cpu_buffers().size()={}, gpu_count={}); "
        "construct the field with MemorySide::Host or HostDevice and download from device "
        "before writing",
        callsite, field.cpu_buffers().size(), n_gpus);
  }
  starts.resize(n_gpus);
  z_extents.resize(n_gpus);
  pitches.resize(n_gpus);
  per_comps.resize(n_gpus);
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = spans[g];
    const int64_t halo_z = span.halo_z();
    const int64_t pitch = span.xy_slab_elements();
    const T* base = field.cpu_buffers()[g].data();
    starts[g] = static_cast<const void*>(base + halo_z * pitch);
    z_extents[g] = span.logic_extent().z;
    pitches[g] = pitch;
    per_comps[g] = span.linear_storage_elements();
  }
}

template <typename T,
          int Components>
void
collect_field_views_mut(const Dense& dense,
                        Dense::Field<T,
                                     Components>& field,
                        std::vector<void*>& starts,
                        std::vector<int64_t>& z_extents,
                        std::vector<int64_t>& pitches,
                        std::vector<int64_t>& per_comps,
                        std::string_view callsite) {
  const auto& spans = dense.spans();
  const size_t n_gpus = spans.size();
  if (field.cpu_buffers().size() != n_gpus) {
    log::throw_with_log<std::invalid_argument>(
        "{}: field has no host mirror (cpu_buffers().size()={}, gpu_count={}); "
        "construct the field with MemorySide::Host or HostDevice before reading",
        callsite, field.cpu_buffers().size(), n_gpus);
  }
  starts.resize(n_gpus);
  z_extents.resize(n_gpus);
  pitches.resize(n_gpus);
  per_comps.resize(n_gpus);
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = spans[g];
    const int64_t halo_z = span.halo_z();
    const int64_t pitch = span.xy_slab_elements();
    T* base = field.cpu_buffers()[g].data();
    starts[g] = static_cast<void*>(base + halo_z * pitch);
    z_extents[g] = span.logic_extent().z;
    pitches[g] = pitch;
    per_comps[g] = span.linear_storage_elements();
  }
}

}  // namespace detail

// ============================================================================
// Backend-specific format surfaces. Each .inl is tail-included here so io.h
// stays focused on the cross-format shared helpers (collect_field_views /
// native_dtype_tag) and the native-binary path. The .inl files are not
// standalone headers — see the file header at the top of each for the
// include-context contract.
// ============================================================================

#include <carbon/detail/io_hdf5.inl>
#include <carbon/detail/io_nanovdb.inl>
#include <carbon/detail/io_vtk.inl>

namespace detail {

// dtype_tag specializations: kept inline for compactness. Each tag is exactly 8 chars.
// clang-format off
template <> [[nodiscard]] constexpr auto native_dtype_tag<float>()         noexcept -> const char* { return "f32\0\0\0\0"; }
template <> [[nodiscard]] constexpr auto native_dtype_tag<double>()        noexcept -> const char* { return "f64\0\0\0\0"; }
template <> [[nodiscard]] constexpr auto native_dtype_tag<std::int8_t>()   noexcept -> const char* { return "i8\0\0\0\0\0"; }
template <> [[nodiscard]] constexpr auto native_dtype_tag<std::int16_t>()  noexcept -> const char* { return "i16\0\0\0\0"; }
template <> [[nodiscard]] constexpr auto native_dtype_tag<std::int32_t>()  noexcept -> const char* { return "i32\0\0\0\0"; }
template <> [[nodiscard]] constexpr auto native_dtype_tag<int64_t>()  noexcept -> const char* { return "i64\0\0\0\0"; }
template <> [[nodiscard]] constexpr auto native_dtype_tag<std::uint8_t>()  noexcept -> const char* { return "u8\0\0\0\0\0"; }
template <> [[nodiscard]] constexpr auto native_dtype_tag<std::uint16_t>() noexcept -> const char* { return "u16\0\0\0\0"; }
template <> [[nodiscard]] constexpr auto native_dtype_tag<std::uint32_t>() noexcept -> const char* { return "u32\0\0\0\0"; }
template <> [[nodiscard]] constexpr auto native_dtype_tag<std::uint64_t>() noexcept -> const char* { return "u64\0\0\0\0"; }
// clang-format on

}  // namespace detail

}  // namespace carbon::io
