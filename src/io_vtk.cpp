#include <carbon/dense.h>
#include <carbon/io.h>
#include <carbon/logging.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/distributed.h>
#endif

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <istream>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace carbon::io::detail {

[[nodiscard]] auto
shard_path(std::string_view base,
           const Dense& dense) -> std::string;

namespace {

/**
 * Byte-swap one element in place. Specialized on the element's machine size — the
 * VTK legacy spec mandates network byte order for binary blocks regardless of host
 * endianness, so writers always swap to BE on LE hosts and readers swap back the
 * same way. No-op when the host is already big-endian.
 */
template <size_t N>
inline void
byte_swap_in_place(void* /*p*/) noexcept {
  static_assert(N == 1 || N == 2 || N == 4 || N == 8,
                "byte_swap_in_place only supports element sizes 1/2/4/8");
}
template <>
inline void
byte_swap_in_place<1>(void*) noexcept {}
template <>
inline void
byte_swap_in_place<2>(void* p) noexcept {
  auto* b = static_cast<std::uint8_t*>(p);
  std::swap(b[0], b[1]);
}
template <>
inline void
byte_swap_in_place<4>(void* p) noexcept {
  auto* b = static_cast<std::uint8_t*>(p);
  std::swap(b[0], b[3]);
  std::swap(b[1], b[2]);
}
template <>
inline void
byte_swap_in_place<8>(void* p) noexcept {
  auto* b = static_cast<std::uint8_t*>(p);
  std::swap(b[0], b[7]);
  std::swap(b[1], b[6]);
  std::swap(b[2], b[5]);
  std::swap(b[3], b[4]);
}

[[nodiscard]] constexpr auto
host_is_little_endian() noexcept -> bool {
  return std::endian::native == std::endian::little;
}

void
maybe_swap_buffer(void* data,
                  size_t element_size,
                  size_t element_count) noexcept {
  if (!host_is_little_endian()) {
    return;
  }
  auto* p = static_cast<std::uint8_t*>(data);
  switch (element_size) {
    case 1:
      return;
    case 2:
      for (size_t i = 0; i < element_count; ++i) {
        byte_swap_in_place<2>(p + 2U * i);
      }
      return;
    case 4:
      for (size_t i = 0; i < element_count; ++i) {
        byte_swap_in_place<4>(p + 4U * i);
      }
      return;
    case 8:
      for (size_t i = 0; i < element_count; ++i) {
        byte_swap_in_place<8>(p + 8U * i);
      }
      return;
    default:
      return;
  }
}

/**
 * Map a carbon native dtype tag (8-char NUL-padded, e.g. @c "f32\0\0\0\0\0") to the
 * VTK legacy dtype keyword. Returned as a literal pointer; null on unknown tag so
 * the caller can throw a contextual error.
 */
[[nodiscard]] auto
vtk_keyword_from_native_tag(const char* tag) noexcept -> const char* {
  // 8-char prefix compare keeps the table compact; tags themselves are NUL-padded
  // out to 8 bytes by the carbon native_dtype_tag specializations.
  const std::string_view t(tag, 8);
  if (t.starts_with("f32"))
    return "float";
  if (t.starts_with("f64"))
    return "double";
  if (t.starts_with("i8"))
    return "char";
  if (t.starts_with("u8"))
    return "unsigned_char";
  if (t.starts_with("i16"))
    return "short";
  if (t.starts_with("u16"))
    return "unsigned_short";
  if (t.starts_with("i32"))
    return "int";
  if (t.starts_with("u32"))
    return "unsigned_int";
  if (t.starts_with("i64"))
    return "vtktypeint64";
  if (t.starts_with("u64"))
    return "vtktypeuint64";
  return nullptr;
}

/**
 * Stream a single field's owned cells out to @p out as a contiguous component-
 * fastest blob. Carbon stores data component-OUTER (per_comp stride between
 * components), so we have to gather each cell's @c Components values into a small
 * temp buffer before swapping + writing — that's the price of matching VTK's
 * AOS layout. A row-sized scratch buffer keeps the inner loop cache-friendly.
 */
void
stream_field_payload(std::ostream& out,
                     size_t element_size,
                     int components,
                     int64_t global_x,
                     const std::vector<const void*>& starts,
                     const std::vector<int64_t>& z_extents,
                     const std::vector<int64_t>& pitches,
                     const std::vector<int64_t>& per_comps) {
  std::vector<std::uint8_t> row_scratch(
      static_cast<size_t>(global_x) * static_cast<size_t>(components) * element_size, 0U);
  for (size_t g = 0; g < z_extents.size(); ++g) {
    const auto* base = static_cast<const std::uint8_t*>(starts[g]);
    const int64_t pitch = pitches[g];
    const int64_t per_comp = per_comps[g];
    const int64_t lim_z = z_extents[g];
    for (int64_t z = 0; z < lim_z; ++z) {
      // (x, y) loop pulls the global pitch from the carbon span, which is global_x*global_y.
      // We don't have global_y on hand here but we don't need it: the row stride in carbon's
      // owned buffer equals haloed.x == global.x, so we walk row-major over (y, x) within the
      // slab using global_x = row width.
      const int64_t row_stride = global_x;
      const int64_t y_count = pitch / row_stride;
      for (int64_t y = 0; y < y_count; ++y) {
        // Gather one row's worth of AOS cells into row_scratch.
        for (int64_t x = 0; x < global_x; ++x) {
          for (int c = 0; c < components; ++c) {
            const int64_t off =
                (static_cast<int64_t>(c) * per_comp + z * pitch + y * row_stride + x) *
                static_cast<int64_t>(element_size);
            const auto dst_off = (static_cast<size_t>(x) * static_cast<size_t>(components) +
                                  static_cast<size_t>(c)) *
                                 element_size;
            std::memcpy(row_scratch.data() + dst_off, base + off, element_size);
          }
        }
        maybe_swap_buffer(row_scratch.data(), element_size,
                          static_cast<size_t>(global_x) * static_cast<size_t>(components));
        out.write(reinterpret_cast<const char*>(row_scratch.data()),
                  static_cast<std::streamsize>(row_scratch.size()));
      }
    }
  }
}

/**
 * Inverse of @ref stream_field_payload: read AOS rows from @p in, swap back from BE,
 * and scatter into the field's component-outer carbon buffers.
 */
void
read_field_payload(std::istream& in,
                   size_t element_size,
                   int components,
                   int64_t global_x,
                   const std::vector<void*>& starts,
                   const std::vector<int64_t>& z_extents,
                   const std::vector<int64_t>& pitches,
                   const std::vector<int64_t>& per_comps) {
  std::vector<std::uint8_t> row_scratch(
      static_cast<size_t>(global_x) * static_cast<size_t>(components) * element_size, 0U);
  for (size_t g = 0; g < z_extents.size(); ++g) {
    auto* base = static_cast<std::uint8_t*>(starts[g]);
    const int64_t pitch = pitches[g];
    const int64_t per_comp = per_comps[g];
    const int64_t lim_z = z_extents[g];
    for (int64_t z = 0; z < lim_z; ++z) {
      const int64_t row_stride = global_x;
      const int64_t y_count = pitch / row_stride;
      for (int64_t y = 0; y < y_count; ++y) {
        in.read(reinterpret_cast<char*>(row_scratch.data()),
                static_cast<std::streamsize>(row_scratch.size()));
        if (!in.good()) {
          log::throw_with_log<std::runtime_error>("io::read_vtk: short read at slab={} z={} y={}",
                                                  g, z, y);
        }
        maybe_swap_buffer(row_scratch.data(), element_size,
                          static_cast<size_t>(global_x) * static_cast<size_t>(components));
        for (int64_t x = 0; x < global_x; ++x) {
          for (int c = 0; c < components; ++c) {
            const int64_t off =
                (static_cast<int64_t>(c) * per_comp + z * pitch + y * row_stride + x) *
                static_cast<int64_t>(element_size);
            const auto src_off = (static_cast<size_t>(x) * static_cast<size_t>(components) +
                                  static_cast<size_t>(c)) *
                                 element_size;
            std::memcpy(base + off, row_scratch.data() + src_off, element_size);
          }
        }
      }
    }
  }
}

/**
 * Per-field record cached when @ref open_for_read parses the file header. @c offset is
 * the byte offset of the field's binary payload (immediately after the
 * @c LOOKUP_TABLE\\ndefault\\n line); the read_field path seeks here before pulling
 * bytes off the stream.
 */
struct ReadFieldRec {
  std::string name;
  std::string vtk_dtype;
  int components = 0;
  std::streampos payload_offset = 0;
};

/**
 * Strip CRs introduced by writers on Windows-y machines and trailing whitespace. The
 * legacy spec lets either LF or CR-LF terminate header lines.
 */
void
rtrim_inplace(std::string& s) noexcept {
  while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
    s.pop_back();
  }
}

}  // namespace

struct VtkSession::Impl {
  std::string path;
  bool for_write = false;
  std::uint32_t num_fields_recorded = 0;

  // Owned dimensions of this shard (per the carbon Dense at open time): used by
  // the writer for the header and by the reader to validate against the file.
  int64_t local_x = 0;
  int64_t local_y = 0;
  int64_t local_z_total = 0;
  int64_t origin_z = 0;

  std::ofstream writer;
  std::ifstream reader;
  std::vector<ReadFieldRec> read_fields;
  size_t read_fields_cursor = 0;
};

VtkSession::VtkSession() noexcept = default;
VtkSession::~VtkSession() = default;
VtkSession::VtkSession(VtkSession&&) noexcept = default;
auto
VtkSession::operator=(VtkSession&&) noexcept -> VtkSession& = default;

auto
VtkSession::shard_path(std::string_view base,
                       const Dense& dense) -> std::string {
  return ::carbon::io::detail::shard_path(base, dense);
}

namespace {

[[nodiscard]] auto
sum_local_z(const Dense& dense) -> int64_t {
  int64_t total = 0;
  for (const auto& span : dense.spans()) {
    total += span.logic_extent().z;
  }
  return total;
}

[[nodiscard]] auto
local_origin_z(const Dense& dense) -> int64_t {
  if (dense.spans().empty()) {
    return 0;
  }
  const auto& s = dense.spans().front();
  return s.z_offset() + s.halo_z();
}

}  // namespace

void
VtkSession::open_for_write(std::string_view path,
                           const Dense& dense,
                           std::uint32_t num_fields) {
  impl_ = std::make_unique<Impl>();
  impl_->path = VtkSession::shard_path(path, dense);
  impl_->for_write = true;
  impl_->num_fields_recorded = num_fields;
  impl_->local_x = dense.global_extent().x;
  impl_->local_y = dense.global_extent().y;
  impl_->local_z_total = sum_local_z(dense);
  impl_->origin_z = local_origin_z(dense);

  impl_->writer.open(impl_->path, std::ios::binary | std::ios::trunc);
  if (!impl_->writer) {
    log::throw_with_log<std::runtime_error>("io::write_vtk: failed to open '{}' for writing",
                                            impl_->path);
  }
  // VTK legacy header lines are LF-terminated. Total point count goes on POINT_DATA
  // — that's the product of the DIMENSIONS, since we use STRUCTURED_POINTS with
  // POINT_DATA (cell-centered values placed on the implicit grid vertices, which
  // is the convention paraview / vtk readers expect for uniform grids).
  const int64_t npoints = impl_->local_x * impl_->local_y * impl_->local_z_total;
  impl_->writer << "# vtk DataFile Version 3.0\n";
  impl_->writer << "carbon dense field shard\n";
  impl_->writer << "BINARY\n";
  impl_->writer << "DATASET STRUCTURED_POINTS\n";
  impl_->writer << "DIMENSIONS " << impl_->local_x << ' ' << impl_->local_y << ' '
                << impl_->local_z_total << '\n';
  impl_->writer << "ORIGIN 0 0 " << impl_->origin_z << '\n';
  impl_->writer << "SPACING 1 1 1\n";
  impl_->writer << "POINT_DATA " << npoints << '\n';
  if (!impl_->writer.good()) {
    log::throw_with_log<std::runtime_error>("io::write_vtk: header write to '{}' failed",
                                            impl_->path);
  }
}

void
VtkSession::write_field(std::string_view name,
                        const char* dtype_tag,
                        size_t element_size,
                        int components,
                        const std::vector<const void*>& starts,
                        const std::vector<int64_t>& z_extents,
                        const std::vector<int64_t>& pitches,
                        const std::vector<int64_t>& per_comps) {
  if (impl_ == nullptr || !impl_->for_write) {
    log::throw_with_log<std::runtime_error>("io::write_vtk: session is not in a writable state");
  }
  const char* vtk_kw = vtk_keyword_from_native_tag(dtype_tag);
  if (vtk_kw == nullptr) {
    log::throw_with_log<std::invalid_argument>(
        "io::write_vtk: dtype '{}' has no VTK legacy mapping", std::string_view(dtype_tag, 8));
  }
  // SCALARS supports up to 4 components per the legacy spec; the on-disk layout for
  // N components is interleaved (component fastest), then x, then y, then z.
  if (components < 1 || components > 4) {
    log::throw_with_log<std::invalid_argument>(
        "io::write_vtk: SCALARS components={} unsupported (legacy VTK accepts 1..4); use one "
        "field per component for higher counts",
        components);
  }
  impl_->writer << "SCALARS " << name << ' ' << vtk_kw << ' ' << components << '\n';
  impl_->writer << "LOOKUP_TABLE default\n";
  if (!impl_->writer.good()) {
    log::throw_with_log<std::runtime_error>("io::write_vtk: SCALARS header write failed for '{}'",
                                            name);
  }
  stream_field_payload(impl_->writer, element_size, components, impl_->local_x, starts, z_extents,
                       pitches, per_comps);
  impl_->writer << '\n';  // legacy spec is loose here; trailing LF keeps a hand-edited file tidy.
  if (!impl_->writer.good()) {
    log::throw_with_log<std::runtime_error>("io::write_vtk: payload write failed for '{}'", name);
  }
}

namespace {

/**
 * Consume the legacy VTK header from @p in, validate it against the carbon-side
 * extents, and append per-SCALARS payload offsets to @p out_fields. Stops at end of
 * file; non-SCALARS attribute blocks (VECTORS, TENSORS, FIELD, ...) are rejected
 * since carbon only writes SCALARS. Kept as a free helper (taking primitive refs
 * rather than the session impl) so it doesn't have to cross the @c VtkSession::Impl
 * access barrier.
 */
void
parse_vtk_header(std::ifstream& in,
                 std::string_view path,
                 int64_t local_x,
                 int64_t local_y,
                 int64_t local_z_total,
                 int64_t origin_z,
                 std::vector<ReadFieldRec>& out_fields) {
  std::string line;

  auto next_line = [&](bool required) -> std::string {
    line.clear();
    if (!std::getline(in, line)) {
      if (required) {
        log::throw_with_log<std::runtime_error>("io::read_vtk: '{}' truncated while parsing header",
                                                path);
      }
      return {};
    }
    rtrim_inplace(line);
    return line;
  };

  const auto magic = next_line(true);
  if (magic.rfind("# vtk DataFile Version", 0) != 0) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' does not start with a VTK legacy magic line (got '{}')", path, magic);
  }
  next_line(true);  // free-form title — carbon ignores it on read.
  const auto fmt = next_line(true);
  if (fmt != "BINARY") {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' is '{}' but only BINARY legacy VTK is supported", path, fmt);
  }
  const auto ds = next_line(true);
  if (ds != "DATASET STRUCTURED_POINTS") {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' is '{}' but only DATASET STRUCTURED_POINTS is supported (carbon "
        "writes uniform grids)",
        path, ds);
  }

  int64_t dims[3] = {0, 0, 0};
  int64_t origin[3] = {0, 0, 0};
  double spacing[3] = {1.0, 1.0, 1.0};
  bool got_point_data = false;
  int64_t point_count = 0;

  while (true) {
    const auto cur = next_line(true);
    if (cur.empty()) {
      continue;
    }
    std::istringstream iss(cur);
    std::string head;
    iss >> head;
    if (head == "DIMENSIONS") {
      iss >> dims[0] >> dims[1] >> dims[2];
    } else if (head == "ORIGIN") {
      iss >> origin[0] >> origin[1] >> origin[2];
    } else if (head == "SPACING" || head == "ASPECT_RATIO") {
      iss >> spacing[0] >> spacing[1] >> spacing[2];
    } else if (head == "POINT_DATA") {
      iss >> point_count;
      got_point_data = true;
      break;
    } else if (head == "CELL_DATA") {
      log::throw_with_log<std::invalid_argument>(
          "io::read_vtk: '{}' uses CELL_DATA; carbon writes POINT_DATA only", path);
    } else {
      log::throw_with_log<std::invalid_argument>(
          "io::read_vtk: '{}' has unexpected header line '{}'", path, cur);
    }
  }
  if (!got_point_data) {
    log::throw_with_log<std::invalid_argument>("io::read_vtk: '{}' has no POINT_DATA section",
                                               path);
  }

  // Validate against the carbon Dense the caller plans to load into. DIMENSIONS must
  // match the local owned shape; ORIGIN must match the local z origin (per-rank under
  // MPI). x/y origin always 0 in carbon's z-only partitioning. SPACING must be unit
  // because carbon's index space is integer-valued.
  if (dims[0] != local_x || dims[1] != local_y || dims[2] != local_z_total) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' DIMENSIONS ({}, {}, {}) do not match Dense local extent ({}, {}, {})",
        path, dims[0], dims[1], dims[2], local_x, local_y, local_z_total);
  }
  if (origin[0] != 0 || origin[1] != 0 || origin[2] != origin_z) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' ORIGIN ({}, {}, {}) does not match Dense origin (0, 0, {})", path,
        origin[0], origin[1], origin[2], origin_z);
  }
  if (spacing[0] != 1.0 || spacing[1] != 1.0 || spacing[2] != 1.0) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' SPACING ({}, {}, {}) is non-unit; carbon's Dense uses integer "
        "voxel coordinates",
        path, spacing[0], spacing[1], spacing[2]);
  }
  if (point_count != local_x * local_y * local_z_total) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' POINT_DATA {} disagrees with DIMENSIONS product {}", path, point_count,
        local_x * local_y * local_z_total);
  }

  // Walk SCALARS sections until EOF, recording each field's payload offset.
  while (true) {
    if (!in) {
      break;
    }
    const auto cur = next_line(false);
    if (cur.empty()) {
      if (in.eof()) {
        break;
      }
      continue;
    }
    std::istringstream iss(cur);
    std::string head;
    iss >> head;
    if (head != "SCALARS") {
      log::throw_with_log<std::invalid_argument>(
          "io::read_vtk: unsupported attribute '{}' in '{}'; carbon's reader only accepts "
          "SCALARS blocks",
          head, path);
    }
    ReadFieldRec rec;
    iss >> rec.name >> rec.vtk_dtype;
    rec.components = 1;
    iss >> rec.components;  // no-op if missing — defaults to 1 per spec
    if (rec.components < 1 || rec.components > 4) {
      log::throw_with_log<std::invalid_argument>(
          "io::read_vtk: SCALARS '{}' has components={} (legacy VTK supports 1..4)", rec.name,
          rec.components);
    }
    const auto lookup = next_line(true);
    if (lookup.rfind("LOOKUP_TABLE", 0) != 0) {
      log::throw_with_log<std::invalid_argument>(
          "io::read_vtk: expected LOOKUP_TABLE after SCALARS '{}', got '{}'", rec.name, lookup);
    }
    rec.payload_offset = in.tellg();
    if (rec.payload_offset == std::streampos(-1)) {
      log::throw_with_log<std::runtime_error>(
          "io::read_vtk: failed to query stream offset for SCALARS '{}'", rec.name);
    }
    // Skip past this field's binary payload + a trailing LF the writer may emit.
    size_t element_size = 0;
    const auto& kw = rec.vtk_dtype;
    if (kw == "char" || kw == "unsigned_char") {
      element_size = 1;
    } else if (kw == "short" || kw == "unsigned_short") {
      element_size = 2;
    } else if (kw == "int" || kw == "unsigned_int" || kw == "float") {
      element_size = 4;
    } else if (kw == "double" || kw == "vtktypeint64" || kw == "vtktypeuint64" || kw == "long" ||
               kw == "unsigned_long") {
      element_size = 8;
    } else {
      log::throw_with_log<std::invalid_argument>(
          "io::read_vtk: SCALARS '{}' has dtype '{}' which carbon does not map back", rec.name, kw);
    }
    const std::streamoff payload_bytes = static_cast<std::streamoff>(point_count) *
                                         static_cast<std::streamoff>(rec.components) *
                                         static_cast<std::streamoff>(element_size);
    in.seekg(rec.payload_offset + payload_bytes);
    out_fields.push_back(std::move(rec));
    // Consume an optional trailing newline so the next SCALARS line parses cleanly.
    const int next_c = in.peek();
    if (next_c == '\n' || next_c == '\r') {
      in.get();
      if (next_c == '\r' && in.peek() == '\n') {
        in.get();
      }
    }
  }
}

}  // namespace

void
VtkSession::open_for_read(std::string_view path,
                          const Dense& dense) {
  impl_ = std::make_unique<Impl>();
  impl_->path = VtkSession::shard_path(path, dense);
  impl_->for_write = false;
  impl_->local_x = dense.global_extent().x;
  impl_->local_y = dense.global_extent().y;
  impl_->local_z_total = sum_local_z(dense);
  impl_->origin_z = local_origin_z(dense);

  impl_->reader.open(impl_->path, std::ios::binary);
  if (!impl_->reader) {
    log::throw_with_log<std::runtime_error>("io::read_vtk: failed to open '{}' for reading",
                                            impl_->path);
  }
  parse_vtk_header(impl_->reader, impl_->path, impl_->local_x, impl_->local_y, impl_->local_z_total,
                   impl_->origin_z, impl_->read_fields);
  (void)dense;  // checked indirectly via the impl_->local_* fields cached at open time.
}

void
VtkSession::read_field(std::string_view name,
                       const char* dtype_tag,
                       size_t element_size,
                       int components,
                       const std::vector<void*>& starts,
                       const std::vector<int64_t>& z_extents,
                       const std::vector<int64_t>& pitches,
                       const std::vector<int64_t>& per_comps) {
  if (impl_ == nullptr || impl_->for_write) {
    log::throw_with_log<std::runtime_error>("io::read_vtk: session is not in a readable state");
  }

  // Match by ordinal (i.e. the i-th SCALARS in the file lines up with the i-th
  // field passed to read_vtk). Using ordinals rather than names lets us round-
  // trip files written by other tools that didn't follow our "field_<i>" naming.
  if (impl_->read_fields.empty()) {
    log::throw_with_log<std::invalid_argument>("io::read_vtk: file '{}' has no SCALARS blocks",
                                               impl_->path);
  }
  size_t& idx = impl_->read_fields_cursor;
  if (idx >= impl_->read_fields.size()) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: more fields requested than '{}' contains", impl_->path);
  }
  const auto& rec = impl_->read_fields[idx++];
  (void)name;  // names diverge between writers; ordinal binding is the contract.

  const char* expected_kw = vtk_keyword_from_native_tag(dtype_tag);
  if (expected_kw == nullptr || rec.vtk_dtype != expected_kw) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: field {} dtype '{}' does not match bound carbon dtype '{}' in '{}'",
        rec.name, rec.vtk_dtype, (expected_kw ? expected_kw : "?"), impl_->path);
  }
  if (rec.components != components) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: field {} has {} components but bound Field has {}", rec.name, rec.components,
        components);
  }

  impl_->reader.clear();
  impl_->reader.seekg(rec.payload_offset);
  if (!impl_->reader.good()) {
    log::throw_with_log<std::runtime_error>("io::read_vtk: seek to payload of '{}' failed in '{}'",
                                            rec.name, impl_->path);
  }
  read_field_payload(impl_->reader, element_size, components, impl_->local_x, starts, z_extents,
                     pitches, per_comps);
}

auto
VtkSession::num_fields() const noexcept -> std::uint32_t {
  if (impl_ == nullptr) {
    return 0U;
  }
  return impl_->for_write ? impl_->num_fields_recorded
                          : static_cast<std::uint32_t>(impl_->read_fields.size());
}

}  // namespace carbon::io::detail

namespace carbon::io {

auto
write_vtk_multiblock_index(std::string_view base,
                           const Dense& dense) -> void {
#ifdef CARBON_HAVE_MPI_NCCL
  const auto* dist = dense.system().distributed();
  if (dist == nullptr) {
    // No distributed handle: single-process call. The legacy `.vtk` written
    // by write_vtk already opens directly; nothing for the master to wrap.
    return;
  }
  const int world_size = dist->world_size();
  const int world_rank = dist->world_rank();
  if (world_size <= 1 || world_rank != 0) {
    return;
  }

  // Master path: <base>.vtm. Per-piece references are stored relative to
  // the master file's parent directory so the bundle stays portable.
  const std::filesystem::path base_path(std::string{base});
  const std::filesystem::path master_path = base_path.string() + ".vtm";
  const std::filesystem::path base_filename = base_path.filename();

  std::ofstream out(master_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    log::throw_with_log<std::runtime_error>(
        "io::write_vtk_multiblock_index: failed to open '{}' for writing", master_path.string());
  }

  // Minimal vtkMultiBlockDataSet XML envelope. ParaView accepts legacy `.vtk`
  // pieces under `<DataSet file="…"/>` despite the format being heterogeneous
  // — each shard is opened as its own STRUCTURED_POINTS block.
  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"";
  out << (std::endian::native == std::endian::big ? "BigEndian" : "LittleEndian");
  out << "\">\n";
  out << "  <vtkMultiBlockDataSet>\n";
  for (int r = 0; r < world_size; ++r) {
    std::ostringstream piece_filename;
    piece_filename << base_filename.string() << ".rank" << r;
    out << "    <Block index=\"" << r << "\" name=\"rank" << r << "\">\n";
    out << "      <DataSet index=\"0\" file=\"" << piece_filename.str() << "\"/>\n";
    out << "    </Block>\n";
  }
  out << "  </vtkMultiBlockDataSet>\n";
  out << "</VTKFile>\n";

  if (!out.good()) {
    log::throw_with_log<std::runtime_error>("io::write_vtk_multiblock_index: write to '{}' failed",
                                            master_path.string());
  }
#else
  // Without MPI/NCCL there is no Distributed handle in carbon's build —
  // the function is a no-op so the call site stays portable across builds.
  (void)base;
  (void)dense;
#endif
}

}  // namespace carbon::io
