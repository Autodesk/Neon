// VTK XML (vti) backend for carbon::DenseDistributed (slab / pencil / brick). 
// See include/carbon/detail/io_vtk_distributed.inl for the public surface.

#include <carbon/dense_distributed.h>
#include <carbon/detail/dense_distributed/layout.h>
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
#include <utility>
#include <vector>
#include <cctype>

namespace carbon::io::detail {

namespace {

namespace dd = ::carbon::detail::dense_distributed;

/** Staging block for payload I/O: rows are batched so the stream sees few, large transfers. */
constexpr size_t kBlockBytes = size_t{4} << 20;

[[nodiscard]] constexpr auto
host_is_little_endian() noexcept -> bool {
  return std::endian::native == std::endian::little;
}

/** Reverse the bytes of @p n consecutive @c ES -byte elements (legacy VTK payloads are BE). */
template <size_t ES>
void
swap_run(std::uint8_t* p,
         size_t n) noexcept {
  if constexpr (ES > 1) {
    for (size_t i = 0; i < n; ++i) {
      std::uint8_t* e = p + i * ES;
      for (size_t k = 0; k < ES / 2; ++k) {
        std::swap(e[k], e[ES - 1 - k]);
      }
    }
  }
}

/** Map a carbon native dtype tag (8-char, NUL padded) to the VTK legacy keyword; null if unknown. */
[[nodiscard]] auto
vtk_keyword_from_native_tag(const char* tag) noexcept -> const char* {
  const std::string_view t(tag, 8);
  if (t.starts_with("f32")) {
    return "float";
  }
  if (t.starts_with("f64")) {
    return "double";
  }
  if (t.starts_with("i8")) {
    return "char";
  }
  if (t.starts_with("u8")) {
    return "unsigned_char";
  }
  if (t.starts_with("i16")) {
    return "short";
  }
  if (t.starts_with("u16")) {
    return "unsigned_short";
  }
  if (t.starts_with("i32")) {
    return "int";
  }
  if (t.starts_with("u32")) {
    return "unsigned_int";
  }
  if (t.starts_with("i64")) {
    return "vtktypeint64";
  }
  if (t.starts_with("u64")) {
    return "vtktypeuint64";
  }
  return nullptr;
}

/** Map a carbon native dtype tag to the VTK XML keyword; null if unknown. */
[[nodiscard]] auto
vtk_xml_keyword_from_native_tag(const char* tag) noexcept -> const char* {
  const std::string_view t(tag, 8);
  if (t.starts_with("f32")) return "Float32";
  if (t.starts_with("f64")) return "Float64";
  if (t.starts_with("i8")) return "Int8";
  if (t.starts_with("u8")) return "UInt8";
  if (t.starts_with("i16")) return "Int16";
  if (t.starts_with("u16")) return "UInt16";
  if (t.starts_with("i32")) return "Int32";
  if (t.starts_with("u32")) return "UInt32";
  if (t.starts_with("i64")) return "Int64";
  if (t.starts_with("u64")) return "UInt64";
  return nullptr;
}

[[nodiscard]] auto
element_size_of_keyword(const std::string& kw) noexcept -> size_t {
  if (kw == "char" || kw == "unsigned_char" || kw == "Int8" || kw == "UInt8") return 1;
  if (kw == "short" || kw == "unsigned_short" || kw == "Int16" || kw == "UInt16") return 2;
  if (kw == "int" || kw == "unsigned_int" || kw == "float" || kw == "Int32" || kw == "UInt32" || kw == "Float32") return 4;
  if (kw == "double" || kw == "vtktypeint64" || kw == "vtktypeuint64" || kw == "long" ||
      kw == "unsigned_long" || kw == "Int64" || kw == "UInt64" || kw == "Float64") return 8;
  return 0;
}

void
rtrim_inplace(std::string& s) noexcept {
  while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
    s.pop_back();
  }
}

/** Legacy SCALARS names are whitespace-delimited tokens: make the label a single token. */
[[nodiscard]] auto
sanitize_name(std::string_view name) -> std::string {
  std::string out(name);
  for (char& c : out) {
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\"' || c == '<' || c == '>') {
      c = '_';
    }
  }
  return out.empty() ? std::string("field") : out;
}

[[nodiscard]] auto
shard_path_of(std::string_view base,
              const DenseDistributed& grid) -> std::string {
#ifdef CARBON_HAVE_MPI_NCCL
  if (grid.system().distributed() != nullptr) {
    std::ostringstream oss;
    oss << base << ".rank" << grid.system().distributed()->world_rank() << ".vti";
    return oss.str();
  }
#else
  (void)grid;
#endif
  return std::string(base) + ".vti";
}

[[nodiscard]] auto
pieces_cells(const std::vector<VtkBoxPiece>& pieces) -> int64_t {
  int64_t n = 0;
  for (const auto& p : pieces) {
    n += p.nx * p.ny * p.nz;
  }
  return n;
}

// ---- payload gather / scatter (compile-time element size) ----------------------------------

/** One owned row of @p nx cells -> interleaved (component-fastest) record bytes. */
template <size_t ES>
void
gather_row(std::uint8_t* dst,
           const std::uint8_t* src,
           int64_t nx,
           int comps,
           int64_t comp_stride_elems) noexcept {
  if (comps == 1) {
    std::memcpy(dst, src, static_cast<size_t>(nx) * ES);
    return;
  }
  const size_t stride = static_cast<size_t>(comps) * ES;
  for (int c = 0; c < comps; ++c) {
    const std::uint8_t* s = src + static_cast<size_t>(c * comp_stride_elems) * ES;
    std::uint8_t* d = dst + static_cast<size_t>(c) * ES;
    for (int64_t x = 0; x < nx; ++x) {
      std::memcpy(d + static_cast<size_t>(x) * stride, s + static_cast<size_t>(x) * ES, ES);
    }
  }
}

/** Interleaved record bytes -> one owned row (inverse of @ref gather_row). */
template <size_t ES>
void
scatter_row(std::uint8_t* dst,
            const std::uint8_t* src,
            int64_t nx,
            int comps,
            int64_t comp_stride_elems) noexcept {
  if (comps == 1) {
    std::memcpy(dst, src, static_cast<size_t>(nx) * ES);
    return;
  }
  const size_t stride = static_cast<size_t>(comps) * ES;
  for (int c = 0; c < comps; ++c) {
    std::uint8_t* d = dst + static_cast<size_t>(c * comp_stride_elems) * ES;
    const std::uint8_t* s = src + static_cast<size_t>(c) * ES;
    for (int64_t x = 0; x < nx; ++x) {
      std::memcpy(d + static_cast<size_t>(x) * ES, s + static_cast<size_t>(x) * stride, ES);
    }
  }
}

[[nodiscard]] inline auto
row_ptr(const VtkBoxPiece& p,
        size_t es,
        int64_t z,
        int64_t y) noexcept -> const std::uint8_t* {
  return static_cast<const std::uint8_t*>(p.start) +
         static_cast<size_t>(z * p.plane_pitch + y * p.row_stride) * es;
}

[[nodiscard]] inline auto
row_ptr(const VtkBoxPieceMut& p,
        size_t es,
        int64_t z,
        int64_t y) noexcept -> std::uint8_t* {
  return static_cast<std::uint8_t*>(p.start) +
         static_cast<size_t>(z * p.plane_pitch + y * p.row_stride) * es;
}

// --- LEGACY PAYLOAD HELPERS (Required for tests) ---

template <size_t ES>
void
write_payload(std::ostream& out,
              int comps,
              const std::vector<VtkBoxPiece>& pieces) {
  size_t total_bytes = 0;
  size_t max_row_bytes = 0;
  for (const auto& p : pieces) {
    const size_t row = static_cast<size_t>(p.nx) * static_cast<size_t>(comps) * ES;
    total_bytes += row * static_cast<size_t>(p.ny) * static_cast<size_t>(p.nz);
    max_row_bytes = std::max(max_row_bytes, row);
  }
  if (total_bytes == 0) {
    return;
  }
  std::vector<std::uint8_t> block(std::max(std::min(total_bytes, kBlockBytes), max_row_bytes));
  size_t used = 0;
  const auto flush = [&] {
    if (used == 0) {
      return;
    }
    if (host_is_little_endian()) {
      swap_run<ES>(block.data(), used / ES);
    }
    out.write(reinterpret_cast<const char*>(block.data()), static_cast<std::streamsize>(used));
    used = 0;
  };
  for (const auto& p : pieces) {
    const size_t row_bytes = static_cast<size_t>(p.nx) * static_cast<size_t>(comps) * ES;
    for (int64_t z = 0; z < p.nz; ++z) {
      for (int64_t y = 0; y < p.ny; ++y) {
        if (used + row_bytes > block.size()) {
          flush();
        }
        gather_row<ES>(block.data() + used, row_ptr(p, ES, z, y), p.nx, comps, p.comp_stride);
        used += row_bytes;
      }
    }
  }
  flush();
}

template <size_t ES>
void
read_payload(std::istream& in,
             int comps,
             const std::vector<VtkBoxPieceMut>& pieces) {
  size_t max_row_bytes = 0;
  size_t total_bytes = 0;
  for (const auto& p : pieces) {
    const size_t row = static_cast<size_t>(p.nx) * static_cast<size_t>(comps) * ES;
    max_row_bytes = std::max(max_row_bytes, row);
    total_bytes += row * static_cast<size_t>(p.ny) * static_cast<size_t>(p.nz);
  }
  if (total_bytes == 0) {
    return;
  }
  std::vector<std::uint8_t> block(std::max(std::min(total_bytes, kBlockBytes), max_row_bytes));
  for (const auto& p : pieces) {
    const size_t row_bytes = static_cast<size_t>(p.nx) * static_cast<size_t>(comps) * ES;
    const int64_t rows_total = p.ny * p.nz;
    const int64_t rows_per_block = static_cast<int64_t>(block.size() / row_bytes);
    for (int64_t r0 = 0; r0 < rows_total; r0 += rows_per_block) {
      const int64_t rows = std::min(rows_per_block, rows_total - r0);
      const size_t bytes = static_cast<size_t>(rows) * row_bytes;
      in.read(reinterpret_cast<char*>(block.data()), static_cast<std::streamsize>(bytes));
      if (!in.good()) {
        log::throw_with_log<std::runtime_error>(
            "io::read_vtk: short read in payload (row {} of {})", r0, rows_total);
      }
      if (host_is_little_endian()) {
        swap_run<ES>(block.data(), bytes / ES);
      }
      for (int64_t i = 0; i < rows; ++i) {
        const int64_t r = r0 + i;
        scatter_row<ES>(row_ptr(p, ES, r / p.ny, r % p.ny),
                        block.data() + static_cast<size_t>(i) * row_bytes, p.nx, comps,
                        p.comp_stride);
      }
    }
  }
}

// --- XML PAYLOAD HELPERS ---

template <size_t ES>
void
write_xml_payload_to_buffer(std::uint8_t* dst,
                            int comps,
                            const std::vector<VtkBoxPiece>& pieces) {
  size_t used = 0;
  for (const auto& p : pieces) {
    const size_t row_bytes = static_cast<size_t>(p.nx) * static_cast<size_t>(comps) * ES;
    for (int64_t z = 0; z < p.nz; ++z) {
      for (int64_t y = 0; y < p.ny; ++y) {
        gather_row<ES>(dst + used, row_ptr(p, ES, z, y), p.nx, comps, p.comp_stride);
        used += row_bytes;
      }
    }
  }
}

}  // namespace

// ================================================================================================
// Box geometry
// ================================================================================================

auto
vtk_box_of(const DenseDistributed& grid) -> VtkBox {
  VtkBox box;
  const auto& spans = grid.spans();
  if (spans.empty()) {
    return box;
  }
  const auto owned_origin = [](const auto& span) -> int64_3d {
    const int64_3d o = span.offset();
    const int64_3d h = span.halo();
    return int64_3d{o.x + h.x, o.y + h.y, o.z + h.z};
  };
  const int64_3d first = owned_origin(spans.front());
  box.origin[0] = first.x;
  box.origin[1] = first.y;
  box.origin[2] = first.z;

  int64_t z_total = 0;
  int64_3d prev_origin{};
  int64_t prev_nz = 0;
  bool have_prev = false;
  int64_t nx = 0;
  int64_t ny = 0;
  for (const auto& span : spans) {
    const int64_3d owned = span.logic_extent();
    if (owned.x <= 0 || owned.y <= 0 || owned.z <= 0) {
      continue;
    }
    const int64_3d o = owned_origin(span);
    if (!have_prev) {
      nx = owned.x;
      ny = owned.y;
      box.origin[0] = o.x;
      box.origin[1] = o.y;
      box.origin[2] = o.z;
    } else if (owned.x != nx || owned.y != ny || o.x != prev_origin.x || o.y != prev_origin.y ||
               o.z != prev_origin.z + prev_nz) {
      log::throw_with_log<std::invalid_argument>(
          "io::vtk: the local spans of this DenseDistributed do not form a single box (only a "
          "z-stacked slab can span several local GPUs)");
    }
    z_total += owned.z;
    prev_origin = o;
    prev_nz = owned.z;
    have_prev = true;
  }
  if (have_prev) {
    box.dims[0] = nx;
    box.dims[1] = ny;
    box.dims[2] = z_total;
  }
  return box;
}

// ================================================================================================
// Stream-level building blocks (Legacy - required for tests)
// ================================================================================================

void
write_vtk_box_header(std::ostream& out,
                     const VtkBox& box) {
  const bool empty = box.empty();
  out << "# vtk DataFile Version 3.0\n";
  out << (empty ? "carbon dense field shard (empty: this rank owns no cells)\n" : "carbon dense field shard\n");
  out << "BINARY\n";
  out << "DATASET STRUCTURED_POINTS\n";
  out << "DIMENSIONS " << (empty ? 0 : box.dims[0]) << ' ' << (empty ? 0 : box.dims[1]) << ' ' << (empty ? 0 : box.dims[2]) << '\n';
  out << "ORIGIN " << box.origin[0] << ' ' << box.origin[1] << ' ' << box.origin[2] << '\n';
  out << "SPACING 1 1 1\n";
  out << "POINT_DATA " << box.points() << '\n';
  if (!out.good()) log::throw_with_log<std::runtime_error>("io::write_vtk: failed to write the shard header");
}

void
write_vtk_box_field(std::ostream& out,
                    std::string_view name,
                    const char* dtype_tag,
                    size_t element_size,
                    int components,
                    const VtkBox& box,
                    const std::vector<VtkBoxPiece>& pieces) {
  const char* kw = vtk_keyword_from_native_tag(dtype_tag);
  if (kw == nullptr) log::throw_with_log<std::invalid_argument>("io::write_vtk: dtype error");
  if (components < 1 || components > 4) log::throw_with_log<std::invalid_argument>("io::write_vtk: components error");
  
  out << "SCALARS " << sanitize_name(name) << ' ' << kw << ' ' << components << '\n';
  out << "LOOKUP_TABLE default\n";
  if (!box.empty()) {
    switch (element_size) {
      case 1: write_payload<1>(out, components, pieces); break;
      case 2: write_payload<2>(out, components, pieces); break;
      case 4: write_payload<4>(out, components, pieces); break;
      case 8: write_payload<8>(out, components, pieces); break;
      default: log::throw_with_log<std::invalid_argument>("io::write_vtk: unsupported size");
    }
  }
  out << '\n';
}

void
parse_vtk_box_header(std::istream& in,
                     std::string_view path,
                     const VtkBox& expected,
                     std::vector<VtkBoxFieldRecord>& out_fields) {
  const auto next_line = [&](bool strict) -> std::string {
    std::string line;
    if (!std::getline(in, line)) {
      if (strict) log::throw_with_log<std::invalid_argument>("io::read_vtk: header ended");
      return {};
    }
    rtrim_inplace(line);
    return line;
  };

  const auto magic = next_line(true);
  if (magic.rfind("# vtk DataFile Version", 0) != 0) log::throw_with_log<std::invalid_argument>("io::read_vtk: bad magic");
  next_line(true); // title
  if (next_line(true) != "BINARY") log::throw_with_log<std::invalid_argument>("io::read_vtk: not BINARY");
  if (next_line(true) != "DATASET STRUCTURED_POINTS") log::throw_with_log<std::invalid_argument>("io::read_vtk: not STRUCTURED_POINTS");

  int64_t dims[3] = {0, 0, 0};
  int64_t origin[3] = {0, 0, 0};
  double spacing[3] = {1.0, 1.0, 1.0};
  int64_t point_count = -1;
  while (point_count < 0) {
    const auto cur = next_line(true);
    if (cur.empty()) continue;
    std::istringstream iss(cur);
    std::string head;
    iss >> head;
    if (head == "DIMENSIONS") iss >> dims[0] >> dims[1] >> dims[2];
    else if (head == "ORIGIN") iss >> origin[0] >> origin[1] >> origin[2];
    else if (head == "SPACING" || head == "ASPECT_RATIO") iss >> spacing[0] >> spacing[1] >> spacing[2];
    else if (head == "POINT_DATA") iss >> point_count;
    else log::throw_with_log<std::invalid_argument>("io::read_vtk: unexpected header");
  }

  while (in) {
    const auto cur = next_line(false);
    if (cur.empty()) {
      if (in.eof()) break;
      continue;
    }
    std::istringstream iss(cur);
    std::string head;
    iss >> head;
    if (head != "SCALARS") log::throw_with_log<std::invalid_argument>("io::read_vtk: expected SCALARS");
    VtkBoxFieldRecord rec;
    iss >> rec.name >> rec.vtk_dtype;
    rec.components = 1;
    iss >> rec.components;
    next_line(true); // LOOKUP_TABLE
    rec.payload_offset = in.tellg();
    const size_t es = element_size_of_keyword(rec.vtk_dtype);
    in.seekg(rec.payload_offset + static_cast<std::streamoff>(point_count * rec.components * es));
    out_fields.push_back(std::move(rec));
    const int next_c = in.peek();
    if (next_c == '\n' || next_c == '\r') {
      in.get();
      if (next_c == '\r' && in.peek() == '\n') in.get();
    }
  }
}

void
read_vtk_box_field_payload(std::istream& in,
                           size_t element_size,
                           int components,
                           const VtkBox& box,
                           const std::vector<VtkBoxPieceMut>& pieces) {
  if (box.empty()) return;
  switch (element_size) {
    case 1: read_payload<1>(in, components, pieces); break;
    case 2: read_payload<2>(in, components, pieces); break;
    case 4: read_payload<4>(in, components, pieces); break;
    case 8: read_payload<8>(in, components, pieces); break;
    default: log::throw_with_log<std::invalid_argument>("io::read_vtk: bad size");
  }
}

// ================================================================================================
// Stream-level building blocks (XML - used by VtkBoxSession)
// ================================================================================================

void
write_vtk_xml_box_header(std::ostream& out,
                         const VtkBox& box) {
  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
  const bool empty = box.empty();
  if (empty) {
    out << "  <ImageData WholeExtent=\"0 0 0 0 0 0\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
    out << "    <Piece Extent=\"0 0 0 0 0 0\">\n";
  } else {
    out << "  <ImageData WholeExtent=\"0 " << box.dims[0]-1 << " 0 " << box.dims[1]-1 << " 0 " << box.dims[2]-1 
        << "\" Origin=\"" << box.origin[0] << " " << box.origin[1] << " " << box.origin[2] << "\" Spacing=\"1 1 1\">\n";
    out << "    <Piece Extent=\"0 " << box.dims[0]-1 << " 0 " << box.dims[1]-1 << " 0 " << box.dims[2]-1 << "\">\n";
  }
  out << "      <PointData>\n";
}

void
write_vtk_xml_box_field(std::ostream& out,
                        std::string_view name,
                        const char* dtype_tag,
                        size_t element_size,
                        int components,
                        const VtkBox& box,
                        const std::vector<VtkBoxPiece>& pieces) {
  const char* kw = vtk_xml_keyword_from_native_tag(dtype_tag);
  if (kw == nullptr) log::throw_with_log<std::invalid_argument>("io::write_vtk: bad xml dtype");
  
  out << "        <DataArray type=\"" << kw << "\" Name=\"" << sanitize_name(name)
      << "\" NumberOfComponents=\"" << components << "\" format=\"ascii\">\n";
      
  if (!box.empty()) {
    size_t payload_bytes = static_cast<size_t>(box.points()) * components * element_size;
    std::vector<std::uint8_t> raw_data(payload_bytes);
    
    switch (element_size) {
      case 1: write_xml_payload_to_buffer<1>(raw_data.data(), components, pieces); break;
      case 2: write_xml_payload_to_buffer<2>(raw_data.data(), components, pieces); break;
      case 4: write_xml_payload_to_buffer<4>(raw_data.data(), components, pieces); break;
      case 8: write_xml_payload_to_buffer<8>(raw_data.data(), components, pieces); break;
      default: log::throw_with_log<std::invalid_argument>("io::write_vtk: bad size");
    }
    
    if (element_size == 4) {
      const float* fdata = reinterpret_cast<const float*>(raw_data.data());
      size_t n = payload_bytes / 4;
      for (size_t i = 0; i < n; ++i) {
        out << fdata[i] << " ";
        if (i % 10 == 9) out << "\n";
      }
    } else if (element_size == 8) {
      const double* ddata = reinterpret_cast<const double*>(raw_data.data());
      size_t n = payload_bytes / 8;
      for (size_t i = 0; i < n; ++i) {
        out << ddata[i] << " ";
        if (i % 10 == 9) out << "\n";
      }
    } else {
      for (size_t i = 0; i < payload_bytes; ++i) {
        out << static_cast<int>(raw_data[i]) << " ";
        if (i % 10 == 9) out << "\n";
      }
    }
  }
  out << "\n        </DataArray>\n";
}

void
parse_vtk_xml_box_header(std::istream& in,
                         std::string_view path,
                         const VtkBox& expected,
                         std::vector<VtkBoxFieldRecord>& out_fields) {
  std::string line;
  auto get_attr = [](const std::string& s, const std::string& key) -> std::string {
    std::string k = key + "=\"";
    auto pos = s.find(k);
    if (pos == std::string::npos) return "";
    pos += k.size();
    auto end = s.find("\"", pos);
    return s.substr(pos, end - pos);
  };

  while (std::getline(in, line)) {
    if (line.find("<ImageData") != std::string::npos) {
      std::string extent_str = get_attr(line, "WholeExtent");
      std::string origin_str = get_attr(line, "Origin");
      
      int64_t dims[3] = {0, 0, 0};
      int64_t origin[3] = {0, 0, 0};
      
      std::stringstream ss_ext(extent_str);
      int64_t e0, e1, e2, e3, e4, e5;
      if (ss_ext >> e0 >> e1 >> e2 >> e3 >> e4 >> e5) {
        dims[0] = e1 - e0 + 1;
        dims[1] = e3 - e2 + 1;
        dims[2] = e5 - e4 + 1;
      }
      
      std::stringstream ss_org(origin_str);
      ss_org >> origin[0] >> origin[1] >> origin[2];

      const bool want_empty = expected.empty();
      const int64_t want_dims[3] = {want_empty ? 0 : expected.dims[0],
                                    want_empty ? 0 : expected.dims[1],
                                    want_empty ? 0 : expected.dims[2]};
      if (dims[0] != want_dims[0] || dims[1] != want_dims[1] || dims[2] != want_dims[2]) {
        log::throw_with_log<std::invalid_argument>(
            "io::read_vtk: '{}' DIMENSIONS ({}, {}, {}) do not match this rank's owned box ({}, {}, {})",
            path, dims[0], dims[1], dims[2], want_dims[0], want_dims[1], want_dims[2]);
      }
      if (origin[0] != expected.origin[0] || origin[1] != expected.origin[1] || origin[2] != expected.origin[2]) {
        log::throw_with_log<std::invalid_argument>(
            "io::read_vtk: '{}' ORIGIN ({}, {}, {}) does not match this rank's box origin ({}, {}, {})",
            path, origin[0], origin[1], origin[2], expected.origin[0], expected.origin[1], expected.origin[2]);
      }
    } else if (line.find("<DataArray") != std::string::npos) {
      VtkBoxFieldRecord rec;
      rec.vtk_dtype = get_attr(line, "type");
      rec.name = get_attr(line, "Name");
      rec.components = std::stoi(get_attr(line, "NumberOfComponents"));
      rec.payload_offset = in.tellg();
      out_fields.push_back(rec);
      
      std::string skip_line;
      while (std::getline(in, skip_line)) {
        if (skip_line.find("</DataArray>") != std::string::npos) break;
      }
    }
  }
}

void
read_vtk_xml_box_field_payload(std::istream& in,
                               size_t element_size,
                               int components,
                               const VtkBox& box,
                               const std::vector<VtkBoxPieceMut>& pieces) {
  if (box.empty()) return;
  size_t total_elements = static_cast<size_t>(box.points()) * components;
  std::vector<float> fdata(total_elements);
  for (size_t i = 0; i < total_elements; ++i) {
    in >> fdata[i];
  }
  
  size_t offset = 0;
  for (const auto& p : pieces) {
    const size_t row_bytes = static_cast<size_t>(p.nx) * static_cast<size_t>(components) * element_size;
    for (int64_t z = 0; z < p.nz; ++z) {
      for (int64_t y = 0; y < p.ny; ++y) {
        std::memcpy(row_ptr(p, element_size, z, y), fdata.data() + offset, row_bytes);
        offset += p.nx * components;
      }
    }
  }
}

// ================================================================================================
// Session
// ================================================================================================

struct VtkBoxSession::Impl {
  std::string path;
  bool for_write = false;
  std::uint32_t num_fields_recorded = 0;
  VtkBox box;
  std::ofstream writer;
  std::ifstream reader;
  std::vector<VtkBoxFieldRecord> read_fields;
  size_t read_fields_cursor = 0;

  void finalize() {
    if (!writer.is_open()) return;
    writer << "      </PointData>\n";
    writer << "    </Piece>\n";
    writer << "  </ImageData>\n";
    writer << "</VTKFile>\n";
    writer.flush();
    if (!writer.good()) {
      log::throw_with_log<std::runtime_error>("io::write_vtk: failed to finalize the shard");
    }
    writer.close();
  }
};

VtkBoxSession::VtkBoxSession() noexcept = default;
VtkBoxSession::~VtkBoxSession() {
  if (impl_ && impl_->for_write) {
    impl_->finalize();
  }
}
VtkBoxSession::VtkBoxSession(VtkBoxSession&&) noexcept = default;
auto
VtkBoxSession::operator=(VtkBoxSession&&) noexcept -> VtkBoxSession& = default;

auto
VtkBoxSession::shard_path(std::string_view base,
                          const DenseDistributed& grid) -> std::string {
  return shard_path_of(base, grid);
}

void
VtkBoxSession::open_for_write(std::string_view path,
                              const DenseDistributed& grid,
                              std::uint32_t /*num_fields*/) {
  if (impl_ && impl_->for_write) {
    impl_->finalize();
  }
  impl_ = std::make_unique<Impl>();
  impl_->path = shard_path_of(path, grid);
  impl_->for_write = true;
  impl_->box = vtk_box_of(grid);
  impl_->writer.open(impl_->path, std::ios::binary | std::ios::trunc);
  if (!impl_->writer) {
    log::throw_with_log<std::runtime_error>("io::write_vtk: failed to open '{}' for writing",
                                            impl_->path);
  }
  write_vtk_xml_box_header(impl_->writer, impl_->box);
}

void
VtkBoxSession::write_field(std::string_view name,
                           const char* dtype_tag,
                           size_t element_size,
                           int components,
                           const std::vector<VtkBoxPiece>& pieces) {
  if (impl_ == nullptr || !impl_->for_write) {
    log::throw_with_log<std::runtime_error>("io::write_vtk: session is not in a writable state");
  }
  write_vtk_xml_box_field(impl_->writer, name, dtype_tag, element_size, components, impl_->box, pieces);
  ++impl_->num_fields_recorded;
}

void
VtkBoxSession::open_for_read(std::string_view path,
                             const DenseDistributed& grid) {
  impl_ = std::make_unique<Impl>();
  impl_->path = shard_path_of(path, grid);
  impl_->for_write = false;
  impl_->box = vtk_box_of(grid);
  impl_->reader.open(impl_->path, std::ios::binary);
  if (!impl_->reader) {
    log::throw_with_log<std::runtime_error>("io::read_vtk: failed to open '{}' for reading",
                                            impl_->path);
  }
  parse_vtk_xml_box_header(impl_->reader, impl_->path, impl_->box, impl_->read_fields);
}

void
VtkBoxSession::read_field(std::string_view /*name*/,
                          const char* dtype_tag,
                          size_t element_size,
                          int components,
                          const std::vector<VtkBoxPieceMut>& pieces) {
  if (impl_ == nullptr || impl_->for_write) {
    log::throw_with_log<std::runtime_error>("io::read_vtk: session is not in a readable state");
  }
  if (impl_->read_fields_cursor >= impl_->read_fields.size()) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: more fields requested than '{}' contains", impl_->path);
  }
  const auto& rec = impl_->read_fields[impl_->read_fields_cursor++];
  const char* expected_kw = vtk_xml_keyword_from_native_tag(dtype_tag);
  if (expected_kw == nullptr || rec.vtk_dtype != expected_kw) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: field {} dtype '{}' does not match bound carbon dtype '{}' in '{}'",
        rec.name, rec.vtk_dtype, (expected_kw != nullptr ? expected_kw : "?"), impl_->path);
  }
  if (rec.components != components) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: field {} has {} components but the bound Field has {}", rec.name,
        rec.components, components);
  }
  impl_->reader.clear();
  impl_->reader.seekg(rec.payload_offset);
  if (!impl_->reader.good()) {
    log::throw_with_log<std::runtime_error>(
        "io::read_vtk: seek to the payload of '{}' failed in '{}'", rec.name, impl_->path);
  }
  read_vtk_xml_box_field_payload(impl_->reader, element_size, components, impl_->box, pieces);
}

auto
VtkBoxSession::num_fields() const noexcept -> std::uint32_t {
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
                           const DenseDistributed& grid) -> void {
#ifdef CARBON_HAVE_MPI_NCCL
  const auto* dist = grid.system().distributed();
  if (dist == nullptr) {
    return;
  }
  const int world_size = dist->world_size();
  if (world_size <= 1 || dist->world_rank() != 0) {
    return;
  }

  const ProcessGrid& pg = grid.process_grid();
  const int64_3d global = grid.global_extent();
  std::vector<int> listed;
  for (int r = 0; r < world_size; ++r) {
    bool owns = true;
    if (pg.world_size() == world_size) {
      const auto chunk = ::carbon::detail::dense_distributed::rank_chunk(
          global, pg, pg.coords_of(r), int64_3d{0, 0, 0});
      owns = chunk.owned.x > 0 && chunk.owned.y > 0 && chunk.owned.z > 0;
    }
    if (owns) {
      listed.push_back(r);
    }
  }

  const std::filesystem::path base_path(std::string{base});
  const std::filesystem::path master_path = base_path.string() + ".vtm";
  const std::filesystem::path base_filename = base_path.filename();

  std::ofstream out(master_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    log::throw_with_log<std::runtime_error>(
        "io::write_vtk_multiblock_index: failed to open '{}' for writing", master_path.string());
  }
  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"";
  out << (std::endian::native == std::endian::big ? "BigEndian" : "LittleEndian");
  out << "\">\n";
  out << "  <vtkMultiBlockDataSet>\n";
  int block = 0;
  for (const int r : listed) {
    out << "    <Block index=\"" << block++ << "\" name=\"rank" << r << "\">\n";
    out << "      <DataSet index=\"0\" file=\"" << base_filename.string() << ".rank" << r << ".vti"
        << "\"/>\n";
    out << "    </Block>\n";
  }
  out << "  </vtkMultiBlockDataSet>\n";
  out << "</VTKFile>\n";
  if (!out.good()) {
    log::throw_with_log<std::runtime_error>("io::write_vtk_multiblock_index: write to '{}' failed",
                                            master_path.string());
  }
#else
  (void)base;
  (void)grid;
#endif
}

}  // namespace carbon::io