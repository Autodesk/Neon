// VTK XML backend for carbon::DenseDistributed (slab / pencil / brick). See
// include/carbon/detail/io_vtk_distributed.inl for the on-disk format and the public surface.
//
// The helpers below (byte order, dtype keywords, header parsing) intentionally mirror the Dense
// backend in io_vtk.cpp instead of sharing it, so the Dense path stays byte-for-byte untouched.
// What differs is the geometry: a shard is the rank's owned *box* (WholeExtent = owned extent,
// Origin = global index of its first cell) and rows are read/written with the real haloed row
// stride instead of assuming full-width XY planes.

#include <carbon/dense_distributed.h>
#include <carbon/detail/dense_distributed/layout.h>
#include <carbon/io.h>
#include <carbon/logging.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/distributed.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
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
#include <streambuf>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

[[nodiscard]] auto
element_size_of_keyword(const std::string& kw) noexcept -> size_t {
  if (kw == "char" || kw == "unsigned_char") {
    return 1;
  }
  if (kw == "short" || kw == "unsigned_short") {
    return 2;
  }
  if (kw == "int" || kw == "unsigned_int" || kw == "float") {
    return 4;
  }
  if (kw == "double" || kw == "vtktypeint64" || kw == "vtktypeuint64" || kw == "long" ||
      kw == "unsigned_long") {
    return 8;
  }
  return 0;
}

/** Map a carbon native dtype tag to the VTK XML keyword; null if unknown. */
[[nodiscard]] auto
vtk_xml_keyword_from_native_tag(const char* tag) noexcept -> const char* {
  const std::string_view t(tag, 8);
  if (t.starts_with("f32")) {
    return "Float32";
  }
  if (t.starts_with("f64")) {
    return "Float64";
  }
  if (t.starts_with("i8")) {
    return "Int8";
  }
  if (t.starts_with("u8")) {
    return "UInt8";
  }
  if (t.starts_with("i16")) {
    return "Int16";
  }
  if (t.starts_with("u16")) {
    return "UInt16";
  }
  if (t.starts_with("i32")) {
    return "Int32";
  }
  if (t.starts_with("u32")) {
    return "UInt32";
  }
  if (t.starts_with("i64")) {
    return "Int64";
  }
  if (t.starts_with("u64")) {
    return "UInt64";
  }
  return nullptr;
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
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      c = '_';
    }
  }
  return out.empty() ? std::string("field") : out;
}

[[nodiscard]] auto
xml_escape(std::string_view value) -> std::string {
  std::string out;
  out.reserve(value.size());
  for (const char c : value) {
    switch (c) {
      case '&':
        out += "&amp;";
        break;
      case '<':
        out += "&lt;";
        break;
      case '>':
        out += "&gt;";
        break;
      case '"':
        out += "&quot;";
        break;
      case '\'':
        out += "&apos;";
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

[[nodiscard]] auto
xml_attribute(std::string_view tag,
              std::string_view key) -> std::string {
  const std::string needle = std::string(key) + "=\"";
  const size_t begin = tag.find(needle);
  if (begin == std::string_view::npos) {
    return {};
  }
  const size_t value_begin = begin + needle.size();
  const size_t end = tag.find('"', value_begin);
  if (end == std::string_view::npos) {
    return {};
  }
  return std::string(tag.substr(value_begin, end - value_begin));
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

template <size_t ES>
void
write_payload_le(std::ostream& out,
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
    if (!host_is_little_endian()) {
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
read_payload_le(std::istream& in,
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
            "io::read_vtk: short read in inline binary payload (row {} of {})", r0, rows_total);
      }
      if (!host_is_little_endian()) {
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

void
write_u64_le(std::ostream& out,
             std::uint64_t value) {
  std::uint8_t bytes[8];
  if (host_is_little_endian()) {
    std::memcpy(bytes, &value, sizeof(value));
  } else {
    for (size_t i = 0; i < 8U; ++i) {
      bytes[i] = static_cast<std::uint8_t>(value >> (8U * i));
    }
  }
  out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

[[nodiscard]] auto
read_u64_le(std::istream& in,
            std::string_view path) -> std::uint64_t {
  std::uint8_t bytes[8];
  in.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  if (!in.good()) {
    log::throw_with_log<std::runtime_error>(
        "io::read_vtk: short read while reading the inline binary block size in '{}'", path);
  }
  std::uint64_t value = 0;
  if (host_is_little_endian()) {
    std::memcpy(&value, bytes, sizeof(value));
  } else {
    for (size_t i = 0; i < 8U; ++i) {
      value |= static_cast<std::uint64_t>(bytes[i]) << (8U * i);
    }
  }
  return value;
}

/** Stream buffer that base64-encodes bytes into @p out as one unwrapped line. */
class Base64EncodeBuf final : public std::streambuf {
public:
  explicit Base64EncodeBuf(std::ostream& out) noexcept
      : out_(out) {}

protected:
  auto
  xsputn(const char* s,
         std::streamsize n) -> std::streamsize override {
    for (std::streamsize i = 0; i < n; ++i) {
      in_[in_len_++] = static_cast<std::uint8_t>(s[i]);
      if (in_len_ == in_.size()) {
        emit(in_.size());
        in_len_ = 0;
      }
    }
    return n;
  }

  auto
  overflow(int_type ch) -> int_type override {
    if (!traits_type::eq_int_type(ch, traits_type::eof())) {
      const char c = traits_type::to_char_type(ch);
      (void)xsputn(&c, 1);
    }
    return traits_type::not_eof(ch);
  }

  auto
  sync() -> int override {
    if (in_len_ != 0) {
      emit(in_len_);
      in_len_ = 0;
    }
    return out_.good() ? 0 : -1;
  }

private:
  void
  emit(size_t n) {
    static constexpr char kTable[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const std::uint8_t b0 = in_[0];
    const std::uint8_t b1 = n > 1 ? in_[1] : 0;
    const std::uint8_t b2 = n > 2 ? in_[2] : 0;
    char quad[4];
    quad[0] = kTable[b0 >> 2];
    quad[1] = kTable[((b0 & 0x03U) << 4U) | (b1 >> 4U)];
    quad[2] = n > 1 ? kTable[((b1 & 0x0fU) << 2U) | (b2 >> 6U)] : '=';
    quad[3] = n > 2 ? kTable[b2 & 0x3fU] : '=';
    out_.write(quad, 4);
  }

  std::ostream& out_;
  std::array<std::uint8_t, 3> in_{};
  size_t in_len_ = 0;
};

/** Stream buffer that base64-decodes the inline DataArray body from @p in. */
class Base64DecodeBuf final : public std::streambuf {
public:
  explicit Base64DecodeBuf(std::istream& in)
      : in_(in) {
    setg(nullptr, nullptr, nullptr);
  }

protected:
  auto
  underflow() -> int_type override {
    if (gptr() < egptr()) {
      return traits_type::to_int_type(*gptr());
    }
    out_len_ = 0;
    while (out_len_ == 0) {
      const int v = next_value();
      if (v < 0) {
        return traits_type::eof();
      }
      quad_[quad_len_++] = v;
      if (quad_len_ == quad_.size()) {
        decode_quad();
        quad_len_ = 0;
      }
    }
    setg(reinterpret_cast<char*>(out_.data()), reinterpret_cast<char*>(out_.data()),
         reinterpret_cast<char*>(out_.data() + out_len_));
    return traits_type::to_int_type(out_[0]);
  }

private:
  static constexpr int kPad = 64;
  static constexpr int kEnd = -1;

  auto
  next_value() -> int {
    if (!error_.empty()) {
      return kEnd;
    }
    for (;;) {
      char c = 0;
      if (!in_.get(c)) {
        if (quad_len_ != 0) {
          fail("truncated base64 quartet");
        }
        return kEnd;
      }
      const auto u = static_cast<unsigned char>(c);
      if (std::isspace(u) != 0) {
        continue;
      }
      if (c == '<') {
        if (quad_len_ != 0) {
          fail("truncated base64 quartet");
        }
        return kEnd;
      }
      if (c == '=') {
        if (quad_len_ < 2) {
          fail("misplaced base64 padding");
        }
        return kPad;
      }
      const int v = decode_char(u);
      if (v < 0) {
        fail("invalid base64 character");
        return kEnd;
      }
      return v;
    }
  }

  static auto
  decode_char(unsigned char c) noexcept -> int {
    if (c >= 'A' && c <= 'Z') {
      return static_cast<int>(c - 'A');
    }
    if (c >= 'a' && c <= 'z') {
      return static_cast<int>(c - 'a') + 26;
    }
    if (c >= '0' && c <= '9') {
      return static_cast<int>(c - '0') + 52;
    }
    if (c == '+') {
      return 62;
    }
    if (c == '/') {
      return 63;
    }
    return -1;
  }

  void
  decode_quad() {
    const bool pad2 = quad_[2] == kPad;
    const bool pad3 = quad_[3] == kPad;
    if (quad_[0] == kPad || quad_[1] == kPad || (pad2 && !pad3)) {
      fail("misplaced base64 padding");
      return;
    }
    out_[0] = static_cast<std::uint8_t>((quad_[0] << 2) | (quad_[1] >> 4));
    out_len_ = 1;
    if (!pad2) {
      out_[1] = static_cast<std::uint8_t>(((quad_[1] & 0x0f) << 4) | (quad_[2] >> 2));
      out_len_ = 2;
      if (!pad3) {
        out_[2] = static_cast<std::uint8_t>(((quad_[2] & 0x03) << 6) | quad_[3]);
        out_len_ = 3;
      }
    }
    if (pad2 || pad3) {
      padded_ = true;
    } else if (padded_) {
      fail("base64 data after padding");
    }
  }

  void
  fail(const char* what) {
    if (error_.empty()) {
      error_ = std::string(what);
    }
  }

  std::istream& in_;
  std::array<int, 4> quad_{};
  size_t quad_len_ = 0;
  std::array<std::uint8_t, 3> out_{};
  size_t out_len_ = 0;
  bool padded_ = false;
  std::string error_;
};

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
  // Global index of the first owned cell of a span = buffer origin + halo.
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
      continue;  // this GPU owns nothing
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
// Stream-level building blocks
// ================================================================================================

void
write_vtk_box_header(std::ostream& out,
                     const VtkBox& box) {
  const bool empty = box.empty();
  out << "# vtk DataFile Version 3.0\n";
  out << (empty ? "carbon dense field shard (empty: this rank owns no cells)\n"
                : "carbon dense field shard\n");
  out << "BINARY\n";
  out << "DATASET STRUCTURED_POINTS\n";
  out << "DIMENSIONS " << (empty ? 0 : box.dims[0]) << ' ' << (empty ? 0 : box.dims[1]) << ' '
      << (empty ? 0 : box.dims[2]) << '\n';
  out << "ORIGIN " << box.origin[0] << ' ' << box.origin[1] << ' ' << box.origin[2] << '\n';
  out << "SPACING 1 1 1\n";
  out << "POINT_DATA " << box.points() << '\n';
  if (!out.good()) {
    log::throw_with_log<std::runtime_error>("io::write_vtk: failed to write the shard header");
  }
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
  if (kw == nullptr) {
    log::throw_with_log<std::invalid_argument>(
        "io::write_vtk: field '{}' has a dtype VTK legacy cannot express", name);
  }
  if (components < 1 || components > 4) {
    log::throw_with_log<std::invalid_argument>(
        "io::write_vtk: field '{}' has {} components (legacy VTK SCALARS supports 1..4)", name,
        components);
  }
  if (!box.empty()) {
    if (pieces_cells(pieces) != box.points()) {
      log::throw_with_log<std::invalid_argument>(
          "io::write_vtk: field '{}' pieces hold {} cells but the shard box has {}", name,
          pieces_cells(pieces), box.points());
    }
    for (const auto& p : pieces) {
      if (p.nx != box.dims[0] || p.ny != box.dims[1]) {
        log::throw_with_log<std::invalid_argument>(
            "io::write_vtk: field '{}' piece ({}x{}) does not match the shard box ({}x{})", name,
            p.nx, p.ny, box.dims[0], box.dims[1]);
      }
    }
  }
  out << "SCALARS " << sanitize_name(name) << ' ' << kw << ' ' << components << '\n';
  out << "LOOKUP_TABLE default\n";
  if (!box.empty()) {
    switch (element_size) {
      case 1:
        write_payload<1>(out, components, pieces);
        break;
      case 2:
        write_payload<2>(out, components, pieces);
        break;
      case 4:
        write_payload<4>(out, components, pieces);
        break;
      case 8:
        write_payload<8>(out, components, pieces);
        break;
      default:
        log::throw_with_log<std::invalid_argument>(
            "io::write_vtk: unsupported element size {} for field '{}'", element_size, name);
    }
  }
  out << '\n';
  if (!out.good()) {
    log::throw_with_log<std::runtime_error>("io::write_vtk: write of field '{}' failed", name);
  }
}

void
parse_vtk_box_header(std::istream& in,
                     std::string_view path,
                     const VtkBox& expected,
                     std::vector<VtkBoxFieldRecord>& out_fields) {
  const auto next_line = [&](bool strict) -> std::string {
    std::string line;
    if (!std::getline(in, line)) {
      if (strict) {
        log::throw_with_log<std::invalid_argument>("io::read_vtk: '{}' ended inside the header",
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
  next_line(true);  // free-form title
  const auto fmt = next_line(true);
  if (fmt != "BINARY") {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' is '{}' but only BINARY legacy VTK is supported", path, fmt);
  }
  const auto ds = next_line(true);
  if (ds != "DATASET STRUCTURED_POINTS") {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' is '{}' but only DATASET STRUCTURED_POINTS is supported", path, ds);
  }

  int64_t dims[3] = {0, 0, 0};
  int64_t origin[3] = {0, 0, 0};
  double spacing[3] = {1.0, 1.0, 1.0};
  int64_t point_count = -1;
  while (point_count < 0) {
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
      if (point_count < 0) {
        log::throw_with_log<std::invalid_argument>("io::read_vtk: '{}' has a bad POINT_DATA line",
                                                   path);
      }
    } else if (head == "CELL_DATA") {
      log::throw_with_log<std::invalid_argument>(
          "io::read_vtk: '{}' uses CELL_DATA; carbon writes POINT_DATA only", path);
    } else {
      log::throw_with_log<std::invalid_argument>(
          "io::read_vtk: '{}' has unexpected header line '{}'", path, cur);
    }
  }

  // Validate against the box this rank is about to load into.
  const bool want_empty = expected.empty();
  const int64_t want_dims[3] = {want_empty ? 0 : expected.dims[0],
                                want_empty ? 0 : expected.dims[1],
                                want_empty ? 0 : expected.dims[2]};
  if (dims[0] != want_dims[0] || dims[1] != want_dims[1] || dims[2] != want_dims[2]) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' DIMENSIONS ({}, {}, {}) do not match this rank's owned box ({}, {}, "
        "{})",
        path, dims[0], dims[1], dims[2], want_dims[0], want_dims[1], want_dims[2]);
  }
  if (origin[0] != expected.origin[0] || origin[1] != expected.origin[1] ||
      origin[2] != expected.origin[2]) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' ORIGIN ({}, {}, {}) does not match this rank's box origin ({}, {}, {})",
        path, origin[0], origin[1], origin[2], expected.origin[0], expected.origin[1],
        expected.origin[2]);
  }
  if (spacing[0] != 1.0 || spacing[1] != 1.0 || spacing[2] != 1.0) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' SPACING ({}, {}, {}) is non-unit; carbon uses integer voxel "
        "coordinates",
        path, spacing[0], spacing[1], spacing[2]);
  }
  if (point_count != expected.points()) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' POINT_DATA {} disagrees with the box size {}", path, point_count,
        expected.points());
  }

  // Walk the SCALARS blocks, recording each payload offset.
  while (in) {
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
          "io::read_vtk: unsupported attribute '{}' in '{}'; only SCALARS blocks are accepted",
          head, path);
    }
    VtkBoxFieldRecord rec;
    iss >> rec.name >> rec.vtk_dtype;
    rec.components = 1;
    iss >> rec.components;
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
          "io::read_vtk: failed to query the stream offset of SCALARS '{}'", rec.name);
    }
    const size_t es = element_size_of_keyword(rec.vtk_dtype);
    if (es == 0) {
      log::throw_with_log<std::invalid_argument>(
          "io::read_vtk: SCALARS '{}' has dtype '{}' which carbon does not map back", rec.name,
          rec.vtk_dtype);
    }
    in.seekg(rec.payload_offset + static_cast<std::streamoff>(point_count) *
                                      static_cast<std::streamoff>(rec.components) *
                                      static_cast<std::streamoff>(es));
    out_fields.push_back(std::move(rec));
    const int next_c = in.peek();  // optional trailing newline after the payload
    if (next_c == '\n' || next_c == '\r') {
      in.get();
      if (next_c == '\r' && in.peek() == '\n') {
        in.get();
      }
    }
  }
}

void
read_vtk_box_field_payload(std::istream& in,
                           size_t element_size,
                           int components,
                           const VtkBox& box,
                           const std::vector<VtkBoxPieceMut>& pieces) {
  if (box.empty()) {
    return;
  }
  int64_t cells = 0;
  for (const auto& p : pieces) {
    cells += p.nx * p.ny * p.nz;
    if (p.nx != box.dims[0] || p.ny != box.dims[1]) {
      log::throw_with_log<std::invalid_argument>(
          "io::read_vtk: piece ({}x{}) does not match the shard box ({}x{})", p.nx, p.ny,
          box.dims[0], box.dims[1]);
    }
  }
  if (cells != box.points()) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: pieces hold {} cells but the shard box has {}", cells, box.points());
  }
  switch (element_size) {
    case 1:
      read_payload<1>(in, components, pieces);
      break;
    case 2:
      read_payload<2>(in, components, pieces);
      break;
    case 4:
      read_payload<4>(in, components, pieces);
      break;
    case 8:
      read_payload<8>(in, components, pieces);
      break;
    default:
      log::throw_with_log<std::invalid_argument>("io::read_vtk: unsupported element size {}",
                                                 element_size);
  }
}

// ---- XML ImageData + inline-binary building blocks -----------------------------------------

void
validate_pieces_for_box(std::string_view name,
                        int64_t cells,
                        int components,
                        const VtkBox& box,
                        const std::vector<VtkBoxPiece>& pieces) {
  if (components < 1 || components > 4) {
    log::throw_with_log<std::invalid_argument>(
        "io::write_vtk: field '{}' has {} components (VTK XML supports 1..4 here)", name,
        components);
  }
  if (!box.empty()) {
    if (cells != box.points()) {
      log::throw_with_log<std::invalid_argument>(
          "io::write_vtk: field '{}' pieces hold {} cells but the shard box has {}", name, cells,
          box.points());
    }
    for (const auto& p : pieces) {
      if (p.nx != box.dims[0] || p.ny != box.dims[1]) {
        log::throw_with_log<std::invalid_argument>(
            "io::write_vtk: field '{}' piece ({}x{}) does not match the shard box ({}x{})", name,
            p.nx, p.ny, box.dims[0], box.dims[1]);
      }
    }
  }
}

void
write_vti_box_header(std::ostream& out,
                     const VtkBox& box) {
  const bool empty = box.empty();
  const int64_t nx = empty ? 0 : box.dims[0];
  const int64_t ny = empty ? 0 : box.dims[1];
  const int64_t nz = empty ? 0 : box.dims[2];
  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" "
         "header_type=\"UInt64\">\n";
  out << "  <ImageData WholeExtent=\"0 " << (empty ? -1 : nx - 1) << " 0 "
      << (empty ? -1 : ny - 1) << " 0 " << (empty ? -1 : nz - 1) << "\" Origin=\""
      << box.origin[0] << ' ' << box.origin[1] << ' ' << box.origin[2]
      << "\" Spacing=\"1 1 1\">\n";
  out << "    <Piece Extent=\"0 " << (empty ? -1 : nx - 1) << " 0 " << (empty ? -1 : ny - 1)
      << " 0 " << (empty ? -1 : nz - 1) << "\">\n";
  out << "      <PointData>\n";
  if (!out.good()) {
    log::throw_with_log<std::runtime_error>("io::write_vtk: failed to write the shard header");
  }
}

void
write_vti_box_field_begin(std::ostream& out,
                          std::string_view name,
                          const char* dtype_tag,
                          int components) {
  const char* kw = vtk_xml_keyword_from_native_tag(dtype_tag);
  if (kw == nullptr) {
    log::throw_with_log<std::invalid_argument>(
        "io::write_vtk: field '{}' has a dtype VTK XML cannot express", name);
  }
  out << "        <DataArray type=\"" << kw << "\" Name=\"" << xml_escape(sanitize_name(name))
      << "\" NumberOfComponents=\"" << components << "\" format=\"binary\">\n";
  if (!out.good()) {
    log::throw_with_log<std::runtime_error>("io::write_vtk: failed to write field metadata");
  }
}

void
write_vti_box_footer(std::ostream& out) {
  out << "      </PointData>\n";
  out << "    </Piece>\n";
  out << "  </ImageData>\n";
  out << "</VTKFile>\n";
  if (!out.good()) {
    log::throw_with_log<std::runtime_error>("io::write_vtk: failed to write the shard footer");
  }
}

[[nodiscard]] auto
parse_u64_attribute(std::string_view tag,
                    std::string_view key,
                    std::string_view path) -> std::uint64_t {
  const std::string text = xml_attribute(tag, key);
  if (text.empty()) {
    log::throw_with_log<std::invalid_argument>("io::read_vtk: '{}' is missing attribute '{}'",
                                               path, key);
  }
  try {
    size_t used = 0;
    const auto value = std::stoull(text, &used);
    if (used != text.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return value;
  } catch (const std::exception&) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' has a non-numeric '{}' attribute ('{}')", path, key, text);
  }
}

void
parse_vti_box_header(std::istream& in,
                     std::string_view path,
                     const VtkBox& expected,
                     std::vector<VtkBoxFieldRecord>& out_fields) {
  int64_t whole[6] = {0, -1, 0, -1, 0, -1};
  int64_t origin[3] = {0, 0, 0};
  double spacing[3] = {1.0, 1.0, 1.0};
  bool saw_vtk_file = false;
  bool saw_image_data = false;
  bool saw_point_data = false;
  bool in_point_data = false;

  std::string line;
  while (std::getline(in, line)) {
    rtrim_inplace(line);
    if (line.find("<VTKFile") != std::string::npos) {
      saw_vtk_file = true;
      if (xml_attribute(line, "type") != "ImageData") {
        log::throw_with_log<std::invalid_argument>(
            "io::read_vtk: '{}' is not a VTK XML ImageData file", path);
      }
      if (xml_attribute(line, "byte_order") != "LittleEndian") {
        log::throw_with_log<std::invalid_argument>(
            "io::read_vtk: '{}' must use little-endian byte order", path);
      }
      if (xml_attribute(line, "header_type") != "UInt64") {
        log::throw_with_log<std::invalid_argument>(
            "io::read_vtk: '{}' must use UInt64 inline-binary headers", path);
      }
    } else if (line.find("<ImageData") != std::string::npos) {
      saw_image_data = true;
      const std::string extent = xml_attribute(line, "WholeExtent");
      std::istringstream es(extent);
      if (!(es >> whole[0] >> whole[1] >> whole[2] >> whole[3] >> whole[4] >> whole[5])) {
        log::throw_with_log<std::invalid_argument>(
            "io::read_vtk: '{}' has a bad WholeExtent attribute ('{}')", path, extent);
      }
      const std::string org = xml_attribute(line, "Origin");
      std::istringstream os(org);
      if (!(os >> origin[0] >> origin[1] >> origin[2])) {
        log::throw_with_log<std::invalid_argument>(
            "io::read_vtk: '{}' has a bad Origin attribute ('{}')", path, org);
      }
      const std::string sp = xml_attribute(line, "Spacing");
      std::istringstream ss(sp);
      if (!(ss >> spacing[0] >> spacing[1] >> spacing[2])) {
        log::throw_with_log<std::invalid_argument>(
            "io::read_vtk: '{}' has a bad Spacing attribute ('{}')", path, sp);
      }
    } else if (line.find("<PointData") != std::string::npos) {
      saw_point_data = true;
      in_point_data = true;
    } else if (line.find("</PointData>") != std::string::npos) {
      in_point_data = false;
      break;
    } else if (in_point_data && line.find("<DataArray") != std::string::npos) {
      if (xml_attribute(line, "format") != "binary") {
        log::throw_with_log<std::invalid_argument>(
            "io::read_vtk: '{}' contains a DataArray that is not inline binary", path);
      }
      VtkBoxFieldRecord rec;
      rec.name = xml_attribute(line, "Name");
      rec.vtk_dtype = xml_attribute(line, "type");
      rec.components = static_cast<int>(parse_u64_attribute(line, "NumberOfComponents", path));
      rec.payload_offset = in.tellg();
      if (rec.payload_offset == std::streampos(-1)) {
        log::throw_with_log<std::runtime_error>(
            "io::read_vtk: failed to locate the inline payload of '{}' in '{}'", rec.name, path);
      }
      if (rec.components < 1 || rec.components > 4) {
        log::throw_with_log<std::invalid_argument>(
            "io::read_vtk: DataArray '{}' has components={} (supported: 1..4)", rec.name,
            rec.components);
      }
      out_fields.push_back(std::move(rec));
    } else if (line.find("<AppendedData") != std::string::npos) {
      log::throw_with_log<std::invalid_argument>(
          "io::read_vtk: '{}' uses appended data; carbon reads inline binary only", path);
    }
  }
  if (!saw_vtk_file || !saw_image_data || !saw_point_data) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' is missing VTKFile/ImageData/PointData", path);
  }

  const bool want_empty = expected.empty();
  if (whole[0] != 0 || whole[1] != (want_empty ? -1 : expected.dims[0] - 1) || whole[2] != 0 ||
      whole[3] != (want_empty ? -1 : expected.dims[1] - 1) || whole[4] != 0 ||
      whole[5] != (want_empty ? -1 : expected.dims[2] - 1)) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' WholeExtent ({}, {}, {}, {}, {}, {}) does not match this rank's "
        "owned box ({}, {}, {})",
        path, whole[0], whole[1], whole[2], whole[3], whole[4], whole[5], expected.dims[0],
        expected.dims[1], expected.dims[2]);
  }
  if (origin[0] != expected.origin[0] || origin[1] != expected.origin[1] ||
      origin[2] != expected.origin[2]) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' Origin ({}, {}, {}) does not match this rank's box origin ({}, {}, {})",
        path, origin[0], origin[1], origin[2], expected.origin[0], expected.origin[1],
        expected.origin[2]);
  }
  if (spacing[0] != 1.0 || spacing[1] != 1.0 || spacing[2] != 1.0) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: '{}' Spacing ({}, {}, {}) is non-unit; carbon uses integer voxel "
        "coordinates",
        path, spacing[0], spacing[1], spacing[2]);
  }
}

void
read_vti_box_field_payload(std::istream& in,
                           std::string_view path,
                           const VtkBoxFieldRecord& record,
                           size_t element_size,
                           int components,
                           const VtkBox& box,
                           const std::vector<VtkBoxPieceMut>& pieces) {
  if (box.empty()) {
    return;
  }
  int64_t cells = 0;
  for (const auto& p : pieces) {
    cells += p.nx * p.ny * p.nz;
    if (p.nx != box.dims[0] || p.ny != box.dims[1]) {
      log::throw_with_log<std::invalid_argument>(
          "io::read_vtk: piece ({}x{}) does not match the shard box ({}x{})", p.nx, p.ny,
          box.dims[0], box.dims[1]);
    }
  }
  if (cells != box.points()) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: pieces hold {} cells but the shard box has {}", cells, box.points());
  }
  in.clear();
  in.seekg(record.payload_offset);
  if (!in.good()) {
    log::throw_with_log<std::runtime_error>(
        "io::read_vtk: seek to the inline payload of '{}' failed in '{}'", record.name, path);
  }
  Base64DecodeBuf decoded(in);
  std::istream encoded(&decoded);
  const auto expected_bytes = static_cast<std::uint64_t>(cells) *
                              static_cast<std::uint64_t>(components) *
                              static_cast<std::uint64_t>(element_size);
  const auto got_bytes = read_u64_le(encoded, path);
  if (got_bytes != expected_bytes) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_vtk: inline block for '{}' is {} bytes, expected {} in '{}'", record.name,
        got_bytes, expected_bytes, path);
  }
  switch (element_size) {
    case 1:
      read_payload_le<1>(encoded, components, pieces);
      break;
    case 2:
      read_payload_le<2>(encoded, components, pieces);
      break;
    case 4:
      read_payload_le<4>(encoded, components, pieces);
      break;
    case 8:
      read_payload_le<8>(encoded, components, pieces);
      break;
    default:
      log::throw_with_log<std::invalid_argument>("io::read_vtk: unsupported element size {}",
                                                 element_size);
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
  bool finalized = false;
  std::vector<VtkBoxFieldRecord> read_fields;
  size_t read_fields_cursor = 0;
};

VtkBoxSession::VtkBoxSession() noexcept = default;
VtkBoxSession::~VtkBoxSession() {
  try {
    finalize();
  } catch (...) {
    // Destructors must not throw; write_vtk calls finalize() explicitly to surface I/O failures.
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
  impl_ = std::make_unique<Impl>();
  impl_->path = shard_path_of(path, grid);
  impl_->for_write = true;
  impl_->box = vtk_box_of(grid);
  impl_->writer.open(impl_->path, std::ios::binary | std::ios::trunc);
  if (!impl_->writer) {
    log::throw_with_log<std::runtime_error>("io::write_vtk: failed to open '{}' for writing",
                                            impl_->path);
  }
  write_vti_box_header(impl_->writer, impl_->box);
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
  int64_t cells = 0;
  for (const auto& p : pieces) {
    cells += p.nx * p.ny * p.nz;
  }
  validate_pieces_for_box(name, cells, components, impl_->box, pieces);
  write_vti_box_field_begin(impl_->writer, name, dtype_tag, components);
  const auto bytes = static_cast<std::uint64_t>(cells) *
                     static_cast<std::uint64_t>(components) *
                     static_cast<std::uint64_t>(element_size);
  Base64EncodeBuf encoded_buf(impl_->writer);
  std::ostream encoded(&encoded_buf);
  write_u64_le(encoded, bytes);
  switch (element_size) {
    case 1:
      write_payload_le<1>(encoded, components, pieces);
      break;
    case 2:
      write_payload_le<2>(encoded, components, pieces);
      break;
    case 4:
      write_payload_le<4>(encoded, components, pieces);
      break;
    case 8:
      write_payload_le<8>(encoded, components, pieces);
      break;
    default:
      log::throw_with_log<std::invalid_argument>("io::write_vtk: unsupported element size {}",
                                                 element_size);
  }
  encoded.flush();
  impl_->writer << "\n        </DataArray>\n";
  if (!impl_->writer.good()) {
    log::throw_with_log<std::runtime_error>("io::write_vtk: failed to write field payload");
  }
  ++impl_->num_fields_recorded;
}

void
VtkBoxSession::finalize() {
  if (impl_ == nullptr || !impl_->for_write || impl_->finalized) {
    return;
  }
  write_vti_box_footer(impl_->writer);
  impl_->finalized = true;
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
  parse_vti_box_header(impl_->reader, impl_->path, impl_->box, impl_->read_fields);
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
  // Fields bind by ordinal (names differ between writers).
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
  read_vti_box_field_payload(impl_->reader, impl_->path, rec, element_size, components,
                             impl_->box, pieces);
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
    return;  // single process: the shard at <base> is the whole dataset
  }
  const int world_size = dist->world_size();
  if (world_size <= 1 || dist->world_rank() != 0) {
    return;
  }

  // Which ranks own cells? Every rank's owned box follows from the global split alone, so the
  // index needs no communication. Ranks with an empty box wrote a placeholder shard that is not
  // listed (there is nothing to display).
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
    out << "      <DataSet index=\"0\" file=\"" << detail::xml_escape(base_filename.string())
        << ".rank" << r << ".vti\"/>\n";
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
