#include <carbon/dense_disg.h>
#include <carbon/distributed.h>
#include <carbon/io.h>
#include <carbon/logging.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace carbon::io::detail {

void
require(bool cond,
        std::string_view what) {
  if (!cond) {
    log::throw_with_log<std::invalid_argument>("io::native: {}", what);
  }
}

namespace {

// Shared implementation for the per-grid shard_path overloads — extracted because both
// Cartesian families speak the same Distributed handle through their owned System.
template <typename Grid>
[[nodiscard]] auto
shard_path_impl(std::string_view base,
                const Grid& grid) -> std::string {
#ifdef CARBON_HAVE_MPI_NCCL
  if (grid.system().distributed() != nullptr) {
    std::ostringstream oss;
    oss << base << ".rank" << grid.system().distributed()->world_rank();
    return oss.str();
  }
#else
  (void)grid;
#endif
  return std::string(base);
}

// Shared implementation for the per-grid make_native_header overloads — both families
// expose identical metadata accessors (global_extent, local_slab_offset, gpu_count,
// spans), so the header bytes match exactly. A Dense-written shard can therefore be
// read into a DenseDisg field of matching global_extent, and vice versa.
template <typename Grid>
[[nodiscard]] auto
make_native_header_impl(const Grid& grid,
                        std::uint32_t num_fields) -> ::carbon::io::detail::NativeFileHeader {
  ::carbon::io::detail::NativeFileHeader hdr{};
  std::memcpy(hdr.magic, ::carbon::io::detail::kNativeMagic,
              sizeof(::carbon::io::detail::kNativeMagic));
  hdr.version = ::carbon::io::detail::kNativeVersion;
  hdr.endianness_check = ::carbon::io::detail::kEndiannessCheck;
  hdr.global_extent[0] = grid.global_extent().x;
  hdr.global_extent[1] = grid.global_extent().y;
  hdr.global_extent[2] = grid.global_extent().z;
  hdr.local_slab_offset = static_cast<std::uint64_t>(grid.local_slab_offset());
  hdr.local_slab_count = static_cast<std::uint64_t>(grid.gpu_count());

  // Local owned extent: (global.x, global.y, sum of owned z across local slabs). x and y
  // are uniform across slabs in carbon's z-only partitioning.
  int64_t local_z_total = 0;
  for (const auto& span : grid.spans()) {
    local_z_total += span.logic_extent().z;
  }
  hdr.local_owned_extent[0] = grid.global_extent().x;
  hdr.local_owned_extent[1] = grid.global_extent().y;
  hdr.local_owned_extent[2] = local_z_total;

  hdr.num_fields = num_fields;
  hdr.reserved = 0;
  return hdr;
}

// Shared implementation for the per-grid validate_native_header overloads. Same
// rationale as make_native_header_impl: the header is family-agnostic, so the
// validator's constraints are too.
template <typename Grid>
void
validate_native_header_impl(const ::carbon::io::detail::NativeFileHeader& hdr,
                            const Grid& grid) {
  if (std::memcmp(hdr.magic, ::carbon::io::detail::kNativeMagic,
                  sizeof(::carbon::io::detail::kNativeMagic)) != 0) {
    ::carbon::log::throw_with_log<std::invalid_argument>(
        "io::read_native: bad magic (expected 'CARBnB01'); not a carbon native file");
  }
  if (hdr.version != ::carbon::io::detail::kNativeVersion) {
    ::carbon::log::throw_with_log<std::invalid_argument>(
        "io::read_native: file version {} unsupported (this build expects {})", hdr.version,
        ::carbon::io::detail::kNativeVersion);
  }
  if (hdr.endianness_check != ::carbon::io::detail::kEndiannessCheck) {
    ::carbon::log::throw_with_log<std::invalid_argument>(
        "io::read_native: endianness check failed (got 0x{:x}, expected 0x{:x}); the file was "
        "produced on a host with the opposite byte order",
        hdr.endianness_check, ::carbon::io::detail::kEndiannessCheck);
  }
  const auto ge = grid.global_extent();
  if (hdr.global_extent[0] != ge.x || hdr.global_extent[1] != ge.y ||
      hdr.global_extent[2] != ge.z) {
    ::carbon::log::throw_with_log<std::invalid_argument>(
        "io::read_native: file global extent ({}, {}, {}) does not match grid global extent "
        "({}, {}, {})",
        hdr.global_extent[0], hdr.global_extent[1], hdr.global_extent[2], ge.x, ge.y, ge.z);
  }
  if (hdr.local_slab_offset != static_cast<std::uint64_t>(grid.local_slab_offset()) ||
      hdr.local_slab_count != static_cast<std::uint64_t>(grid.gpu_count())) {
    ::carbon::log::throw_with_log<std::invalid_argument>(
        "io::read_native: file slab range [{}, {}) does not match grid local slab range "
        "[{}, {})",
        hdr.local_slab_offset, hdr.local_slab_offset + hdr.local_slab_count,
        grid.local_slab_offset(), grid.local_slab_offset() + grid.gpu_count());
  }
  int64_t local_z_total = 0;
  for (const auto& span : grid.spans()) {
    local_z_total += span.logic_extent().z;
  }
  if (hdr.local_owned_extent[0] != ge.x || hdr.local_owned_extent[1] != ge.y ||
      hdr.local_owned_extent[2] != local_z_total) {
    ::carbon::log::throw_with_log<std::invalid_argument>(
        "io::read_native: file local owned extent ({}, {}, {}) does not match grid's "
        "({}, {}, {})",
        hdr.local_owned_extent[0], hdr.local_owned_extent[1], hdr.local_owned_extent[2], ge.x, ge.y,
        local_z_total);
  }
}

}  // namespace

auto
shard_path(std::string_view base,
           const Dense& dense) -> std::string {
  return shard_path_impl(base, dense);
}

auto
shard_path(std::string_view base,
           const DenseDisg& dense_disg) -> std::string {
  return shard_path_impl(base, dense_disg);
}

auto
make_native_header(const Dense& dense,
                   std::uint32_t num_fields) -> NativeFileHeader {
  return make_native_header_impl(dense, num_fields);
}

auto
make_native_header(const DenseDisg& dense_disg,
                   std::uint32_t num_fields) -> NativeFileHeader {
  return make_native_header_impl(dense_disg, num_fields);
}

void
write_native_header(std::ofstream& out,
                    const NativeFileHeader& hdr) {
  out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
  if (!out.good()) {
    log::throw_with_log<std::runtime_error>("io::write_native: failed to write {}-byte file header",
                                            sizeof(hdr));
  }
}

auto
read_native_header(std::ifstream& in) -> NativeFileHeader {
  NativeFileHeader hdr{};
  in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
  if (!in.good()) {
    log::throw_with_log<std::runtime_error>("io::read_native: failed to read {}-byte file header",
                                            sizeof(hdr));
  }
  return hdr;
}

void
validate_native_header(const NativeFileHeader& hdr,
                       const Dense& dense) {
  validate_native_header_impl(hdr, dense);
}

void
validate_native_header(const NativeFileHeader& hdr,
                       const DenseDisg& dense_disg) {
  validate_native_header_impl(hdr, dense_disg);
}

void
write_native_field_meta(std::ofstream& out,
                        std::string_view name,
                        const char* dtype_tag,
                        std::int32_t components) {
  // Layout: uint32 name_length | name bytes | char dtype[8] | int32 components.
  const auto name_len = static_cast<std::uint32_t>(std::min<size_t>(name.size(), 256U));
  out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
  out.write(name.data(), static_cast<std::streamsize>(name_len));
  out.write(dtype_tag, 8);
  out.write(reinterpret_cast<const char*>(&components), sizeof(components));
  if (!out.good()) {
    log::throw_with_log<std::runtime_error>("io::write_native: failed to write field meta");
  }
}

auto
read_native_field_meta(std::ifstream& in) -> NativeFieldMeta {
  NativeFieldMeta meta{};
  std::uint32_t name_len = 0;
  in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
  if (!in.good() || name_len > 4096U) {
    log::throw_with_log<std::runtime_error>(
        "io::read_native: failed to read field meta (name_len={})", name_len);
  }
  meta.name.resize(name_len);
  if (name_len > 0U) {
    in.read(meta.name.data(), static_cast<std::streamsize>(name_len));
  }
  in.read(meta.dtype, 8);
  meta.dtype[8] = '\0';
  in.read(reinterpret_cast<char*>(&meta.components), sizeof(meta.components));
  if (!in.good()) {
    log::throw_with_log<std::runtime_error>("io::read_native: failed to read field meta");
  }
  return meta;
}

}  // namespace carbon::io::detail
