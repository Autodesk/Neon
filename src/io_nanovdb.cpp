#include <carbon/dense.h>
#include <carbon/io.h>
#include <carbon/logging.h>

// nanoVDB v11.0.0 ships its writer / builder under nanovdb/util/. Newer (post-OpenVDB
// v12) drops moved the headers under nanovdb/tools/ and nanovdb/io/; carbon currently
// pins to v11.0.0 via FetchContent in CMakeLists.txt so the v11 paths are the canonical
// ones here. If you bump CARBON_OPENVDB_GIT_TAG and the build breaks on these includes,
// switch them to <nanovdb/tools/CreateNanoGrid.h> + <nanovdb/io/IO.h>.
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/HostBuffer.h>
#include <nanovdb/util/IO.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace carbon::io::detail {

[[nodiscard]] auto
shard_path(std::string_view base,
           const Dense& dense) -> std::string;

namespace {

using NanoHandle = nanovdb::GridHandle<nanovdb::HostBuffer>;

/**
 * Compose the per-(field, component) grid name. Single-component fields drop the @c _c0
 * suffix so a 1-component carbon field round-trips as one nanoVDB grid named exactly
 * @c "field_<i>".
 */
[[nodiscard]] auto
component_grid_name(std::string_view field_name,
                    int component,
                    int total_components) -> std::string {
  std::string out(field_name);
  if (total_components > 1) {
    out += "_c";
    out += std::to_string(component);
  }
  return out;
}

/**
 * Build one nanoVDB grid covering every owned cell on this rank's local slabs for a
 * single (field, component). Uses @c nanovdb::build::Grid<T>'s functor overload, which
 * walks the requested CoordBBox and calls our lambda once per voxel — cheaper than
 * setValue() into the WriteAccessor and matches the GridBuilder docs' recommended path.
 *
 * @param starts        Per-slab base pointer to the owned region (component 0).
 * @param z_extents     Per-slab owned-z extent.
 * @param pitches       Per-slab xy_slab_elements (row-stride in element units).
 * @param per_comps     Per-slab linear_storage_elements (component stride).
 * @param z_start       Global z of slab 0's first owned cell on this rank.
 */
template <typename T>
[[nodiscard]] auto
build_grid_for_component(std::string_view name,
                         int component,
                         int64_t global_x,
                         int64_t global_y,
                         const std::vector<const void*>& starts,
                         const std::vector<int64_t>& z_extents,
                         const std::vector<int64_t>& pitches,
                         const std::vector<int64_t>& per_comps,
                         int64_t z_start) -> NanoHandle {
  using BuildGrid = nanovdb::build::Grid<T>;
  BuildGrid grid(/*background=*/T{0}, std::string(name), nanovdb::GridClass::Unknown);

  int64_t total_local_z = 0;
  for (auto ze : z_extents) {
    total_local_z += ze;
  }
  if (total_local_z == 0 || global_x == 0 || global_y == 0) {
    return nanovdb::createNanoGrid(grid);
  }

  // Snapshot per-slab origins for fast lookup inside the functor; the functor
  // dispatches on global z to find which slab owns the cell.
  std::vector<int64_t> slab_z_origin(z_extents.size());
  int64_t z_acc = z_start;
  for (size_t g = 0; g < z_extents.size(); ++g) {
    slab_z_origin[g] = z_acc;
    z_acc += z_extents[g];
  }
  const int64_t z_end = z_start + total_local_z;

  const nanovdb::Coord lo(0, 0, static_cast<int32_t>(z_start));
  const nanovdb::Coord hi(static_cast<int32_t>(global_x - 1), static_cast<int32_t>(global_y - 1),
                          static_cast<int32_t>(z_end - 1));
  const nanovdb::CoordBBox bbox(lo, hi);

  const int64_t comp_stride = component < 0 ? int64_t{0} : static_cast<int64_t>(component);

  grid(
      [&](const nanovdb::Coord& ijk) -> T {
        const int64_t gx = static_cast<int64_t>(ijk[0]);
        const int64_t gy = static_cast<int64_t>(ijk[1]);
        const int64_t gz = static_cast<int64_t>(ijk[2]);
        // Find the slab that owns gz. Linear scan: rank-local slab counts are tiny (one per
        // GPU under carbon's one-rank-per-GPU model), so a binary search is overkill.
        size_t g = 0;
        while (g + 1 < z_extents.size() && gz >= slab_z_origin[g] + z_extents[g]) {
          ++g;
        }
        const int64_t z_local = gz - slab_z_origin[g];
        const T* base = static_cast<const T*>(starts[g]);
        const int64_t off = comp_stride * per_comps[g] + z_local * pitches[g] + gy * global_x + gx;
        return base[off];
      },
      bbox);

  return nanovdb::createNanoGrid(grid);
}

/**
 * Inverse of @ref build_grid_for_component: for one nanoVDB grid, scatter values over
 * every owned cell on every local slab into the field buffer.
 */
template <typename T>
void
scatter_grid_to_component(const NanoHandle& handle,
                          int component,
                          int64_t global_x,
                          int64_t global_y,
                          const std::vector<void*>& starts,
                          const std::vector<int64_t>& z_extents,
                          const std::vector<int64_t>& pitches,
                          const std::vector<int64_t>& per_comps,
                          int64_t z_start) {
  const auto* nano = handle.template grid<T>();
  if (nano == nullptr) {
    log::throw_with_log<std::runtime_error>(
        "io::read_nanovdb: grid handle has wrong build type for the requested scalar; "
        "the writer's T must match the reader's T");
  }
  const auto acc = nano->getAccessor();

  int64_t z_acc = z_start;
  const int64_t comp_stride = component < 0 ? int64_t{0} : static_cast<int64_t>(component);
  for (size_t g = 0; g < z_extents.size(); ++g) {
    const int64_t lim_z = z_extents[g];
    T* base = static_cast<T*>(starts[g]);
    for (int64_t z = 0; z < lim_z; ++z) {
      const int64_t gz = z_acc + z;
      for (int64_t y = 0; y < global_y; ++y) {
        for (int64_t x = 0; x < global_x; ++x) {
          const T v = acc.getValue(nanovdb::Coord(static_cast<int32_t>(x), static_cast<int32_t>(y),
                                                  static_cast<int32_t>(gz)));
          const int64_t off = comp_stride * per_comps[g] + z * pitches[g] + y * global_x + x;
          base[off] = v;
        }
      }
    }
    z_acc += lim_z;
  }
}

[[nodiscard]] auto
dtype_matches(const char* tag,
              const char* want) noexcept -> bool {
  return std::strncmp(tag, want, 8) == 0;
}

}  // namespace

struct NanoVdbSession::Impl {
  std::string path;
  bool for_write = false;
  // Carbon's z-only partition keeps (x, y) uniform across slabs, so we cache the
  // global (x, y) extents at open time — they're what nanoVDB's CoordBBox needs and
  // what the per-slab index arithmetic in write_field / read_field uses.
  int64_t global_x = 0;
  int64_t global_y = 0;
  // Write side: handles accumulate per write_field call; close_write flushes them.
  std::vector<NanoHandle> write_handles;
  std::uint32_t num_fields_recorded = 0;
  // Read side: every grid in the file pre-loaded; read_field looks up by name.
  std::vector<NanoHandle> read_handles;
  std::uint32_t num_fields_inferred = 0;
};

NanoVdbSession::NanoVdbSession() noexcept = default;
NanoVdbSession::~NanoVdbSession() = default;
NanoVdbSession::NanoVdbSession(NanoVdbSession&&) noexcept = default;
auto
NanoVdbSession::operator=(NanoVdbSession&&) noexcept -> NanoVdbSession& = default;

auto
NanoVdbSession::shard_path(std::string_view base,
                           const Dense& dense) -> std::string {
  return ::carbon::io::detail::shard_path(base, dense);
}

void
NanoVdbSession::open_for_write(std::string_view path,
                               const Dense& dense,
                               std::uint32_t num_fields) {
  impl_ = std::make_unique<Impl>();
  impl_->path = NanoVdbSession::shard_path(path, dense);
  impl_->for_write = true;
  impl_->num_fields_recorded = num_fields;
  impl_->global_x = dense.global_extent().x;
  impl_->global_y = dense.global_extent().y;
  impl_->write_handles.reserve(num_fields);
}

void
NanoVdbSession::open_for_read(std::string_view path,
                              const Dense& dense) {
  impl_ = std::make_unique<Impl>();
  impl_->path = NanoVdbSession::shard_path(path, dense);
  impl_->for_write = false;
  impl_->global_x = dense.global_extent().x;
  impl_->global_y = dense.global_extent().y;

  std::vector<NanoHandle> handles;
  try {
    handles = nanovdb::io::readGrids<nanovdb::HostBuffer>(impl_->path);
  } catch (const std::exception& e) {
    log::throw_with_log<std::runtime_error>("io::read_nanovdb: failed to read '{}': {}",
                                            impl_->path, e.what());
  }

  // Infer field count from grid names: any name "field_<i>" or "field_<i>_c<c>" votes
  // for max field index + 1. Other grids (e.g. user-added grids in a hand-edited file)
  // are tolerated but do not bump the count.
  std::uint32_t max_field_seen = 0;
  bool any_field_seen = false;
  for (const auto& h : handles) {
    const auto* meta = h.gridMetaData();
    if (meta == nullptr) {
      continue;
    }
    const std::string name = meta->shortGridName();
    if (name.rfind("field_", 0) != 0) {
      continue;
    }
    const size_t prefix = std::strlen("field_");
    size_t pos = prefix;
    while (pos < name.size() && std::isdigit(static_cast<unsigned char>(name[pos]))) {
      ++pos;
    }
    if (pos == prefix) {
      continue;
    }
    const std::uint32_t idx =
        static_cast<std::uint32_t>(std::stoul(name.substr(prefix, pos - prefix)));
    max_field_seen = std::max(max_field_seen, idx);
    any_field_seen = true;
  }
  impl_->num_fields_inferred = any_field_seen ? max_field_seen + 1U : 0U;
  impl_->read_handles = std::move(handles);
}

void
NanoVdbSession::write_field(std::string_view name,
                            const char* dtype_tag,
                            size_t /*element_size*/,
                            int components,
                            const std::vector<const void*>& starts,
                            const std::vector<int64_t>& z_extents,
                            const std::vector<int64_t>& pitches,
                            const std::vector<int64_t>& per_comps,
                            int64_t z_start) {
  if (impl_ == nullptr || !impl_->for_write) {
    log::throw_with_log<std::runtime_error>(
        "io::write_nanovdb: session is not in a writable state");
  }
  if (z_extents.empty()) {
    return;
  }

  const int64_t global_x = impl_->global_x;
  const int64_t global_y = impl_->global_y;
  const int64_t pitch = pitches.front();
  if (pitch != global_x * global_y) {
    log::throw_with_log<std::runtime_error>(
        "io::write_nanovdb: slab pitch {} disagrees with global_x*global_y {} on field "
        "'{}'; non-tight pitches aren't supported by the nanoVDB backend yet",
        pitch, global_x * global_y, name);
  }

  for (int c = 0; c < components; ++c) {
    const std::string gname = component_grid_name(name, c, components);
    NanoHandle handle;
    if (dtype_matches(dtype_tag, "f32\0\0\0\0")) {
      handle = build_grid_for_component<float>(gname, c, global_x, global_y, starts, z_extents,
                                               pitches, per_comps, z_start);
    } else if (dtype_matches(dtype_tag, "f64\0\0\0\0")) {
      handle = build_grid_for_component<double>(gname, c, global_x, global_y, starts, z_extents,
                                                pitches, per_comps, z_start);
    } else if (dtype_matches(dtype_tag, "i32\0\0\0\0")) {
      handle = build_grid_for_component<int32_t>(gname, c, global_x, global_y, starts, z_extents,
                                                 pitches, per_comps, z_start);
    } else {
      log::throw_with_log<std::invalid_argument>(
          "io::write_nanovdb: dtype '{}' not supported by nanoVDB; pick float, double, or "
          "int32_t (or use carbon::io::write_hdf5 / write_native for arbitrary dtypes)",
          std::string_view(dtype_tag, 8));
    }
    impl_->write_handles.push_back(std::move(handle));
  }
}

void
NanoVdbSession::read_field(std::string_view name,
                           const char* dtype_tag,
                           size_t /*element_size*/,
                           int components,
                           const std::vector<void*>& starts,
                           const std::vector<int64_t>& z_extents,
                           const std::vector<int64_t>& pitches,
                           const std::vector<int64_t>& per_comps,
                           int64_t z_start) {
  if (impl_ == nullptr || impl_->for_write) {
    log::throw_with_log<std::runtime_error>("io::read_nanovdb: session is not in a readable state");
  }
  if (z_extents.empty()) {
    return;
  }
  const int64_t global_x = impl_->global_x;
  const int64_t global_y = impl_->global_y;
  const int64_t pitch = pitches.front();
  if (pitch != global_x * global_y) {
    log::throw_with_log<std::runtime_error>(
        "io::read_nanovdb: slab pitch {} disagrees with global_x*global_y {} on field "
        "'{}'; non-tight pitches aren't supported by the nanoVDB backend yet",
        pitch, global_x * global_y, name);
  }

  for (int c = 0; c < components; ++c) {
    const std::string gname = component_grid_name(name, c, components);
    // Find the matching handle by short grid name.
    const NanoHandle* match = nullptr;
    for (const auto& h : impl_->read_handles) {
      const auto* meta = h.gridMetaData();
      if (meta != nullptr && std::string_view(meta->shortGridName()) == gname) {
        match = &h;
        break;
      }
    }
    if (match == nullptr) {
      log::throw_with_log<std::invalid_argument>("io::read_nanovdb: grid '{}' not found in '{}'",
                                                 gname, impl_->path);
    }

    if (dtype_matches(dtype_tag, "f32\0\0\0\0")) {
      scatter_grid_to_component<float>(*match, c, global_x, global_y, starts, z_extents, pitches,
                                       per_comps, z_start);
    } else if (dtype_matches(dtype_tag, "f64\0\0\0\0")) {
      scatter_grid_to_component<double>(*match, c, global_x, global_y, starts, z_extents, pitches,
                                        per_comps, z_start);
    } else if (dtype_matches(dtype_tag, "i32\0\0\0\0")) {
      scatter_grid_to_component<int32_t>(*match, c, global_x, global_y, starts, z_extents, pitches,
                                         per_comps, z_start);
    } else {
      log::throw_with_log<std::invalid_argument>(
          "io::read_nanovdb: dtype '{}' not supported by nanoVDB", std::string_view(dtype_tag, 8));
    }
  }
}

void
NanoVdbSession::close_write() {
  if (impl_ == nullptr || !impl_->for_write) {
    return;
  }
  if (impl_->write_handles.empty()) {
    // Nothing was written. Truncate the file so a stale read_nanovdb fails fast.
    std::ofstream truncate(impl_->path, std::ios::binary | std::ios::trunc);
    return;
  }
  try {
    nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>(impl_->path, impl_->write_handles);
  } catch (const std::exception& e) {
    log::throw_with_log<std::runtime_error>("io::write_nanovdb: failed to write '{}': {}",
                                            impl_->path, e.what());
  }
  impl_->write_handles.clear();
}

auto
NanoVdbSession::num_fields() const noexcept -> std::uint32_t {
  if (impl_ == nullptr) {
    return 0U;
  }
  return impl_->for_write ? impl_->num_fields_recorded : impl_->num_fields_inferred;
}

}  // namespace carbon::io::detail
