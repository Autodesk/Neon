#include <carbon/distributed.h>
#include <carbon/io.h>
#include <carbon/logging.h>

#include <hdf5.h>

#ifdef CARBON_HAVE_PHDF5
#include <mpi.h>
#endif

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace carbon::io::detail {

namespace {

constexpr std::uint32_t kHdf5FormatVersion = 1U;

[[nodiscard]] auto
hdf5_native_type(std::string_view dtype_tag) -> hid_t {
  // Tags are 8-byte NUL-padded; compare on the first NUL-terminated chars only.
  const std::string_view t(dtype_tag.data(),
                           std::min<size_t>(dtype_tag.size(), std::strlen(dtype_tag.data())));
  if (t == "f32") {
    return H5T_NATIVE_FLOAT;
  }
  if (t == "f64") {
    return H5T_NATIVE_DOUBLE;
  }
  if (t == "i8") {
    return H5T_NATIVE_INT8;
  }
  if (t == "i16") {
    return H5T_NATIVE_INT16;
  }
  if (t == "i32") {
    return H5T_NATIVE_INT32;
  }
  if (t == "i64") {
    return H5T_NATIVE_INT64;
  }
  if (t == "u8") {
    return H5T_NATIVE_UINT8;
  }
  if (t == "u16") {
    return H5T_NATIVE_UINT16;
  }
  if (t == "u32") {
    return H5T_NATIVE_UINT32;
  }
  if (t == "u64") {
    return H5T_NATIVE_UINT64;
  }
  log::throw_with_log<std::invalid_argument>(
      "io::hdf5: unsupported dtype tag '{}' (no native HDF5 type mapping)", t);
  return -1;  // unreachable
}

// Best-effort dtype tag from an HDF5 type id; reverse of hdf5_native_type. Compares against
// the same set of native types so it round-trips cleanly with what we wrote.
[[nodiscard]] auto
hdf5_dtype_tag(hid_t dtype) -> std::string {
  const std::array<std::pair<hid_t, const char*>, 10> table = {{
      {H5T_NATIVE_FLOAT, "f32"},
      {H5T_NATIVE_DOUBLE, "f64"},
      {H5T_NATIVE_INT8, "i8"},
      {H5T_NATIVE_INT16, "i16"},
      {H5T_NATIVE_INT32, "i32"},
      {H5T_NATIVE_INT64, "i64"},
      {H5T_NATIVE_UINT8, "u8"},
      {H5T_NATIVE_UINT16, "u16"},
      {H5T_NATIVE_UINT32, "u32"},
      {H5T_NATIVE_UINT64, "u64"},
  }};
  for (const auto& entry : table) {
    if (H5Tequal(dtype, entry.first) > 0) {
      return entry.second;
    }
  }
  return "?";
}

}  // namespace

struct Hdf5Session::Impl {
  hid_t file{H5I_INVALID_HID};
  hid_t fapl{H5I_INVALID_HID};
  hid_t dxpl{H5I_INVALID_HID};
  std::array<hsize_t, 4> file_shape{};  // (Components, global.z, global.y, global.x)
  int64_t global_extent[3]{};
  std::uint32_t num_fields{0};
  bool reading{false};
  bool collective{false};

  ~Impl() {
    if (dxpl != H5I_INVALID_HID) {
      H5Pclose(dxpl);
    }
    if (file != H5I_INVALID_HID) {
      H5Fclose(file);
    }
    if (fapl != H5I_INVALID_HID) {
      H5Pclose(fapl);
    }
  }
};

Hdf5Session::Hdf5Session() noexcept = default;
Hdf5Session::~Hdf5Session() = default;
Hdf5Session::Hdf5Session(Hdf5Session&&) noexcept = default;
auto
Hdf5Session::operator=(Hdf5Session&&) noexcept -> Hdf5Session& = default;

auto
Hdf5Session::num_fields() const noexcept -> std::uint32_t {
  return impl_ ? impl_->num_fields : 0U;
}

void
Hdf5Session::open_for_write(std::string_view path,
                            const Dense& dense,
                            std::uint32_t num_fields) {
  impl_ = std::make_unique<Impl>();
  impl_->reading = false;
  impl_->num_fields = num_fields;
  impl_->global_extent[0] = dense.global_extent().x;
  impl_->global_extent[1] = dense.global_extent().y;
  impl_->global_extent[2] = dense.global_extent().z;

  const bool distributed = (dense.system().distributed() != nullptr);
  if (distributed) {
#ifdef CARBON_HAVE_PHDF5
    impl_->collective = true;
    impl_->fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(impl_->fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    impl_->dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(impl_->dxpl, H5FD_MPIO_COLLECTIVE);
#else
    log::throw_with_log<std::runtime_error>(
        "io::write_hdf5: this carbon build was compiled without parallel HDF5 support, but "
        "the Dense is distributed across MPI ranks. Rebuild with CARBON_ENABLE_HDF5=ON and a "
        "parallel HDF5 install (e.g. libhdf5-mpich-dev) to enable collective writes.");
#endif
  } else {
    impl_->fapl = H5P_DEFAULT;  // not closed in dtor (sentinel)
    impl_->dxpl = H5P_DEFAULT;
  }

  const std::string path_owned(path);
  impl_->file = H5Fcreate(path_owned.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                          impl_->fapl == H5I_INVALID_HID ? H5P_DEFAULT : impl_->fapl);
  if (impl_->file < 0) {
    log::throw_with_log<std::runtime_error>("io::write_hdf5: H5Fcreate('{}') failed", path);
  }

  // Root attributes: global extent + format version. Created from rank 0 only when
  // collective, otherwise from every (single) writer.
  hsize_t three = 3;
  hid_t space = H5Screate_simple(1, &three, nullptr);
  hid_t attr =
      H5Acreate2(impl_->file, "global_extent", H5T_NATIVE_INT64, space, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr, H5T_NATIVE_INT64, impl_->global_extent);
  H5Aclose(attr);
  H5Sclose(space);

  hid_t scalar_space = H5Screate(H5S_SCALAR);
  hid_t ver_attr =
      H5Acreate2(impl_->file, "version", H5T_NATIVE_UINT32, scalar_space, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(ver_attr, H5T_NATIVE_UINT32, &kHdf5FormatVersion);
  H5Aclose(ver_attr);

  hid_t nf_attr = H5Acreate2(impl_->file, "num_fields", H5T_NATIVE_UINT32, scalar_space,
                             H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(nf_attr, H5T_NATIVE_UINT32, &num_fields);
  H5Aclose(nf_attr);
  H5Sclose(scalar_space);

  // file_shape will be set per-field (Components varies); just record the global volume part.
  impl_->file_shape[1] = static_cast<hsize_t>(impl_->global_extent[2]);
  impl_->file_shape[2] = static_cast<hsize_t>(impl_->global_extent[1]);
  impl_->file_shape[3] = static_cast<hsize_t>(impl_->global_extent[0]);
}

void
Hdf5Session::open_for_read(std::string_view path,
                           const Dense& dense) {
  impl_ = std::make_unique<Impl>();
  impl_->reading = true;
  impl_->global_extent[0] = dense.global_extent().x;
  impl_->global_extent[1] = dense.global_extent().y;
  impl_->global_extent[2] = dense.global_extent().z;

  const bool distributed = (dense.system().distributed() != nullptr);
  if (distributed) {
#ifdef CARBON_HAVE_PHDF5
    impl_->collective = true;
    impl_->fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(impl_->fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    impl_->dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(impl_->dxpl, H5FD_MPIO_COLLECTIVE);
#else
    log::throw_with_log<std::runtime_error>(
        "io::read_hdf5: this carbon build was compiled without parallel HDF5 support, but "
        "the Dense is distributed across MPI ranks. Rebuild with CARBON_ENABLE_HDF5=ON and a "
        "parallel HDF5 install (e.g. libhdf5-mpich-dev) to enable collective reads.");
#endif
  } else {
    impl_->fapl = H5P_DEFAULT;
    impl_->dxpl = H5P_DEFAULT;
  }

  const std::string path_owned(path);
  impl_->file = H5Fopen(path_owned.c_str(), H5F_ACC_RDONLY,
                        impl_->fapl == H5I_INVALID_HID ? H5P_DEFAULT : impl_->fapl);
  if (impl_->file < 0) {
    log::throw_with_log<std::runtime_error>("io::read_hdf5: H5Fopen('{}') failed", path);
  }

  // Validate global_extent attribute against the in-memory Dense.
  hid_t attr = H5Aopen(impl_->file, "global_extent", H5P_DEFAULT);
  if (attr < 0) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_hdf5: file '{}' missing required 'global_extent' root attribute", path);
  }
  int64_t file_extent[3] = {0, 0, 0};
  H5Aread(attr, H5T_NATIVE_INT64, file_extent);
  H5Aclose(attr);
  if (file_extent[0] != impl_->global_extent[0] || file_extent[1] != impl_->global_extent[1] ||
      file_extent[2] != impl_->global_extent[2]) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_hdf5: file '{}' global extent ({}, {}, {}) does not match Dense ({}, {}, {})",
        path, file_extent[0], file_extent[1], file_extent[2], impl_->global_extent[0],
        impl_->global_extent[1], impl_->global_extent[2]);
  }

  hid_t nf_attr = H5Aopen(impl_->file, "num_fields", H5P_DEFAULT);
  if (nf_attr < 0) {
    log::throw_with_log<std::invalid_argument>(
        "io::read_hdf5: file '{}' missing required 'num_fields' root attribute", path);
  }
  H5Aread(nf_attr, H5T_NATIVE_UINT32, &impl_->num_fields);
  H5Aclose(nf_attr);

  impl_->file_shape[1] = static_cast<hsize_t>(impl_->global_extent[2]);
  impl_->file_shape[2] = static_cast<hsize_t>(impl_->global_extent[1]);
  impl_->file_shape[3] = static_cast<hsize_t>(impl_->global_extent[0]);
}

void
Hdf5Session::write_field(std::string_view name,
                         const char* dtype_tag,
                         size_t /*element_size*/,
                         int components,
                         const std::vector<const void*>& slab_owned_starts,
                         const std::vector<int64_t>& slab_owned_z_extents,
                         const std::vector<int64_t>& slab_pitches,
                         const std::vector<int64_t>& slab_per_comp_strides,
                         int64_t slab_global_z_start) {
  const hid_t dtype = hdf5_native_type(dtype_tag);

  std::array<hsize_t, 4> file_shape = {static_cast<hsize_t>(components), impl_->file_shape[1],
                                       impl_->file_shape[2], impl_->file_shape[3]};
  hid_t fspace = H5Screate_simple(4, file_shape.data(), nullptr);
  const std::string name_owned(name);
  hid_t dxpl_use = (impl_->dxpl == H5I_INVALID_HID) ? H5P_DEFAULT : impl_->dxpl;
  hid_t dataset = H5Dcreate2(impl_->file, name_owned.c_str(), dtype, fspace, H5P_DEFAULT,
                             H5P_DEFAULT, H5P_DEFAULT);
  if (dataset < 0) {
    H5Sclose(fspace);
    log::throw_with_log<std::runtime_error>("io::write_hdf5: H5Dcreate2('{}') failed", name);
  }

  // For each (slab, component) pair, write a contiguous block of shape (1, lim_z, gy, gx).
  // The source layout in memory: per slab, plane c at offset c * per_comp_stride from the
  // slab's owned-start pointer; within a plane, lim_z * gy * gx elements (carbon's z-only
  // halo means the per-z-row stride is gx, no padding).
  int64_t z_cursor = slab_global_z_start;
  for (size_t g = 0; g < slab_owned_starts.size(); ++g) {
    const int64_t lim_z = slab_owned_z_extents[g];
    const int64_t pitch = slab_pitches[g];
    const int64_t per_comp = slab_per_comp_strides[g];
    if (pitch != impl_->global_extent[0] * impl_->global_extent[1]) {
      H5Dclose(dataset);
      H5Sclose(fspace);
      log::throw_with_log<std::runtime_error>(
          "io::write_hdf5: span pitch {} differs from global x*y {} on slab {}; only z-only "
          "halo layouts are supported by the HDF5 writer today",
          pitch, impl_->global_extent[0] * impl_->global_extent[1], g);
    }
    for (int c = 0; c < components; ++c) {
      std::array<hsize_t, 4> mem_dims = {1U, static_cast<hsize_t>(lim_z), impl_->file_shape[2],
                                         impl_->file_shape[3]};
      hid_t mspace = H5Screate_simple(4, mem_dims.data(), nullptr);

      std::array<hsize_t, 4> file_offset = {static_cast<hsize_t>(c), static_cast<hsize_t>(z_cursor),
                                            0U, 0U};
      H5Sselect_hyperslab(fspace, H5S_SELECT_SET, file_offset.data(), nullptr, mem_dims.data(),
                          nullptr);

      const auto* slab_base = static_cast<const std::byte*>(slab_owned_starts[g]);
      const size_t plane_byte_offset =
          static_cast<size_t>(c) * static_cast<size_t>(per_comp) * H5Tget_size(dtype);
      const void* src = slab_base + plane_byte_offset;
      H5Dwrite(dataset, dtype, mspace, fspace, dxpl_use, src);
      H5Sclose(mspace);
    }
    z_cursor += lim_z;
  }

  H5Dclose(dataset);
  H5Sclose(fspace);
}

void
Hdf5Session::read_field(std::string_view name,
                        const char* dtype_tag,
                        size_t /*element_size*/,
                        int components,
                        const std::vector<void*>& slab_owned_starts,
                        const std::vector<int64_t>& slab_owned_z_extents,
                        const std::vector<int64_t>& slab_pitches,
                        const std::vector<int64_t>& slab_per_comp_strides,
                        int64_t slab_global_z_start) {
  const hid_t dtype = hdf5_native_type(dtype_tag);
  const std::string name_owned(name);
  hid_t dataset = H5Dopen2(impl_->file, name_owned.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    log::throw_with_log<std::runtime_error>("io::read_hdf5: H5Dopen2('{}') failed", name);
  }

  // Validate dataset dtype + rank + shape against the binding (T, Components).
  hid_t file_dtype = H5Dget_type(dataset);
  if (H5Tequal(file_dtype, dtype) <= 0) {
    const std::string file_tag = hdf5_dtype_tag(file_dtype);
    H5Tclose(file_dtype);
    H5Dclose(dataset);
    log::throw_with_log<std::invalid_argument>(
        "io::read_hdf5: dataset '{}' dtype '{}' does not match bound Field's dtype '{}'", name,
        file_tag, dtype_tag);
  }
  H5Tclose(file_dtype);

  hid_t fspace = H5Dget_space(dataset);
  if (H5Sget_simple_extent_ndims(fspace) != 4) {
    H5Sclose(fspace);
    H5Dclose(dataset);
    log::throw_with_log<std::invalid_argument>(
        "io::read_hdf5: dataset '{}' rank != 4 (carbon HDF5 layout is "
        "(Components, z, y, x))",
        name);
  }
  std::array<hsize_t, 4> dims{};
  H5Sget_simple_extent_dims(fspace, dims.data(), nullptr);
  if (dims[0] != static_cast<hsize_t>(components)) {
    H5Sclose(fspace);
    H5Dclose(dataset);
    log::throw_with_log<std::invalid_argument>(
        "io::read_hdf5: dataset '{}' components {} does not match bound Field's Components {}",
        name, dims[0], components);
  }
  if (dims[1] != impl_->file_shape[1] || dims[2] != impl_->file_shape[2] ||
      dims[3] != impl_->file_shape[3]) {
    H5Sclose(fspace);
    H5Dclose(dataset);
    log::throw_with_log<std::invalid_argument>(
        "io::read_hdf5: dataset '{}' shape ({}, {}, {}, {}) does not match Dense's "
        "(Components, {}, {}, {})",
        name, dims[0], dims[1], dims[2], dims[3], impl_->file_shape[1], impl_->file_shape[2],
        impl_->file_shape[3]);
  }

  hid_t dxpl_use = (impl_->dxpl == H5I_INVALID_HID) ? H5P_DEFAULT : impl_->dxpl;
  int64_t z_cursor = slab_global_z_start;
  for (size_t g = 0; g < slab_owned_starts.size(); ++g) {
    const int64_t lim_z = slab_owned_z_extents[g];
    const int64_t pitch = slab_pitches[g];
    const int64_t per_comp = slab_per_comp_strides[g];
    if (pitch != impl_->global_extent[0] * impl_->global_extent[1]) {
      H5Sclose(fspace);
      H5Dclose(dataset);
      log::throw_with_log<std::runtime_error>(
          "io::read_hdf5: span pitch {} differs from global x*y {} on slab {}; only z-only "
          "halo layouts are supported by the HDF5 reader today",
          pitch, impl_->global_extent[0] * impl_->global_extent[1], g);
    }
    for (int c = 0; c < components; ++c) {
      std::array<hsize_t, 4> mem_dims = {1U, static_cast<hsize_t>(lim_z), impl_->file_shape[2],
                                         impl_->file_shape[3]};
      hid_t mspace = H5Screate_simple(4, mem_dims.data(), nullptr);
      std::array<hsize_t, 4> file_offset = {static_cast<hsize_t>(c), static_cast<hsize_t>(z_cursor),
                                            0U, 0U};
      H5Sselect_hyperslab(fspace, H5S_SELECT_SET, file_offset.data(), nullptr, mem_dims.data(),
                          nullptr);

      auto* slab_base = static_cast<std::byte*>(slab_owned_starts[g]);
      const size_t plane_byte_offset =
          static_cast<size_t>(c) * static_cast<size_t>(per_comp) * H5Tget_size(dtype);
      void* dst = slab_base + plane_byte_offset;
      H5Dread(dataset, dtype, mspace, fspace, dxpl_use, dst);
      H5Sclose(mspace);
    }
    z_cursor += lim_z;
  }

  H5Sclose(fspace);
  H5Dclose(dataset);
}

}  // namespace carbon::io::detail
