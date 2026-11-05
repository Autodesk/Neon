// io_vtk_distributed.inl — VTK XML backend for carbon::DenseDistributed, split out of
// <carbon/io.h>. Tail include of io.h (see io_vtk.inl for the include-context contract); not a
// standalone header.
//
// One XML ImageData `.vti` shard per rank under MPI, plus a `.vtm` multiblock index written by
// rank 0. Geometry is the rank's owned bounding box, generalised from "full XY planes, z-slab":
//
//   <VTKFile type="ImageData" byte_order="LittleEndian" header_type="UInt64">
//     <ImageData WholeExtent="0 nx-1 0 ny-1 0 nz-1"
//                Origin="x0 y0 z0" Spacing="1 1 1">
//       <Piece Extent="0 nx-1 0 ny-1 0 nz-1">
//         <PointData>
//           <DataArray type="Float32" Name="rho" NumberOfComponents="1"
//                      format="binary">
//             <base64 block>
//           </DataArray>
//         </PointData>
//       </Piece>
//     </ImageData>
//   </VTKFile>
//
// Each inline block is base64-encoded and starts with an 8-byte little-endian byte count,
// followed by the little-endian component-fastest payload (x fastest, then y, then z). The
// base64 is written as one line: VTK's inline reader mishandles wrapped blocks once the payload
// grows past a few MiB. Inline binary avoids the VTK 9.2 XML parser's raw-appended-data buffer
// limit while remaining a normal VTK XML file. A rank that owns no cells writes a valid empty
// shard (`WholeExtent` / `Piece Extent` are `0 -1` on every axis); the `.vtm` index lists only
// the non-empty shards.
//
// Index space (Spacing 1, Origin in cells) is deliberate: it is what the Dense writer/reader use,
// costs nothing extra to write, and keeps shards of both grid families interchangeable in
// ParaView. The span's physical origin/spacing are not applied.
//
// The legacy stream-level helpers below are retained for the host VTK tests. The session itself
// uses the XML helpers in io_vtk_distributed.cpp.
namespace detail {

/** Pointer-parametrised block of owned cells inside a haloed per-GPU buffer. */
template <typename P>
struct VtkBoxPieceT {
  /** First @b owned cell of component 0. */
  P start{};
  /** Owned extent of the block. */
  int64_t nx{0};
  int64_t ny{0};
  int64_t nz{0};
  /** Elements between consecutive rows (the haloed x extent). */
  int64_t row_stride{0};
  /** Elements between consecutive z planes. */
  int64_t plane_pitch{0};
  /** Elements between consecutive components. */
  int64_t comp_stride{0};
};
using VtkBoxPiece = VtkBoxPieceT<const void*>;
using VtkBoxPieceMut = VtkBoxPieceT<void*>;

/** Bounding box of one rank's shard in global index space; all-zero @c dims means "empty". */
struct VtkBox {
  int64_t dims[3] = {0, 0, 0};
  int64_t origin[3] = {0, 0, 0};

  [[nodiscard]] auto
  empty() const noexcept -> bool {
    return dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0;
  }
  [[nodiscard]] auto
  points() const noexcept -> int64_t {
    return empty() ? 0 : dims[0] * dims[1] * dims[2];
  }
};

/** Per-DataArray record found while parsing a shard header. */
struct VtkBoxFieldRecord {
  std::string name;
  std::string vtk_dtype;
  int components = 0;
  std::streampos payload_offset = 0;
};

/**
 * The shard box of @p grid on this rank: the union of its local spans' owned cells. Single-GPU
 * ranks give their span's owned box; a single-process multi-GPU slab stacks the per-GPU z ranges.
 * Throws @c std::invalid_argument if the local spans do not form one box. A rank owning no cells
 * returns an empty box whose @c origin is still the global index of its (empty) owned region.
 */
[[nodiscard]] CARBON_API auto
vtk_box_of(const DenseDistributed& grid) -> VtkBox;

// ---- stream-level building blocks (what the session and the unit tests share) ---------------

/** Write the legacy header for @p box (placeholder header when @c box.empty()). */
CARBON_API void
write_vtk_box_header(std::ostream& out,
                     const VtkBox& box);

/**
 * Write one SCALARS block: header line, LOOKUP_TABLE, then (unless @p box is empty) the payload
 * gathered from @p pieces, in file order, as big-endian component-fastest records.
 */
CARBON_API void
write_vtk_box_field(std::ostream& out,
                    std::string_view name,
                    const char* dtype_tag,
                    size_t element_size,
                    int components,
                    const VtkBox& box,
                    const std::vector<VtkBoxPiece>& pieces);

/**
 * Parse a shard header and record every SCALARS block. DIMENSIONS / ORIGIN / POINT_DATA must match
 * @p expected (an empty @p expected requires the placeholder form); SPACING must be unit.
 */
CARBON_API void
parse_vtk_box_header(std::istream& in,
                     std::string_view path,
                     const VtkBox& expected,
                     std::vector<VtkBoxFieldRecord>& out_fields);

/** Read one field's payload (stream already positioned) and scatter it into @p pieces. */
CARBON_API void
read_vtk_box_field_payload(std::istream& in,
                           size_t element_size,
                           int components,
                           const VtkBox& box,
                           const std::vector<VtkBoxPieceMut>& pieces);

/** Per-rank VTK shard session for a @c DenseDistributed grid. */
class CARBON_API VtkBoxSession {
public:
  VtkBoxSession() noexcept;
  ~VtkBoxSession();
  VtkBoxSession(const VtkBoxSession&) = delete;
  auto
  operator=(const VtkBoxSession&) -> VtkBoxSession& = delete;
  VtkBoxSession(VtkBoxSession&&) noexcept;
  auto
  operator=(VtkBoxSession&&) noexcept -> VtkBoxSession&;

  void
  open_for_write(std::string_view path,
                 const DenseDistributed& grid,
                 std::uint32_t num_fields);
  void
  open_for_read(std::string_view path,
                const DenseDistributed& grid);

  void
  write_field(std::string_view name,
              const char* dtype_tag,
              size_t element_size,
              int components,
              const std::vector<VtkBoxPiece>& pieces);

  /** Finish a writable session: close PointData/ImageData/VTKFile. Idempotent. */
  void
  finalize();

  void
  read_field(std::string_view name,
             const char* dtype_tag,
             size_t element_size,
             int components,
             const std::vector<VtkBoxPieceMut>& pieces);

  [[nodiscard]] auto
  num_fields() const noexcept -> std::uint32_t;

  [[nodiscard]] static auto
  shard_path(std::string_view base,
             const DenseDistributed& grid) -> std::string;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * Owned-cell blocks of @p field, one per local span (the host mirror, halo excluded). Const
 * variant for writers, mutable variant for readers.
 */
template <typename T,
          int Components>
void
collect_vtk_pieces(const DenseDistributed& grid,
                   const DenseDistributed::Field<T,
                                                 Components>& field,
                   std::vector<VtkBoxPiece>& pieces,
                   std::string_view callsite) {
  const auto& spans = grid.spans();
  if (field.cpu_buffers().size() != spans.size()) {
    log::throw_with_log<std::invalid_argument>(
        "{}: field has no host mirror (cpu_buffers().size()={}, gpu_count={}); construct the field "
        "with MemorySide::Host or HostDevice and download from device before writing",
        callsite, field.cpu_buffers().size(), spans.size());
  }
  pieces.clear();
  for (size_t g = 0; g < spans.size(); ++g) {
    const auto& span = spans[g];
    const int64_3d owned = span.logic_extent();
    if (owned.x <= 0 || owned.y <= 0 || owned.z <= 0) {
      continue;
    }
    const int64_3d halo = span.halo();
    VtkBoxPiece p;
    p.row_stride = span.haloed_extent().x;
    p.plane_pitch = span.xy_slab_elements();
    p.comp_stride = span.linear_storage_elements();
    p.nx = owned.x;
    p.ny = owned.y;
    p.nz = owned.z;
    p.start = static_cast<const void*>(field.cpu_buffers()[g].data() + halo.z * p.plane_pitch +
                                       halo.y * p.row_stride + halo.x);
    pieces.push_back(p);
  }
}

template <typename T,
          int Components>
void
collect_vtk_pieces_mut(const DenseDistributed& grid,
                       DenseDistributed::Field<T,
                                               Components>& field,
                       std::vector<VtkBoxPieceMut>& pieces,
                       std::string_view callsite) {
  const auto& spans = grid.spans();
  if (field.cpu_buffers().size() != spans.size()) {
    log::throw_with_log<std::invalid_argument>(
        "{}: field has no host mirror (cpu_buffers().size()={}, gpu_count={}); construct the field "
        "with MemorySide::Host or HostDevice before reading",
        callsite, field.cpu_buffers().size(), spans.size());
  }
  pieces.clear();
  for (size_t g = 0; g < spans.size(); ++g) {
    const auto& span = spans[g];
    const int64_3d owned = span.logic_extent();
    if (owned.x <= 0 || owned.y <= 0 || owned.z <= 0) {
      continue;
    }
    const int64_3d halo = span.halo();
    VtkBoxPieceMut p;
    p.row_stride = span.haloed_extent().x;
    p.plane_pitch = span.xy_slab_elements();
    p.comp_stride = span.linear_storage_elements();
    p.nx = owned.x;
    p.ny = owned.y;
    p.nz = owned.z;
    p.start = static_cast<void*>(field.cpu_buffers()[g].data() + halo.z * p.plane_pitch +
                                 halo.y * p.row_stride + halo.x);
    pieces.push_back(p);
  }
}

}  // namespace detail

/**
 * Write @p fields (host mirrors, halo excluded) of @p grid as one XML ImageData `.vti` shard per
 * rank (`<path>.rank<N>.vti` under MPI, `<path>.vti` otherwise). Works for every decomposition
 * (slab / pencil / brick). Download device data to the host mirror first. Call
 * @ref write_vtk_multiblock_index afterwards (any rank may call it; rank 0 writes) to get a
 * single `<path>.vtm` that ParaView opens as one multiblock dataset.
 */
template <typename... Fields>
void
write_vtk(std::string_view path,
          const DenseDistributed& grid,
          const Fields&... fields) {
  detail::VtkBoxSession sess;
  sess.open_for_write(path, grid, static_cast<std::uint32_t>(sizeof...(Fields)));
  size_t i = 0;
  (
      [&] {
        using FieldT = std::remove_cv_t<std::remove_reference_t<decltype(fields)>>;
        using ScalarT = typename FieldT::value_type;
        constexpr int kComponents = FieldT::components;
        std::vector<detail::VtkBoxPiece> pieces;
        detail::collect_vtk_pieces<ScalarT, kComponents>(grid, fields, pieces, "io::write_vtk");
        const std::string& field_name = fields.name();
        std::string vtk_name = field_name.empty() ? ("field_" + std::to_string(i)) : field_name;
        sess.write_field(vtk_name, detail::native_dtype_tag<ScalarT>(), sizeof(ScalarT),
                         kComponents, pieces);
        ++i;
      }(),
      ...);
  sess.finalize();
}

/**
 * Read @p fields of @p grid from the shard @ref write_vtk wrote for this rank. WholeExtent and
 * Origin must match this rank's owned box; fields bind by ordinal. Only owned cells are touched
 * (halo cells keep their contents).
 */
template <typename... Fields>
void
read_vtk(std::string_view path,
         const DenseDistributed& grid,
         Fields&... fields) {
  detail::VtkBoxSession sess;
  sess.open_for_read(path, grid);
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
        std::vector<detail::VtkBoxPieceMut> pieces;
        detail::collect_vtk_pieces_mut<ScalarT, kComponents>(grid, fields, pieces, "io::read_vtk");
        sess.read_field("field_" + std::to_string(i), detail::native_dtype_tag<ScalarT>(),
                        sizeof(ScalarT), kComponents, pieces);
        ++i;
      }(),
      ...);
}

/**
 * Write `<base>.vtm`, a vtkMultiBlockDataSet index over the per-rank shards of
 * @ref write_vtk, so ParaView opens the whole decomposed domain at once. Only rank 0 writes;
 * ranks that own no cells are not listed. No-op without a distributed System.
 */
CARBON_API auto
write_vtk_multiblock_index(std::string_view base,
                           const DenseDistributed& grid) -> void;
