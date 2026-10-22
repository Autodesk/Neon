#pragma once

/**
 * @file checkpoint.h
 * @brief Named-checkpoint writer / reader for @c carbon::Dense field sets.
 *
 * A checkpoint is a single directory holding a JSON manifest plus one binary file per
 * field. The directory's name is the checkpoint identity (the user supplies it); the
 * manifest carries the step / sim-time metadata and the list of fields with their
 * dtype + components + on-disk filename. Each per-field file is a single-field
 * @c carbon::io native binary file (full @c NativeFileHeader + one @c NativeFieldMeta
 * + the SoA payload), so any reader can spot-check the file independently of the
 * manifest.
 *
 * @par Disk layout
 * @code
 *   <dir>/
 *     manifest.json             // rank-0 only; presence == commit marker
 *     <field_name>.bin          // single-process
 *     <field_name>.bin.rank0    // under MPI: one shard per rank per field
 *     <field_name>.bin.rank1
 * @endcode
 *
 * @par Atomic commit (writer)
 * Under MPI the writer ctor barrier-syncs the @c mkdir on rank 0 before any rank
 * starts writing shards, every @c add() runs on every rank in lock-step (no global
 * sync per-field — the per-rank shards are independent), and @c commit() emits the
 * @c manifest.json from rank 0 only, fenced by two barriers (one before so all
 * shards are visible, one after so subsequent reader ctors on every rank see the
 * manifest). Single-process mode skips the barriers (the wrapper in
 * @ref Distributed is a no-op when the @c System is non-distributed).
 *
 * @par Sync from / to device
 * The default surface is explicit (matches @ref io::write_native ): the caller
 * downloads / uploads device data with @c carbon::Transfer<T> and the checkpoint
 * layer only touches @c cpu_buffers . Opt in to layer-side device-host sync with
 * @ref CheckpointWriter::sync_from_device or
 * @ref CheckpointReader::sync_to_device . Both flags short-circuit on fields
 * with no device buffers ( @c MemorySide::Host ), so toggling them on for a
 * mixed-side field set is safe.
 *
 * @par Out of scope today
 * @c Block and @c DenseDisg checkpoints, multi-checkpoint discovery
 * ( @c list / @c latest ), compression, async I/O, schema migration. Single
 * @c schema_version=1 is the only manifest schema understood.
 */

#include <carbon/dense.h>
#include <carbon/distributed.h>
#include <carbon/export.h>
#include <carbon/io.h>
#include <carbon/logging.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#include <cstdint>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace carbon {

namespace checkpoint_detail {

/**
 * Manifest schema version this build understands. Bumped on any incompatible change
 * to the on-disk JSON layout; readers reject any other value.
 */
inline constexpr std::uint32_t kSchemaVersion = 1U;

/**
 * Per-field manifest entry. Order in the manifest's @c fields array matches the
 * order @ref CheckpointWriter::add was called.
 */
struct FieldEntry {
  std::string name;  ///< User-supplied symbolic name; primary key for @c load.
  std::string
      dtype;  ///< Native dtype tag (e.g. "f32", "i32"); matches @ref io::detail::native_dtype_tag .
  std::int32_t components{0};  ///< Components per cell; matches the @c Field's template parameter.
  std::string file;            ///< Filename relative to the checkpoint directory (e.g. "rho.bin").
};

/**
 * In-memory mirror of @c manifest.json . Rank-0 builds and writes one of these in
 * @ref CheckpointWriter::commit ; every reader rank parses one in
 * @ref CheckpointReader::CheckpointReader .
 */
struct Manifest {
  std::uint32_t schema_version{kSchemaVersion};
  int64_t step{0};
  double time{0.0};
  int64_t global_extent[3]{0, 0, 0};
  int world_size{1};
  std::vector<FieldEntry> fields{};
};

/**
 * Build the per-field shard path. Single-process mode returns @c <dir>/<file> ;
 * MPI mode returns @c <dir>/<file>.rank<world_rank> .
 */
[[nodiscard]] CARBON_API auto
field_shard_path(std::string_view dir,
                 std::string_view file,
                 const Dense& dense) -> std::string;

/** Create the checkpoint directory if missing; rank 0 only under MPI; barrier-synced. */
CARBON_API void
ensure_directory(std::string_view dir,
                 const Dense& dense);

/** Serialise @p manifest to @c <dir>/manifest.json . Caller is responsible for rank gating. */
CARBON_API void
write_manifest(std::string_view dir,
               const Manifest& manifest);

/** Parse @c <dir>/manifest.json and return the in-memory mirror. Throws on malformed JSON or
 *  schema mismatch. */
[[nodiscard]] CARBON_API auto
read_manifest(std::string_view dir) -> Manifest;

/** Validate a freshly read manifest against the bound @ref Dense (schema version,
 *  global_extent, world_size when distributed). */
CARBON_API void
validate_manifest(const Manifest& manifest,
                  const Dense& dense);

/** Look up @p name in @p manifest ; throw @c std::out_of_range with a clear message
 *  when absent (lists every present field name in the message). */
[[nodiscard]] CARBON_API auto
find_field(const Manifest& manifest,
           std::string_view name) -> const FieldEntry&;

/** Sanitise a user-supplied field name for use as a path component: rejects empty
 *  strings, path separators, and parent traversals. */
CARBON_API void
require_safe_field_name(std::string_view name);

}  // namespace checkpoint_detail

/**
 * @brief Write a named checkpoint of one or more @c Dense::Field instances.
 *
 * Construct, call @ref add for every field to persist (each may have its own @c T
 * and @c Components ), then call @ref commit to seal the checkpoint. Without
 * @ref commit the checkpoint is treated as in-flight by readers (the
 * @c manifest.json marker is not written) and the destructor logs a warning.
 *
 * @par Lifetime / threading
 * Single-shot: build, add, commit. Not thread-safe. Pass by value into a scope and
 * destroy.
 */
class CARBON_API CheckpointWriter {
public:
  /**
   * @param dir   Checkpoint directory. Created if absent (rank 0 under MPI).
   * @param dense Layout the fields live on. Captured by const-reference; must outlive
   *              the writer.
   * @param step  User-meaningful step / iteration index. Round-tripped through the
   *              manifest and surfaced by @ref CheckpointReader::step .
   * @param time  User-meaningful sim-time. Round-tripped through the manifest and
   *              surfaced by @ref CheckpointReader::time .
   */
  CheckpointWriter(std::string_view dir,
                   const Dense& dense,
                   int64_t step = 0,
                   double time = 0.0);

  ~CheckpointWriter();

  CheckpointWriter(const CheckpointWriter&) = delete;
  CheckpointWriter&
  operator=(const CheckpointWriter&) = delete;
  CheckpointWriter(CheckpointWriter&&) = delete;
  CheckpointWriter&
  operator=(CheckpointWriter&&) = delete;

  /**
   * @brief Persist @p field's owned cells under the symbolic name @p name .
   *
   * Writes @c <dir>/<name>.bin (or @c <dir>/<name>.bin.rank<N> under MPI) as a
   * single-field native binary file using the @c io::detail::write_native_*
   * helpers. The host mirror ( @c field.cpu_buffers ) must be populated unless
   * @ref sync_from_device was set, in which case the writer issues a
   * @c Transfer<T> device->host first.
   *
   * @throws std::invalid_argument when @p name is unsafe as a path component, when
   *         a field with @p name was already added, or when the field has no host
   *         mirror.
   * @throws std::runtime_error    on filesystem errors during the per-shard write.
   */
  template <typename T,
            int Components>
  auto
  add(std::string_view name,
      const Dense::Field<T,
                         Components>& field) -> CheckpointWriter& {
    checkpoint_detail::require_safe_field_name(name);
    for (const auto& e : entries_) {
      if (e.name == name) {
        log::throw_with_log<std::invalid_argument>(
            "CheckpointWriter::add: field '{}' already added to checkpoint '{}'", name, dir_);
      }
    }

    if (sync_from_device_ && !field.gpu_buffers().empty()) {
      // Pull device -> host so the host mirror reflects the current device state.
      // Host-only fields ( @c MemorySide::Host ) carry no device buffers and short-
      // circuit harmlessly: the host mirror IS the authoritative copy. Pure-Device
      // fields surface a clear throw via the @c cpu_buffers() guard inside
      // @c write_native_field_payload below. System is a shared-state handle
      // (cheap copy) so we can call the non-const @c stream() factory through a
      // local copy without losing const-correctness on @c dense_ .
      System sys = dense_.system();
      const Stream stream = sys.stream();
      Transfer<T>(sys, field.gpu_buffers(), field.cpu_buffers())(stream);
      stream.wait();
    }

    const std::string path =
        checkpoint_detail::field_shard_path(dir_, std::string(name) + ".bin", dense_);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
      log::throw_with_log<std::runtime_error>(
          "CheckpointWriter::add: failed to open '{}' for writing", path);
    }

    const ::carbon::io::detail::NativeFileHeader hdr =
        ::carbon::io::detail::make_native_header(dense_, /*num_fields=*/1U);
    ::carbon::io::detail::write_native_header(out, hdr);
    const char* dtype_tag = ::carbon::io::detail::native_dtype_tag<T>();
    ::carbon::io::detail::write_native_field_meta(out, name, dtype_tag,
                                                  static_cast<std::int32_t>(Components));
    ::carbon::io::detail::write_native_field_payload(out, dense_, field);
    if (!out.good()) {
      log::throw_with_log<std::runtime_error>("CheckpointWriter::add: write to '{}' failed", path);
    }

    checkpoint_detail::FieldEntry entry;
    entry.name = std::string(name);
    entry.dtype = std::string(dtype_tag);
    entry.components = static_cast<std::int32_t>(Components);
    entry.file = std::string(name) + ".bin";
    entries_.push_back(std::move(entry));
    return *this;
  }

  /**
   * @brief Opt in to layer-side device->host sync inside @ref add .
   *
   * Off by default to match @ref io::write_native semantics. When on, every
   * subsequent @ref add issues a @c Transfer<T> from @c gpu_buffers to
   * @c cpu_buffers and waits on the per-call @c Stream before reading the host
   * mirror. Useful for callers that hold @c MemorySide::HostDevice fields and want
   * the checkpoint API to do the obvious transfer for them.
   */
  auto
  sync_from_device(bool yes) noexcept -> CheckpointWriter& {
    sync_from_device_ = yes;
    return *this;
  }

  /**
   * @brief Seal the checkpoint by emitting @c <dir>/manifest.json .
   *
   * Under MPI the call collectively barriers, rank 0 writes the manifest, and a
   * second barrier ensures every rank's subsequent @ref CheckpointReader sees the
   * commit. Idempotent: subsequent @ref commit calls are no-ops.
   */
  void
  commit();

private:
  std::string dir_;
  const Dense& dense_;
  int64_t step_{0};
  double time_{0.0};
  bool sync_from_device_{false};
  bool committed_{false};
  std::vector<checkpoint_detail::FieldEntry> entries_{};
};

/**
 * @brief Read a named checkpoint produced by @ref CheckpointWriter .
 *
 * The constructor loads @c <dir>/manifest.json , validates schema_version /
 * global_extent / world_size against the bound @ref Dense , and caches the field
 * list. @ref load looks up a field by name and populates the bound
 * @c Dense::Field 's host mirror; with @ref sync_to_device on, the host mirror is
 * pushed back to device on every load.
 */
class CARBON_API CheckpointReader {
public:
  /**
   * @param dir   Checkpoint directory containing a @c manifest.json . Throws if
   *              absent.
   * @param dense Layout the fields live on; must match the manifest's
   *              @c global_extent (and @c world_size when distributed).
   *
   * @throws std::runtime_error    on malformed JSON or missing manifest.
   * @throws std::invalid_argument on schema mismatch / extent mismatch.
   */
  CheckpointReader(std::string_view dir,
                   const Dense& dense);

  ~CheckpointReader() = default;

  CheckpointReader(const CheckpointReader&) = delete;
  CheckpointReader&
  operator=(const CheckpointReader&) = delete;
  CheckpointReader(CheckpointReader&&) = delete;
  CheckpointReader&
  operator=(CheckpointReader&&) = delete;

  /** User-supplied @c step from the writer's ctor; round-tripped through the manifest. */
  [[nodiscard]] auto
  step() const noexcept -> int64_t {
    return manifest_.step;
  }

  /** User-supplied @c time from the writer's ctor; round-tripped through the manifest. */
  [[nodiscard]] auto
  time() const noexcept -> double {
    return manifest_.time;
  }

  /** Names of every field in the checkpoint, in the order @ref CheckpointWriter::add
   *  was called. Useful for introspection / discovery before any @ref load . */
  [[nodiscard]] auto
  field_names() const -> std::vector<std::string> {
    std::vector<std::string> names;
    names.reserve(manifest_.fields.size());
    for (const auto& e : manifest_.fields) {
      names.push_back(e.name);
    }
    return names;
  }

  /**
   * @brief Load the field stored under @p name into @p field .
   *
   * Validates the on-disk dtype + component count against the bound
   * @c Field<T, Components> 's template parameters before reading. The field's
   * host mirror is populated; with @ref sync_to_device on, a host->device
   * @c Transfer<T> is issued after the read and waited on before returning.
   *
   * @throws std::out_of_range     when @p name is not present in the manifest.
   * @throws std::invalid_argument on dtype / component / header mismatch.
   * @throws std::runtime_error    on filesystem errors.
   */
  template <typename T,
            int Components>
  void
  load(std::string_view name,
       Dense::Field<T,
                    Components>& field) {
    const auto& entry = checkpoint_detail::find_field(manifest_, name);
    const char* expected_tag = ::carbon::io::detail::native_dtype_tag<T>();
    if (entry.dtype != expected_tag) {
      log::throw_with_log<std::invalid_argument>(
          "CheckpointReader::load: field '{}' dtype '{}' does not match bound Field's dtype '{}'",
          name, entry.dtype, expected_tag);
    }
    if (entry.components != static_cast<std::int32_t>(Components)) {
      log::throw_with_log<std::invalid_argument>(
          "CheckpointReader::load: field '{}' components {} does not match bound Field's "
          "Components {}",
          name, entry.components, Components);
    }

    const std::string path = checkpoint_detail::field_shard_path(dir_, entry.file, dense_);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      log::throw_with_log<std::runtime_error>(
          "CheckpointReader::load: failed to open '{}' for reading", path);
    }

    const ::carbon::io::detail::NativeFileHeader hdr = ::carbon::io::detail::read_native_header(in);
    ::carbon::io::detail::validate_native_header(hdr, dense_);
    if (hdr.num_fields != 1U) {
      log::throw_with_log<std::invalid_argument>(
          "CheckpointReader::load: '{}' has num_fields={}, expected 1", path, hdr.num_fields);
    }
    const ::carbon::io::detail::NativeFieldMeta meta =
        ::carbon::io::detail::read_native_field_meta(in);
    if (std::string_view(meta.dtype) != expected_tag) {
      log::throw_with_log<std::invalid_argument>(
          "CheckpointReader::load: '{}' field meta dtype '{}' does not match bound Field's dtype "
          "'{}'",
          path, meta.dtype, expected_tag);
    }
    if (meta.components != static_cast<std::int32_t>(Components)) {
      log::throw_with_log<std::invalid_argument>(
          "CheckpointReader::load: '{}' field meta components {} does not match bound Field's "
          "Components {}",
          path, meta.components, Components);
    }
    ::carbon::io::detail::read_native_field_payload(in, dense_, field);

    if (sync_to_device_ && !field.gpu_buffers().empty()) {
      // Push host -> device so any subsequent device-side kernel sees the freshly
      // loaded values. Host-only fields ( @c MemorySide::Host ) carry no device
      // buffers and short-circuit harmlessly. System is a shared-state handle
      // (cheap copy); call @c stream() through a local copy to keep @c dense_ const.
      System sys = dense_.system();
      const Stream stream = sys.stream();
      Transfer<T>(sys, field.cpu_buffers(), field.gpu_buffers())(stream);
      stream.wait();
    }
  }

  /**
   * @brief Opt in to layer-side host->device sync inside @ref load .
   *
   * Off by default to match @ref io::read_native semantics. When on, every
   * subsequent @ref load issues a @c Transfer<T> from @c cpu_buffers to
   * @c gpu_buffers and waits on the per-call @c Stream before returning.
   */
  auto
  sync_to_device(bool yes) noexcept -> CheckpointReader& {
    sync_to_device_ = yes;
    return *this;
  }

private:
  std::string dir_;
  const Dense& dense_;
  bool sync_to_device_{false};
  checkpoint_detail::Manifest manifest_{};
};

}  // namespace carbon
