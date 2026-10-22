#include <carbon/checkpoint.h>

#include <carbon/distributed.h>
#include <carbon/logging.h>
#include <carbon/system.h>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include <filesystem>
#include <fstream>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace carbon {

namespace checkpoint_detail {

namespace {

namespace fs = std::filesystem;

/**
 * Flush @p path 's contents (or, for @p is_directory, its directory entries) to stable storage.
 *
 * Best-effort and deliberately non-throwing: this hardens the commit against power loss, and a
 * platform or filesystem that cannot honour it should not fail an otherwise complete
 * checkpoint. The C++ standard library has no fsync equivalent, so this is POSIX-only; on other
 * platforms the rename still gives atomicity, just not durability.
 */
void
fsync_path(const fs::path& path,
           bool is_directory) {
#if defined(__unix__) || defined(__APPLE__)
  const int flags = is_directory ? O_RDONLY | O_DIRECTORY : O_RDONLY;
  const int fd = ::open(path.c_str(), flags);
  if (fd < 0) {
    CARBON_LOG_DEBUG("checkpoint: fsync skipped, could not open '{}'", path.string());
    return;
  }
  if (::fsync(fd) != 0) {
    CARBON_LOG_DEBUG("checkpoint: fsync('{}') failed", path.string());
  }
  ::close(fd);
#else
  (void)path;
  (void)is_directory;
#endif
}

// [[maybe_unused]]: every caller sits behind #ifdef CARBON_HAVE_MPI_NCCL, so a
// non-MPI build compiles this out of every use site and would otherwise warn.
[[maybe_unused]] [[nodiscard]] auto
maybe_distributed(const Dense& dense) -> const Distributed* {
  // System::distributed() returns nullptr unconditionally on non-MPI builds (the
  // shared_ptr was never set), so the accessor itself is always safe; only the
  // methods on the returned pointer are MPI-only and must be gated below.
  return dense.system().distributed();
}

[[nodiscard]] auto
maybe_world_rank(const Dense& dense) -> int {
#ifdef CARBON_HAVE_MPI_NCCL
  if (const auto* d = maybe_distributed(dense); d != nullptr) {
    return d->world_rank();
  }
#else
  (void)dense;
#endif
  return 0;
}

[[nodiscard]] auto
maybe_world_size(const Dense& dense) -> int {
#ifdef CARBON_HAVE_MPI_NCCL
  if (const auto* d = maybe_distributed(dense); d != nullptr) {
    return d->world_size();
  }
#else
  (void)dense;
#endif
  return 1;
}

void
maybe_barrier(const Dense& dense) {
#ifdef CARBON_HAVE_MPI_NCCL
  if (const auto* d = maybe_distributed(dense); d != nullptr) {
    d->barrier();
  }
#else
  (void)dense;
#endif
}

/**
 * Read the full contents of a file as a string. Used for manifest.json (small,
 * always fits in memory). Throws std::runtime_error on open failure.
 */
[[nodiscard]] auto
slurp_file(const fs::path& path) -> std::string {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    log::throw_with_log<std::runtime_error>("checkpoint: failed to open '{}' for reading",
                                            path.string());
  }
  std::ostringstream oss;
  oss << in.rdbuf();
  return oss.str();
}

}  // namespace

auto
field_shard_path(std::string_view dir,
                 std::string_view file,
                 const Dense& dense) -> std::string {
  fs::path p = fs::path(std::string(dir)) / std::string(file);
  std::string result = p.string();
#ifdef CARBON_HAVE_MPI_NCCL
  if (const auto* d = maybe_distributed(dense); d != nullptr) {
    std::ostringstream oss;
    oss << result << ".rank" << d->world_rank();
    return oss.str();
  }
#else
  (void)dense;
#endif
  return result;
}

void
ensure_directory(std::string_view dir,
                 const Dense& dense) {
  // The comment that used to sit here described deferring the throw past the barrier, but the
  // throw was issued inline: rank 0 left the collective while every peer was already committed
  // to maybe_barrier below, so an unwritable output directory hung the job instead of failing
  // it. Log at the point of failure, join the barrier, and only then throw.
  std::error_code ec;
  if (maybe_world_rank(dense) == 0) {
    fs::create_directories(fs::path(std::string(dir)), ec);
    if (ec) {
      CARBON_LOG_ERROR("checkpoint: failed to create directory '{}': {} (throwing after barrier)",
                       std::string(dir), ec.message());
    }
  }
  // Synchronise: every reader / shard writer needs to see the directory before it
  // tries to open files inside it. No-op single-process.
  maybe_barrier(dense);
  if (ec) {
    log::throw_with_log<std::runtime_error>("checkpoint: failed to create directory '{}': {}",
                                            std::string(dir), ec.message());
  }
}

void
require_safe_field_name(std::string_view name) {
  if (name.empty()) {
    log::throw_with_log<std::invalid_argument>("checkpoint: field name must not be empty");
  }
  if (name.find('/') != std::string_view::npos || name.find('\\') != std::string_view::npos) {
    log::throw_with_log<std::invalid_argument>(
        "checkpoint: field name '{}' must not contain path separators ('/' or '\\\\')", name);
  }
  if (name == "." || name == "..") {
    log::throw_with_log<std::invalid_argument>(
        "checkpoint: field name '{}' must not be '.' or '..'", name);
  }
  if (name == "manifest") {
    log::throw_with_log<std::invalid_argument>(
        "checkpoint: field name 'manifest' is reserved (clashes with manifest.json)");
  }
}

void
require_safe_manifest_file(std::string_view file,
                           int field_index) {
  // The writer derives this from a name already vetted by require_safe_field_name, but the
  // reader takes it from JSON on disk, and field_shard_path feeds it straight to
  // fs::path::operator/ — which *replaces* the directory outright when the right-hand side is
  // absolute. So an edited or corrupt manifest could otherwise redirect the reader to any path
  // on the host and load it into a field buffer. Vet it on the way in with the same rules the
  // writer applies to names.
  const auto reject = [&](const char* why) {
    log::throw_with_log<std::runtime_error>(
        "checkpoint::read_manifest: fields[{}] 'file' entry '{}' {}; the manifest must reference "
        "plain shard filenames within the checkpoint directory",
        field_index, std::string(file), why);
  };
  if (file.empty()) {
    reject("is empty");
  }
  if (file.find('/') != std::string_view::npos || file.find('\\') != std::string_view::npos) {
    reject("contains a path separator");
  }
  if (file == "." || file == "..") {
    reject("is a directory reference");
  }
  if (fs::path(std::string(file)).is_absolute()) {
    reject("is an absolute path");
  }
}

void
write_manifest(std::string_view dir,
               const Manifest& manifest) {
  rapidjson::Document doc;
  doc.SetObject();
  auto& alloc = doc.GetAllocator();

  doc.AddMember("schema_version",
                rapidjson::Value(static_cast<unsigned>(manifest.schema_version)).Move(), alloc);
  doc.AddMember("step", rapidjson::Value(manifest.step).Move(), alloc);
  doc.AddMember("time", rapidjson::Value(manifest.time).Move(), alloc);

  rapidjson::Value extent(rapidjson::kArrayType);
  extent.PushBack(rapidjson::Value(manifest.global_extent[0]).Move(), alloc);
  extent.PushBack(rapidjson::Value(manifest.global_extent[1]).Move(), alloc);
  extent.PushBack(rapidjson::Value(manifest.global_extent[2]).Move(), alloc);
  doc.AddMember("global_extent", extent, alloc);

  doc.AddMember("world_size", rapidjson::Value(manifest.world_size).Move(), alloc);

  rapidjson::Value fields(rapidjson::kArrayType);
  for (const auto& e : manifest.fields) {
    rapidjson::Value entry(rapidjson::kObjectType);
    entry.AddMember("name", rapidjson::Value(e.name.c_str(), alloc).Move(), alloc);
    entry.AddMember("dtype", rapidjson::Value(e.dtype.c_str(), alloc).Move(), alloc);
    entry.AddMember("components", rapidjson::Value(e.components).Move(), alloc);
    entry.AddMember("file", rapidjson::Value(e.file.c_str(), alloc).Move(), alloc);
    fields.PushBack(entry, alloc);
  }
  doc.AddMember("fields", fields, alloc);

  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);

  // The presence of manifest.json is the commit marker (see the class docs on
  // CheckpointWriter), so it has to appear whole or not at all. Writing in place gave a window
  // in which a reader — or a restart after a crash mid-write — found a manifest.json that
  // looked committed but was truncated JSON, which read_manifest then rejects, turning a
  // recoverable "no checkpoint yet" into a hard parse failure. Stage to a sibling temp file,
  // make its contents durable, then rename: rename within a directory is atomic, so the marker
  // becomes visible only once complete.
  const fs::path dir_path = fs::path(std::string(dir));
  const fs::path path = dir_path / "manifest.json";
  const fs::path tmp_path = dir_path / "manifest.json.tmp";

  {
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    if (!out) {
      log::throw_with_log<std::runtime_error>("checkpoint: failed to open '{}' for writing",
                                              tmp_path.string());
    }
    out << buffer.GetString();
    out.flush();
    if (!out.good()) {
      log::throw_with_log<std::runtime_error>("checkpoint: write to '{}' failed",
                                              tmp_path.string());
    }
  }

  fsync_path(tmp_path, /*is_directory=*/false);

  std::error_code ec;
  fs::rename(tmp_path, path, ec);
  if (ec) {
    fs::remove(tmp_path, ec);
    log::throw_with_log<std::runtime_error>("checkpoint: failed to commit manifest '{}': {}",
                                            path.string(), ec.message());
  }

  // Without this the rename can still be lost to a power failure even though the file contents
  // were synced: the directory entry lives in its own block.
  fsync_path(dir_path, /*is_directory=*/true);
}

namespace {

[[nodiscard]] auto
require_string(const rapidjson::Value& v,
               const char* field_name,
               const char* what) -> std::string {
  if (!v.HasMember(field_name) || !v[field_name].IsString()) {
    log::throw_with_log<std::runtime_error>(
        "checkpoint::read_manifest: {} missing or wrong type for '{}'", what, field_name);
  }
  return std::string(v[field_name].GetString(), v[field_name].GetStringLength());
}

[[nodiscard]] auto
require_int(const rapidjson::Value& v,
            const char* field_name,
            const char* what) -> int64_t {
  if (!v.HasMember(field_name) || !v[field_name].IsInt64()) {
    if (v.HasMember(field_name) && v[field_name].IsInt()) {
      return static_cast<int64_t>(v[field_name].GetInt());
    }
    log::throw_with_log<std::runtime_error>(
        "checkpoint::read_manifest: {} missing or wrong type for '{}'", what, field_name);
  }
  return v[field_name].GetInt64();
}

}  // namespace

auto
read_manifest(std::string_view dir) -> Manifest {
  const fs::path path = fs::path(std::string(dir)) / "manifest.json";
  if (!fs::exists(path)) {
    log::throw_with_log<std::runtime_error>(
        "checkpoint::read_manifest: '{}' does not exist (no committed checkpoint)", path.string());
  }
  const std::string text = slurp_file(path);

  rapidjson::Document doc;
  // ParseInsitu would be faster but mutates the buffer; this is small JSON so we
  // pay the copy and keep `text` intact for diagnostic logging on failure.
  doc.Parse(text.c_str(), text.size());
  if (doc.HasParseError()) {
    log::throw_with_log<std::runtime_error>(
        "checkpoint::read_manifest: '{}' is not valid JSON: {} (offset {})", path.string(),
        rapidjson::GetParseError_En(doc.GetParseError()), doc.GetErrorOffset());
  }
  if (!doc.IsObject()) {
    log::throw_with_log<std::runtime_error>(
        "checkpoint::read_manifest: '{}' top-level value is not an object", path.string());
  }

  Manifest m;

  if (!doc.HasMember("schema_version") || !doc["schema_version"].IsUint()) {
    log::throw_with_log<std::runtime_error>(
        "checkpoint::read_manifest: '{}' missing 'schema_version' (uint)", path.string());
  }
  m.schema_version = doc["schema_version"].GetUint();

  m.step = require_int(doc, "step", "manifest");
  if (!doc.HasMember("time") || !doc["time"].IsNumber()) {
    log::throw_with_log<std::runtime_error>(
        "checkpoint::read_manifest: '{}' missing 'time' (number)", path.string());
  }
  m.time = doc["time"].GetDouble();

  if (!doc.HasMember("global_extent") || !doc["global_extent"].IsArray() ||
      doc["global_extent"].Size() != 3U) {
    log::throw_with_log<std::runtime_error>(
        "checkpoint::read_manifest: '{}' missing 'global_extent' (array of 3)", path.string());
  }
  for (rapidjson::SizeType i = 0; i < 3; ++i) {
    const auto& v = doc["global_extent"][i];
    if (!v.IsInt64() && !v.IsInt()) {
      log::throw_with_log<std::runtime_error>(
          "checkpoint::read_manifest: '{}' global_extent[{}] is not an integer", path.string(),
          static_cast<int>(i));
    }
    m.global_extent[i] = v.IsInt64() ? v.GetInt64() : static_cast<int64_t>(v.GetInt());
  }

  if (!doc.HasMember("world_size") || !doc["world_size"].IsInt()) {
    log::throw_with_log<std::runtime_error>(
        "checkpoint::read_manifest: '{}' missing 'world_size' (int)", path.string());
  }
  m.world_size = doc["world_size"].GetInt();

  if (!doc.HasMember("fields") || !doc["fields"].IsArray()) {
    log::throw_with_log<std::runtime_error>(
        "checkpoint::read_manifest: '{}' missing 'fields' (array)", path.string());
  }
  const auto& fields = doc["fields"];
  m.fields.reserve(fields.Size());
  for (rapidjson::SizeType i = 0; i < fields.Size(); ++i) {
    const auto& f = fields[i];
    if (!f.IsObject()) {
      log::throw_with_log<std::runtime_error>(
          "checkpoint::read_manifest: '{}' fields[{}] is not an object", path.string(),
          static_cast<int>(i));
    }
    FieldEntry e;
    e.name = require_string(f, "name", "manifest field entry");
    e.dtype = require_string(f, "dtype", "manifest field entry");
    if (!f.HasMember("components") || !f["components"].IsInt()) {
      log::throw_with_log<std::runtime_error>(
          "checkpoint::read_manifest: '{}' fields[{}] missing 'components' (int)", path.string(),
          static_cast<int>(i));
    }
    e.components = f["components"].GetInt();
    e.file = require_string(f, "file", "manifest field entry");
    require_safe_manifest_file(e.file, static_cast<int>(i));
    m.fields.push_back(std::move(e));
  }

  return m;
}

void
validate_manifest(const Manifest& manifest,
                  const Dense& dense) {
  if (manifest.schema_version != kSchemaVersion) {
    log::throw_with_log<std::invalid_argument>(
        "checkpoint: manifest schema_version {} does not match this build's {}",
        manifest.schema_version, kSchemaVersion);
  }
  const auto ge = dense.global_extent();
  if (manifest.global_extent[0] != ge.x || manifest.global_extent[1] != ge.y ||
      manifest.global_extent[2] != ge.z) {
    log::throw_with_log<std::invalid_argument>(
        "checkpoint: manifest global_extent ({}, {}, {}) does not match Dense ({}, {}, {})",
        manifest.global_extent[0], manifest.global_extent[1], manifest.global_extent[2], ge.x, ge.y,
        ge.z);
  }
  const int ws = maybe_world_size(dense);
  if (manifest.world_size != ws) {
    log::throw_with_log<std::invalid_argument>(
        "checkpoint: manifest world_size {} does not match this run's {} — restart with the "
        "same number of ranks the checkpoint was written under",
        manifest.world_size, ws);
  }
}

auto
find_field(const Manifest& manifest,
           std::string_view name) -> const FieldEntry& {
  for (const auto& e : manifest.fields) {
    if (e.name == name) {
      return e;
    }
  }
  std::ostringstream avail;
  for (size_t i = 0; i < manifest.fields.size(); ++i) {
    if (i > 0U) {
      avail << ", ";
    }
    avail << '\'' << manifest.fields[i].name << '\'';
  }
  log::throw_with_log<std::out_of_range>(
      "checkpoint: field '{}' not present in manifest; available: [{}]", name, avail.str());
}

}  // namespace checkpoint_detail

CheckpointWriter::CheckpointWriter(std::string_view dir,
                                   const Dense& dense,
                                   int64_t step,
                                   double time)
    : dir_(dir),
      dense_(dense),
      step_(step),
      time_(time) {
  checkpoint_detail::ensure_directory(dir_, dense_);
}

CheckpointWriter::~CheckpointWriter() {
  if (!committed_ && !entries_.empty()) {
    CARBON_LOG_WARN(
        "CheckpointWriter('{}') destroyed with {} field(s) added but no commit() — manifest.json "
        "was not written; the checkpoint will be treated as in-flight by readers",
        dir_, entries_.size());
  }
}

void
CheckpointWriter::commit() {
  if (committed_) {
    return;
  }
  // First barrier: every rank's per-shard writes are flushed before rank 0 emits the
  // manifest marker. Single-process: no-op.
  checkpoint_detail::maybe_barrier(dense_);

  if (checkpoint_detail::maybe_world_rank(dense_) == 0) {
    checkpoint_detail::Manifest manifest;
    manifest.schema_version = checkpoint_detail::kSchemaVersion;
    manifest.step = step_;
    manifest.time = time_;
    const auto ge = dense_.global_extent();
    manifest.global_extent[0] = ge.x;
    manifest.global_extent[1] = ge.y;
    manifest.global_extent[2] = ge.z;
    manifest.world_size = checkpoint_detail::maybe_world_size(dense_);
    manifest.fields = entries_;
    checkpoint_detail::write_manifest(dir_, manifest);
  }

  // Second barrier: every peer rank that immediately constructs a CheckpointReader
  // must observe rank 0's manifest.
  checkpoint_detail::maybe_barrier(dense_);
  committed_ = true;
}

CheckpointReader::CheckpointReader(std::string_view dir,
                                   const Dense& dense)
    : dir_(dir),
      dense_(dense) {
  manifest_ = checkpoint_detail::read_manifest(dir_);
  checkpoint_detail::validate_manifest(manifest_, dense_);
}

}  // namespace carbon
