#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
#include <carbon/report.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <carbon/logging.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>

#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

namespace carbon {

namespace {

std::unique_ptr<Report> g_report;

}  // namespace

void
init_report(const std::string& record_name) {
  if (g_report) {
    log::throw_with_log<std::runtime_error>("carbon::init_report: already initialized");
  }
  g_report = std::make_unique<Report>(record_name);
}

void
init_report(const std::string& record_name,
            int argc,
            char** argv) {
  init_report(record_name);
  g_report->command_line(argc, argv);
}

Report&
global_report() {
  if (!g_report) {
    log::throw_with_log<std::runtime_error>("carbon::global_report: call init_report first");
  }
  return *g_report;
}

bool
report_initialized() noexcept {
  return g_report != nullptr;
}

namespace report {

void
init(const std::string& record_name) {
  carbon::init_report(record_name);
}

void
init(const std::string& record_name,
     int argc,
     char** argv) {
  carbon::init_report(record_name, argc, argv);
}

void
command_line(int argc,
             char** argv) {
  if (!report_initialized()) {
    log::throw_with_log<std::runtime_error>(
        "carbon::report::command_line: global report not initialized");
  }
  global_report().command_line(argc, argv);
}

void
set_token(const std::string& token) {
  if (!report_initialized()) {
    log::throw_with_log<std::runtime_error>(
        "carbon::report::set_token: global report not initialized");
  }
  global_report().set_token(token);
}

void
write(const std::string& output_filename,
      bool append_time_to_file_name,
      const std::string& output_folder) {
  if (!report_initialized()) {
    log::throw_with_log<std::runtime_error>("carbon::report::write: global report not initialized");
  }
  global_report().write(output_filename, append_time_to_file_name, output_folder);
}

Report::SubBlock
get_subdoc() {
  if (!report_initialized()) {
    log::throw_with_log<std::runtime_error>(
        "carbon::report::get_subdoc: global report not initialized");
  }
  return global_report().get_subdoc();
}

void
add_subdoc(const std::string& name,
           rapidjson::Document& subdoc) {
  if (!report_initialized()) {
    log::throw_with_log<std::runtime_error>(
        "carbon::report::add_subdoc: global report not initialized");
  }
  global_report().add_subdoc(name, subdoc);
}

void
add_subdoc(const std::string& name,
           rapidjson::Document& subdoc,
           rapidjson::Document* parent) {
  if (!report_initialized()) {
    log::throw_with_log<std::runtime_error>(
        "carbon::report::add_subdoc: global report not initialized");
  }
  global_report().add_subdoc(name, subdoc, parent);
}

}  // namespace report

Report::Report()
    : doc_(std::make_shared<rapidjson::Document>()) {
  doc_->SetObject();
}

Report::Report(const std::string& record_name)
    : Report() {
  output_name_suffix_ = record_name;
  add_member("record", record_name);
}

void
Report::command_line(int argc,
                     char** argv) {
  auto& alloc = doc_->GetAllocator();
  rapidjson::Value arr(rapidjson::kArrayType);
  for (int i = 0; i < argc; ++i) {
    const char* s = argv[i] != nullptr ? argv[i] : "";
    arr.PushBack(rapidjson::Value(s, alloc).Move(), alloc);
  }
  rapidjson::Value key("command_line", alloc);
  doc_->AddMember(key, arr, alloc);
}

void
Report::set_token(const std::string& token) {
  add_member("token", token);
}

namespace {

std::string
timestamp_suffix() {
  using namespace std::chrono;
  const auto t = system_clock::to_time_t(system_clock::now());
  std::tm buf{};
#if defined(_WIN32)
  localtime_s(&buf, &t);
#else
  localtime_r(&t, &buf);
#endif
  std::ostringstream oss;
  oss << std::put_time(&buf, "%Y%m%d_%H%M%S");
  return oss.str();
}

}  // namespace

void
Report::write(const std::string& output_filename,
              bool append_time_to_file_name,
              const std::string& output_folder) {
  namespace fs = std::filesystem;
  const fs::path folder(output_folder);
  std::string base = output_filename;
  if (append_time_to_file_name) {
    base += "_" + timestamp_suffix();
  }
  base += ".json";
  const fs::path out = folder / base;
  std::error_code ec;
  fs::create_directories(folder, ec);
  std::ofstream ofs(out);
  if (!ofs) {
    log::throw_with_log<std::runtime_error>("carbon::Report::write: failed to open {}",
                                            out.string());
  }
  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  doc_->Accept(writer);
  ofs << buffer.GetString();
}

rapidjson::Document
Report::get_subdoc() {
  rapidjson::Document d;
  d.SetObject();
  return d;
}

void
Report::add_subdoc(const std::string& name,
                   rapidjson::Document& subdoc) {
  add_subdoc(name, subdoc, nullptr);
}

void
Report::add_subdoc(const std::string& name,
                   rapidjson::Document& subdoc,
                   rapidjson::Document* parent) {
  rapidjson::Document* const target = parent != nullptr ? parent : doc_.get();
  auto& alloc = target->GetAllocator();
  rapidjson::Value key(name.c_str(), alloc);
  rapidjson::Value val;
  val.CopyFrom(subdoc, alloc);
  target->AddMember(key, val, alloc);
}

// Explicit instantiations (Neon-style): single definition in this translation unit.
template void
Report::add_member<int32_t>(const std::string&,
                            const int32_t&,
                            rapidjson::Document*);
template void
Report::add_member<int32_t>(const std::string&,
                            const int32_t&);
template void
Report::add_member<uint32_t>(const std::string&,
                             const uint32_t&,
                             rapidjson::Document*);
template void
Report::add_member<uint32_t>(const std::string&,
                             const uint32_t&);
template void
Report::add_member<int64_t>(const std::string&,
                            const int64_t&,
                            rapidjson::Document*);
template void
Report::add_member<int64_t>(const std::string&,
                            const int64_t&);
template void
Report::add_member<uint64_t>(const std::string&,
                             const uint64_t&,
                             rapidjson::Document*);
template void
Report::add_member<uint64_t>(const std::string&,
                             const uint64_t&);
template void
Report::add_member<double>(const std::string&,
                           const double&,
                           rapidjson::Document*);
template void
Report::add_member<double>(const std::string&,
                           const double&);
template void
Report::add_member<float>(const std::string&,
                          const float&,
                          rapidjson::Document*);
template void
Report::add_member<float>(const std::string&,
                          const float&);
template void
Report::add_member<bool>(const std::string&,
                         const bool&,
                         rapidjson::Document*);
template void
Report::add_member<bool>(const std::string&,
                         const bool&);
template void
Report::add_member<const char*>(const std::string&,
                                const char* const&,
                                rapidjson::Document*);
template void
Report::add_member<const char*>(const std::string&,
                                const char* const&);
template void
Report::add_member<std::string>(const std::string&,
                                const std::string&,
                                rapidjson::Document*);
template void
Report::add_member<std::string>(const std::string&,
                                const std::string&);

template void
Report::add_member<uint32_t>(const std::string&,
                             const std::vector<uint32_t>&,
                             rapidjson::Document*);
template void
Report::add_member<uint32_t>(const std::string&,
                             const std::vector<uint32_t>&);
template void
Report::add_member<int32_t>(const std::string&,
                            const std::vector<int32_t>&,
                            rapidjson::Document*);
template void
Report::add_member<int32_t>(const std::string&,
                            const std::vector<int32_t>&);
template void
Report::add_member<double>(const std::string&,
                           const std::vector<double>&,
                           rapidjson::Document*);
template void
Report::add_member<double>(const std::string&,
                           const std::vector<double>&);
template void
Report::add_member<float>(const std::string&,
                          const std::vector<float>&,
                          rapidjson::Document*);
template void
Report::add_member<float>(const std::string&,
                          const std::vector<float>&);
template void
Report::add_member<bool>(const std::string&,
                         const std::vector<bool>&,
                         rapidjson::Document*);
template void
Report::add_member<bool>(const std::string&,
                         const std::vector<bool>&);

}  // namespace carbon
