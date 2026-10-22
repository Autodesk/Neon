#pragma once

#include <carbon/detail/backend/select.h>
#include <carbon/system.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace carbon::bench {

struct TimingStats {
  double median_seconds = 0.0;
  double min_seconds = 0.0;
  std::vector<double> samples{};
};

[[nodiscard]] inline auto
json_escape(const std::string& s) -> std::string {
  std::string out;
  out.reserve(s.size() + 8U);
  for (const char c : s) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
        break;
    }
  }
  return out;
}

[[nodiscard]] inline auto
utc_timestamp_now() -> std::string {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

[[nodiscard]] inline auto
parse_string_flag(std::string_view flag,
                  int argc,
                  char** argv) -> std::string {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string_view{argv[i]} == flag) {
      return std::string{argv[i + 1]};
    }
  }
  return {};
}

[[nodiscard]] inline auto
parse_int_flag(std::string_view flag,
               int argc,
               char** argv,
               int fallback) -> int {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string_view{argv[i]} == flag) {
      return std::atoi(argv[i + 1]);
    }
  }
  return fallback;
}

[[nodiscard]] inline auto
interconnect_topo() -> std::string {
  if (const char* env = std::getenv("CARBON_BENCH_TOPO")) {
    return std::string{env};
  }
  return "unknown";
}

[[nodiscard]] inline auto
compute_timing_stats(std::vector<double> samples) -> TimingStats {
  TimingStats stats{};
  stats.samples = std::move(samples);
  if (stats.samples.empty()) {
    return stats;
  }
  stats.min_seconds = *std::min_element(stats.samples.begin(), stats.samples.end());
  std::sort(stats.samples.begin(), stats.samples.end());
  const size_t mid = stats.samples.size() / 2U;
  if (stats.samples.size() % 2U == 0U) {
    stats.median_seconds = 0.5 * (stats.samples[mid - 1U] + stats.samples[mid]);
  } else {
    stats.median_seconds = stats.samples[mid];
  }
  return stats;
}

inline auto
write_system_json(std::ostream& out,
                  const carbon::System& sys) -> void {
  const size_t n_gpus = sys.gpu_group().size();
  out << "  \"system\": {\n";
  out << "    \"backend\": \"" << CARBON_BACKEND_NAME << "\",\n";
  out << "    \"interconnect_topo\": \"" << json_escape(interconnect_topo()) << "\",\n";
  out << "    \"gpu_count\": " << n_gpus << ",\n";
  out << "    \"gpus\": [\n";
  const auto& gpu_devices = sys.gpu_group().devices();
  for (size_t i = 0; i < gpu_devices.size(); ++i) {
    const auto& props = gpu_devices[i].properties();
    out << "      {\n";
    out << "        \"device_id\": " << gpu_devices[i].device_id() << ",\n";
    out << "        \"name\": \"" << json_escape(props.name) << "\",\n";
    out << "        \"compute_capability\": \"" << props.compute_major << "." << props.compute_minor
        << "\",\n";
    out << "        \"total_memory_bytes\": " << props.total_global_mem_bytes << ",\n";
    out << "        \"multiprocessor_count\": " << props.multiprocessor_count << "\n";
    out << "      }" << (i + 1U == gpu_devices.size() ? "" : ",") << "\n";
  }
  out << "    ]\n";
  out << "  }";
}

inline auto
write_timing_results_json(std::ostream& out,
                          const TimingStats& stats,
                          double headline,
                          const char* headline_key) -> void {
  out << "  \"results\": {\n";
  out << "    \"repeats\": " << stats.samples.size() << ",\n";
  out << "    \"timing_seconds\": " << stats.median_seconds << ",\n";
  out << "    \"timing_seconds_min\": " << stats.min_seconds << ",\n";
  out << "    \"timing_samples_seconds\": [";
  for (size_t i = 0; i < stats.samples.size(); ++i) {
    if (i > 0U) {
      out << ", ";
    }
    out << stats.samples[i];
  }
  out << "],\n";
  out << "    \"" << headline_key << "\": " << headline << "\n";
  out << "  }\n";
}

[[nodiscard]] inline auto
write_json_report(const std::string& path,
                  std::string_view benchmark,
                  const std::string& config_json,
                  const carbon::System& sys,
                  const TimingStats& stats,
                  double headline,
                  const char* headline_key) -> bool {
  std::ofstream out(path);
  if (!out.is_open()) {
    std::cerr << benchmark << ": failed to open --report path '" << path << "'\n";
    return false;
  }
  out << std::setprecision(10);
  out << "{\n";
  out << "  \"schema\": \"carbon-bench-report-v1\",\n";
  out << "  \"benchmark\": \"" << json_escape(std::string(benchmark)) << "\",\n";
  out << "  \"timestamp_utc\": \"" << utc_timestamp_now() << "\",\n";
  out << "  \"config\": " << config_json << ",\n";
  write_system_json(out, sys);
  out << ",\n";
  write_timing_results_json(out, stats, headline, headline_key);
  out << "}\n";
  return !out.fail();
}

}  // namespace carbon::bench
