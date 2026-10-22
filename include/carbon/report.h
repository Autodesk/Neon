#pragma once

#include <carbon/export.h>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <rapidjson/document.h>

namespace carbon::detail {

template <typename>
struct is_vector : std::false_type {};

template <typename T, typename A>
struct is_vector<std::vector<T, A>> : std::true_type {};

template <typename T>
inline constexpr bool is_vector_v = is_vector<std::remove_cvref_t<T>>::value;

}  // namespace carbon::detail

namespace carbon {

/**
 * JSON report container (similar in spirit to Neon::core::Report): collect key/value data
 * and write pretty-printed JSON to disk.
 */
class CARBON_API Report {
public:
  using SubBlock = rapidjson::Document;

  Report();

  /** Construct and store @p record_name as member `"record"` and as the output name suffix. */
  explicit Report(const std::string& record_name);

  /** Add argv entries as a JSON array under `"command_line"`. */
  void
  command_line(int argc,
               char** argv);

  /** Store @p token as member `"token"`. */
  void
  set_token(const std::string& token);

  /**
   * Write the document to @p output_folder / (@p output_filename + optional timestamp + ".json").
   */
  void
  write(const std::string& output_filename,
        bool append_time_to_file_name = true,
        const std::string& output_folder = ".");

  template <typename T>
  std::enable_if_t<!detail::is_vector_v<T>,
                   void>
  add_member(const std::string& member_key,
             const T& member_val);

  template <typename T>
  std::enable_if_t<!detail::is_vector_v<T>,
                   void>
  add_member(const std::string& member_key,
             const T& member_val,
             rapidjson::Document* doc);

  template <typename T>
  void
  add_member(const std::string& member_key,
             const std::vector<T>& member_val,
             rapidjson::Document* doc = nullptr);

  template <typename T>
  void
  add_member(const std::string& member_key,
             const std::vector<T>& member_val);

  /** New empty JSON object; fill it and pass to add_subdoc. */
  SubBlock
  get_subdoc();

  /** Attach @p subdoc as an object member named @p name on the root document. */
  void
  add_subdoc(const std::string& name,
             rapidjson::Document& subdoc);

  /**
   * Attach @p subdoc as a member named @p name on @p parent. If @p parent is null, behaves like
   * add_subdoc(name, subdoc) on the root document.
   */
  void
  add_subdoc(const std::string& name,
             rapidjson::Document& subdoc,
             rapidjson::Document* parent);

protected:
  std::shared_ptr<rapidjson::Document> doc_;
  std::string output_name_suffix_;
};

}  // namespace carbon

#include "carbon/detail/report.inl"

namespace carbon {

// Explicit instantiation declarations (definitions in src/report.cpp); pattern from Neon Report_imp.h.
extern template void
Report::add_member<int32_t>(const std::string&,
                            const int32_t&,
                            rapidjson::Document*);
extern template void
Report::add_member<int32_t>(const std::string&,
                            const int32_t&);
extern template void
Report::add_member<uint32_t>(const std::string&,
                             const uint32_t&,
                             rapidjson::Document*);
extern template void
Report::add_member<uint32_t>(const std::string&,
                             const uint32_t&);
extern template void
Report::add_member<int64_t>(const std::string&,
                            const int64_t&,
                            rapidjson::Document*);
extern template void
Report::add_member<int64_t>(const std::string&,
                            const int64_t&);
extern template void
Report::add_member<uint64_t>(const std::string&,
                             const uint64_t&,
                             rapidjson::Document*);
extern template void
Report::add_member<uint64_t>(const std::string&,
                             const uint64_t&);
extern template void
Report::add_member<double>(const std::string&,
                           const double&,
                           rapidjson::Document*);
extern template void
Report::add_member<double>(const std::string&,
                           const double&);
extern template void
Report::add_member<float>(const std::string&,
                          const float&,
                          rapidjson::Document*);
extern template void
Report::add_member<float>(const std::string&,
                          const float&);
extern template void
Report::add_member<bool>(const std::string&,
                         const bool&,
                         rapidjson::Document*);
extern template void
Report::add_member<bool>(const std::string&,
                         const bool&);
extern template void
Report::add_member<const char*>(const std::string&,
                                const char* const&,
                                rapidjson::Document*);
extern template void
Report::add_member<const char*>(const std::string&,
                                const char* const&);
extern template void
Report::add_member<std::string>(const std::string&,
                                const std::string&,
                                rapidjson::Document*);
extern template void
Report::add_member<std::string>(const std::string&,
                                const std::string&);

extern template void
Report::add_member<uint32_t>(const std::string&,
                             const std::vector<uint32_t>&,
                             rapidjson::Document*);
extern template void
Report::add_member<uint32_t>(const std::string&,
                             const std::vector<uint32_t>&);
extern template void
Report::add_member<int32_t>(const std::string&,
                            const std::vector<int32_t>&,
                            rapidjson::Document*);
extern template void
Report::add_member<int32_t>(const std::string&,
                            const std::vector<int32_t>&);
extern template void
Report::add_member<double>(const std::string&,
                           const std::vector<double>&,
                           rapidjson::Document*);
extern template void
Report::add_member<double>(const std::string&,
                           const std::vector<double>&);
extern template void
Report::add_member<float>(const std::string&,
                          const std::vector<float>&,
                          rapidjson::Document*);
extern template void
Report::add_member<float>(const std::string&,
                          const std::vector<float>&);
extern template void
Report::add_member<bool>(const std::string&,
                         const std::vector<bool>&,
                         rapidjson::Document*);
extern template void
Report::add_member<bool>(const std::string&,
                         const std::vector<bool>&);

/**
 * Create the process-wide Report singleton (exactly once). @p record_name is forwarded to
 * Report(const std::string&). Throws std::runtime_error if called again after success.
 * Not thread-safe: call from a single thread during startup (or otherwise externally
 * synchronized).
 */
CARBON_API void
init_report(const std::string& record_name);

/** Convenience: init_report(record_name) and command_line(argc, argv). Not thread-safe. */
CARBON_API void
init_report(const std::string& record_name,
            int argc,
            char** argv);

/** Reference to the global Report; throws if init_report was not called. Not thread-safe. */
CARBON_API Report&
global_report();

/** True after a successful init_report. */
CARBON_API bool
report_initialized() noexcept;

/**
 * Wrappers that forward to global_report(); each checks report_initialized() first.
 * Not thread-safe (same as init_report / global_report).
 */
namespace report {

/** Same as init_report(@p record_name). Not thread-safe. */
CARBON_API void
init(const std::string& record_name);

/** Same as init_report(@p record_name, argc, argv). Not thread-safe. */
CARBON_API void
init(const std::string& record_name,
     int argc,
     char** argv);

CARBON_API void
command_line(int argc,
             char** argv);

CARBON_API void
set_token(const std::string& token);

CARBON_API void
write(const std::string& output_filename,
      bool append_time_to_file_name = true,
      const std::string& output_folder = ".");

template <typename T>
std::enable_if_t<!detail::is_vector_v<T>,
                 void>
add_member(const std::string& member_key,
           const T& member_val);

template <typename T>
std::enable_if_t<!detail::is_vector_v<T>,
                 void>
add_member(const std::string& member_key,
           const T& member_val,
           rapidjson::Document* doc);

template <typename T>
void
add_member(const std::string& member_key,
           const std::vector<T>& member_val,
           rapidjson::Document* doc = nullptr);

template <typename T>
void
add_member(const std::string& member_key,
           const std::vector<T>& member_val);

CARBON_API Report::SubBlock
get_subdoc();

CARBON_API void
add_subdoc(const std::string& name,
           rapidjson::Document& subdoc);

CARBON_API void
add_subdoc(const std::string& name,
           rapidjson::Document& subdoc,
           rapidjson::Document* parent);

}  // namespace report

}  // namespace carbon

#include "carbon/detail/report_global.inl"
