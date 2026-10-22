#pragma once

#include <carbon/logging.h>

#include <stdexcept>
#include <string>

namespace carbon {
namespace report {

namespace {

inline void
require_initialized(const char* fn) {
  if (!::carbon::report_initialized()) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "carbon::report::{}: global report not initialized", fn);
  }
}

}  // namespace

template <typename T>
std::enable_if_t<!::carbon::detail::is_vector_v<T>,
                 void>
add_member(const std::string& member_key,
           const T& member_val,
           rapidjson::Document* doc) {
  require_initialized("add_member");
  global_report().add_member(member_key, member_val, doc);
}

template <typename T>
void
add_member(const std::string& member_key,
           const std::vector<T>& member_val,
           rapidjson::Document* doc) {
  require_initialized("add_member");
  global_report().add_member(member_key, member_val, doc);
}

template <typename T>
std::enable_if_t<!::carbon::detail::is_vector_v<T>,
                 void>
add_member(const std::string& member_key,
           const T& member_val) {
  require_initialized("add_member");
  global_report().add_member(member_key, member_val);
}

template <typename T>
void
add_member(const std::string& member_key,
           const std::vector<T>& member_val) {
  require_initialized("add_member");
  global_report().add_member(member_key, member_val);
}

}  // namespace report
}  // namespace carbon
