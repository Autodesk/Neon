#pragma once

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <rapidjson/document.h>

namespace carbon::detail {

template <typename T>
void
scalar_to_json(rapidjson::Value& out,
               const T& val,
               rapidjson::Document::AllocatorType& alloc) {
  if constexpr (std::is_same_v<std::remove_cvref_t<T>, bool>) {
    out.SetBool(val);
  } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, float>) {
    out.SetDouble(static_cast<double>(val));
  } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, double>) {
    out.SetDouble(val);
  } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, std::string>) {
    out.SetString(val.c_str(), static_cast<rapidjson::SizeType>(val.size()), alloc);
  } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, const char*> ||
                       std::is_same_v<std::remove_cvref_t<T>, char*>) {
    const char* s = val;
    out.SetString(s != nullptr ? s : "", alloc);
  } else if constexpr (std::is_integral_v<std::remove_cvref_t<T>>) {
    using U = std::remove_cvref_t<T>;
    if constexpr (std::is_signed_v<U>) {
      if constexpr (sizeof(U) <= sizeof(int)) {
        out.SetInt(static_cast<int>(val));
      } else {
        out.SetInt64(static_cast<int64_t>(val));
      }
    } else {
      if constexpr (sizeof(U) <= sizeof(unsigned)) {
        out.SetUint(static_cast<unsigned>(val));
      } else {
        out.SetUint64(static_cast<uint64_t>(val));
      }
    }
  } else {
    static_assert(sizeof(T) == 0, "unsupported scalar type for Report::add_member");
  }
}

}  // namespace carbon::detail

namespace carbon {

template <typename T>
std::enable_if_t<!detail::is_vector_v<T>,
                 void>
Report::add_member(const std::string& member_key,
                   const T& member_val,
                   rapidjson::Document* doc) {
  rapidjson::Document* target = doc != nullptr ? doc : doc_.get();
  auto& alloc = target->GetAllocator();
  rapidjson::Value key(member_key.c_str(), alloc);
  rapidjson::Value jv;
  detail::scalar_to_json(jv, member_val, alloc);
  target->AddMember(key, jv, alloc);
}

template <typename T>
void
Report::add_member(const std::string& member_key,
                   const std::vector<T>& member_val,
                   rapidjson::Document* doc) {
  rapidjson::Document* target = doc != nullptr ? doc : doc_.get();
  auto& alloc = target->GetAllocator();
  rapidjson::Value arr(rapidjson::kArrayType);
  for (const auto& item : member_val) {
    rapidjson::Value elem;
    detail::scalar_to_json(elem, item, alloc);
    arr.PushBack(elem, alloc);
  }
  rapidjson::Value key(member_key.c_str(), alloc);
  target->AddMember(key, arr, alloc);
}

template <typename T>
std::enable_if_t<!detail::is_vector_v<T>,
                 void>
Report::add_member(const std::string& member_key,
                   const T& member_val) {
  add_member(member_key, member_val, nullptr);
}

template <typename T>
void
Report::add_member(const std::string& member_key,
                   const std::vector<T>& member_val) {
  add_member(member_key, member_val, nullptr);
}

}  // namespace carbon
