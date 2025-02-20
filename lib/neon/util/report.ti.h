#pragma once

#include <string>
#include <vector>

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include "neon/util/report.h"
namespace neon::util {

template <typename T>
auto Report::addMember(const std::string& memberKey, const T memberVal) -> void
{
    addMember(memberKey, memberVal, mDoc.get());
}

template <typename T>
auto Report::addMember(const std::string& memberKey, const T memberVal, rapidjson::Document* doc) -> void
{
    rapidjson::Value key(memberKey.c_str(), doc->GetAllocator());
    if constexpr (std::is_same_v<T, int32_t>) {
        doc->AddMember(key, rapidjson::Value().SetInt(memberVal), doc->GetAllocator());
        return;
    }

    if constexpr (std::is_same_v<T, int64_t>) {
        doc->AddMember(key, rapidjson::Value().SetInt64(memberVal), doc->GetAllocator());
        return;
    }

    if constexpr (std::is_same_v<T, uint64_t>) {
        doc->AddMember(key, rapidjson::Value().SetUint64(memberVal), doc->GetAllocator());
        return;
    }

    if constexpr (std::is_same_v<T, uint32_t>) {
        doc->AddMember(key, rapidjson::Value().SetUint(memberVal), doc->GetAllocator());
        return;
    }

    if constexpr (std::is_same_v<T, double>) {
        doc->AddMember(key, rapidjson::Value().SetDouble(memberVal), doc->GetAllocator());
        return;
    }

    if constexpr (std::is_same_v<T, float>) {
        doc->AddMember(key, rapidjson::Value().SetFloat(memberVal), doc->GetAllocator());
        return;
    }

    if constexpr (std::is_same_v<T, bool>) {
        doc->AddMember(key, rapidjson::Value().SetBool(memberVal), doc->GetAllocator());
        return;
    }

    if constexpr (std::is_same_v<T, std::string>) {
        doc->AddMember(key,
                       rapidjson::Value().SetString(
                           memberVal.c_str(), rapidjson::SizeType(memberVal.length()), doc->GetAllocator()),
                       doc->GetAllocator());
        return;
    }

    if constexpr (std::is_same_v<T, const char*>) {
        std::string memberValStr = std::string(memberVal);
        doc->AddMember(key,
                       rapidjson::Value().SetString(
                           memberValStr.c_str(), rapidjson::SizeType(memberValStr.length()), doc->GetAllocator()),
                       doc->GetAllocator());
        return;
    }


    static_assert(true, "Neon::Report only handles values of type int32_t, unit32_t, int64_t, unit64_t, double, float, bool, const char*, or std::string");
}


template <typename T>
auto Report::addMember(const std::string&    memberKey,
                       const std::vector<T>& memberVal,
                       rapidjson::Document*  doc) -> void
{
    rapidjson::Value val(rapidjson::kArrayType);
    rapidjson::Value key(memberKey.c_str(), doc->GetAllocator());

    for (size_t i = 0; i < memberVal.size(); ++i) {
        val.PushBack(rapidjson::Value(memberVal[i]).Move(),
                     doc->GetAllocator());
    }
    doc->AddMember(key, val, doc->GetAllocator());
}




}  // namespace neon::util