#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include "Neon/core/tools/Logger.h"
#include "Neon/core/tools/Report.h"
#include "Neon/core/tools/git_sha1.h"

namespace Neon {

namespace core {

Report::Report(const std::string& record_name)
{
    mDoc = std::make_shared<rapidjson::Document>();
    mDoc->SetObject();
    mDoc->AddMember("Record Name",
                    rapidjson::Value().SetString(record_name.c_str(),
                                                 rapidjson::SizeType(record_name.length()),
                                                 mDoc->GetAllocator()),
                    mDoc->GetAllocator());

    //git metadata
    std::string str = g_GIT_SHA1;
    mDoc->AddMember("git_sha",
                    rapidjson::Value().SetString(
                        str.c_str(), rapidjson::SizeType(str.length()), mDoc->GetAllocator()),
                    mDoc->GetAllocator());

    std::string str_status = g_GIT_LOCAL_CHANGES_STATUS;
    mDoc->AddMember(
        "git_local_changes_status",
        rapidjson::Value().SetString(
            str_status.c_str(), rapidjson::SizeType(str_status.length()), mDoc->GetAllocator()),
        mDoc->GetAllocator());

    std::string str_refspec = g_GIT_REFSPEC;
    mDoc->AddMember("git_refspec",
                    rapidjson::Value().SetString(str_refspec.c_str(),
                                                 rapidjson::SizeType(str_refspec.length()),
                                                 mDoc->GetAllocator()),
                    mDoc->GetAllocator());

    // Time
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    {
        std::ostringstream oss;
        oss << std::put_time(&tm, "_D%d_%m_%Y__T%H_%M_%S");
        mOutputNameSuffix = oss.str() + ".json";
    }

    {
        std::ostringstream oss;
        oss << std::put_time(&tm, "%a %d:%m:%Y %H:%M:%S");
        std::string str_out = oss.str();

        mDoc->AddMember(
            "date",
            rapidjson::Value().SetString(
                str_out.c_str(), rapidjson::SizeType(str_out.length()), mDoc->GetAllocator()),
            mDoc->GetAllocator());
    }
}

auto Report::setToken(const std::string& token) -> void
{
    mDoc->AddMember("token",
                    rapidjson::Value().SetString(
                        token.c_str(), rapidjson::SizeType(token.length()), mDoc->GetAllocator()),
                    mDoc->GetAllocator());
}

auto Report::commandLine(int argc, char** argv) -> void
{
    std::string cmd(argv[0]);
    for (int i = 1; i < argc; i++) {
        cmd = cmd + " " + std::string(argv[i]);
    }
    mDoc->AddMember("command_line",
                    rapidjson::Value().SetString(
                        cmd.c_str(), rapidjson::SizeType(cmd.length()), mDoc->GetAllocator()),
                    mDoc->GetAllocator());
}


auto Report::write(const std::string& outputFilename,
                   bool               appendTimeToFileName /*= true*/,
                   const std::string&  outputFolder /* = "."*/) -> void
{
    // https://stackoverflow.com/a/6417908/1608232
    auto remove_extension = [](const std::string& filename) -> std::string {
        size_t lastdot = filename.find_last_of(".");
        if (lastdot == std::string::npos)
            return filename;
        return filename.substr(0, lastdot);
    };

    std::string full_name =
        outputFolder + "/" + remove_extension(outputFilename) +
        (appendTimeToFileName ? mOutputNameSuffix : ".json");

    // create the folder if it does not exist
    if (!std::filesystem::is_directory(outputFolder) ||
        !std::filesystem::exists(outputFolder)) {
        std::filesystem::create_directories(outputFolder);
    }

    std::ofstream ofs(full_name);
    if (!ofs.is_open()) {
        NEON_ERROR("Report::write() can not open {}", full_name);
    }
    rapidjson::OStreamWrapper                          osw(ofs);
    rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(osw);
    mDoc->Accept(writer);
}

auto Report::getSubdoc() -> rapidjson::Document
{
    rapidjson::Document subdoc(&mDoc->GetAllocator());
    subdoc.SetObject();
    return subdoc;
}

auto Report::addSubdoc(const std::string& name, rapidjson::Document& subdoc) -> void
{
    mDoc->AddMember(rapidjson::Value().SetString(
                        name.c_str(), rapidjson::SizeType(name.length()), mDoc->GetAllocator()),
                    subdoc, mDoc->GetAllocator());
}


template void Report::addMember<uint32_t>(const std::string&, const uint32_t);
template void Report::addMember<int32_t>(const std::string&, const int32_t);
template void Report::addMember<uint64_t>(const std::string&, const uint64_t);
template void Report::addMember<int64_t>(const std::string&, const int64_t);
template void Report::addMember<const char*>(const std::string&, const char*);
template void Report::addMember<double>(const std::string&, const double);
template void Report::addMember<float>(const std::string&, const float);
template void Report::addMember<bool>(const std::string&, const bool);
template void Report::addMember<std::string>(const std::string&, const std::string);

template void Report::addMember<std::vector<uint32_t>>(const std::string&, const std::vector<uint32_t>);
template void Report::addMember<std::vector<int32_t>>(const std::string&, const std::vector<int32_t>);
template void Report::addMember<std::vector<double>>(const std::string&, const std::vector<double>);
template void Report::addMember<std::vector<float>>(const std::string&, const std::vector<float>);
template void Report::addMember<std::vector<bool>>(const std::string&, const std::vector<bool>);

}  // namespace core
}  // namespace Neon