#pragma once
#include <memory>
#include <string>
#include <vector>

#include <rapidjson/document.h>

namespace Neon::core {

/**
 * Base class for storing test results in JSON files. This class only implements
 * basic methods like adding a member. It should not be used as it is if the user
 * is looking for more information about the host or device systems
 */
class Report
{
   public:
    using SubBlock = rapidjson::Document;

    Report() = default;

    /**
     * Constructor with name of the record
     */
    Report(const std::string& record_name);

    /**
     * Add the command line arguments to the report
     */
    auto commandLine(int argc, char** argv) -> void;

    /**
     * Set a token for this report. Token will be added as a new member
     */
    auto setToken(const std::string& token) -> void;

    /**
     * Write the report to a file with possibility to specify the output directory
     * and appending the time to the file name
     */
    auto write(const std::string& outputFilename,
               bool               appendTimeToFileName = true,
               const std::string& outputFolder = ".") -> void;

    /**
     * Add a new key-value to the report. Key must be a string while value could be
     * int32_t, uint32_t, double, float, bool, std::string, or std::vector of any
     * of these types
     */
    template <typename T>
    auto addMember(const std::string& memberKey, const T memberVal) -> void;


    /**
     * Add a new key-value to the report. This method is only sensible if the new key-value is added to a subdoc,
     * Key must be a string while value could be int32_t, uint32_t, double, float, bool,
     * std::string, or std::vector of any of these types
     */
    template <typename T>
    auto addMember(const std::string&   memberKey,
                   const T              memberVal,
                   rapidjson::Document* doc) -> void;

    /**
     * Generate a new subdoc. Subdoc groups a set of relevant information together
     * where these information are the value
     */
    auto getSubdoc() -> rapidjson::Document;

    /**
     * Add a subdoc to this report
     */
    auto addSubdoc(const std::string& name, rapidjson::Document& subdoc) -> void;


    template <typename T>
    auto addMember(const std::string&    memberKey,
                   const std::vector<T>& memberVal,
                   rapidjson::Document*  doc) -> void;

   protected:

    std::shared_ptr<rapidjson::Document> mDoc;
    std::string                          mOutputNameSuffix;
};
}  // namespace Neon::core

#include "Neon/core/tools/Report_imp.h"