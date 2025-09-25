#pragma once
#include "Neon/core/tools/Report.h"


namespace Neon {

/**
 * Extends the base report in libNeonCore by adding host and device information to the report. 
 * This is what the user should be using 
*/
class Report : public core::Report
{
   public:
    Report() = default;

    /**
     * Constructor with name of the record
    */
    Report(const std::string& record_name);

   protected:
    /**
     * Adding gpus information to this report
    */
    auto device() -> void;

    /**
     * Adding host information to this report      
    */
    auto system() -> void;
};


}  // namespace Neon
