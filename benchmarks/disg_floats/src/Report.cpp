#include <string>
#include <vector>
#include "Repoert.h"

Report::Report(const Config& c)
    : mReport("lbm-lid-driven-cavity-flow")
{
    mFname = c.reportName;

    mReport.addMember("argv", c.argv);

    mReport.addMember("Re", c.n);
    mReport.addMember("ulb", c.iterations);
    mReport.addMember("N", c.cardinality);
    mReport.addMember("benchmark", OpUtils::toString(c.op));

}

auto Report::
    recordLoopTime(double             time,
                   const std::string& unit)
        -> void
{
    if (mtimeUnit.length() == 0) {
        mtimeUnit = unit;
    }
    if (unit.length() != mtimeUnit.length()) {
        NEON_THROW_UNSUPPORTED_OPERATION("Time unit inconsistency");
    }
    mLoopTime.push_back(time);
}


auto Report::
    save(std::stringstream& testCode)
        -> void
{
    mReport.addMember(std::string("Loop Time (") + mtimeUnit + ")", mLoopTime);

    mReport.write(mFname + testCode.str(), true);
}

auto Report::helpGetReport() -> Neon::Report&
{
    return mReport;
}