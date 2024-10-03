#include "report.h"
#include <string>
#include <vector>

Report::Report(const Config& c)
    : mReport("lbm-lid-driven-cavity-flow")
{
    mFname = c.reportFile;

    mReport.addMember("argv", c.mArgv);

    mReport.addMember("N", c.N);
    mReport.addMember("benchmark", c.benchmark);
    mReport.addMember("repetitions", c.repetitions);
    mReport.addMember("vti", c.vti);


    mReport.addMember("benchIniIter", c.benchIniIter);
    mReport.addMember("benchMaxIter", c.benchMaxIter);

    mReport.addMember("deviceType", c.deviceType);
    mReport.addMember("numDevices", c.devices.size());
    mReport.addMember("devices", c.devices);
    mReport.addMember("reportFile", c.reportFile);
    mReport.addMember("gridType", c.gridType);


    c.occCli.addToReport(mReport);
    c.transferModeCli.addToReport(mReport);
    c.stencilSemanticCli.addToReport(mReport);
    c.spaceCurveCli.addToReport(mReport);

    mReport.addMember("computeTypeStr", c.computeTypeStr);

}

auto Report::
    recordMLUPS(double mlups)
        -> void
{
    mMLUPS.push_back(mlups);
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

auto Report::recordNeonGridInitTime(double time, const std::string& unit) -> void
{
    if (mtimeUnit.length() == 0) {
        mtimeUnit = unit;
    }
    if (unit.length() != mtimeUnit.length()) {
        NEON_THROW_UNSUPPORTED_OPERATION("Time unit inconsistency");
    }
    mNeonGridInitTime.push_back(time);
}

auto Report::recordProblemSetupTime(double time, const std::string& unit) -> void
{
    if (mtimeUnit.length() == 0) {
        mtimeUnit = unit;
    }
    if (unit.length() != mtimeUnit.length()) {
        NEON_THROW_UNSUPPORTED_OPERATION("Time unit inconsistency");
    }
    mProblemSetupTime.push_back(time);
}

auto Report::
    save(std::stringstream& testCode)
        -> void
{
    mReport.addMember("MLUPS", mMLUPS);
    mReport.addMember(std::string("Loop Time (") + mtimeUnit + ")", mLoopTime);
    mReport.addMember(std::string("Problem Setup Time (") + mtimeUnit + ")", mProblemSetupTime);
    mReport.addMember(std::string("Neon Grid Init Time (") + mtimeUnit + ")", mNeonGridInitTime);

    mReport.write(mFname + testCode.str(), true);
}

auto Report::recordBk(Neon::Backend& backend) -> void
{
    backend.toReport(mReport);
}

auto Report::recordGrid(Neon::domain::interface::GridBase& g) -> void
{
    g.toReport(mReport, true);
}

auto Report::helpGetReport() -> Neon::Report&
{
    return mReport;
}