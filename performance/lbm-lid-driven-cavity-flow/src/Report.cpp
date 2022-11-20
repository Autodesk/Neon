#include <string>
#include <vector>
#include "Repoert.h"

Report::Report(const Config& c)
    : mReport("lbm-lid-driven-cavity-flow")
{
    mFname = c.reportFile;

    mReport.addMember("Re", c.Re);
    mReport.addMember("ulb", c.ulb);
    mReport.addMember("N", c.N);
    mReport.addMember("benchmark", c.benchmark);
    mReport.addMember("max_t", c.max_t);
    mReport.addMember("outFrequency", c.outFrequency);
    mReport.addMember("dataFrequency", c.dataFrequency);
    mReport.addMember("repetitions", c.repetitions);

    mReport.addMember("benchIniIter", c.benchIniIter);
    mReport.addMember("benchMaxIter", c.benchMaxIter);

    mReport.addMember("numDevices", c.devices.size());
    mReport.addMember("devices", c.devices);
    mReport.addMember("reportFile", c.reportFile);
    mReport.addMember("gridType", c.gridType);

    mReport.addMember("occ", Neon::skeleton::OccUtils::toString(c.occ));
    mReport.addMember("transferMode", Neon::set::TransferModeUtils::toString(c.transferMode));
    mReport.addMember("transferSemantic", Neon::set::TransferSemanticUtils::toString(c.transferSemantic));

    mReport.addMember("nu", c.mLbmParameters.nu);
    mReport.addMember("omega", c.mLbmParameters.omega);
    mReport.addMember("dx", c.mLbmParameters.dx);
    mReport.addMember("dt", c.mLbmParameters.dt);

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
    if (unit.length() != 0) {
        NEON_THROW_UNSUPPORTED_OPERATION("Time unit is missing");
    }
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
    if (unit.length() != 0) {
        NEON_THROW_UNSUPPORTED_OPERATION("Time unit is missing");
    }
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
    if (unit.length() != 0) {
        NEON_THROW_UNSUPPORTED_OPERATION("Time unit is missing");
    }
    if (mtimeUnit.length() == 0) {
        mtimeUnit = unit;
    }
    if (unit.length() != mtimeUnit.length()) {
        NEON_THROW_UNSUPPORTED_OPERATION("Time unit inconsistency");
    }
    mProblemSetupTime.push_back(time);
}

auto Report::
    save()
        -> void
{
    mReport.addMember("MLUPS", mMLUPS);
    mReport.addMember(std::string("Loop Time (") + mtimeUnit + ")", mLoopTime);

    mReport.write(mFname, true);
}


