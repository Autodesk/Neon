#include <string>
#include <vector>
#include "Repoert.h"

Report::Report(const Config& c)
    : mReport("lbm-lid-driven-cavity-flow")
{
    mReport.addMember("Re", c.Re);
    mReport.addMember("ulb", c.ulb);
    mReport.addMember("N", c.N);
    mReport.addMember("benchmark", c.benchmark);
    mReport.addMember("max_t", c.max_t);
    mReport.addMember("outFrequency", c.outFrequency);
    mReport.addMember("dataFrequency", c.dataFrequency);

    mReport.addMember("benchIniIter", c.benchIniIter);
    mReport.addMember("benchMaxIter", c.benchMaxIter);

    mReport.addMember("numDevices", c.devices.size());
    mReport.addMember("devices", c.devices);
    mReport.addMember("reportFile", c.reportFile);
    mReport.addMember("gridType", c.gridType);

    mReport.addMember("occ", Neon::skeleton::OccUtils::toString(c.occ));
    mReport.addMember("transferMode", Neon::set::TransferModeUtils::toString(c.transferMode));
    mReport.addMember("transferSemantic", Neon::set::TransferSemanticUtils::toString(c.transferSemantic));
}

auto Report::
    recordMLUPS(double mlups)
        -> void
{
    mMLUPS.push_back(mlups);
}

auto Report::
    recordTime(double             time,
               const std::string& unit)
        -> void
{
    if (mtimeUnit.length() == 0) {
        if (unit.length() == 0) {
            NEON_THROW_UNSUPPORTED_OPERATION("Time unit missing");
        }
        mtimeUnit = unit;
    }
    mTime.push_back(time);
}

auto Report::
    save()
        -> void
{
    mReport.addMember("MLUPS", mMLUPS);
    mReport.addMember(std::string("Time (") + mtimeUnit + ")", mTime);

    mReport.write(mFname, true);
}
