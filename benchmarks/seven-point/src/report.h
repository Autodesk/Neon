#pragma once

#include <string>
#include <vector>
#include "Neon/domain/interface/GridBase.h"
#include "config.h"
struct Report
{
    Neon::Report mReport;
    std::string  mFname;

    std::vector<double> mMLUPS;
    std::vector<double> mLoopTime;
    std::vector<double> mNeonGridInitTime;
    std::vector<double> mProblemSetupTime;

    std::string mtimeUnit = "";

    explicit Report(const Config& c);

    auto recordMLUPS(double mlups)
        -> void;

    auto recordLoopTime(double             time,
                        const std::string& unit)
        -> void;

    auto recordNeonGridInitTime(double             time,
                                const std::string& unit)
        -> void;

    auto recordProblemSetupTime(double             time,
                                const std::string& unit)
        -> void;

    auto save(std::stringstream& testCode)
        -> void;
    void recordBk(Neon::Backend& backend);

    void recordGrid(Neon::domain::interface::GridBase& g);

    auto helpGetReport() -> Neon::Report&;
};
