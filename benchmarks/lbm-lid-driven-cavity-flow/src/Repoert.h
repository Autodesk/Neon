#pragma once

#include <string>
#include <vector>
#include "Config.h"

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

    auto save()
        -> void;
    void recordBk(Neon::Backend& backend);
};
