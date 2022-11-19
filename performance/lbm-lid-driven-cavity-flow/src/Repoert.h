#pragma once

#include <string>
#include <vector>
#include "Config.h"

struct Report
{
    Neon::Report mReport;
    std::string  mFname;

    std::vector<double> mMLUPS;
    std::vector<double> mTime;

    std::string mtimeUnit = "";

    explicit Report(const Config& c);

    auto recordMLUPS(double mlups)
        -> void;

    auto recordTime(double             time,
                    const std::string& unit)
        -> void;

    auto save()
        -> void;
};
