#pragma once

#include <string>
#include <vector>
#include "Neon/core/tools/clipp.h"
#include "Neon/domain/tools/SpaceCurves.h"
#include "Neon/skeleton/Skeleton.h"

enum class Op
{
    axpy,
    apy
};

class OpUtils
{
   public:
    static auto formString(std::string const& option)
        -> Op
    {
        if (option == "axpy")
            return Op::axpy;
        if (option == "apy")
            return Op::apy;
        Neon::NeonException exp("");
        exp << "wrong option";
        NEON_THROW(exp);
    }

    static auto toString(Op const& op)
        -> std::string
    {
        switch (op) {
            case Op::axpy:
                return "axpy";
            case Op::apy:
                return "apy";
        }
        Neon::NeonException exp("");
        exp << "wrong option";
        NEON_THROW(exp);
    }
};

struct Config
{
    int         n;
    Op          op;
    int         iterations;
    int         repetitions;
    int         cardinality;
    int         blockSize;
    std::string reportName;
    std::string argv;


    auto toString()
        const -> std::string;

    auto parseArgs(int argc, char* argv[])
        -> int;
};
