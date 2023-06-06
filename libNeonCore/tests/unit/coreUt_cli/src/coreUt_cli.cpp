
#include "gtest/gtest.h"

#include "Neon/core/core.h"
#include "Neon/core/tools/clipp.h"

#include <Neon/core/types/vec.h>
#include <cstring>
#include <iostream>


struct setting_t
{

    std::vector<int> gpuIds;
    bool             maxGpuSet;
    int              nIterations;
    int              cudeEdge;

    setting_t()
    {
        maxGpuSet = false;
        nIterations = 0;
        cudeEdge = 0;
    }
    std::string toString()
    {
        std::ostringstream msg;
        msg << "Config:" << std::endl;
        msg << "nIterations " << nIterations << std::endl;
        msg << "cudeEdge " << cudeEdge << std::endl;
        msg << "GpuSet ";
        for (int i = 0; i < (int)gpuIds.size(); i++) {
            msg << gpuIds[i] << " ";
        }
        msg << std::endl;
        return msg.str();
    }
};

setting_t cmdline_settings(int argc, char* argv[])
{
    setting_t s;

    auto cli = (clipp::required("-n", "--iter") & clipp::value("iterations", s.nIterations),
                clipp::required("-b", "--boxSize") & clipp::value("Size of the box (n voxels)", s.cudeEdge),
                clipp::option("-g", "--gpuSet") & clipp::integers("gpuSet", s.gpuIds));

    auto res = parse(argc, argv, cli);

    if (res.any_error()) {
        auto fmt = clipp::doc_formatting{}
                       .first_column(7)
                       .doc_column(15)
                       .last_column(99);
        std::cout << make_man_page(cli, "testProgram", fmt) << '\n';
        Neon::NeonException ecp("Command Line Parsing");
        ecp << "Invalid Parsing";
        NEON_THROW(ecp);
    }

    std::cout << s.toString();

    return s;
}

TEST(meta, typedField)
{
    int         iterations = 10;
    int         boxSize = 22;
    std::string programName("coreUt_cli");
    std::string nOpt("-n10");
    std::string boxOpt("-b22");
    char*       argv[3];
    argv[0] = &programName[0];
    argv[1] = &nOpt[0];
    argv[2] = &boxOpt[0];

    [[maybe_unused]] setting_t set;
    ASSERT_NO_THROW(set = cmdline_settings(3, argv));
    ASSERT_TRUE(set.nIterations == iterations);
    ASSERT_TRUE(set.cudeEdge == boxSize);
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
