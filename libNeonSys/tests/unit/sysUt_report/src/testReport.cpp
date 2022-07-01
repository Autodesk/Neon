#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/Report.h"


int    argc_;
char** argv_;

TEST(sys, report)
{
    Neon::Report report("Testing Report");
    report.setToken("Token404");
    report.commandLine(argc_, argv_);

    int32_t i = 0;
    report.addMember("my_int32", i);

    uint32_t ui = 1;
    report.addMember("my_uint32", ui);

    double d = 2;
    report.addMember("my_double", d);

    float f = 3;
    report.addMember("my_float", f);

    bool b = true;
    report.addMember("my_bool", b);

    std::string s = "sss";
    report.addMember("my_string", s);


    std::vector<int32_t> v_i{0, 1, 2, 3, 4};
    report.addMember("my_int32_vector", v_i);

    std::vector<uint32_t> v_ui{0, 1, 2, 3, 4};
    report.addMember("my_uint32_vector", v_ui);

    std::vector<double> v_d{0, 1, 2, 3, 4};
    report.addMember("my_double_vector", v_d);

    std::vector<float> v_f{0, 1, 2, 3, 4};
    report.addMember("my_float_vector", v_f);

    std::vector<bool> v_b{true, false, true, false, true};
    report.addMember("my_bool_vector", v_b);


    auto subdoc = report.getSubdoc();
    report.addMember("int32_subdoc", int32_t(55), &subdoc);
    report.addMember("double_subdoc", double(77), &subdoc);
    report.addMember("str_subdoc", std::string("sub"), &subdoc);
    report.addSubdoc("my_subdoc", subdoc);


    report.write("ReportUnitTest", false);
}

int main(int argc, char** argv)
{
    argc_ = argc;
    argv_ = argv;

    ::testing::InitGoogleTest(&argc, argv);

    Neon::init();
    return RUN_ALL_TESTS();
}