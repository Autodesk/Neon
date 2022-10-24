#pragma once
namespace {
struct TestInformation
{
    static auto prefix()
        -> std::string
    {
        return "domain-unit-test-staggered-grid";
    }

    static auto fullName(const std::string& gridName,
                         const std::string& subTestName)
        -> std::string
    {
        return prefix() + "-" +subTestName+"-"+ gridName;
    }
};
}  // namespace