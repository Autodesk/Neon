#pragma once
namespace {
struct TestInformation
{
    static auto prefix()
        -> std::string
    {
        return "domain-unit-test-patterns-containers";
    }

    static auto fullName(const std::string& gridName)
        -> std::string
    {
        return prefix() + "-" + gridName;
    }
};
}  // namespace