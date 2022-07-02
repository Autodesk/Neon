#include "gtest/gtest.h"

#include "Neon/core/core.h"

#include "Neon/domain/tools/Geometries.h"
#include "Neon/domain/tools/IODomain.h"

auto IoDomainTest(Neon::index_3d&              dimension,
                  Neon::domain::tool::Geometry geometryName) -> void
{
    Neon::domain::tool::GeometryMask geometryMask(geometryName, dimension);
    auto                             denseMask = geometryMask.getIODenseMask();
    auto                             geometryStr = Neon::domain::tool::GeometryUtils::toString(geometryName);

    denseMask.ioVtk<int>(std::string("gUt_IODomain_mask") + "_" + geometryStr, "mask", Neon::ioToVTKns::VtiDataType_e::voxel);
    Neon::domain::tool::testing::IODomain<int> ioDomain(dimension, 1, denseMask);

    ioDomain.resetValuesToLinear(10);

    ioDomain.ioToVti("gUt_IODomain_voxel", "fieldTest", Neon::ioToVTKns::VtiDataType_e::voxel);
    ioDomain.ioToVti("gUt_IODomain_node", "fieldTest", Neon::ioToVTKns::VtiDataType_e::node);

    bool isInside = true;
    auto value = ioDomain.getValue(dimension * 100, 0, &isInside);

    ASSERT_FALSE(isInside);
    ASSERT_EQ(value, ioDomain.getOutsideValue());

    ioDomain.forEachActive([&](const Neon::index_3d& ids,
                               [[maybe_unused]] int  cardinality,
                               int&                  value) {
        {
            bool isValid = false;
            auto nghVal = ioDomain.nghVal(ids, Neon::int8_3d(0, 0, 0), 0, &isValid);
            ASSERT_TRUE(isValid);
            ASSERT_EQ(value, nghVal);
        }
        {
            bool isValid = true;
            auto nghVal = ioDomain.nghVal(dimension * 100, Neon::int8_3d(0, 0, 0), 0, &isValid);
            ASSERT_FALSE(isValid);
            ASSERT_EQ(nghVal, ioDomain.getOutsideValue());
        }
    });
}

TEST(gUt_tools_IODomain, HollowSphere)
{
    Neon::index_3d dimension(10, 20, 10);
    IoDomainTest(dimension, Neon::domain::tool::Geometry::HollowSphere);
}

TEST(gUt_tools_IODomain, HollowCube)
{
    Neon::index_3d dimension(10, 20, 10);
    IoDomainTest(dimension, Neon::domain::tool::Geometry::HollowCube);
}

TEST(gUt_tools_IODomain, LowerBottomLeft)
{
    Neon::index_3d dimension(10, 20, 10);
    IoDomainTest(dimension, Neon::domain::tool::Geometry::LowerBottomLeft);
}
