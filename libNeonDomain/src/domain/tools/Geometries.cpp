#include "Neon/domain/tools/Geometries.h"
#include "Neon/core/core.h"

namespace Neon::domain::tool {


// Removing unreachable code warning for windows (4702)
// Fixing the warning in VS, creates issue in linux
#if defined(NEON_COMPILER_VS)
#pragma warning(push)
#pragma warning(disable : 4702)
#endif
GeometryMask::GeometryMask(Geometry       geo,
                           Neon::index_3d gridDimension,
                           double         domainRatio,
                           double         hollowRatio)
{
    mGeoName = geo;

    mGridDimension = gridDimension;
    mDimension = gridDimension.newType<double>();
    mCenter = mDimension.newType<double>() / 2;

    mDomainRatio = domainRatio;
    mHollowRatio = hollowRatio;
    mDomainRadia = (mDimension / 2) * mDomainRatio;
    mHollowRadia = mDomainRadia * mHollowRatio;

    switch (mGeoName) {
        case Geometry::FullDomain: {
            mFun = [](const index_3d& idx) -> bool {
                (void)idx;
                return true;
            };
            return;
        }
        case Geometry::Cube: {
            mFun = [&](const index_3d& idx) -> bool {
                bool isOk = true;
                auto t = idx.newType<double>();

                for (int direction = 0; direction < index_3d::num_axis; direction++) {
                    bool newTest = std::abs(mCenter.v[direction] - t.v[direction]) <= mDomainRadia.v[direction];
                    isOk = isOk && newTest;
                }
                return isOk;
            };
            return;
        }

        case Geometry::HollowCube: {
            mFun = [&](const index_3d& idx) -> bool {
                auto t = idx.newType<double>();
                bool testA;
                {
                    bool isOk = true;

                    for (int direction = 0; direction < index_3d::num_axis; direction++) {
                        bool newTest = std::abs(mCenter.v[direction] - t.v[direction]) <= mDomainRadia.v[direction];
                        isOk = isOk && newTest;
                    }
                    testA = isOk;
                }
                bool testB;
                {
                    bool isOk = false;

                    for (int direction = 0; direction < index_3d::num_axis; direction++) {
                        bool newTest = std::abs(mCenter.v[direction] - t.v[direction]) >= mHollowRadia.v[direction];
                        isOk = isOk || newTest;
                    }
                    testB = isOk;
                }
                return testA && testB;
            };
            return;
        }
        case Geometry::LowerBottomLeft: {
            mFun = [&](const index_3d& idx) -> bool {
                auto   t = idx.newType<double>();
                double a = 0;
                double b = 0;
                for (int direction = 0; direction < index_3d::num_axis; direction++) {
                    a += mCenter.v[direction];
                    b += t.v[direction] + idx.v[direction];
                }
                return b <= a;
            };
            return;
        }

        case Geometry::Sphere: {
            mFun = [&](const index_3d& idx) -> bool {
                auto t = idx.newType<double>();

                double sum = 0;
                for (int direction = 0; direction < index_3d::num_axis; direction++) {
                    double tmp = (t.v[direction] - mCenter.v[direction]) / mDomainRadia.v[direction];
                    sum += std::pow(tmp, 2);
                }
                // The norm of geoSize is the diameter of the sphere
                // radius^2 = (diameter^2 / 2^2)
                return sum <= 1;
            };
            return;
        }
        case Geometry::HollowSphere: {
            mFun = [&](const index_3d& idx) -> bool {
                auto t = idx.newType<double>();
                bool testA;
                {
                    double sum = 0;
                    for (int direction = 0; direction < index_3d::num_axis; direction++) {
                        double tmp = (t.v[direction] - mCenter.v[direction]) / mDomainRadia.v[direction];
                        sum += std::pow(tmp, 2);
                    }
                    testA = (sum <= 1.0);
                }
                bool testB;
                {
                    double sum = 0;
                    for (int direction = 0; direction < index_3d::num_axis; direction++) {
                        double tmp = (t.v[direction] - mCenter.v[direction]) / mHollowRadia.v[direction];
                        sum += std::pow(tmp, 2);
                    }
                    // geoSize is the diameter of the sphere
                    testB = sum >= 1.0;
                }
                return testA && testB;
            };
            return;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("GeoActiveMask_t");
        }
    }
}
#if defined(NEON_COMPILER_VS)
#pragma warning(pop)
#endif


bool GeometryMask::operator()(const index_3d& idx)
{
    return mFun(idx);
}

auto GeometryUtils::toString(Neon::domain::tool::Geometry name) -> std::string
{
    switch (name) {
        case Neon::domain::tool::Geometry::FullDomain: {
            return "FullDomain";
        }
        case Neon::domain::tool::Geometry::LowerBottomLeft: {
            return "LowerLeft";
        }
        case Neon::domain::tool::Geometry::Cube: {
            return "Cube";
        }
        case Neon::domain::tool::Geometry::HollowCube: {
            return "HollowCube";
        }
        case Neon::domain::tool::Geometry::Sphere: {
            return "Sphere";
        }
        case Neon::domain::tool::Geometry::HollowSphere: {
            return "HollowSphere";
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION();
        }
    }
}

auto GeometryUtils::toInt(Neon::domain::tool::Geometry name)
    -> int
{
    return static_cast<int>(name);
}

auto GeometryUtils::fromString(const std::string& name)
    -> Neon::domain::tool::Geometry
{
#define ADD_OPTION(NAME)                           \
    if (name == #NAME) {                           \
        return Neon::domain::tool::Geometry::NAME; \
    }
    ADD_OPTION(FullDomain);
    ADD_OPTION(LowerBottomLeft);
    ADD_OPTION(Cube);
    ADD_OPTION(HollowCube);
    ADD_OPTION(Sphere);
    ADD_OPTION(HollowSphere);
    NEON_THROW_UNSUPPORTED_OPTION("GeometryNamesUtils");
#undef ADD_OPTION
}

auto GeometryUtils::fromInt(int id)
    -> Neon::domain::tool::Geometry
{
#define ADD_OPTION(NAME)                                                                      \
    if (id == Neon::domain::tool::GeometryUtils::toInt(Neon::domain::tool::Geometry::NAME)) { \
        return Neon::domain::tool::Geometry::NAME;                                            \
    }
    ADD_OPTION(FullDomain);
    ADD_OPTION(LowerBottomLeft);
    ADD_OPTION(Cube);
    ADD_OPTION(HollowCube);
    ADD_OPTION(Sphere);
    ADD_OPTION(HollowSphere);
    NEON_THROW_UNSUPPORTED_OPTION("GeometryNamesUtils");
#undef ADD_OPTION
}
auto GeometryUtils::getOptions() -> std::array<Geometry, nOptions>
{
    std::array<Geometry, nOptions> options{Geometry::FullDomain,
                                           Geometry::LowerBottomLeft,
                                           Geometry::Cube,
                                           Geometry::HollowCube,
                                           Geometry::Sphere,
                                           Geometry::HollowSphere};
    return options;
}

auto GeometryMask::getIODenseMask()
    -> IODense<uint8_t, int>
{
    using IODomain = Neon::domain::tool::testing::IODomain<uint8_t, int>;
    IODense<uint8_t, int> mask(mGridDimension, 1);
    auto                  fun = mFun;
    mask.forEach([&](const Neon::index_3d&        idx,
                     [[maybe_unused]] int         c,
                     typename IODomain::FlagType& val) {
        bool isIn = fun(idx);
        val = isIn ? IODomain::InsideFlag
                   : IODomain::OutsideFlag;
    });

    return mask;
}

auto GeometryMask::getMaskFunction() const
    -> std::function<bool(const index_3d& idx)>
{
    return mFun;
}


GeometryUtils::Cli::Cli(std::string str)
{
    set(str);
}

auto GeometryUtils::Cli::getOption() -> Geometry
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "Geometry model was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return mOption;
}

GeometryUtils::Cli::Cli()
{
    mSet = false;
}
GeometryUtils::Cli::Cli(Geometry model)
{
    mSet = true;
    mOption = model;
}

auto GeometryUtils::Cli::set(const std::string& opt)
    -> void
{
    try {
        mOption = GeometryUtils::fromString(opt);
    } catch (...) {
        std::stringstream errorMsg;
        errorMsg << "ForkJoin: " << opt << " is not a valid option (valid options are {";
        auto options = GeometryUtils::getOptions();
        int  i = 0;
        for (auto o : options) {
            if (i != 0) {
                errorMsg << ", " << GeometryUtils::toString(o);
            }
            errorMsg << GeometryUtils::toString(o);
            i = 1;
        }
        errorMsg << "})";
        NEON_ERROR(errorMsg.str());
    }
    mSet = true;
}
auto GeometryUtils::Cli::getStringOptions() -> std::string
{
    std::stringstream s;
    auto              options = GeometryUtils::getOptions();
    int               i = 0;
    for (auto o : options) {
        if (i != 0) {
            s << ", ";
        }
        s << GeometryUtils::toString(o);
        i = 1;
    }
    std::string msg = s.str();
    return msg;
}
auto GeometryUtils::Cli::addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) -> void
{
    report.addMember("Geometry", GeometryUtils::toString(this->getOption()), &subBlock);
}

auto GeometryUtils::Cli::addToReport(Neon::Report& report) -> void
{
    report.addMember("Geometry", GeometryUtils::toString(this->getOption()));
}

}  // namespace Neon::domain::tool