#pragma once
#include <string>
#include "Neon/core/core.h"
#include "Neon/core/tools/io/IODense.h"
#include "Neon/domain/tools/IODomain.h"
#include "Neon/Report.h"

namespace Neon::domain::tool {

/**
 * An Enum to define different geometries
 */
enum class Geometry
{
    FullDomain = 0,      /** The entire bounding box is part of the geometry */
    LowerBottomLeft = 1, /** The lover left bottom part of the bounding box is part of the geometry. Equivalent to a pyramid.*/
    Cube = 2,            /** A a cube inside the bounding box */
    HollowCube = 3,      /** A cube inside the bounding box, where a centered smaller cube has been removed. */
    Sphere = 4,          /** A sphere */
    HollowSphere = 5,    /** A hollow sphere */
};

class GeometryUtils
{
   public:
    static constexpr int nOptions = static_cast<int>(Geometry::HollowSphere) + 1;

    static auto toString(Geometry name) -> std::string;
    static auto toInt(Geometry name) -> int;
    static auto fromString(const std::string&) -> Geometry;
    static auto fromInt(int) -> Geometry;
    static auto getOptions() -> std::array<Geometry, nOptions>;

    struct Cli
    {
        explicit Cli(std::string);
        explicit Cli(Geometry model);
        Cli();

        auto getOption() -> Geometry;
        auto getStringOptions() -> std::string;
        auto set(const std::string& opt) -> void;

        auto addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) -> void;
        auto addToReport(Neon::Report& report) -> void;

       private:
        bool     mSet = false;
        Geometry mOption;
    };
};

struct GeometryMask
{

   private:
    Geometry        mGeoName;
    Neon::index_3d  mGridDimension;
    Neon::double_3d mDimension;
    Neon::double_3d mCenter;

    double          mDomainRatio /** [User] 0 -> no domain, 1 full domain */;
    double          mHollowRatio /** [User] 0 -> no hollow, 1 Max hollow */;
    Neon::double_3d mHollowRadia /** [Computed] Radia (on x,y,z) of the hollow  */;
    Neon::double_3d mDomainRadia /** [Computed] Radia (on x,y,z) of the domain */;

    std::function<bool(const index_3d& idx)> mFun;


   public:
    GeometryMask(Geometry       geo,
                  Neon::index_3d gridDimension,
                  double         domainRatio = .8,
                  double         hollowRatio = .5);

    auto getMaskFunction() const
        -> std::function<bool(const index_3d& idx)>;
    /**
     * Function that specifies if a voxel is inside of outside the geometry
     * @param idx
     * @return
     */
    auto operator()(const index_3d& idx) -> bool;

    auto getIODenseMask()
        -> IODense<uint8_t, int>;
};

}  // namespace Neon::domain::tool