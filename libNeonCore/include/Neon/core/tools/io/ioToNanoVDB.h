#pragma once

#include <algorithm>
#include <cfloat>
#include <cinttypes>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <variant>
#include <regex>
#include <sstream>
#include <streambuf>
#include <string>
#include <typeinfo>
#include <vector>
#include "cuda_fp16.h"
#include <iomanip>


#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>


#include "Neon/core/types/vec.h"

namespace Neon {

/**
 * Namespace for this tool
 */
namespace ioToNanoVDBns {

/*
 * nanovdb's 3-component vector
 */
template <class real_tt>
using Vec3 = nanovdb::math::Vec3<real_tt>;

/*
 * nanovdb's 4-component vector
 */
template <class real_tt>
using Vec4 = nanovdb::math::Vec4<real_tt>;

/**
 * Implicit function that defines the data stores by a user fields
 */
template <class real_tt, typename intType_ta>
using UserFieldAccessGenericFunction_t = std::function<real_tt(const Neon::Integer_3d<intType_ta>&, int componentIdx)>;

/**
 * Implicit function that takes in an index, and returns a boolean for if that index is a valid index in the field or not.
 */
template <class real_tt, typename intType_ta>
using UserFieldAccessMask = std::function<bool(const Neon::Integer_3d<intType_ta>&)>;

/**
 * Number of components of the field
 */
using nComponent_t = int;

/**
 * Name of the file where the field will be exported into
 */
using FieldName_t = std::string;

template <class intType_ta, typename real_tt>
struct UserFieldInformation
{
    UserFieldAccessGenericFunction_t<real_tt, intType_ta> m_userFieldAccessGenericFunction;
    nComponent_t                                          m_cardinality;
    
    UserFieldInformation(const UserFieldAccessGenericFunction_t<real_tt, intType_ta>& fun, nComponent_t card)
        : m_userFieldAccessGenericFunction(fun),
          m_cardinality(card)
    {
    }

    UserFieldInformation(const std::tuple<UserFieldAccessGenericFunction_t<real_tt, intType_ta>, nComponent_t>& tuple)
        : m_userFieldAccessGenericFunction(std::get<0>(tuple)),
          m_cardinality(std::get<1>(tuple))
    {
    }
};

/*
 * Alias for NanoVDB grid types
 */
template <typename real_tt>
using Grid1Component = std::shared_ptr<nanovdb::tools::build::Grid<real_tt>>;

template <typename real_tt>
using Grid3Components = std::shared_ptr<nanovdb::tools::build::Grid<Vec3<real_tt>>>;

template <typename real_tt>
using Grid4Components = std::shared_ptr<nanovdb::tools::build::Grid<Vec4<real_tt>>>;

// The possible kinds of grids (containing 1, 3, or 4 components) that can be used
template<typename real_tt>
using GridPtrVariant = std::variant<
    std::shared_ptr<nanovdb::tools::build::Grid<real_tt>>,
    std::shared_ptr<nanovdb::tools::build::Grid<Vec3<real_tt>>>,
    std::shared_ptr<nanovdb::tools::build::Grid<Vec4<real_tt>>>
>;

namespace helpNs {

template<typename real_tt>
struct BackgroundValueVisitor {
    real_tt operator()(const typename Neon::ioToNanoVDBns::Grid1Component<real_tt>) {
        return std::numeric_limits<real_tt>::min();
    }
    Vec3<real_tt> operator()(const typename Neon::ioToNanoVDBns::Grid3Components<real_tt>) {
        return Vec3<real_tt>(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());
    }
    Vec4<real_tt> operator()(const typename Neon::ioToNanoVDBns::Grid4Components<real_tt>) {
        return Vec4<real_tt>(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());
    }
};

template<typename real_tt, typename intType_ta>
struct GetDataVisitor {
    real_tt operator()(const typename Neon::ioToNanoVDBns::Grid1Component<real_tt>, Neon::Integer_3d<intType_ta> xyz, const ioToNanoVDBns::UserFieldAccessGenericFunction_t<real_tt, intType_ta>& fieldData) const {
        return fieldData(xyz, 0);
    }
    Vec3<real_tt> operator()(const typename Neon::ioToNanoVDBns::Grid3Components<real_tt>, Neon::Integer_3d<intType_ta> xyz, const ioToNanoVDBns::UserFieldAccessGenericFunction_t<real_tt, intType_ta>& fieldData) const {
        return Vec3<real_tt>(fieldData(xyz, 0), fieldData(xyz, 1), fieldData(xyz, 2));
    }
    Vec4<real_tt> operator()(const typename Neon::ioToNanoVDBns::Grid4Components<real_tt>, Neon::Integer_3d<intType_ta> xyz, const ioToNanoVDBns::UserFieldAccessGenericFunction_t<real_tt, intType_ta>& fieldData) const {
        return Vec4<real_tt>(fieldData(xyz, 0), fieldData(xyz, 1), fieldData(xyz, 2), fieldData(xyz, 3));
    }
};

template<typename real_tt, typename intType_ta>
struct BuildGridVisitor {
    real_tt operator()(const typename Neon::ioToNanoVDBns::Grid1Component<real_tt> grid, Neon::Integer_3d<intType_ta> xyz, const ioToNanoVDBns::UserFieldAccessGenericFunction_t<real_tt, intType_ta>& fieldData) const {
        return fieldData(xyz, 0);
    }
    Vec3<real_tt> operator()(const typename Neon::ioToNanoVDBns::Grid3Components<real_tt> grid, Neon::Integer_3d<intType_ta> xyz, const ioToNanoVDBns::UserFieldAccessGenericFunction_t<real_tt, intType_ta>& fieldData) const {
        return Vec3<real_tt>(fieldData(xyz, 0), fieldData(xyz, 1), fieldData(xyz, 2));
    }
    Vec4<real_tt> operator()(const typename Neon::ioToNanoVDBns::Grid4Components<real_tt> grid, Neon::Integer_3d<intType_ta> xyz, const ioToNanoVDBns::UserFieldAccessGenericFunction_t<real_tt, intType_ta>& fieldData) const {
        return Vec4<real_tt>(fieldData(xyz, 0), fieldData(xyz, 1), fieldData(xyz, 2), fieldData(xyz, 3));
    }
};


}  // namespace helpNs

template <typename intType_ta, typename real_tt = double>
void ioToNanoVDB(const ioToNanoVDBns::UserFieldInformation<intType_ta, real_tt>&      fieldData /*!                             User data that defines the field */,
             ioToNanoVDBns::UserFieldAccessMask<real_tt, intType_ta>                  mask /*!                                  Stores a mask for which indices in the field should be outputted*/,
             const std::string&                                                       filename /*!                              File name */,
             const Neon::Integer_3d<intType_ta>&                                      dim /*!                                   Dimension of the field */,
             double                                                                   spacingScale = 1.0 /*! Spacing, i.e. size of a voxel */, /* TODOMATT: add spacing data transformation */
             const Neon::Integer_3d<intType_ta>&                                      origin = Neon::Integer_3d<intType_ta>(0, 0, 0) /*!      Origin  */,
             [[maybe_unused]] int                                                     iterationId = -1)
{
    // Create our grid
    Neon::ioToNanoVDBns::GridPtrVariant<real_tt> outputGrid;
    
    // Based on the cardinality, use a scalar, Vec3, or Vec4
    switch (fieldData.m_cardinality) {
        case 1:
            outputGrid = std::make_shared<nanovdb::tools::build::Grid<real_tt>>(
                std::numeric_limits<real_tt>::min(), filename, nanovdb::GridClass::LevelSet);
            break;
        case 3:
            outputGrid = std::make_shared<nanovdb::tools::build::Grid<Vec3<real_tt>>>(
                Vec3<real_tt>(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min()), filename, nanovdb::GridClass::LevelSet);
            break;
        case 4:
            outputGrid = std::make_shared<nanovdb::tools::build::Grid<Vec4<real_tt>>>(
                Vec4<real_tt>(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min()), filename, nanovdb::GridClass::LevelSet);
            break;
        default:            std::string msg = std::string("Too many components specified during attempt at creating nanoVDB output");
            NeonException exception("ioToNanoVDB");
            exception << msg;
            NEON_THROW(exception);
    }

    // through experimentation, it seems that both coordinates are inclusive. So, the upper corner should be subtracted by one.
    const int bboxOffset = -1;
    nanovdb::CoordBBox bbox(nanovdb::Coord(origin.x, origin.y, origin.z), nanovdb::Coord(origin.x + dim.x + bboxOffset, origin.y + dim.y + bboxOffset, origin.z + dim.z + bboxOffset));

    // Write the data to the nanoVDB grid
    std::visit([&](auto& grid) {

        // Set the voxel scale:
        grid->setTransform(spacingScale);
 
        // Use a nanoVDB functor to set the grid values
        (*grid)([&](const nanovdb::Coord &ijk) {
            const Neon::Integer_3d<intType_ta> xyz(ijk[0], ijk[1], ijk[2]);

            // If the mask shows that the current coordinate is empty, then we will set it with the background value.
            // This background value makes this index unset (to accomodate sparse grids)
            if (!mask(xyz)) {
                return Neon::ioToNanoVDBns::helpNs::BackgroundValueVisitor<real_tt>{}(grid);
            }

            return Neon::ioToNanoVDBns::helpNs::GetDataVisitor<real_tt, intType_ta>{}(grid, xyz, fieldData.m_userFieldAccessGenericFunction);
        }, bbox);

    }, outputGrid);


    try {
        // Write the grid out to a file
        std::visit([&](auto& grid) {
            nanovdb::io::writeGrid(filename, nanovdb::tools::createNanoGrid(*grid), nanovdb::io::Codec::NONE);
        }, outputGrid); 

    // catch possible file IO errors    
    } catch (...) {
        std::string msg = std::string("An error on file operations where encountered when writing nanoVDB data");
        NeonException exception("ioToNanoVDB output exception");
        exception << msg;
        NEON_THROW(exception);
    }
}
}  // namespace ioToNanoVDBns

template <typename intType_ta = int, class real_tt = double>
struct ioToNanoVDB
{
    ioToNanoVDB(const std::string&                                                               filename /*!                                                 File name */,
            const Neon::Integer_3d<intType_ta>&                                                  dim /*!                                                      IoDense dimension of the field */,
            const std::function<real_tt(const Neon::Integer_3d<intType_ta>&, int componentIdx)>& fun /*!                                                      Implicit defintion of the user field */,
            const nComponent_t                                                                   card /*!                                                     Field cardinality */,
            const double                                                                         scalingData = 1.0 /*!                                        Spacing, i.e. size of a voxel */,
            const Neon::Integer_3d<intType_ta>&                                                  origin = Neon::Integer_3d<intType_ta>(0, 0, 0) /*!                         Minimum Corner && Origin  */,
            const Neon::ioToNanoVDBns::UserFieldAccessMask<real_tt, intType_ta>                  mask = [](const Neon::index_3d& idx){return (idx.x == idx.x) ? true: false;}) /*! Used for sparce matrices; returns true for indices that should be stored in vdb output */
        : m_filename(filename),
          m_dim(dim),
          m_scalingData(scalingData),
          m_origin(origin),
          m_field(ioToNanoVDBns::UserFieldInformation<intType_ta, real_tt>(fun, card)),
          m_mask(mask)
    {
        std::ofstream out("metadata2");
        out << "dim: " << m_dim.x << " " << m_dim.y << " " << m_dim.z << std::endl;
    }

    virtual ~ioToNanoVDB()
    {
    }

    auto flush() -> void
    {
        std::string filename;
        if (m_iteration == -1) {
            filename = m_filename;
        } else {
            std::stringstream ss;
            ss << std::setw(5) << std::setfill('0') << m_iteration;
            std::string s = ss.str();
            filename = m_filename + s;
        }
        filename = filename + ".nvdb";
        ioToNanoVDBns::ioToNanoVDB<intType_ta, real_tt>(m_field,
                                                m_mask,
                                                filename,
                                                m_dim,
                                                m_scalingData,
                                                m_origin,
                                                m_iteration);
    }


    auto setIteration(int iteration)
    {
        m_iteration = iteration;
    }

    auto setFileName(const std::string& fname)
    {
        m_filename = fname;
    }


   private:
    std::string                                                       m_filename /*!                              File name */;
    Neon::Integer_3d<intType_ta>                                      m_dim /*!                                   IoDense dimension of the field */;
    double                                                            m_scalingData = 1.0 /*!                     Spacing, i.e. size of a voxel */;
    Neon::Integer_3d<intType_ta>                                      m_origin = Neon::Integer_3d<intType_ta>(0, 0, 0) /*!      Origin  */;
    ioToNanoVDBns::UserFieldInformation<intType_ta, real_tt>          m_field /*!                                 Field data*/;
    ioToNanoVDBns::UserFieldAccessMask<real_tt, intType_ta>           m_mask;
    int                                                               m_iteration = -1;
};

}  // namespace Neon
