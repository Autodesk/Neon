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

#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>


#include "Neon/core/types/vec.h"

namespace Neon {

/**
 * Namespace for this tool
 */
namespace ioToHDF5ns {

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


namespace helpNs {

}  // namespace helpNs

template <typename intType_ta, typename real_tt = double>
void ioToHDF5(const ioToHDF5ns::UserFieldInformation<intType_ta, real_tt>&      fieldData /*!                             User data that defines the field */,
             ioToHDF5ns::UserFieldAccessMask<real_tt, intType_ta>                  mask /*!                                  Stores a mask for which indices in the field should be outputted*/,
             const std::string&                                                       filename /*!                              File name */,
             const Neon::Integer_3d<intType_ta>&                                      dim /*!                                   Dimension of the field */,
             [[maybe_unused]] double                                                  spacingScale = 1.0 /*! Spacing, i.e. size of a voxel */,
             const Neon::Integer_3d<intType_ta>&                                      origin = Neon::Integer_3d<intType_ta>(0, 0, 0) /*!      Origin  */,
             const Neon::Integer_3d<intType_ta>&                                      chunking = Neon::Integer_3d<intType_ta>(10, 10, 10) /*!      Chunking  */,
             [[maybe_unused]] int                                                     iterationId = -1)
{

    if (fieldData.m_cardinality != 1) {
        std::string msg = std::string("Too many components specified during attempt at creating HDF5 output. It currently only supports 1 component.");
        NeonException exception("ioToHDF5");
        exception << msg;
        NEON_THROW(exception);    
    }

    // create the dataset
    HighFive::File file(filename, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);
    HighFive::DataSetCreateProps props;
    props.add(HighFive::Chunking(std::vector<intType_ta>{chunking.x, chunking.y, chunking.z}));
    HighFive::DataSet dataset = file.createDataSet<real_tt>(filename, HighFive::DataSpace({dim.x, dim.y, dim.z}), props); 

    // write the values to the dataset
    for (int i = origin.x; i < origin.x + dim.x; ++i) {
        for (int j = origin.y; j < origin.y + dim.y; ++j) {
            for (int k = origin.z; k < origin.z + dim.z; ++k) {
                if (!mask(Neon::Integer_3d<intType_ta>(i, j, k)) {
                    dataset.select({i - origin.x, j - origin.y, k - origin.z}, {1, 1, 1}).write(fieldData.m_userFieldAccessGenericFunction(Neon::Integer_3d<intType_ta>(i, j, k), 0));
                })
            }
        }
    }

}
}  // namespace ioToHDF5ns

template <typename intType_ta = int, class real_tt = double>
struct ioToHDF5
{
    ioToHDF5(const std::string&                                                                  filename /*!                                                 File name */,
            const Neon::Integer_3d<intType_ta>&                                                  dim /*!                                                      IoDense dimension of the field */,
            const std::function<real_tt(const Neon::Integer_3d<intType_ta>&, int componentIdx)>& fun /*!                                                      Implicit defintion of the user field */,
            const nComponent_t                                                                   card /*!                                                     Field cardinality */,
            const double                                                                         scalingData = 1.0 /*!                                        Spacing, i.e. size of a voxel */,
            const Neon::Integer_3d<intType_ta>&                                                  origin = Neon::Integer_3d<intType_ta>(0, 0, 0) /*!           Minimum Corner && Origin  */,
            const Neon::Integer_3d<intType_ta>&                                                  chunking = Neon::Integer_3d<intType_ta>(10, 10, 10) /*1      Chunking size of the output file */,
            const Neon::ioToHDF5ns::UserFieldAccessMask<real_tt, intType_ta>                     mask = [](const Neon::index_3d& idx){return (idx.x == idx.x) ? true: false;}) /*! Used for sparce matrices; returns true for indices that should be included in the output */
        : m_filename(filename),
          m_dim(dim),
          m_scalingData(scalingData),
          m_origin(origin),
          m_field(ioToHDF5ns::UserFieldInformation<intType_ta, real_tt>(fun, card)),
          m_chunking(chunking),
          m_mask(mask)
    {
        std::ofstream out("metadata2");
        out << "dim: " << m_dim.x << " " << m_dim.y << " " << m_dim.z << std::endl;
    }

    virtual ~ioToHDF5()
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
        filename = filename + ".h5";
        ioToHDF5ns::ioToHDF5<intType_ta, real_tt>(m_field,
                                                m_mask,
                                                filename,
                                                m_dim,
                                                m_scalingData,
                                                m_origin,
                                                m_chunking,
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
    std::string                                                       m_filename /*!                                             File name */;
    Neon::Integer_3d<intType_ta>                                      m_dim /*!                                                  IoDense dimension of the field */;
    double                                                            m_scalingData = 1.0 /*!                                    Spacing, i.e. size of a voxel */;
    Neon::Integer_3d<intType_ta>                                      m_origin = Neon::Integer_3d<intType_ta>(0, 0, 0) /*!       Origin  */;
    Neon::Integer_3d<intType_ta>                                      m_chunking = Neon::Integer_3d<intType_ta>(10, 10, 10) /*!  Chunking  */;
    ioToHDF5ns::UserFieldInformation<intType_ta, real_tt>             m_field /*!                                                Field data*/;
    ioToHDF5ns::UserFieldAccessMask<real_tt, intType_ta>              m_mask;
    int                                                               m_iteration = -1;
};

}  // namespace Neon
