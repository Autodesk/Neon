#pragma once

#include <algorithm>
#include <cfloat>
#include <cinttypes>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <regex>
#include <sstream>
#include <streambuf>
#include <string>
#include <typeinfo>
#include <vector>
#include "cuda_fp16.h"

#include <iomanip>
#include "Neon/core/tools/io/ioToVti.h"
#include "Neon/core/types/vec.h"

namespace Neon {

/**
 * Namespace for a legacy type of VTI tool
 */
namespace ioToVTKns {
/**
 * Implicit function that defines the data stores by a user fields
 */
template <class real_tt, typename intType_ta>
using UserFieldAccessGenericFunction_t = std::function<real_tt(const Neon::Integer_3d<intType_ta>&, int componentIdx)>;

/**
 * Number of components of the field
 */
using nComponent_t = int;

/**
 * Name of the file where the field will be exported into
 */
using FieldName_t = std::string;

/**
 * Type of data. node or voxel
 */
enum VtiDataType_e
{
    node,
    voxel
};

/**
 * Tuple to store all information associated to a field
 */
// template <class real_tt, typename intType_ta>
// using UserFieldInformation = std::tuple<UserFieldAccessGenericFunction_t<real_tt, intType_ta>, nComponent_t, FieldName_t, VtiDataType_e>;

template <class intType_ta, typename real_tt>
struct UserFieldInformation
{
    UserFieldInformation(const UserFieldAccessGenericFunction_t<real_tt, intType_ta>& fun, nComponent_t card, const FieldName_t& fname, VtiDataType_e vtiType)
        : m_userFieldAccessGenericFunction(fun),
          m_cardinality(card),
          m_fieldName(fname),
          m_vtiDataType(vtiType)
    {
    }

    UserFieldInformation(const std::tuple<UserFieldAccessGenericFunction_t<real_tt, intType_ta>, nComponent_t, FieldName_t, VtiDataType_e>& tuple)
        : m_userFieldAccessGenericFunction(std::get<0>(tuple)),
          m_cardinality(std::get<1>(tuple)),
          m_fieldName(std::get<2>(tuple)),
          m_vtiDataType(std::get<3>(tuple))
    {
    }


    UserFieldAccessGenericFunction_t<real_tt, intType_ta> m_userFieldAccessGenericFunction;
    nComponent_t                                          m_cardinality;
    FieldName_t                                           m_fieldName;
    VtiDataType_e                                         m_vtiDataType;
};


namespace helpNs {
template <typename T>
void SwapEnd(T& var)
{
    char* varArray = reinterpret_cast<char*>(&var);
    for (long i = 0; i < static_cast<long>(sizeof(var) / 2); i++)
        std::swap(varArray[sizeof(var) - 1 - i], varArray[i]);
}
namespace numerical_chars {
inline std::ostream& operator<<(std::ostream& os, char c)
{
    return os << (std::is_signed<char>::value ? static_cast<int>(c)
                                              : static_cast<unsigned int>(c));
}

inline std::ostream& operator<<(std::ostream& os, signed char c)
{
    return os << static_cast<int>(c);
}

inline std::ostream& operator<<(std::ostream& os, unsigned char c)
{
    return os << static_cast<unsigned int>(c);
}
}  // namespace numerical_chars

/**
 * Dump data in a binary format into a file
 * @tparam real_tt
 * @tparam intType_ta
 * @param out: open file
 * @param fieldData: user field data defined as an implicit function
 * @param nComponents: number of components
 * @param space: dimension of the grid
 */
template <typename intType_ta, typename real_tt>
void dumpRawDataIntoFile(std::ofstream&                                                   b_stream,
                         ioToVTKns::UserFieldAccessGenericFunction_t<real_tt, intType_ta> grid,
                         ioToVTKns::nComponent_t                                          nComponents,
                         const Integer_3d<intType_ta>&                                    space)
{
#if 0
    //TODO
    // Put back online the following version that adds buffering.
    const int                    BUF_SIZE = 1024*2;
    real_tt                      raw[BUF_SIZE];
    size_t                       counter = 0;
    Neon::Integer_3d<intType_ta> idx;

    for (intType_ta z = 0; z < space.z; z++) {
        for (intType_ta y = 0; y < space.y; y++) {
            for (intType_ta x = 0; x < space.x; x++) {
                idx.set(x, y, z);
                for (int v = 0; v < nComponents; v++) {

                    real_tt val = (grid(idx, v));
                    SwapEnd(val);
                    size_t     jump = size_t(v) +
                                  size_t(x) * nComponents +
                                  size_t(y) * nComponents * space.x +
                                  size_t(z) * nComponents * space.x * space.y;
                    raw[jump] = val;

                    counter++;
                    if (counter == BUF_SIZE) {
                        b_stream.write((char*)raw, sizeof(real_tt) * counter);
                        counter = 0;
                    }
                }
            }
        }
    }
    if (counter != 0) {
        b_stream.write((char*)raw, sizeof(real_tt) * counter);
        counter = 0;
    }
    b_stream << "\n";
#else
    Neon::Integer_3d<intType_ta> idx;

    for (intType_ta z = 0; z < space.z; z++) {
        for (intType_ta y = 0; y < space.y; y++) {
            for (intType_ta x = 0; x < space.x; x++) {
                idx.set(x, y, z);
                for (int v = 0; v < nComponents; v++) {

                    real_tt val = grid(idx, v);
                    SwapEnd(val);
                    b_stream.write((char*)&val, sizeof(real_tt) * 1);
                }
            }
        }
    }
    b_stream << "\n";
#endif
}

/**
 * Dump data in a ascii format into a file
 * @tparam real_tt
 * @tparam intType_ta
 * @param out: open file
 * @param fieldData: user field data defined as an implicit function
 * @param nComponents: number of components
 * @param space: dimension of the grid
 */
template <typename intType_ta, typename real_tt>
void dumpTextDataIntoFile(std::ofstream&                                                   out,
                          ioToVTKns::UserFieldAccessGenericFunction_t<real_tt, intType_ta> fieldData,
                          ioToVTKns::nComponent_t                                          nComponents,
                          const Integer_3d<intType_ta>&                                    space)
{
    Neon::Integer_3d<intType_ta> idx;
    for (intType_ta z = 0; z < space.z; z++) {
        for (intType_ta y = 0; y < space.y; y++) {
            for (intType_ta x = 0; x < space.x; x++) {
                idx.set(x, y, z);
                for (int v = 0; v < nComponents; v++) {
                    if constexpr (std::is_same_v<real_tt, __half>) {
                        float val = __half2float(fieldData(idx, v));
                        out << val << " ";
                    } else if constexpr (!std::is_same_v<real_tt, __half>) {
                        real_tt val = static_cast<real_tt>(fieldData(idx, v));
                        out << val << " ";
                    }
                }
            }
            out << "\n";
        }
    }
}

/**
 * Write the data of a field into a file.
 *
 * @tparam real_tt
 * @tparam intType_ta
 * @param out: an open file
 * @param fieldData: user field data defined as an implicit function
 * @param nComponents: number of components
 * @param fieldName: name of the field
 * @param space: dimension of the grid
 * @param vtiIO: BINARY or ASCII
 */
template <typename intType_ta, typename real_tt>
void writeData(std::ofstream&                                                          out,
               const ioToVTKns::UserFieldAccessGenericFunction_t<real_tt, intType_ta>& fieldData,
               ioToVTKns::nComponent_t                                                 nComponents,
               const ioToVTKns::FieldName_t&                                           fieldName,
               const Integer_3d<intType_ta>&                                           space,
               IoFileType                                                              vtiIO)
{
    out << "SCALARS " << fieldName << " ";
    if constexpr (std::is_same<real_tt, double>::value) {
        out << "double ";
    } else if constexpr (std::is_same<real_tt, float>::value) {
        out << "float ";
    } else if constexpr (std::is_same<real_tt, int>::value || std::is_same<real_tt, uint32_t>::value) {
        out << "int ";
    } else if constexpr (std::is_same<real_tt, char>::value) {
        out << "short ";
    } else if constexpr (std::is_same<real_tt, __half>::value) {
        out << "half ";
    } else {
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
    if (nComponents != 1) {
        out << " " << nComponents << "\n";
    } else {
        out << "\n";
    }
    out << "LOOKUP_TABLE default\n";

    if (vtiIO == IoFileType::ASCII) {
        dumpTextDataIntoFile<intType_ta, real_tt>(out, fieldData, nComponents, space);
    } else {
        dumpRawDataIntoFile<intType_ta, real_tt>(out, fieldData, nComponents, space);
    }
    out << "METADATA\n";
    out << "INFORMATION 0\n\n";
}

/**
 *
 * @tparam real_tt
 * @tparam intType_ta
 * @param out: an open file
 * @param fieldsData: vector with the field data
 * @param nodeSpace: dimension of the node grid
 * @param voxSpace: dimension of the voxel grid
 * @param vtiIOe: format of the output - ASCII or BINARY
 */
template <typename intType_ta, typename real_tt>
void WriteNodeAndVoxelData(std::ofstream&                                                           out,
                           const std::vector<ioToVTKns::UserFieldInformation<intType_ta, real_tt>>& fieldsData,
                           const Integer_3d<intType_ta>&                                            nodeSpace,
                           const Integer_3d<intType_ta>&                                            voxSpace,
                           IoFileType                                                               vtiIOe)
{
    ioToVTKns::VtiDataType_e filteringNodeOrVoxels;

    Integer_3d<intType_ta> space[2];
    space[ioToVTKns::node] = nodeSpace;
    space[ioToVTKns::voxel] = voxSpace;

    auto writeDataUnwrap = [&](const ioToVTKns::UserFieldInformation<intType_ta, real_tt>& t) -> void {
        if (t.m_vtiDataType == filteringNodeOrVoxels) {
            writeData<intType_ta, real_tt>(out,
                                           t.m_userFieldAccessGenericFunction,
                                           t.m_cardinality,
                                           t.m_fieldName,
                                           space[filteringNodeOrVoxels],
                                           vtiIOe);
        }
    };

    //    auto unpackAndWriteData = [&](auto&... args) {
    //        //Neon::meta::debug::printType(args...);
    //        return writeDataUnwrap(args...);
    //    };

    auto addDataFiltered = [&](ioToVTKns::VtiDataType_e nodeOrVox) {
        filteringNodeOrVoxels = nodeOrVox;
        for (int i = 0; i < int(fieldsData.size()); i++) {
            // Neon::meta::debug::printType(fieldsData[i]);
            // std::apply(writeDataUnwrap, fieldsData[i]);
            writeDataUnwrap(fieldsData[i]);
        }
    };

    out << "CELL_DATA " << size_t(voxSpace.x) * size_t(voxSpace.y) * size_t(voxSpace.z) << std::endl;
    // CELL DATA FIRST
    addDataFiltered(ioToVTKns::voxel);
    out << "POINT_DATA " << size_t(nodeSpace.x) * size_t(nodeSpace.y) * size_t(nodeSpace.z) << std::endl;
    // NODE DATA AFTER
    addDataFiltered(ioToVTKns::node);
}

/**
 * Given an already open files this function writes the field data into the file.
 *
 * @tparam real_tt: type of the field
 * @tparam intType_ta: type of the indexes for the grid
 * @param out: an open file
 * @param fieldsData: vector with the field data
 * @param nodeSpace: dimension of the node grid
 * @param voxSpace: dimension of the voxel grid
 * @param spacingData: grid spacing
 * @param origin: origin of the grid
 * @param vtiIOe: format of the output - ASCII or BINARY
 */
template <typename intType_ta, typename real_tt>
void WriteGridInfoAndAllFields(std::ofstream&                                                           out,
                               const std::vector<ioToVTKns::UserFieldInformation<intType_ta, real_tt>>& fieldsData,
                               const Integer_3d<intType_ta>&                                            nodeSpace,
                               const Integer_3d<intType_ta>&                                            voxSpace,
                               const Vec_3d<double>&                                                    spacingData,
                               const Vec_3d<double>&                                                    origin,
                               IoFileType                                                               vtiIOe)
{


    out << "DIMENSIONS " << nodeSpace.x << " " << nodeSpace.y << " " << nodeSpace.z << "\n";
    out << "ORIGIN " << origin.x << " " << origin.y << " " << origin.z << "\n";
    out << "SPACING " << spacingData.x << " " << spacingData.y << " " << spacingData.z << "\n";

    WriteNodeAndVoxelData<intType_ta, real_tt>(out,
                                               fieldsData,
                                               nodeSpace,
                                               voxSpace,
                                               vtiIOe);
}

}  // namespace helpNs

/**
 * Export a set of data fields to the legacy vtk file format.
 * It supports both binary and ascii format.
 *
 * Example:
 *
 * Suppose we have a dense 10x10x10 grid with a velocity field and density field.
 * Let v the function that given a position in the grid and a component id it returns the requested velocity component
 * Let d the function that given a position in the grid and a component id it returns the requested density component
 * In our case velocity has 3 component, density only one. v is defined on the nodes, d in the voxels

    Neon::index_3d voxDim(5, 5, 5);
    Neon::index_3d nodDim = voxDim + 1;

    auto velocityNorm = [&](const Neon::Integer_3d<int>& idx,
                            int vIdx ) -> double {
        return idx.x;
    };

    auto density = [&](const Neon::index_3d& idx, int vIdx) -> double {
        return -idx.x + idx.y - idx.z;
    };

    Neon::ioToVTILegacy({{velocityNorm, 1, "velocityNorm", Neon::ioToVTKLegacyNs::node},
                         {density, 1, "density", Neon::ioToVTKLegacyNs::voxel}},
                        "coreUt_vti_test_legacy_implicit_tuple_ASCII",
                        nodDim,
                        voxDim,
                        1.0,
                        0.0,
                        Neon::IoFileType::ASCII);

    Neon::ioToVTILegacy({{velocityNorm, 1, "velocityNorm", Neon::ioToVTKLegacyNs::node},
                         {density, 1, "density", Neon::ioToVTKLegacyNs::voxel}},
                        "coreUt_vti_test_legacy_implicit_tuple_BINARY",
                        nodDim,
                        voxDim,
                        1.0,
                        0.0,
                        Neon::IoFileType::BINARY);

 *
 *
 */
template <typename intType_ta, class real_tt = double>
void ioToVTK(const std::vector<ioToVTKns::UserFieldInformation<intType_ta, real_tt>>& fieldsData /*!                            User data that defines the field */,
             const std::string&                                                       filename /*!                              File name */,
             const Integer_3d<intType_ta>&                                            nodeSpace /*!                             IoDense dimension of the node space (nodeSpace = voxelSpace +1) */,
             const Vec_3d<double>&                                                    spacingData = Vec_3d<double>(1, 1, 1) /*! Spacing, i.e. size of a voxel */,
             const Vec_3d<double>&                                                    origin = Vec_3d<double>(0, 0, 0) /*!      Origin  */,
             IoFileType                                                               vtiIOe = IoFileType::ASCII /*!            Binary or ASCII file  */,
             [[maybe_unused]] int                                                     iterationId = -1)
{
    const Integer_3d<intType_ta> voxSpace = nodeSpace - 1;
    // const auto&                  extendedSpace = voxSpace;

    std::ofstream out(filename + ".vtk", std::ios::out | std::ios::binary);
    if (!out.is_open()) {
        std::string   msg = std::string("[VoxelGrid::WriteToBin] File ") + filename + std::string(" could not be open!!!");
        NeonException exception("ioToVTI");
        exception << msg;
        NEON_THROW(exception);
    }

    try {
        out << "# vtk DataFile Version 3.0" << std::endl;
        out << "Title Neon" << std::endl;
        //        if (iterationId != -1) {
        //            out << "iteration " << iterationId << std::endl;
        //            ;
        //        }

        if (vtiIOe == IoFileType::ASCII) {
            out << "ASCII" << std::endl;
        } else {
            out << "BINARY" << std::endl;
        }
        out << "DATASET STRUCTURED_POINTS" << std::endl;

        ioToVTKns::helpNs::WriteGridInfoAndAllFields<intType_ta, real_tt>(out,
                                                                          fieldsData,
                                                                          nodeSpace,
                                                                          voxSpace, spacingData, origin, vtiIOe);
    } catch (...) {
        std::string   msg = std::string("An error on file operations where encountered when writing field data");
        NeonException exception("ioToVTI");
        exception << msg;
        NEON_THROW(exception);
    }
}
}  // namespace ioToVTKns

template <typename intType_ta = int, class real_tt = double>
struct IoToVTK
{
    IoToVTK(const std::string&            filename /*!                              File name */,
            const Integer_3d<intType_ta>& nodeSpace /*!                             IoDense dimension of the node space (nodeSpace = voxelSpace +1) */,
            const Vec_3d<double>&         spacingData = Vec_3d<double>(1, 1, 1) /*! Spacing, i.e. size of a voxel */,
            const Vec_3d<double>&         origin = Vec_3d<double>(0, 0, 0) /*!      Origin  */,
            IoFileType                    vtiIOe = IoFileType::ASCII /*!            Binary or ASCII file  */)
        : m_filename(filename),
          m_nodeSpace(nodeSpace),
          m_spacingData(spacingData),
          m_origin(origin),
          m_vtiIOe(vtiIOe)
    {
    }

    virtual ~IoToVTK()
    {
        flush();
    }

    auto addField(const std::function<real_tt(const Neon::Integer_3d<intType_ta>&, int componentIdx)>& fun /*!    Implicit defintion of the user field */,
                  nComponent_t                                                                         card /*!    Field cardinality */,
                  const std::string&                                                                   fname /*!   Name of the field */,
                  ioToVTKns::VtiDataType_e                                                             vtiType /*! Type of vti element */) -> void
    {
        // ioToVTKns::UserFieldInformation userField(fun, card, fname, vtiType);
        m_fiedVec.emplace_back(fun, card, fname, vtiType);
    }

    auto flush() -> void
    {
        if (m_fiedVec.size() != 0) {
            std::string filename;
            if (m_iteration == -1) {
                filename = m_filename;
            } else {
                std::stringstream ss;
                ss << std::setw(5) << std::setfill('0') << m_iteration;
                std::string s = ss.str();
                filename = m_filename + s;
            }
            ioToVTKns::ioToVTK<intType_ta, real_tt>(m_fiedVec,
                                                    filename,
                                                    m_nodeSpace,
                                                    m_spacingData,
                                                    m_origin,
                                                    m_vtiIOe,
                                                    m_iteration);
        }
    }

    auto clear() -> void
    {
        m_fiedVec.clear();
    }

    auto flushAndClear() -> void
    {
        flush();
        clear();
    }

    auto setIteration(int iteration)
    {
        m_iteration = iteration;
    }

    auto setFormat(IoFileType vtiIOe = IoFileType::ASCII)
    {
        m_vtiIOe = vtiIOe;
    }

    auto setFileName(const std::string& fname)
    {
        m_filename = fname;
    }


   private:
    std::string                                                       m_filename /*!                              File name */;
    Integer_3d<intType_ta>                                            m_nodeSpace /*!                             IoDense dimension of the node space (nodeSpace = voxelSpace +1) */;
    Vec_3d<double>                                                    m_spacingData = Vec_3d<double>(1, 1, 1) /*! Spacing, i.e. size of a voxel */;
    Vec_3d<double>                                                    m_origin = Vec_3d<double>(0, 0, 0) /*!      Origin  */;
    IoFileType                                                        m_vtiIOe = IoFileType::ASCII /*!            Binary or ASCII file  */;
    std::vector<ioToVTKns::UserFieldInformation<intType_ta, real_tt>> m_fiedVec /*!                               Vector of field data*/;
    int                                                               m_iteration = -1;
};

}  // namespace Neon
