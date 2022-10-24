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

#include "Neon/core/types/vec.h"

namespace Neon {

namespace internal {
namespace ns_help_write_vti {

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
namespace {
template <class real_tt>
void writePieceExtentByComponent(std::ofstream&               out,
                                const std::vector<real_tt*>& mGrids,
                                const index_3d               PieceExtentData,
                                const bool                   isNode,
                                const std::string&           dataName)
{

    std::string PieceExtent;
    if (isNode) {
        PieceExtent = std::to_string(0) + std::string(" ") + std::to_string(PieceExtentData.x - 1) + std::string(" ");
        PieceExtent += std::to_string(0) + std::string(" ") + std::to_string(PieceExtentData.y - 1) + std::string(" ");
        PieceExtent += std::to_string(0) + std::string(" ") + std::to_string(PieceExtentData.z - 1);
    } else {
        PieceExtent = std::to_string(0) + std::string(" ") + std::to_string(PieceExtentData.x) + std::string(" ");
        PieceExtent += std::to_string(0) + std::string(" ") + std::to_string(PieceExtentData.y) + std::string(" ");
        PieceExtent += std::to_string(0) + std::string(" ") + std::to_string(PieceExtentData.z);
    }
    out << std::string("<Piece Extent=\"") + PieceExtent + std::string("\" >") << std::endl;

    if (isNode) {
        out << std::string("<CellData>\n");
        out << std::string("</CellData>\n");
        out << std::string("<PointData>\n");
    } else {
        out << std::string("<PointData>\n");
        out << std::string("</PointData>\n");
        out << std::string("<CellData>\n");
    }
    using namespace ns_help_write_vti::numerical_chars;
    out << std::string("<DataArray type=\"Float64\" NumberOfComponents=\"");
    out << mGrids.size();
    out << std::string("\" Name=\"");
    out << dataName;
    out << std::string("\" format=\"ascii\">\n");
    for (int z = 0; z < PieceExtentData.z; z++) {
        for (int y = 0; y < PieceExtentData.y; y++) {
            for (int x = 0; x < PieceExtentData.x; x++) {
                for (size_t m = 0; m < mGrids.size(); m++) {
                    out.precision(std::numeric_limits<real_tt>::max_digits10);
                    out << (mGrids[m])[x + y * PieceExtentData.x + z * PieceExtentData.x * PieceExtentData.y]
                        << " ";
                }
            }
        }
    }

    out << std::string("\n</DataArray>\n");

    if (isNode) {
        out << std::string("</PointData>\n");
    } else {
        out << std::string("</CellData>\n");
    }

    out << std::string("</Piece>\n");
}

template<typename dummy = int>
void openPieceExtent(std::ofstream& out, const index_3d& PieceExtentData, const bool isNode)
{
    std::string PieceExtent;
    if (isNode) {
        PieceExtent = "0 " + std::to_string(PieceExtentData.x - 1) + " ";
        PieceExtent += "0 " + std::to_string(PieceExtentData.y - 1) + " ";
        PieceExtent += "0 " + std::to_string(PieceExtentData.z - 1);
    } else {
        PieceExtent = "0 " + std::to_string(PieceExtentData.x) + " ";
        PieceExtent += "0 " + std::to_string(PieceExtentData.y) + " ";
        PieceExtent += "0 " + std::to_string(PieceExtentData.z);
    }
    out << std::string("<Piece Extent=\"") + PieceExtent + std::string("\" >") << std::endl;
}

void closePieceExtent(std::ofstream& out)
{
    out << std::string("</Piece>\n");
}

template<typename dummy = int>
void openPointOrCellSection(std::ofstream& out, bool isNode)
{
    if (isNode) {
        out << std::string("<PointData>\n");
    } else {
        out << std::string("<CellData>\n");
    }
}

void closePointOrCellSection(std::ofstream& out, bool isNode)
{
    if (isNode) {
        out << std::string("</PointData>\n");
    } else {
        out << std::string("</CellData>\n");
    }
}
template <class real_tt>
void addNodeOrVoxelData(std::ofstream& out,
                        const real_tt* mGrids,
                        const int32_t  nComponents,
                        const index_3d
                            PieceExtentData,
                        const std::string dataName)
{

    out << std::string("<DataArray type=\"Float64\" NumberOfComponents=\"");
    out << nComponents;
    out << "\" Name=\"";
    out << dataName;
    out << "\" format=\"ascii\">\n";

    size_t nEl = ((size_t)PieceExtentData.rMulTyped<size_t>()) * nComponents;

    out.precision(std::numeric_limits<real_tt>::max_digits10);
    for (size_t i = 0; i < nEl; i++) {
        out << mGrids[i] << " ";
    }

    out << "\n</DataArray>\n";
}
template <typename userReadType_ta, typename vtiWriteType_ta = userReadType_ta>
void writePieceExtent(std::ofstream&         out,
                     const userReadType_ta* mGrids,
                     const int32_t          nComponents,
                     const index_3d         PieceExtentData,
                     const bool             isNode,
                     const std::string      dataName)
{

    std::string PieceExtent;
    if (isNode) {
        PieceExtent = std::to_string(0) + std::string(" ") + std::to_string(PieceExtentData.x - 1) + std::string(" ");
        PieceExtent += std::to_string(0) + std::string(" ") + std::to_string(PieceExtentData.y - 1) + std::string(" ");
        PieceExtent += std::to_string(0) + std::string(" ") + std::to_string(PieceExtentData.z - 1);
    } else {
        PieceExtent = std::to_string(0) + std::string(" ") + std::to_string(PieceExtentData.x) + std::string(" ");
        PieceExtent += std::to_string(0) + std::string(" ") + std::to_string(PieceExtentData.y) + std::string(" ");
        PieceExtent += std::to_string(0) + std::string(" ") + std::to_string(PieceExtentData.z);
    }
    out << std::string("<Piece Extent=\"") + PieceExtent + std::string("\" >") << std::endl;

    if (isNode) {
        out << "<CellData>\n";
        out << "</CellData>\n";
        out << "<PointData>\n";
    } else {
        out << "<PointData>\n";
        out << "</PointData>\n";
        out << "<CellData>\n";
    }
    out << "<DataArray type=\"Float64\" NumberOfComponents=\"";
    out << nComponents;
    out << "\" Name=\"";
    out << dataName;
    out << "\" format=\"ascii\">\n";

    size_t nEl = ((size_t)PieceExtentData.rMulTyped<size_t>()) * nComponents;

    using namespace ns_help_write_vti::numerical_chars;
    out.precision(std::numeric_limits<vtiWriteType_ta>::max_digits10);
    int retCharCounter = 1;
    int doubleRetCharCounter = 1;
    for (size_t i = 0; i < nEl; i++) {
        vtiWriteType_ta val = vtiWriteType_ta(mGrids[i]);
        out << val << " ";
        if (retCharCounter == PieceExtentData.x) {
            retCharCounter = 0;
            out << "\n";
        }
        if (doubleRetCharCounter == PieceExtentData.x * PieceExtentData.y) {
            doubleRetCharCounter = 0;
            out << "\n";
        }
        doubleRetCharCounter++;
        retCharCounter++;
    }

    out << "\n</DataArray>\n";

    if (isNode) {
        out << "</PointData>\n";
    } else {
        out << "</CellData>\n";
    }

    out << "</Piece>\n";
}
}  // namespace
}  // namespace ns_help_write_vti


/**
 * Function to export a vector 3D field to vti format that can be open with Paraview.
 * Data of the field are treated as nodes.
 *
 * @param[in] memGrid: vector of pointers. Each one points to the value of the field for a component.
 * @param[in] nComponents: 
 * @param[in] filename: output file name
 * @param[in] mat_space: x,y,z dimension of the 3D field.
 * @param[in] spacingData: x, y, z voxel size
 * @param[in] origin: x, y, z coordinates of the origin.
 * @param[in] dataName
 * @throw Runtime exception to signal errors with file operations.
 * */
template <typename userReadType_ta, typename vtiWriteType_ta = userReadType_ta, typename vtiGridLocationType_ta = double>
void writeNodesToVTI(const userReadType_ta*               memGrid,
                     const int32_t                        nComponents,
                     const std::string&                   filename,
                     const index_3d                       mat_space,
                     const Vec_3d<vtiGridLocationType_ta> spacingData,
                     const Vec_3d<vtiGridLocationType_ta> origin,
                     const std::string&                   dataName = std::string("Data"))
{
    using namespace ns_help_write_vti;

    auto extendedSpace = mat_space;

    std::string wholeExtent = std::string("0 ") + std::to_string(extendedSpace.x - 1) + std::string(" ") + std::string("0 ") + std::to_string(extendedSpace.y - 1) + std::string(" ") + std::string("0 ") + std::to_string(extendedSpace.z - 1) + std::string(" ");

    std::string spacing = std::to_string(spacingData.x) + std::string(" ") + std::to_string(spacingData.y) + std::string(" ") + std::to_string(spacingData.z) + std::string(" ");

    std::ofstream out(filename);
    if (!out.is_open()) {
        std::string msg = std::string("[VoxelGrid::WriteToBin] File ") + filename + std::string(" could not be open!!!");
        throw std::runtime_error(msg);
    }

    out << "<?xml version=\"1.0\"?>" << std::endl;
    out << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << std::string("<ImageData WholeExtent=\"") + wholeExtent + std::string("\" Origin=\"") + std::to_string(origin.x) + std::string(" ") + std::to_string(origin.y) + std::string(" ") + std::to_string(origin.z) + std::string("\" Spacing=\"") + spacing + std::string("\">\n");

    try {
        ns_help_write_vti::writePieceExtent<userReadType_ta, vtiWriteType_ta>(out, memGrid, nComponents, mat_space, true, dataName);
    } catch (...) {
        std::string msg = std::string("An error on file operations where encountered when writing field data");
        throw std::runtime_error(msg);
    }
    out << std::string(" </ImageData>\n");
    out << std::string(" </VTKFile>\n");
}
/**
 * Function to export a vector 3D field to vti format that can be open with Paraview.
 * Data of the field are treated as voxels.
 *
 * @param[in] chIdVec: vector of pointers. Each one points to the value of the field for a component.
 * @param[in] nComponents: 
 * @param[in] filename: output file name
 * @param[in] mat_space: x,y,z dimension of the 3D field.
 * @param[in] spacingData: x, y, z voxel size
 * @param[in] origin: x, y, z coordinates of the origin.
 * @param[in] dataName
 * @throw Runtime exception to signal errors with file operations.
 */
template <typename userType_ta, typename real_ta = userType_ta, typename vtiGridLocationType_ta = double>
void writeVoxelToVTI(const userType_ta*                   chIdVec,
                     const int32_t                        nComponents,
                     const std::string                    filename,
                     const index_3d                       mat_space,
                     const Vec_3d<vtiGridLocationType_ta> spacingData,
                     const Vec_3d<vtiGridLocationType_ta> origin,
                     const std::string&                   dataName = std::string("Data"))
{
    using namespace ns_help_write_vti;

    auto extendedSpace = mat_space;

    std::string wholeExtent = std::string("0 ") + std::to_string(extendedSpace.x) + std::string(" ") + std::string("0 ") + std::to_string(extendedSpace.y) + std::string(" ") + std::string("0 ") + std::to_string(extendedSpace.z) + std::string(" ");

    std::string spacing = std::to_string(spacingData.x) + std::string(" ") + std::to_string(spacingData.y) + std::string(" ") + std::to_string(spacingData.z) + std::string(" ");

    std::ofstream out(filename);
    if (!out.is_open()) {
        std::string msg = std::string("[VoxelGrid::WriteToBin] File ") + filename + std::string(" could not be open!!!");
        throw std::runtime_error(msg);
    }

    out << "<?xml version=\"1.0\"?>" << std::endl;
    out << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
    out << "<ImageData WholeExtent=\"" << wholeExtent + std::string("\" Origin=\"") + std::to_string(origin.x) + std::string(" ") + std::to_string(origin.y) + std::string(" ") + std::to_string(origin.z) + std::string("\" Spacing=\"") + spacing + std::string("\">")
        << std::endl;
    try {
        writePieceExtent<userType_ta>(out, chIdVec, nComponents, mat_space, false, dataName);
    } catch (...) {
        std::string msg = std::string("An error on file operations where encountered when writing field data");
        throw std::runtime_error(msg);
    }

    out << std::string(" </ImageData>") << std::endl;
    out << std::string(" </VTKFile>") << std::endl;
}
}  // namespace internal

struct vti_e
{
    enum e : int
    {
        NODE,
        VOXEL,
    };

   private:
    e m_config{e::NODE};

   public:
    static const int nConfig = 2;
    vti_e() = default;
    vti_e(e conf)
        : m_config(conf) {}

    e config() const
    {
        return m_config;
    }
};

template <vti_e::e vti_ta, typename userReadType_ta, typename vtiWriteType_ta = userReadType_ta, typename vtiGridLocationType_ta = double>
void exportVti(const userReadType_ta*               memGrid,
               const int32_t                        nComponents,
               const std::string&                   filename,
               const index_3d                       mat_space,
               const Vec_3d<vtiGridLocationType_ta> spacingData,
               const Vec_3d<vtiGridLocationType_ta> origin,
               const std::string&                   dataName = std::string("Data"))
{
    switch (vti_ta) {
        case vti_e::NODE: {
            internal::writeNodesToVTI<userReadType_ta, vtiWriteType_ta, vtiGridLocationType_ta>(memGrid, nComponents, filename, mat_space, spacingData, origin, dataName);
            return;
        }
        case vti_e::VOXEL: {
            internal::writeVoxelToVTI<userReadType_ta, vtiWriteType_ta, vtiGridLocationType_ta>(memGrid, nComponents, filename, mat_space, spacingData, origin, dataName);
            return;
        }
    }
}

namespace internal {
namespace ns_help_read_vti {

namespace {
template<typename dummy = int>
void split(const std::string& s, char delim, std::vector<std::string>& elems)
{
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
}
}  // namespace
template <typename uType>
std::vector<uType> getVectorEntry(std::string entryName, std::string& text)
{
    std::smatch m;

    std::string reg = entryName + std::string("([^=]*)=\"([^\"]*)\"");
    std::regex  e(reg);
    std::regex_search(text, m, e);
    if (m.size() != 3) {
        const char* msg = "Error: parsing failed.\n";
        std::cout << msg << std::endl;
        throw std::runtime_error(msg);
    }

    std::string values = m[2];

    std::vector<std::string> elems;
    split(values, ' ', elems);

    std::vector<uType> ret;

    for (auto&& p : elems) {
        if (std::is_same<uType, int>::value) {
            int val = std::stoi(p);
            ret.push_back(uType(val));
        } else if (std::is_same<uType, float>::value) {
            float val = std::stof(p);
            ret.push_back(uType(val));
        } else if (std::is_same<uType, double>::value) {
            double val = std::stod(p);
            ret.push_back(uType(val));
        } else {
            const char* msg = "Error: unsupported type.\n";
            std::cout << msg << std::endl;
            throw std::runtime_error(msg);
        }
    }
    //        for (auto &&p : ret) {
    //                std::cout << "[" << entryName << "]-> " << p << std::endl;
    //        }
    return ret;
}

namespace {
namespace dataArrayParsing {
namespace help {

template<typename dummy = int>
size_t findEnd(const std::string& filedName, const std::string& closeTag, const std::string& text, size_t startingPos)
{
    size_t foundPos = text.find(closeTag, startingPos);
    if (foundPos == std::string::npos) {
        // return std::string::npos;
        std::string msg = std::string("Error: Unable to find h_closure of ") + filedName + std::string("field of the DataArray tag.\n");
        throw std::runtime_error(msg);
    } else {
        return foundPos - 1;
    }
}

template<typename dummy = int>
size_t findBeg(const std::string& openTag, const std::string& text, size_t startingPos)
{
    size_t foundPos = text.find(openTag, startingPos);
    if (foundPos == std::string::npos) {
        return std::string::npos;
    } else {
        return foundPos + openTag.size();
    }
}

template<typename dummy = int>
size_t findBegOpenArray(const std::string& text, size_t startingPos)
{
    size_t ret = findBeg("<DataArray", text, startingPos);
    return ret;
}

template<typename dummy = int>
size_t findEndOpenArray(const std::string& text, size_t startingPos)
{
    std::string tag("format=\"ascii\">");
    size_t      ret = findEnd("DataArrayCloseHeader", tag, text, startingPos);
    ret += tag.size() + 1;

    return ret;
}

template<typename dummy = int>
size_t findEndCloseArray(const std::string& text, size_t startingPos)
{
    size_t ret = findEnd("DataArrayCloseFull", "</DataArray>", text, startingPos);
    ret -= 1;
    return ret;
}

template<typename dummy = int>
size_t findBegNumber(const std::string& text, size_t startingPos)
{
    size_t ret = findBeg("NumberOfComponents=\"", text, startingPos);
    return ret;
}

template<typename dummy = int>
size_t findEndNumber(const std::string& text, size_t startingPos)
{
    size_t ret = findEnd("Number", "\"", text, startingPos);
    return ret;
}

template<typename dummy = int>
size_t findBegName(const std::string& text, size_t startingPos)
{
    size_t ret = findBeg("Name=\"", text, startingPos);
    return ret;
}

template<typename dummy = int>
size_t findEndName(const std::string& text, size_t startingPos)
{
    size_t ret = findEnd("Name", "\"", text, startingPos);
    return ret;
}
}  // namespace help

template<typename dummy = int>
bool findName(size_t& arrayPosition, size_t& namePosition, const std::string& name, const std::string& text)
{
    size_t foundBegPos = 0;

    while (true) {
        foundBegPos = help::findBegName(text, foundBegPos);
        if (foundBegPos == std::string::npos) {
            return false;
        }

        size_t      foundEndPos = help::findEndName(text, foundBegPos);
        std::string foundName = text.substr(foundBegPos, foundEndPos - foundBegPos + 1);
        if (foundName == name) {
            size_t foundBegArrayPos = 0;
            size_t foundBegArrayPosOld = std::string::npos;

            while (foundBegArrayPos < foundBegPos) {
                foundBegArrayPosOld = foundBegArrayPos;
                foundBegArrayPos = help::findBegOpenArray(text, foundBegArrayPos);

                if (foundBegArrayPos == std::string::npos) {
                    break;
                }
            }

            namePosition = foundEndPos;
            arrayPosition = foundBegArrayPosOld;
            return true;
        }
    }
}

template<typename dummy = int>
bool getNumComponents(int32_t& nComponents, const std::string& text, const size_t arrayPos)
{
    size_t dataArrayEndHeader = help::findEndOpenArray(text, arrayPos);
    if (dataArrayEndHeader == std::string::npos) {
        return false;
    }

    size_t begNumberPos = help::findBegNumber(text, arrayPos);
    if (begNumberPos == std::string::npos) {
        return false;
    }

    size_t endNumberPos = help::findEndNumber(text, begNumberPos);
    if (endNumberPos > dataArrayEndHeader || begNumberPos > endNumberPos) {
        return false;
    }
    std::string nComponentStr = text.substr(begNumberPos, endNumberPos - begNumberPos + 1);
    nComponents = std::stoi(nComponentStr);
    return true;
}

template<typename dummy = int>
bool getValuesPos(size_t& begPos, size_t& endPos, const std::string& text, const size_t arrayPos)
{
    size_t dataBeg = help::findEndOpenArray(text, arrayPos);
    if (dataBeg == std::string::npos) {
        return false;
    }

    size_t dataEnd = help::findEndCloseArray(text, arrayPos);
    if (dataEnd == std::string::npos) {
        return false;
    }

    while (true) {
        if (text.data()[dataBeg] > '9' || text.data()[dataBeg] < '0') {
            dataBeg++;
            continue;
        } else {
            break;
        }
    }

    while (true) {
        if (text.data()[dataEnd] > '9' || text.data()[dataEnd] < '0') {
            dataEnd--;
            continue;
        } else {
            break;
        }
    }
    begPos = dataBeg;
    endPos = dataEnd + 1;  // We leave the last space to easily parse later...

    return true;
}

template <typename uType>
bool fillVolume(uType*             mem,
                const size_t       nEl,
                const size_t       dataBeg,
                const size_t       dataEnd,
                const std::string& text,
                const std::string& dataName)
{
    size_t dataBegNumPos = dataBeg;

    for (size_t i = 0; i < nEl; i++) {

        if (dataBegNumPos >= dataEnd && i != nEl - 1) {
            std::string msg = std::string(
                                  "Error: parsing failed, unable to load numbers (less elements than requested) for ") +
                              dataName + std::string(".");
            throw std::runtime_error(msg);
        }

        size_t      dataEndNumPos = text.find(" ", dataBegNumPos) - 1;
        std::string numStr = text.substr(dataBegNumPos, dataEndNumPos - dataBegNumPos + 1);
        if (std::is_same<uType, int>::value) {
            mem[i] = uType(std::stoi(numStr));
        } else if (std::is_same<uType, float>::value) {
            mem[i] = uType(std::stof(numStr));
        } else if (std::is_same<uType, double>::value) {
            mem[i] = uType(std::stod(numStr));
        } else {
            const char* msg = "Error: unsupported type.\n";
            throw std::runtime_error(msg);
        }

        dataBegNumPos = dataEndNumPos + 2;
    }

    if (dataBegNumPos < dataEnd) {
        free(mem);
        std::string msg = std::string("Error: parsing failed, unable to load numbers (more elements than requested) for ") + dataName + std::string(".");
        throw std::runtime_error(msg);
    }
    return true;
}
}  // namespace dataArrayParsing
}  // namespace
template <typename uType>
uType*
allocVolume(int32_t& nComponents, const std::string dataName, const int nElementInSpace, const std::string& text)
{

    /* Example for dataName = Genesis
         * <DataArray type="Float64" NumberOfComponents="1" Name="Genesis" format="ascii">
         <DataArray type="Float64" NumberOfComponents="1" Name="Genesis" format="ascii">\n
         *           */

    std::smatch m;
    std::string reg;
    std::regex  e;

    size_t dataBeg;
    size_t dataEnd;

    {  /// Searching for the name
        size_t namePosition;
        size_t arrayPosition;
        bool   found = dataArrayParsing::findName(arrayPosition, namePosition, dataName, text);
        if (!found) {
            std::string msg = std::string("Error: parsing failed, unable to find component called ") + dataName + std::string(".");
            throw std::runtime_error(msg);
        }

        nComponents = 0;
        found = dataArrayParsing::getNumComponents(nComponents, text, arrayPosition);
        if (!found) {
            std::string msg = std::string("Error: parsing failed, unable to find the number of component field for ") + dataName + std::string(".");
            throw std::runtime_error(msg);
        }

        found = dataArrayParsing::getValuesPos(dataBeg, dataEnd, text, arrayPosition);
        if (!found) {
            std::string msg = std::string("Error: parsing failed, unable to load value for ") + dataName + std::string(".");
            throw std::runtime_error(msg);
        }
    }

    uType* volume = (uType*)malloc(sizeof(uType) * nElementInSpace * nComponents);
    if (volume == nullptr) {
        const char* msg = "Error: allocation failed.\n";
        throw std::runtime_error(msg);
    }

    dataArrayParsing::fillVolume(volume, ((size_t)nElementInSpace * nComponents), dataBeg, dataEnd, text, dataName);

    return volume;
}

/**
 * The function imports a scalar or vector 3D field from a vti format file.
 * Components of the files are colocated in memory like x1, y1, z1, x2, y2, z2, ...
 * Data of the field are stored as nodes or voxel, depending on the parameter isNode.
 * Important: the memory that is returned must be deleted with a free.
 *
 * TODO(Max) Would make more sense to return a smart pointer?
 *
 * @param[out] mGrid: pointer to a pointer where to return the location of the read volume.
 * @param[out] nComponents: number of components.
 * @param[out] spaceDim: dimension of the 3D field.
 * @param[out] voxel_size: size of the voxel
 * @param[out] origin: coordinate of the origin.
 * @param[in] filename: name of the file containing the data.
 * @param[in] fieldName: name of the field containing the data. This is the names that is specified in the vti file.
 * @param[out] isNode: true if node, false if voxel
 * @throw Runtime exception to signal errors with file operations.
 * */
template <typename DataType, typename SpaceType>
void readFromVTI(DataType**         mGrid,
                 int32_t&           nComponents,
                 index_3d&          spaceDim,
                 Vec_3d<SpaceType>& voxel_size,
                 Vec_3d<SpaceType>& origin,
                 const std::string  filename,
                 const std::string  fieldName,
                 const bool         isNode)
{
    namespace help = ns_help_read_vti;
    std::ifstream t(filename);
    std::string   xmlText;

    {  /// Loading file in memory

        t.seekg(0, std::ios::end);
        xmlText.reserve(t.tellg());
        t.seekg(0, std::ios::beg);

        xmlText.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    }

    auto wholeExtentVec = help::getVectorEntry<int>("ImageData WholeExtent", xmlText);
    auto pieceExtentVec = help::getVectorEntry<int>("Piece Extent", xmlText);

    auto originVec = help::getVectorEntry<SpaceType>("Origin", xmlText);
    auto spacingVec = help::getVectorEntry<SpaceType>("Spacing", xmlText);

    if (pieceExtentVec != wholeExtentVec) {
        const char* msg = "Error: multi blocks are not supported yet.\n";
        throw std::runtime_error(msg);
    }

    /// We add one as we are managing nodes and not voxels
    if (isNode) {
        spaceDim.x = wholeExtentVec[2 * 0 + 1] - wholeExtentVec[2 * 0] + 1;
        spaceDim.y = wholeExtentVec[2 * 1 + 1] - wholeExtentVec[2 * 1] + 1;
        spaceDim.z = wholeExtentVec[2 * 2 + 1] - wholeExtentVec[2 * 1] + 1;
    } else {
        spaceDim.x = wholeExtentVec[2 * 0 + 1] - wholeExtentVec[2 * 0];
        spaceDim.y = wholeExtentVec[2 * 1 + 1] - wholeExtentVec[2 * 1];
        spaceDim.z = wholeExtentVec[2 * 2 + 1] - wholeExtentVec[2 * 2];
    }
    origin.x = originVec[0];
    origin.y = originVec[1];
    origin.z = originVec[2];
    voxel_size.x = spacingVec[0];
    voxel_size.y = spacingVec[1];
    voxel_size.z = spacingVec[2];
    int nElementInSpace = spaceDim.x * spaceDim.y * spaceDim.z;
    *mGrid = help::allocVolume<DataType>(nComponents, fieldName, nElementInSpace, xmlText);
}
}  // namespace ns_help_read_vti

/**
   * The function imports a scalar or vector 3D field from a vti format file.
   * Components of the files are colocated in memory like x1, y1, z1, x2, y2, z2, ...
   * Data of the field are stored as nodes.
   * Important: the memory that is returned must be deleted with a free.
   *
   * TODO(Max) Would make more sense to return a smart pointer?
   *
   * @param[out] mGrid: pointer to a pointer where to return the location of the read volume.
   * @param[out] nComponents: number of components.
   * @param[out] spaceDim: x dimension of the 3D field.
   * @param[out] voxel_size: size of the voxel in the x dimension. 
   * @param[out] origin: coordinate of the origin. 
   * @param[in] filename
   * @param[in] fieldName: name of the field containing the data. This is the names that is specified in the vti file.
   * @throw Runtime exception to signal errors with file operations.
   * */
template <typename DataType, typename SpaceType>
void readNodesFromVTI(DataType**         mGrid,
                      int32_t&           nComponents,
                      index_3d&          spaceDim,
                      Vec_3d<SpaceType>& voxel_size,
                      Vec_3d<SpaceType>& origin,
                      const std::string  filename,
                      const std::string  fieldName)
{

    namespace help = internal::ns_help_read_vti;

    bool isNode = true;
    help::readFromVTI(mGrid,
                      nComponents,
                      spaceDim,
                      voxel_size,
                      origin,
                      filename,
                      fieldName,
                      isNode);
}

/**
 * The function imports a scalar or vector 3D field from a vti format file.
 * Components of the files are colocated in memory like x1, y1, z1, x2, y2, z2, ...
 * Data of the field are stored as voxels.
 * Important: the memory that is returned must be deleted with a free.
 *
 * TODO(Max) Would make more sense to return a smart pointer?
 *
 * @param[out] mGrid: pointer to a pointer where to return the location of the read volume.
 * @param[out] nComponents: number of components.
 * @param[out] spaceDim: dimension of the 3D field.
 * @param[out] voxel_size: size of the voxel in the x dimension.
 * @param[out] origin: coordinate of the origin.
 * @param[in] filename: name of the file containing the data.
 * @param[in] fieldName: name of the field containing the data. This is the names that is specified in the vti file.
 * */
template <typename DataType, typename SpaceType>
void readVoxelsFromVTI(DataType**         mGrid,
                       int32_t&           nComponents,
                       index_3d&          spaceDim,
                       Vec_3d<SpaceType>& voxel_size,
                       Vec_3d<SpaceType>& origin,
                       const std::string  filename,
                       const std::string  fieldName)
{

    namespace help = internal::ns_help_read_vti;
    bool isNode = false;
    help::readFromVTI(mGrid,
                      nComponents,
                      spaceDim.x,
                      spaceDim.y,
                      spaceDim.z,
                      voxel_size.x,
                      voxel_size.y,
                      voxel_size.z,
                      origin.x,
                      origin.y,
                      origin.z,
                      filename,
                      fieldName,
                      isNode);
}

namespace wrapperVTI {
/*
 * TODO@Max For now all the get function redone the input file. Fix it.
 */
class vtiInput_t
{
    std::unique_ptr<std::ifstream> m_inStreamPtr{nullptr};
    std::string                    m_fileName;
    index_3d                       m_voxelGridDim;
    index_3d                       m_nodeGridDim;
    Vec_3d<double>                 m_gridOrigin;
    Vec_3d<double>                 m_gridSpacing;

   public:
    void open(const std::string& fileName)
    {
        this->m_fileName = fileName;
        //                inStreamPtr.reset(new std::ifstream(fileName));
        //                if (!inStreamPtr->is_open( )) {
        //                        std::string msg = std::string("File ") + fileName + std::string(" could not be open!!!");
        //                        throw std::runtime_error(msg);
        //                }
    }

    void close()
    {
        m_fileName = "";
        m_voxelGridDim(0, 0, 0);
        m_nodeGridDim(0, 0, 0);
        m_gridOrigin(0, 0, 0);
        m_gridSpacing(0, 0, 0);
    }

    template <typename DataType, typename SpaceType>
    void getNodeVolume(DataType**         mGrid,
                       int32_t&           nComponents,
                       index_3d&          spaceDim,
                       Vec_3d<SpaceType>& voxel_size,
                       Vec_3d<SpaceType>& origin,
                       const std::string  fieldName)
    {
        namespace help = ns_help_read_vti;

        bool isNode = true;
        help::readFromVTI(mGrid,
                          nComponents,
                          spaceDim,
                          voxel_size,
                          origin,
                          this->m_fileName,
                          fieldName,
                          isNode);
    }
    template <typename DataType, typename SpaceType>
    void getVoxelVolume(DataType**         mGrid,
                        int32_t&           nComponents,
                        index_3d&          spaceDim,
                        Vec_3d<SpaceType>& voxel_size,
                        Vec_3d<SpaceType>& origin,
                        const std::string  fieldName)
    {
        namespace help = ns_help_read_vti;

        bool isNode = false;
        help::readFromVTI(mGrid,
                          nComponents,
                          spaceDim.x,
                          spaceDim.y,
                          spaceDim.z,
                          voxel_size.x,
                          voxel_size.y,
                          voxel_size.z,
                          origin.x,
                          origin.y,
                          origin.z,
                          this->m_fileName,
                          fieldName,
                          isNode);
    }

    template <typename T_tt>
    void getExtraInfo(const std::string& name, T_tt& value)
    {
        std::ifstream t(m_fileName);
        std::string   xmlText;

        {  /// Loading file in memory

            t.seekg(0, std::ios::end);
            xmlText.reserve(t.tellg());
            t.seekg(0, std::ios::beg);

            xmlText.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        }

        std::string search = "<!-- libNeonExtraInfo" + name + ":";
        auto        foundBeginPosition = xmlText.find(search);
        auto        foundEndPosition = xmlText.find(" -->", foundBeginPosition);

        if (foundBeginPosition == std::string::npos || foundEndPosition == std::string::npos) {
            std::string msg = "Error: unable to parse the vti vile correctly. Extra info " + name + " was not found \n";
            throw std::runtime_error(msg);
        }

        std::string valueStr = xmlText.substr(foundBeginPosition + search.size(), foundEndPosition - foundBeginPosition - search.size());

        if (std::is_same<T_tt, int>::value) {
            value = T_tt(std::stoi(valueStr));
        } else if (std::is_same<T_tt, float>::value) {
            value = T_tt(std::stof(valueStr));
        } else if (std::is_same<T_tt, double>::value) {
            value = T_tt(std::stof(valueStr));
        } else {
            const char* msg = "Error: unsupported type.\n";
            throw std::runtime_error(msg);
        }

        t.close();
    }
    void getExtraInfo(const std::string& name, std::string& value)
    {
        std::ifstream t(m_fileName);
        std::string   xmlText;

        {  /// Loading file in memory

            t.seekg(0, std::ios::end);
            xmlText.reserve(t.tellg());
            t.seekg(0, std::ios::beg);

            xmlText.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        }

        std::string search = "<!-- libNeonExtraInfo" + name + ":";
        auto        foundBeginPosition = xmlText.find(search);
        auto        foundEndPosition = xmlText.find(" -->", foundBeginPosition);

        if (foundBeginPosition == std::string::npos || foundEndPosition == std::string::npos) {
            std::string msg = "Error: unable to parse the vti vile correctly. Extra info " + name + " was not found \n";
            throw std::runtime_error(msg);
        }

        value = xmlText.substr(foundBeginPosition + search.size(), foundEndPosition - foundBeginPosition - search.size());

        t.close();
    }
};

class vtiOutput_t
{
   private:
    std::unique_ptr<std::ofstream> m_outStreamPtr{nullptr};
    index_3d                       m_voxelGridDim;
    index_3d                       m_nodeGridDim;
    Vec_3d<double>                 m_gridOrigin;
    Vec_3d<double>                 m_gridSpacing;

    bool m_nodeSectionOpen{false};
    bool m_nodeSectionDone{false};
    bool m_voxelSectionOpen{false};
    bool m_voxelSectionDone{false};

   public:
    vtiOutput_t() = default;

    template <typename T_tt>
    void addExtraInfo(std::string name, T_tt value)
    {
        if (m_nodeSectionOpen || m_nodeSectionDone || m_voxelSectionOpen || m_voxelSectionDone) {
            std::string msg = std::string("Extra data can be added only before adding volume data.");
            throw std::runtime_error(msg);
        }
        (*m_outStreamPtr) << "<!-- libNeonExtraInfo" + name + ":" << value << " -->\n";
    }

    template <typename T_tt>
    void open(std::string         fileName,
              const index_3d      gridDim,
              bool                isNode,
              const Vec_3d<T_tt>& gridSpacing,
              const Vec_3d<T_tt>& gridOrigin)
    {

        this->m_nodeGridDim = gridDim;
        this->m_voxelGridDim = gridDim;
        if (isNode) {
            this->m_voxelGridDim = gridDim - 1;
        } else {
            this->m_nodeGridDim = gridDim + 1;
        }

        this->m_gridOrigin = gridOrigin.template newType<double>();
        this->m_gridSpacing = gridSpacing.template newType<double>();

        m_outStreamPtr.reset(new std::ofstream(fileName));
        if (!m_outStreamPtr->is_open()) {
            std::string msg = std::string("[VoxelGrid::WriteToBin] File ") + fileName + std::string(" could not be open!!!");
            throw std::runtime_error(msg);
        }

        (*m_outStreamPtr) << "<?xml version=\"1.0\"?>" << std::endl;
        (*m_outStreamPtr) << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
                          << std::endl;

        {  /// Adding preamble
            std::string wholeExtent = "0 " + std::to_string(m_voxelGridDim.x) + " " + "0 " + std::to_string(m_voxelGridDim.y) + " " + "0 " + std::to_string(m_voxelGridDim.z) + " ";

            std::string spacing = std::to_string(gridSpacing.x) + " " + std::to_string(gridSpacing.y) + " " + std::to_string(gridSpacing.z) + " ";

            (*m_outStreamPtr) << "<ImageData WholeExtent=\"" + wholeExtent + "\" Origin=\"" + std::to_string(gridOrigin.x) + " " + std::to_string(gridOrigin.y) + " " + std::to_string(gridOrigin.z) + "\" Spacing=\"" + spacing + "\">\n";
        }

        internal::ns_help_write_vti::openPieceExtent(*m_outStreamPtr, this->m_voxelGridDim, false);
    }

    void close()
    {
        if (m_nodeSectionOpen) {
            internal::ns_help_write_vti::closePointOrCellSection(*m_outStreamPtr, true);
        }
        if (m_voxelSectionOpen) {
            internal::ns_help_write_vti::closePointOrCellSection(*m_outStreamPtr, false);
        }

        internal::ns_help_write_vti::closePieceExtent(*m_outStreamPtr);
        {  /// Closing volume
            (*m_outStreamPtr) << std::string(" </ImageData>\n\n\n");
        }

        (*m_outStreamPtr) << std::string(" </VTKFile>\n");
        m_outStreamPtr->close();
        m_outStreamPtr.reset();
    }

   private:
    template <typename T_tt>
    void addVolume(T_tt* gridMem, int32_t nComponents, std::string dataName, const bool isNode)
    {

        bool interlivingError = false;
        interlivingError = interlivingError || (isNode && m_nodeSectionDone);
        interlivingError = interlivingError || ((!isNode) && m_voxelSectionDone);
        if (interlivingError) {
            std::string msg = std::string(
                "Voxel and Node data can not be interleaved in the same VTI file. Pleas add "
                "first all the data for one of the two, then add the remaining one.");
            throw std::runtime_error(msg);
        }

        if (isNode && (!m_nodeSectionOpen)) {
            if (m_voxelSectionOpen) {
                ns_help_write_vti::closePointOrCellSection(*m_outStreamPtr, false);
                m_voxelSectionDone = true;
            }
            ns_help_write_vti::openPointOrCellSection(*m_outStreamPtr, isNode);
            m_nodeSectionOpen = true;
        }
        if ((!isNode) && (!m_voxelSectionOpen)) {
            if (m_nodeSectionOpen) {
                ns_help_write_vti::closePointOrCellSection(*m_outStreamPtr, true);
                m_nodeSectionDone = true;
            }
            ns_help_write_vti::openPointOrCellSection(*m_outStreamPtr, isNode);
            m_voxelSectionOpen = true;
        }

        try {
            if (isNode) {
                ns_help_write_vti::addNodeOrVoxelData(
                    *m_outStreamPtr, gridMem, nComponents, m_nodeGridDim, dataName);
            } else {
                ns_help_write_vti::addNodeOrVoxelData(
                    *m_outStreamPtr, gridMem, nComponents, m_voxelGridDim, dataName);
            }
        } catch (...) {
            std::string msg = std::string("An error on file operations where encountered when writing field data");
            throw std::runtime_error(msg);
        }
    }

   public:
    template <typename T_tt>
    void addNodeVolume(T_tt* gridMem, int32_t nComponents, std::string dataName)
    {
        return this->addVolume<T_tt>(gridMem, nComponents, dataName, true);
    }

    template <typename T_tt>
    void addVoxelVolume(T_tt* gridMem, int32_t nComponents, std::string dataName)
    {
        return this->addVolume<T_tt>(gridMem, nComponents, dataName, false);
    }
};
}  // namespace wrapperVTI
}  // namespace internal

}  // namespace Neon
