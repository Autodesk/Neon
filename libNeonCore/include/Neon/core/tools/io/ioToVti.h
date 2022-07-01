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

struct ioVTI_e
{
    enum e
    {
        ASCII,
        BINARY
    };
};

template <class real_tt, typename intType_ta>
using userGridFun_t = std::function<real_tt(const Neon::Integer_3d<intType_ta>&, int componentIdx)>;
using nComponent_t = int;
using fieldName_t = std::string;
using isNodeFlags_t = bool;

template <class real_tt, typename intType_ta>
struct VtiInputData_t
{
    userGridFun_t<real_tt, intType_ta> func; // Function that takes 3d index and component id and returns the field value
    nComponent_t                       nComponents;  // Number of components in the field
    fieldName_t                        fieldName;     // Name for the field
    isNodeFlags_t                      isNode;       // Whether this data is node based (or voxel based)
    ioVTI_e::e                         asciiOrBinary; // Whether to write data as ASCII or binary
};

namespace internal_implicit {
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


namespace base64 {

#if 0

static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";


static inline bool is_base64(unsigned char c)
{
    return (isalnum(c) || (c == '+') || (c == '/'));
}

std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len)
{
    std::string   ret;
    int           i = 0;
    int           j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; (i < 4); i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];

        while ((i++ < 3))
            ret += '=';
    }

    return ret;
}
std::string base64_decode(std::string const& encoded_string)
{
    int           in_len = encoded_string.size();
    int           i = 0;
    int           j = 0;
    int           in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::string   ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_];
        in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++)
                char_array_4[i] = base64_chars.find(char_array_4[i]);

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++)
                ret += char_array_3[i];
            i = 0;
        }
    }

    if (i) {
        for (j = 0; j < i; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]);

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

        for (j = 0; (j < i - 1); j++)
            ret += char_array_3[j];
    }

    return ret;
}
#endif

namespace beast {
namespace detail {

namespace base64 {

inline char const*
get_alphabet()
{
    static char constexpr tab[] = {
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"};
    return &tab[0];
}

inline signed char const*
get_inverse()
{
    static signed char constexpr tab[] = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  //   0-15
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  //  16-31
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63,  //  32-47
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1,  //  48-63
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,            //  64-79
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1,  //  80-95
        -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,  //  96-111
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1,  // 112-127
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  // 128-143
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  // 144-159
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  // 160-175
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  // 176-191
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  // 192-207
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  // 208-223
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  // 224-239
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1   // 240-255
    };
    return &tab[0];
}


/// Returns max chars needed to encode a base64 string
inline std::size_t constexpr encoded_size(std::size_t n)
{
    return 4 * ((n + 2) / 3);
}

/// Returns max bytes needed to decode a base64 string
inline std::size_t constexpr decoded_size(std::size_t n)
{
    return n / 4 * 3;  // requires n&3==0, smaller
    //return 3 * n / 4;
}

/** Encode a series of octets as a padded, base64 string.

    The resulting string will not be null terminated.

    @par Requires

    The memory pointed to by `out` points to valid memory
    of at least `encoded_size(len)` bytes.

    @return The number of characters written to `out`. This
    will exclude any null termination.
*/
template <class = void>
std::size_t
encode(void* dest, void const* src, std::size_t len)
{
    char*       out = static_cast<char*>(dest);
    char const* in = static_cast<char const*>(src);
    auto const  tab = base64::get_alphabet();

    for (auto n = len / 3; n--;) {
        *out++ = tab[(in[0] & 0xfc) >> 2];
        *out++ = tab[((in[0] & 0x03) << 4) + ((in[1] & 0xf0) >> 4)];
        *out++ = tab[((in[2] & 0xc0) >> 6) + ((in[1] & 0x0f) << 2)];
        *out++ = tab[in[2] & 0x3f];
        in += 3;
    }

    switch (len % 3) {
        case 2:
            *out++ = tab[(in[0] & 0xfc) >> 2];
            *out++ = tab[((in[0] & 0x03) << 4) + ((in[1] & 0xf0) >> 4)];
            *out++ = tab[(in[1] & 0x0f) << 2];
            *out++ = '=';
            break;

        case 1:
            *out++ = tab[(in[0] & 0xfc) >> 2];
            *out++ = tab[((in[0] & 0x03) << 4)];
            *out++ = '=';
            *out++ = '=';
            break;

        case 0:
            break;
    }

    return out - static_cast<char*>(dest);
}

/** Decode a padded base64 string into a series of octets.

    @par Requires

    The memory pointed to by `out` points to valid memory
    of at least `decoded_size(len)` bytes.

    @return The number of octets written to `out`, and
    the number of characters read from the input string,
    expressed as a pair.
*/
template <class = void>
std::pair<std::size_t, std::size_t>
decode(void* dest, char const* src, std::size_t len)
{
    char*         out = static_cast<char*>(dest);
    auto          in = reinterpret_cast<unsigned char const*>(src);
    unsigned char c3[3], c4[4];
    int           i = 0;
    int           j = 0;

    auto const inverse = base64::get_inverse();

    while (len-- && *in != '=') {
        auto const v = inverse[*in];
        if (v == -1)
            break;
        ++in;
        c4[i] = v;
        if (++i == 4) {
            c3[0] = (c4[0] << 2) + ((c4[1] & 0x30) >> 4);
            c3[1] = ((c4[1] & 0xf) << 4) + ((c4[2] & 0x3c) >> 2);
            c3[2] = ((c4[2] & 0x3) << 6) + c4[3];

            for (i = 0; i < 3; i++)
                *out++ = c3[i];
            i = 0;
        }
    }

    if (i) {
        c3[0] = (c4[0] << 2) + ((c4[1] & 0x30) >> 4);
        c3[1] = ((c4[1] & 0xf) << 4) + ((c4[2] & 0x3c) >> 2);
        c3[2] = ((c4[2] & 0x3) << 6) + c4[3];

        for (j = 0; j < i - 1; j++)
            *out++ = c3[j];
    }

    return {out - static_cast<char*>(dest),
            in - reinterpret_cast<unsigned char const*>(src)};
}

}  // namespace base64

template <class = void>
std::string
base64_encode(std::uint8_t const* data,
              std::size_t         len)
{
    std::string dest;
    dest.resize(base64::encoded_size(len));
    dest.resize(base64::encode(&dest[0], data, len));
    return dest;
}

inline std::string
base64_encode(std::string const& s)
{
    return base64_encode(reinterpret_cast<
                             std::uint8_t const*>(s.data()),
                         s.size());
}

template <class = void>
std::string
base64_decode(std::string const& data)
{
    std::string dest;
    dest.resize(base64::decoded_size(data.size()));
    auto const result = base64::decode(
        &dest[0], data.data(), data.size());
    dest.resize(result.first);
    return dest;
}

}  // namespace detail
}  // namespace beast
}  // namespace base64
void openPointOrCellSection(std::ofstream& out, bool isNode) NEON_ATTRIBUTE_UNUSED;
void openPointOrCellSection(std::ofstream& out, bool isNode)
{
    if (isNode) {
        out << std::string("<PointData>\n");
    } else {
        out << std::string("<CellData>\n");
    }
}
void closePointOrCellSection(std::ofstream& out, bool isNode) NEON_ATTRIBUTE_UNUSED;
void closePointOrCellSection(std::ofstream& out, bool isNode)
{
    if (isNode) {
        out << std::string("</PointData>\n");
    } else {
        out << std::string("</CellData>\n");
    }
}

template <class real_tt, typename intType_ta>
void writeData(std::ofstream&                                   out,
               userGridFun_t<real_tt, intType_ta> grid,
               nComponent_t                       nComponents,
               const fieldName_t&                 fieldName,
               const Integer_3d<intType_ta>&                    space,
               ioVTI_e::e                                       vtiIO = ioVTI_e::e::ASCII)
{
    switch (vtiIO) {
        case ioVTI_e::e::BINARY: {
            out << "<DataArray type=\"Float64\" NumberOfComponents=\"";
            out << nComponents;
            out << "\" Name=\"";
            out << fieldName;
            out << "\" format=\"binary\">\n";

            using namespace ns_help_write_vti::numerical_chars;
            out.setf(std::ios_base::fixed, std::ios_base::floatfield);
            //out.precision(std::numeric_limits<real_tt>::max_digits10);
            out.precision(17);

            Neon::Integer_3d<intType_ta> idx;
            size_t                       tmpSize = space.template mSize<intType_ta>() * nComponents;
            unsigned char*               tmp = (unsigned char*)malloc(tmpSize + 4);
            intType_ta*                  tmpData = (intType_ta*)(tmp + 4);
            if (tmp == nullptr) {
                NeonException ex("Allocation issue.");
                NEON_THROW(ex);
            }
            for (intType_ta z = 0; z < space.z; z++) {
                for (intType_ta y = 0; y < space.y; y++) {
                    for (intType_ta x = 0; x < space.x; x++) {
                        idx.set(x, y, z);
                        for (int v = 0; v < nComponents; v++) {
                            intType_ta val = static_cast<intType_ta>(grid(idx, v));
                            //int32_t size = sizeof(double);
                            //out.write((char*)&size, sizeof(size));
                            //out << base64::base64_encode((unsigned char*)&val, sizeof(val));
                            size_t jump = size_t(v) +
                                          size_t(x) * nComponents +
                                          size_t(y) * nComponents * space.x +
                                          size_t(z) * nComponents * space.x * space.y;
                            tmpData[jump] = val;
                        }
                    }
                }
            }
            int len = static_cast<int>(tmpSize);
            tmp[0] = (unsigned char)len;
            tmp[1] = (unsigned char)(len >> 8);
            tmp[2] = (unsigned char)(len >> 16);
            tmp[3] = (unsigned char)(len >> 24);

            out << base64::beast::detail::base64_encode((unsigned char*)tmp, tmpSize + 4);
            //            //out << base64::base64_encode((unsigned char*)tmp, tmpSize);
            //            out << base64::beast::detail::base64_encode((unsigned char*)tmpData, tmpSize);
            //base64_encode((unsigned char*)tmp, tmpSize);
            out << "\n</DataArray>\n\n";
            free(tmp);
            return;
        }
        case ioVTI_e::e::ASCII: {
            out << "<DataArray type=\"Float64\" NumberOfComponents=\"";
            out << nComponents;
            out << "\" Name=\"";
            out << fieldName;
            out << "\" format=\"ascii\">\n";

            using namespace ns_help_write_vti::numerical_chars;
            out.setf(std::ios_base::fixed, std::ios_base::floatfield);
            //out.precision(std::numeric_limits<real_tt>::max_digits10);
            out.precision(17);

            Neon::Integer_3d<intType_ta> idx;

            for (intType_ta z = 0; z < space.z; z++) {
                for (intType_ta y = 0; y < space.y; y++) {
                    for (intType_ta x = 0; x < space.x; x++) {
                        idx.set(x, y, z);
                        for (int v = 0; v < nComponents; v++) {
                            auto val = grid(idx, v);
                            out << val << " ";
                        }
                        out << "\t";
                    }
                    out << "\n";
                }
                out << "\n";
            }
            out << "</DataArray>\n\n";
            return;
        }
    }
}

template <class real_tt, typename intType_ta>
void WriteNodeAndVoxelData(std::ofstream&                                                                              out,
                           const std::vector<std::function<real_tt(const Integer_3d<intType_ta>&, int componentIdx)>>& grids,
                           const std::vector<int32_t>&                                                                 nComponents,
                           const std::vector<std::string>&                                                             fieldName,
                           const std::vector<bool>&                                                                    isNodeFlags,
                           const Integer_3d<intType_ta>&                                                               nodeSpace,
                           const Integer_3d<intType_ta>&                                                               voxSpace)
{
    auto addDataFiltered = [&](bool doNodes) {
        openPointOrCellSection(out, doNodes);
        for (int i = 0; i < int(grids.size()); i++) {
            if (isNodeFlags[i] == doNodes) {
                writeData(out,
                          grids[i],
                          nComponents[i],
                          fieldName[i],
                          (doNodes ? nodeSpace : voxSpace));
            }
        }
        closePointOrCellSection(out, doNodes);
    };  // namespace

    addDataFiltered(true);
    addDataFiltered(false);
}  // namespace ns_help_write_vti

template <class real_tt, typename intType_ta>
void WriteNodeAndVoxelData(std::ofstream&                                                     out,
                           const std::vector<VtiInputData_t<real_tt, intType_ta>>& gridsInfo,
                           const Integer_3d<intType_ta>&                                      nodeSpace,
                           const Integer_3d<intType_ta>&                                      voxSpace)
{
    bool doNodes;
    //    auto writeDataUnwrap = [&](ioToVTIns::userGridFun_t<real_tt, intType_ta> grid,
    //                               const int32_t&                                   nComponents,
    //                               const std::string&                               fieldName,
    //                               bool                                             isNodeFlags) -> void {
    //        if (isNodeFlags == doNodes) {
    //            writeData<real_tt, intType_ta>(out,
    //                                           grid,
    //                                           nComponents,
    //                                           fieldName,
    //                                           (doNodes ? nodeSpace : voxSpace));
    //        }
    //    };

    auto writeDataUnwrap = [&](VtiInputData_t<real_tt, intType_ta> inp) -> void {
        if (inp.isNode == doNodes) {
            writeData<real_tt, intType_ta>(out,
                                           inp.func,
                                           inp.nComponents,
                                           inp.fieldName,
                                           (doNodes ? nodeSpace : voxSpace),
                                           inp.asciiOrBinary);
        }
    };

    //    auto unpackAndWriteData = [&](auto&... args) {
    //        //Neon::meta::debug::printType(args...);
    //        return writeDataUnwrap(args...);
    //    };

    auto addDataFiltered = [&](bool doNodes_) {
        doNodes = doNodes_;
        openPointOrCellSection(out, doNodes);
        for (int i = 0; i < int(gridsInfo.size()); i++) {
            //Neon::meta::debug::printType(gridsInfo[i]);
            //std::apply(writeDataUnwrap, gridsInfo[i]);
            writeDataUnwrap(gridsInfo[i]);
        }
        closePointOrCellSection(out, doNodes);
    };

    addDataFiltered(true);
    addDataFiltered(false);
}


}  // namespace

template <class real_tt, typename intType_ta>
void writePieceExtent(std::ofstream&                                                                              out,
                      const std::vector<std::function<real_tt(const Integer_3d<intType_ta>&, int componentIdx)>>& grids,
                      const std::vector<int32_t>&                                                                 nComponents,
                      const std::vector<std::string>&                                                             fieldName,
                      const std::vector<bool>&                                                                    isNodeFlags,
                      const Integer_3d<intType_ta>&                                                               nodeSpace,
                      const Integer_3d<intType_ta>&                                                               voxSpace)
{
    auto        extendedSpace = voxSpace;
    std::string PieceExtent;

    PieceExtent = std::to_string(0) + std::string(" ") + std::to_string(extendedSpace.x) + std::string(" ");
    PieceExtent += std::to_string(0) + std::string(" ") + std::to_string(extendedSpace.y) + std::string(" ");
    PieceExtent += std::to_string(0) + std::string(" ") + std::to_string(extendedSpace.z);

    out << std::string("<Piece Extent=\"") + PieceExtent + std::string("\" >") << std::endl;
    WriteNodeAndVoxelData<real_tt, intType_ta>(out,
                                               grids,
                                               nComponents,
                                               fieldName,
                                               isNodeFlags,
                                               nodeSpace,
                                               voxSpace);
    out << std::string("</Piece>\n");
}

template <class real_tt, typename intType_ta>
void writePieceExtent(std::ofstream&                                                     out,
                      const std::vector<VtiInputData_t<real_tt, intType_ta>>& gridsInfo,
                      const Integer_3d<intType_ta>&                                      nodeSpace,
                      const Integer_3d<intType_ta>&                                      voxSpace)
{
    auto        extendedSpace = voxSpace;
    std::string PieceExtent;

    PieceExtent = std::to_string(0) + std::string(" ") + std::to_string(extendedSpace.x) + std::string(" ");
    PieceExtent += std::to_string(0) + std::string(" ") + std::to_string(extendedSpace.y) + std::string(" ");
    PieceExtent += std::to_string(0) + std::string(" ") + std::to_string(extendedSpace.z);

    out << std::string("<Piece Extent=\"") + PieceExtent + std::string("\" >") << std::endl;
    WriteNodeAndVoxelData<real_tt, intType_ta>(out,
                                               gridsInfo,
                                               nodeSpace,
                                               voxSpace);
    out << std::string("</Piece>\n");
}
}  // namespace ns_help_write_vti


}  // namespace internal_implicit


/**
 * Function to export a vector 3D field to vti format that can be open with Paraview.
 * Data of the field are treated as nodes.
 *
 * @tparam intType_ta
 * @tparam real_tt
 * @param grids
 * @param nComponents
 * @param fieldName
 * @param isNodeFlags
 * @param filename
 * @param nodeSpace
 * @param voxSpace
 * @param spacingData
 * @param origin
 */
template <typename intType_ta, typename real_tt = double>
[[deprecated]]
void ioToVTI(const std::vector<std::function<real_tt(const Integer_3d<intType_ta>&, int componentIdx)>>& grids,
             const std::vector<int32_t>&                                                                 nComponents,
             const std::vector<std::string>&                                                             fieldName,
             const std::vector<bool>&                                                                    isNodeFlags,
             const std::string&                                                                          filename,
             const Integer_3d<intType_ta>&                                                               nodeSpace,
             const Integer_3d<intType_ta>&                                                               voxSpace,
             const Vec_3d<double>&                                                                       spacingData,
             const Vec_3d<double>&                                                                       origin)
{
    using namespace internal_implicit::ns_help_write_vti;

    if (!(voxSpace == (nodeSpace - 1))) {
        NeonException exception("ioToVTI");
        exception << "Inconsistent data. Node space dimension should be bigger than voxel of one unit.";
        NEON_THROW(exception);
    }

    auto extendedSpace = voxSpace;

    std::string wholeExtent = std::string("0 ") + std::to_string(extendedSpace.x) + std::string(" ") + std::string("0 ") + std::to_string(extendedSpace.y) + std::string(" ") + std::string("0 ") + std::to_string(extendedSpace.z) + std::string(" ");
    std::string spacing = std::to_string(spacingData.x) + std::string(" ") + std::to_string(spacingData.y) + std::string(" ") + std::to_string(spacingData.z) + std::string(" ");

    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::string     msg = std::string("[VoxelGrid::WriteToBin] File ") + filename + std::string(" could not be open!!!");
        NeonException   exception("ioToVTI");
        exception << msg;
        NEON_THROW(exception);
    }

    try {
        out << "<?xml version=\"1.0\"?>" << std::endl;
        out << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        out << std::string("<ImageData WholeExtent=\"") + wholeExtent + std::string("\" Origin=\"") + std::to_string(origin.x) + std::string(" ") + std::to_string(origin.y) + std::string(" ") + std::to_string(origin.z) + std::string("\" Spacing=\"") + spacing + std::string("\">\n");

        writePieceExtent<real_tt>(out,
                                  grids,
                                  nComponents,
                                  fieldName,
                                  isNodeFlags,
                                  nodeSpace,
                                  voxSpace);
    } catch (...) {
        std::string     msg = std::string("An error on file operations where encountered when writing field data");
        NeonException   exception("ioToVTI");
        exception << msg;
        NEON_THROW(exception);
    }
    out << std::string(" </ImageData>\n");
    out << std::string(" </VTKFile>\n");
}


template < typename intType_ta, class real_tt = double>
void ioToVTI(const std::vector<VtiInputData_t<real_tt, intType_ta>>& gridsInfo,
                const std::string&                                                 filename,
                const Integer_3d<intType_ta>&                                      nodeSpace,
                const Integer_3d<intType_ta>&                                      voxSpace,
                const Vec_3d<double>&                                              spacingData = Vec_3d<double>(1,1,1),
                const Vec_3d<double>&                                              origin = Vec_3d<double>(0,0,0))
{
    using namespace internal_implicit::ns_help_write_vti;
    if (!(voxSpace == (nodeSpace - 1))) {
        NeonException exception("ioToVTI");
        exception << "Inconsistent data. Node space dimension should be bigger than voxel of one unit.";
        NEON_THROW(exception);
    }

    auto extendedSpace = voxSpace;

    std::string wholeExtent = std::string("0 ") + std::to_string(extendedSpace.x) + std::string(" ") + std::string("0 ") + std::to_string(extendedSpace.y) + std::string(" ") + std::string("0 ") + std::to_string(extendedSpace.z) + std::string(" ");
    std::string spacing = std::to_string(spacingData.x) + std::string(" ") + std::to_string(spacingData.y) + std::string(" ") + std::to_string(spacingData.z) + std::string(" ");

    std::ofstream out(filename, std::ios::out | std::ios::binary);
    if (!out.is_open()) {
        std::string     msg = std::string("[VoxelGrid::WriteToBin] File ") + filename + std::string(" could not be open!!!");
        NeonException   exception("ioToVTI");
        exception << msg;
        NEON_THROW(exception);
    }

    try {
        out << "<?xml version=\"1.0\"?>" << std::endl;
        out << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        out << std::string("<ImageData WholeExtent=\"") + wholeExtent + std::string("\" Origin=\"") + std::to_string(origin.x) + std::string(" ") + std::to_string(origin.y) + std::string(" ") + std::to_string(origin.z) + std::string("\" Spacing=\"") + spacing + std::string("\">\n");

        writePieceExtent<real_tt, intType_ta>(out,
                                              gridsInfo,
                                              nodeSpace,
                                              voxSpace);
    } catch (...) {
        std::string     msg = std::string("An error on file operations where encountered when writing field data");
        NeonException   exception("ioToVTI");
        exception << msg;
        NEON_THROW(exception);
    }
    out << std::string(" </ImageData>\n");
    out << std::string(" </VTKFile>\n");
}



}  // namespace Neon
