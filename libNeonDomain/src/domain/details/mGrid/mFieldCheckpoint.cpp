#include "Neon/domain/details/mGrid/mGrid.h"

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>
using float16 = __half;

namespace Neon::domain::details::mGrid {
namespace {

template <typename T>
auto checkpointDtypeName() -> const char*
{
    if constexpr (std::is_same_v<T, int8_t>) {
        return "int8";
    }
    if constexpr (std::is_same_v<T, uint8_t>) {
        return "uint8";
    }
    if constexpr (std::is_same_v<T, bool>) {
        return "bool";
    }
    if constexpr (std::is_same_v<T, int32_t>) {
        return "int32";
    }
    if constexpr (std::is_same_v<T, uint32_t>) {
        return "uint32";
    }
    if constexpr (std::is_same_v<T, int64_t>) {
        return "int64";
    }
    if constexpr (std::is_same_v<T, uint64_t>) {
        return "uint64";
    }
    if constexpr (std::is_same_v<T, float>) {
        return "float32";
    }
    if constexpr (std::is_same_v<T, double>) {
        return "float64";
    }
    if constexpr (std::is_same_v<T, float16>) {
        return "float16";
    }
    return "unknown";
}

template <typename T>
auto npyDescr() -> const char*
{
    if constexpr (std::is_same_v<T, int8_t>) {
        return "|i1";
    }
    if constexpr (std::is_same_v<T, uint8_t>) {
        return "|u1";
    }
    if constexpr (std::is_same_v<T, bool>) {
        return "|b1";
    }
    if constexpr (std::is_same_v<T, int32_t>) {
        return "<i4";
    }
    if constexpr (std::is_same_v<T, uint32_t>) {
        return "<u4";
    }
    if constexpr (std::is_same_v<T, int64_t>) {
        return "<i8";
    }
    if constexpr (std::is_same_v<T, uint64_t>) {
        return "<u8";
    }
    if constexpr (std::is_same_v<T, float>) {
        return "<f4";
    }
    if constexpr (std::is_same_v<T, double>) {
        return "<f8";
    }
    if constexpr (std::is_same_v<T, float16>) {
        return "<f2";
    }
    return "unknown";
}

// std::vector<bool> is bit-packed, so bool values are staged as one byte each.
template <typename T>
using Storage = std::conditional_t<std::is_same_v<T, bool>, uint8_t, T>;

template <typename T>
auto parseDtype(const char* name) -> bool
{
    return std::strcmp(name, checkpointDtypeName<T>()) == 0;
}

auto topologyIdToJson(uint64_t topologyId, rapidjson::Document::AllocatorType& alloc) -> rapidjson::Value
{
    const std::string asString = std::to_string(topologyId);
    return rapidjson::Value(asString.c_str(), alloc);
}

auto topologyIdFromJson(const rapidjson::Value& value) -> uint64_t
{
    if (value.IsString()) {
        return std::strtoull(value.GetString(), nullptr, 10);
    }
    return value.GetUint64();
}

[[noreturn]] auto fail(const char* where, const std::string& message) -> void
{
    NeonException exp(where);
    exp << message;
    NEON_THROW(exp);
}

auto formatName(CheckpointFormat format) -> const char*
{
    switch (format) {
        case CheckpointFormat::bin:
            return "bin";
        case CheckpointFormat::npy:
            return "npy";
        case CheckpointFormat::npz:
            return "npz";
    }
    return "bin";
}

auto formatFromName(const std::string& name, const char* where) -> CheckpointFormat
{
    if (name == "bin") {
        return CheckpointFormat::bin;
    }
    if (name == "npy") {
        return CheckpointFormat::npy;
    }
    if (name == "npz") {
        return CheckpointFormat::npz;
    }
    fail(where, "Unknown checkpoint format '" + name + "'");
}

template <typename U>
auto putLE(std::string& out, U value) -> void
{
    for (size_t i = 0; i < sizeof(U); ++i) {
        out.push_back(static_cast<char>((static_cast<uint64_t>(value) >> (8 * i)) & 0xFF));
    }
}

template <typename U>
auto getLE(const char* p) -> U
{
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(U); ++i) {
        value |= static_cast<uint64_t>(static_cast<unsigned char>(p[i])) << (8 * i);
    }
    return static_cast<U>(value);
}

auto crc32Update(uint32_t crc, const char* data, uint64_t size) -> uint32_t
{
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            t[i] = c;
        }
        return t;
    }();
    crc = ~crc;
    for (uint64_t i = 0; i < size; ++i) {
        crc = table[(crc ^ static_cast<unsigned char>(data[i])) & 0xFFu] ^ (crc >> 8);
    }
    return ~crc;
}

// Magic, version 1.0, header length and the dict, padded so the data starts on a 64-byte boundary.
auto npyHeader(const std::string& descr, const std::vector<uint64_t>& shape) -> std::string
{
    std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            dict += ", ";
        }
        dict += std::to_string(shape[i]);
    }
    if (shape.size() == 1) {
        dict += ",";
    }
    dict += "), }";
    const size_t unpadded = 10 + dict.size() + 1;
    dict.append((64 - unpadded % 64) % 64, ' ');
    dict.push_back('\n');

    std::string header;
    header.push_back(static_cast<char>(0x93));
    header += "NUMPY";
    header.push_back(1);
    header.push_back(0);
    putLE<uint16_t>(header, static_cast<uint16_t>(dict.size()));
    return header + dict;
}

struct NpyInfo
{
    std::string           descr;
    bool                  fortranOrder = false;
    std::vector<uint64_t> shape;
};

auto parseNpyDict(const std::string& dict, const char* where) -> NpyInfo
{
    NpyInfo info;

    const size_t descrKey = dict.find("'descr'");
    if (descrKey == std::string::npos) {
        fail(where, "npy header has no 'descr'");
    }
    const size_t descrOpen = dict.find('\'', dict.find(':', descrKey) + 1);
    const size_t descrClose = dict.find('\'', descrOpen + 1);
    if (descrOpen == std::string::npos || descrClose == std::string::npos) {
        fail(where, "npy header has a malformed 'descr'");
    }
    info.descr = dict.substr(descrOpen + 1, descrClose - descrOpen - 1);

    const size_t fortranKey = dict.find("'fortran_order'");
    if (fortranKey == std::string::npos) {
        fail(where, "npy header has no 'fortran_order'");
    }
    size_t fortranValue = dict.find(':', fortranKey) + 1;
    while (fortranValue < dict.size() && dict[fortranValue] == ' ') {
        ++fortranValue;
    }
    info.fortranOrder = dict.compare(fortranValue, 4, "True") == 0;

    const size_t shapeKey = dict.find("'shape'");
    const size_t shapeOpen = dict.find('(', shapeKey);
    const size_t shapeClose = dict.find(')', shapeOpen);
    if (shapeKey == std::string::npos || shapeOpen == std::string::npos || shapeClose == std::string::npos) {
        fail(where, "npy header has a malformed 'shape'");
    }
    const std::string dims = dict.substr(shapeOpen + 1, shapeClose - shapeOpen - 1);
    size_t            start = 0;
    while (start < dims.size()) {
        size_t end = dims.find(',', start);
        if (end == std::string::npos) {
            end = dims.size();
        }
        const std::string token = dims.substr(start, end - start);
        if (token.find_first_not_of(' ') != std::string::npos) {
            info.shape.push_back(std::strtoull(token.c_str(), nullptr, 10));
        }
        start = end + 1;
    }
    return info;
}

auto shapeToString(const std::vector<uint64_t>& shape) -> std::string
{
    std::string s = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        s += (i > 0 ? ", " : "") + std::to_string(shape[i]);
    }
    return s + ")";
}

auto descrMatches(const std::string& found, const std::string& expected) -> bool
{
    if (found.size() != expected.size() || found.empty()) {
        return false;
    }
    // '<', '|' and '=' all mean little-endian/native on the platforms Neon supports; '>' does not.
    if (found[0] == '>' && expected[0] != '|') {
        return false;
    }
    return found.compare(1, std::string::npos, expected, 1, std::string::npos) == 0;
}

/**
 * Read one .npy array from the current stream position into dest, checking dtype and shape.
 * Returns the CRC-32 of every byte consumed (header and payload), for npz entry verification.
 */
auto readNpy(std::istream&                in,
             uint64_t                     availableBytes,
             const std::string&           expectedDescr,
             const std::vector<uint64_t>& expectedShape,
             char*                        dest,
             uint64_t                     destBytes,
             const std::string&           what,
             const char*                  where) -> uint32_t
{
    char preamble[12];
    in.read(preamble, 10);
    if (!in || static_cast<unsigned char>(preamble[0]) != 0x93 || std::memcmp(preamble + 1, "NUMPY", 5) != 0) {
        fail(where, what + " is not a .npy array");
    }
    const int major = static_cast<unsigned char>(preamble[6]);
    size_t    preambleSize = 10;
    uint64_t  dictSize = 0;
    if (major == 1) {
        dictSize = getLE<uint16_t>(preamble + 8);
    } else if (major == 2 || major == 3) {
        in.read(preamble + 10, 2);
        preambleSize = 12;
        dictSize = getLE<uint32_t>(preamble + 8);
    } else {
        fail(where, what + " uses unsupported .npy version " + std::to_string(major));
    }
    std::string dict(dictSize, '\0');
    in.read(dict.data(), static_cast<std::streamsize>(dictSize));
    if (!in) {
        fail(where, what + " has a truncated .npy header");
    }

    const NpyInfo info = parseNpyDict(dict, where);
    if (!descrMatches(info.descr, expectedDescr)) {
        fail(where, what + " has dtype '" + info.descr + "', expected '" + expectedDescr + "'");
    }
    if (info.fortranOrder) {
        fail(where, what + " is Fortran-ordered; save it with C order");
    }
    if (info.shape != expectedShape) {
        fail(where, what + " has shape " + shapeToString(info.shape) + ", expected " + shapeToString(expectedShape));
    }
    if (preambleSize + dictSize + destBytes > availableBytes) {
        fail(where, what + " is truncated");
    }
    in.read(dest, static_cast<std::streamsize>(destBytes));
    if (!in) {
        fail(where, what + " is truncated");
    }

    uint32_t crc = crc32Update(0, preamble, preambleSize);
    crc = crc32Update(crc, dict.data(), dictSize);
    return crc32Update(crc, dest, destBytes);
}

struct NpzEntry
{
    std::string name;
    std::string header;
    const char* data;
    uint64_t    dataBytes;
};

/**
 * Write a stored (uncompressed) zip archive. ZIP64 records are always written, as numpy's
 * savez does, so a level array may exceed 4 GiB.
 */
auto writeNpz(const std::string& path, const std::vector<NpzEntry>& entries, const char* where) -> void
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        fail(where, "Unable to open " + path);
    }
    constexpr uint16_t kVersion = 45;
    constexpr uint16_t kDosDate = (1 << 5) | 1;  // 1980-01-01

    struct Written
    {
        uint32_t crc;
        uint64_t size;
        uint64_t offset;
    };
    std::vector<Written> written;
    uint64_t             position = 0;

    for (auto const& e : entries) {
        const uint64_t size = e.header.size() + e.dataBytes;
        uint32_t       crc = crc32Update(0, e.header.data(), e.header.size());
        crc = crc32Update(crc, e.data, e.dataBytes);

        std::string local;
        putLE<uint32_t>(local, 0x04034b50);
        putLE<uint16_t>(local, kVersion);
        putLE<uint16_t>(local, 0);
        putLE<uint16_t>(local, 0);
        putLE<uint16_t>(local, 0);
        putLE<uint16_t>(local, kDosDate);
        putLE<uint32_t>(local, crc);
        putLE<uint32_t>(local, 0xFFFFFFFFu);
        putLE<uint32_t>(local, 0xFFFFFFFFu);
        putLE<uint16_t>(local, static_cast<uint16_t>(e.name.size()));
        putLE<uint16_t>(local, 20);
        local += e.name;
        putLE<uint16_t>(local, 0x0001);
        putLE<uint16_t>(local, 16);
        putLE<uint64_t>(local, size);
        putLE<uint64_t>(local, size);

        out.write(local.data(), static_cast<std::streamsize>(local.size()));
        out.write(e.header.data(), static_cast<std::streamsize>(e.header.size()));
        out.write(e.data, static_cast<std::streamsize>(e.dataBytes));
        written.push_back({crc, size, position});
        position += local.size() + size;
    }

    std::string central;
    for (size_t i = 0; i < entries.size(); ++i) {
        putLE<uint32_t>(central, 0x02014b50);
        putLE<uint16_t>(central, kVersion);
        putLE<uint16_t>(central, kVersion);
        putLE<uint16_t>(central, 0);
        putLE<uint16_t>(central, 0);
        putLE<uint16_t>(central, 0);
        putLE<uint16_t>(central, kDosDate);
        putLE<uint32_t>(central, written[i].crc);
        putLE<uint32_t>(central, 0xFFFFFFFFu);
        putLE<uint32_t>(central, 0xFFFFFFFFu);
        putLE<uint16_t>(central, static_cast<uint16_t>(entries[i].name.size()));
        putLE<uint16_t>(central, 28);
        putLE<uint16_t>(central, 0);
        putLE<uint16_t>(central, 0);
        putLE<uint16_t>(central, 0);
        putLE<uint32_t>(central, 0);
        putLE<uint32_t>(central, 0xFFFFFFFFu);
        central += entries[i].name;
        putLE<uint16_t>(central, 0x0001);
        putLE<uint16_t>(central, 24);
        putLE<uint64_t>(central, written[i].size);
        putLE<uint64_t>(central, written[i].size);
        putLE<uint64_t>(central, written[i].offset);
    }
    const uint64_t centralOffset = position;
    const uint64_t zip64EndOffset = centralOffset + central.size();

    std::string end;
    putLE<uint32_t>(end, 0x06064b50);
    putLE<uint64_t>(end, 44);
    putLE<uint16_t>(end, kVersion);
    putLE<uint16_t>(end, kVersion);
    putLE<uint32_t>(end, 0);
    putLE<uint32_t>(end, 0);
    putLE<uint64_t>(end, entries.size());
    putLE<uint64_t>(end, entries.size());
    putLE<uint64_t>(end, central.size());
    putLE<uint64_t>(end, centralOffset);

    putLE<uint32_t>(end, 0x07064b50);
    putLE<uint32_t>(end, 0);
    putLE<uint64_t>(end, zip64EndOffset);
    putLE<uint32_t>(end, 1);

    putLE<uint32_t>(end, 0x06054b50);
    putLE<uint16_t>(end, 0);
    putLE<uint16_t>(end, 0);
    putLE<uint16_t>(end, static_cast<uint16_t>(entries.size()));
    putLE<uint16_t>(end, static_cast<uint16_t>(entries.size()));
    putLE<uint32_t>(end, 0xFFFFFFFFu);
    putLE<uint32_t>(end, 0xFFFFFFFFu);
    putLE<uint16_t>(end, 0);

    out.write(central.data(), static_cast<std::streamsize>(central.size()));
    out.write(end.data(), static_cast<std::streamsize>(end.size()));
    if (!out) {
        fail(where, "Failed writing " + path);
    }
}

struct ZipEntryInfo
{
    uint16_t flags = 0;
    uint16_t method = 0;
    uint32_t crc = 0;
    uint64_t compressedSize = 0;
    uint64_t localOffset = 0;
};

auto readZipDirectory(std::ifstream& in, const std::string& path, const char* where) -> std::map<std::string, ZipEntryInfo>
{
    in.seekg(0, std::ios::end);
    const uint64_t fileSize = static_cast<uint64_t>(in.tellg());
    const uint64_t tailSize = std::min<uint64_t>(fileSize, 22 + 65535);
    std::string    tail(tailSize, '\0');
    in.seekg(static_cast<std::streamoff>(fileSize - tailSize), std::ios::beg);
    in.read(tail.data(), static_cast<std::streamsize>(tailSize));

    int64_t eocd = -1;
    for (int64_t i = static_cast<int64_t>(tailSize) - 22; i >= 0; --i) {
        if (getLE<uint32_t>(tail.data() + i) == 0x06054b50u) {
            eocd = i;
            break;
        }
    }
    if (eocd < 0) {
        fail(where, path + " is not a zip/npz archive");
    }
    const uint64_t eocdOffset = fileSize - tailSize + static_cast<uint64_t>(eocd);
    uint64_t       numEntries = getLE<uint16_t>(tail.data() + eocd + 10);
    uint64_t       centralSize = getLE<uint32_t>(tail.data() + eocd + 12);
    uint64_t       centralOffset = getLE<uint32_t>(tail.data() + eocd + 16);

    if (numEntries == 0xFFFF || centralSize == 0xFFFFFFFFu || centralOffset == 0xFFFFFFFFu) {
        char locator[20];
        in.seekg(static_cast<std::streamoff>(eocdOffset - 20), std::ios::beg);
        in.read(locator, 20);
        if (!in || getLE<uint32_t>(locator) != 0x07064b50u) {
            fail(where, path + " has no ZIP64 end-of-directory locator");
        }
        char record[56];
        in.seekg(static_cast<std::streamoff>(getLE<uint64_t>(locator + 8)), std::ios::beg);
        in.read(record, 56);
        if (!in || getLE<uint32_t>(record) != 0x06064b50u) {
            fail(where, path + " has a corrupt ZIP64 end-of-directory record");
        }
        numEntries = getLE<uint64_t>(record + 32);
        centralSize = getLE<uint64_t>(record + 40);
        centralOffset = getLE<uint64_t>(record + 48);
    }

    std::string central(centralSize, '\0');
    in.seekg(static_cast<std::streamoff>(centralOffset), std::ios::beg);
    in.read(central.data(), static_cast<std::streamsize>(centralSize));
    if (!in) {
        fail(where, path + " has a truncated central directory");
    }

    std::map<std::string, ZipEntryInfo> entries;
    size_t                              p = 0;
    for (uint64_t n = 0; n < numEntries; ++n) {
        if (p + 46 > central.size() || getLE<uint32_t>(central.data() + p) != 0x02014b50u) {
            fail(where, path + " has a corrupt central directory");
        }
        const char*  h = central.data() + p;
        ZipEntryInfo info;
        info.flags = getLE<uint16_t>(h + 8);
        info.method = getLE<uint16_t>(h + 10);
        info.crc = getLE<uint32_t>(h + 16);
        info.compressedSize = getLE<uint32_t>(h + 20);
        uint64_t       uncompressedSize = getLE<uint32_t>(h + 24);
        const uint16_t nameLength = getLE<uint16_t>(h + 28);
        const uint16_t extraLength = getLE<uint16_t>(h + 30);
        const uint16_t commentLength = getLE<uint16_t>(h + 32);
        info.localOffset = getLE<uint32_t>(h + 42);
        const std::string name(h + 46, nameLength);

        const char* extra = h + 46 + nameLength;
        for (size_t e = 0; e + 4 <= static_cast<size_t>(extraLength);) {
            const uint16_t id = getLE<uint16_t>(extra + e);
            const uint16_t size = getLE<uint16_t>(extra + e + 2);
            if (id == 0x0001) {
                const char* q = extra + e + 4;
                if (uncompressedSize == 0xFFFFFFFFu) {
                    uncompressedSize = getLE<uint64_t>(q);
                    q += 8;
                }
                if (info.compressedSize == 0xFFFFFFFFu) {
                    info.compressedSize = getLE<uint64_t>(q);
                    q += 8;
                }
                if (info.localOffset == 0xFFFFFFFFu) {
                    info.localOffset = getLE<uint64_t>(q);
                }
            }
            e += 4 + size;
        }
        entries[name] = info;
        p += 46 + nameLength + extraLength + commentLength;
    }
    return entries;
}

template <typename SBlock>
auto checkpointBlocksPerLevel(mGrid<SBlock>& grid) -> std::vector<uint64_t>
{
    std::vector<uint64_t> counts;
    for (int l = 0; l < grid.getDescriptor().getDepth(); ++l) {
        uint64_t n = 0;
        grid(l).helpGetPartitioner1D().forEachSeq(SetIdx(0), [&](const uint32_t, const Neon::int32_3d, auto) { ++n; });
        counts.push_back(n);
    }
    return counts;
}

template <typename SBlock>
auto arrayShape(uint64_t numBlocks, int cardinality) -> std::vector<uint64_t>
{
    return {numBlocks, SBlock::memBlockSizeZ, SBlock::memBlockSizeY, SBlock::memBlockSizeX, static_cast<uint64_t>(cardinality)};
}

}  // namespace

template <typename T, int C, typename SBlock>
auto mField<T, C, SBlock>::ioToCheckpoint(const std::string& basePath,
                                          const std::string& name,
                                          CheckpointFormat   format) const -> void
{
    constexpr const char* kWhere = "mField::ioToCheckpoint";
    auto&                 grid = *mData->grid;

    const int                   card = (*this)(0).getCardinality();
    const std::vector<uint64_t> blocksPerLevel = checkpointBlocksPerLevel(grid);
    const uint64_t              numElements = grid.getCheckpointFieldNumElementsPerCard() * static_cast<uint64_t>(card);
    std::vector<Storage<T>>     buffer(numElements, Storage<T>(0));

    uint64_t writeIdx = 0;
    for (int l = 0; l < grid.getDescriptor().getDepth(); ++l) {
        const int   voxelSpacing = grid.getDescriptor().getSpacing(l - 1);
        auto&       bgrid = (*mData->grid)(l);
        auto&       xfld = const_cast<Self*>(this)->operator()(l);
        const auto& part = xfld.getPartition(Neon::Execution::host, SetIdx(0), Neon::DataView::STANDARD);
        SetIdx      devID(0);

        bgrid.helpGetPartitioner1D().forEachSeq(devID, [&](const uint32_t blockIdx, const Neon::int32_3d memBlockOrigin, auto /*byPartition*/) {
            Neon::index_3d blockOrigin = memBlockOrigin;
            blockOrigin.x *= SBlock::memBlockSizeX * voxelSpacing;
            blockOrigin.y *= SBlock::memBlockSizeY * voxelSpacing;
            blockOrigin.z *= SBlock::memBlockSizeZ * voxelSpacing;

            for (uint32_t z = 0; z < SBlock::memBlockSizeZ; ++z) {
                for (uint32_t y = 0; y < SBlock::memBlockSizeY; ++y) {
                    for (uint32_t x = 0; x < SBlock::memBlockSizeX; ++x) {
                        const Neon::index_3d voxelGlobalID(static_cast<int>(x * static_cast<uint32_t>(voxelSpacing) + blockOrigin.x),
                                                           static_cast<int>(y * static_cast<uint32_t>(voxelSpacing) + blockOrigin.y),
                                                           static_cast<int>(z * static_cast<uint32_t>(voxelSpacing) + blockOrigin.z));
                        const bool active = bgrid.isInsideDomain(voxelGlobalID);
                        Idx        idx(blockIdx, static_cast<int8_t>(x), static_cast<int8_t>(y), static_cast<int8_t>(z));
                        for (int c = 0; c < card; ++c) {
                            if (active) {
                                buffer[writeIdx] = part(idx, c);
                            } else {
                                buffer[writeIdx] = Storage<T>(0);
                            }
                            ++writeIdx;
                        }
                    }
                }
            }
        });
    }

    const char*    bytes = reinterpret_cast<const char*>(buffer.data());
    const uint64_t numBytes = numElements * sizeof(Storage<T>);
    const uint64_t bytesPerBlock = SBlock::memBlockCountElements * static_cast<uint64_t>(card) * sizeof(Storage<T>);
    const uint64_t totalBlocks = numElements / (SBlock::memBlockCountElements * static_cast<uint64_t>(card));

    if (format == CheckpointFormat::bin || format == CheckpointFormat::npy) {
        const std::string dataPath = basePath + "." + formatName(format);
        std::ofstream     out(dataPath, std::ios::binary);
        if (!out) {
            fail(kWhere, "Unable to open " + dataPath);
        }
        if (format == CheckpointFormat::npy) {
            const std::string header = npyHeader(npyDescr<T>(), arrayShape<SBlock>(totalBlocks, card));
            out.write(header.data(), static_cast<std::streamsize>(header.size()));
        }
        out.write(bytes, static_cast<std::streamsize>(numBytes));
        if (!out) {
            fail(kWhere, "Failed writing " + dataPath);
        }
    } else {
        std::vector<NpzEntry> entries;
        uint64_t              offset = 0;
        for (size_t l = 0; l < blocksPerLevel.size(); ++l) {
            const uint64_t levelBytes = blocksPerLevel[l] * bytesPerBlock;
            entries.push_back({"level_" + std::to_string(l) + ".npy",
                               npyHeader(npyDescr<T>(), arrayShape<SBlock>(blocksPerLevel[l], card)),
                               bytes + offset,
                               levelBytes});
            offset += levelBytes;
        }
        writeNpz(basePath + ".npz", entries, kWhere);
    }

    rapidjson::Document doc;
    doc.SetObject();
    auto& alloc = doc.GetAllocator();
    doc.AddMember("name", rapidjson::Value(name.c_str(), alloc), alloc);
    doc.AddMember("cardinality", card, alloc);
    doc.AddMember("dtype", rapidjson::Value(checkpointDtypeName<T>(), alloc), alloc);
    doc.AddMember("num_elements", numElements, alloc);
    doc.AddMember("topology_id", topologyIdToJson(grid.getTopologyId(), alloc), alloc);
    doc.AddMember("format", rapidjson::Value(formatName(format), alloc), alloc);
    doc.AddMember("layout", "block, z, y, x, component", alloc);
    {
        rapidjson::Value blockShape(rapidjson::kArrayType);
        blockShape.PushBack(SBlock::memBlockSizeZ, alloc);
        blockShape.PushBack(SBlock::memBlockSizeY, alloc);
        blockShape.PushBack(SBlock::memBlockSizeX, alloc);
        doc.AddMember("block_shape_zyx", blockShape, alloc);
    }
    {
        rapidjson::Value counts(rapidjson::kArrayType);
        for (auto n : blocksPerLevel) {
            counts.PushBack(n, alloc);
        }
        doc.AddMember("num_blocks_per_level", counts, alloc);
    }

    rapidjson::StringBuffer                    sb;
    rapidjson::Writer<rapidjson::StringBuffer> wr(sb);
    doc.Accept(wr);

    const std::string jsonPath = basePath + ".json";
    std::ofstream     jout(jsonPath);
    if (!jout) {
        fail(kWhere, "Unable to open " + jsonPath);
    }
    jout << sb.GetString();
}

template <typename T, int C, typename SBlock>
auto mField<T, C, SBlock>::ioFromCheckpoint(const std::string& basePath) -> void
{
    constexpr const char* kWhere = "mField::ioFromCheckpoint";
    const std::string     jsonPath = basePath + ".json";
    std::ifstream         jin(jsonPath);
    if (!jin) {
        fail(kWhere, "Unable to open " + jsonPath);
    }
    std::string json((std::istreambuf_iterator<char>(jin)), std::istreambuf_iterator<char>());

    rapidjson::Document doc;
    doc.Parse(json.c_str());
    if (doc.HasParseError()) {
        fail(kWhere, "Invalid JSON in " + jsonPath);
    }

    if (!parseDtype<T>(doc["dtype"].GetString())) {
        fail(kWhere, "dtype mismatch");
    }

    const int card = (*this)(0).getCardinality();
    if (doc["cardinality"].GetInt() != card) {
        fail(kWhere, "cardinality mismatch");
    }

    auto& grid = *mData->grid;
    if (topologyIdFromJson(doc["topology_id"]) != grid.getTopologyId()) {
        fail(kWhere, "topology_id mismatch");
    }

    const uint64_t numElements = doc["num_elements"].GetUint64();
    if (numElements != grid.getCheckpointFieldNumElementsPerCard() * static_cast<uint64_t>(card)) {
        fail(kWhere, "num_elements mismatch");
    }

    const CheckpointFormat format = doc.HasMember("format") ? formatFromName(doc["format"].GetString(), kWhere)
                                                            : CheckpointFormat::bin;
    const std::vector<uint64_t> blocksPerLevel = checkpointBlocksPerLevel(grid);
    std::vector<Storage<T>>     buffer(numElements);
    char*                       bytes = reinterpret_cast<char*>(buffer.data());
    const uint64_t              numBytes = numElements * sizeof(Storage<T>);
    const uint64_t              bytesPerBlock = SBlock::memBlockCountElements * static_cast<uint64_t>(card) * sizeof(Storage<T>);
    const uint64_t              totalBlocks = numElements / (SBlock::memBlockCountElements * static_cast<uint64_t>(card));

    const std::string dataPath = basePath + "." + formatName(format);
    std::ifstream     in(dataPath, std::ios::binary);
    if (!in) {
        fail(kWhere, "Unable to open " + dataPath);
    }

    if (format == CheckpointFormat::bin) {
        in.read(bytes, static_cast<std::streamsize>(numBytes));
        if (!in) {
            fail(kWhere, dataPath + " is truncated");
        }
    } else if (format == CheckpointFormat::npy) {
        in.seekg(0, std::ios::end);
        const uint64_t fileSize = static_cast<uint64_t>(in.tellg());
        in.seekg(0, std::ios::beg);
        readNpy(in, fileSize, npyDescr<T>(), arrayShape<SBlock>(totalBlocks, card), bytes, numBytes, dataPath, kWhere);
    } else {
        const auto entries = readZipDirectory(in, dataPath, kWhere);
        uint64_t   offset = 0;
        for (size_t l = 0; l < blocksPerLevel.size(); ++l) {
            const std::string entryName = "level_" + std::to_string(l) + ".npy";
            const std::string what = dataPath + ":" + entryName;
            auto              it = entries.find(entryName);
            if (it == entries.end()) {
                fail(kWhere, dataPath + " has no array 'level_" + std::to_string(l) + "'");
            }
            const ZipEntryInfo& info = it->second;
            if ((info.flags & 1u) != 0) {
                fail(kWhere, what + " is encrypted");
            }
            if (info.method != 0) {
                fail(kWhere, what + " is compressed; only uncompressed npz (numpy.savez) is supported");
            }
            char local[30];
            in.seekg(static_cast<std::streamoff>(info.localOffset), std::ios::beg);
            in.read(local, 30);
            if (!in || getLE<uint32_t>(local) != 0x04034b50u) {
                fail(kWhere, what + " has a corrupt local header");
            }
            const uint64_t dataStart = info.localOffset + 30 + getLE<uint16_t>(local + 26) + getLE<uint16_t>(local + 28);
            in.seekg(static_cast<std::streamoff>(dataStart), std::ios::beg);

            const uint64_t levelBytes = blocksPerLevel[l] * bytesPerBlock;
            const uint32_t crc = readNpy(in, info.compressedSize, npyDescr<T>(), arrayShape<SBlock>(blocksPerLevel[l], card),
                                         bytes + offset, levelBytes, what, kWhere);
            if (crc != info.crc) {
                fail(kWhere, what + " failed its CRC-32 check");
            }
            offset += levelBytes;
        }
    }

    uint64_t readIdx = 0;
    for (int l = 0; l < grid.getDescriptor().getDepth(); ++l) {
        const int voxelSpacing = grid.getDescriptor().getSpacing(l - 1);
        auto&     bgrid = (*mData->grid)(l);
        auto&     xfld = (*this)(l);
        auto&     part = xfld.getPartition(Neon::Execution::host, SetIdx(0), Neon::DataView::STANDARD);
        SetIdx    devID(0);

        bgrid.helpGetPartitioner1D().forEachSeq(devID, [&](const uint32_t blockIdx, const Neon::int32_3d memBlockOrigin, auto /*byPartition*/) {
            Neon::index_3d blockOrigin = memBlockOrigin;
            blockOrigin.x *= SBlock::memBlockSizeX * voxelSpacing;
            blockOrigin.y *= SBlock::memBlockSizeY * voxelSpacing;
            blockOrigin.z *= SBlock::memBlockSizeZ * voxelSpacing;

            for (uint32_t z = 0; z < SBlock::memBlockSizeZ; ++z) {
                for (uint32_t y = 0; y < SBlock::memBlockSizeY; ++y) {
                    for (uint32_t x = 0; x < SBlock::memBlockSizeX; ++x) {
                        const Neon::index_3d voxelGlobalID(static_cast<int>(x * static_cast<uint32_t>(voxelSpacing) + blockOrigin.x),
                                                           static_cast<int>(y * static_cast<uint32_t>(voxelSpacing) + blockOrigin.y),
                                                           static_cast<int>(z * static_cast<uint32_t>(voxelSpacing) + blockOrigin.z));
                        const bool active = bgrid.isInsideDomain(voxelGlobalID);
                        Idx        idx(blockIdx, static_cast<int8_t>(x), static_cast<int8_t>(y), static_cast<int8_t>(z));
                        for (int c = 0; c < card; ++c) {
                            if (active) {
                                part(idx, c) = static_cast<T>(buffer[readIdx]);
                            }
                            ++readIdx;
                        }
                    }
                }
            }
        });
        xfld.updateDeviceData(0);
    }
}

using SB = Neon::domain::details::StaticBlockDefault;

#define INST_FIELD(T)                                                                                                          \
    template auto mField<T, 0, SB>::ioToCheckpoint(const std::string&, const std::string&, CheckpointFormat) const -> void;    \
    template auto mField<T, 0, SB>::ioFromCheckpoint(const std::string&) -> void;

INST_FIELD(int8_t)
INST_FIELD(uint8_t)
INST_FIELD(bool)
INST_FIELD(int32_t)
INST_FIELD(uint32_t)
INST_FIELD(int64_t)
INST_FIELD(uint64_t)
INST_FIELD(float)
INST_FIELD(double)
INST_FIELD(float16)

#undef INST_FIELD

}  // namespace Neon::domain::details::mGrid
