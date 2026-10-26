#include "Neon/domain/details/mGrid/mGrid.h"

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
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

}  // namespace

template <typename T, int C, typename SBlock>
auto mField<T, C, SBlock>::ioToCheckpoint(const std::string& basePath, const std::string& name) const -> void
{
    auto& grid = *mData->grid;
    grid.getTopologyId();

    const int        card = (*this)(0).getCardinality();
    const uint64_t   numElements = grid.getCheckpointFieldNumElementsPerCard() * static_cast<uint64_t>(card);
    std::vector<T>   buffer(numElements, T(0));

    uint64_t writeIdx = 0;
    for (int l = 0; l < grid.getDescriptor().getDepth(); ++l) {
        const int voxelSpacing = grid.getDescriptor().getSpacing(l - 1);
        auto&     bgrid = (*mData->grid)(l);
        auto& xfld = const_cast<Self*>(this)->operator()(l);
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
                                buffer[writeIdx] = T(0);
                            }
                            ++writeIdx;
                        }
                    }
                }
            }
        });
    }

    const std::string binPath = basePath + ".bin";
    std::ofstream     bout(binPath, std::ios::binary);
    if (!bout) {
        NeonException exp("mField::ioToCheckpoint");
        exp << "Unable to open " << binPath;
        NEON_THROW(exp);
    }
    if constexpr (std::is_same_v<T, bool>) {
        std::vector<uint8_t> bytes(numElements);
        for (uint64_t i = 0; i < numElements; ++i) {
            bytes[i] = buffer[i] ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
        }
        bout.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    } else {
        bout.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(T)));
    }

    rapidjson::Document doc;
    doc.SetObject();
    auto& alloc = doc.GetAllocator();
    doc.AddMember("name", rapidjson::Value(name.c_str(), alloc), alloc);
    doc.AddMember("cardinality", card, alloc);
    doc.AddMember("dtype", rapidjson::Value(checkpointDtypeName<T>(), alloc), alloc);
    doc.AddMember("num_elements", numElements, alloc);
    doc.AddMember("topology_id", topologyIdToJson(grid.getTopologyId(), alloc), alloc);

    rapidjson::StringBuffer                    sb;
    rapidjson::Writer<rapidjson::StringBuffer> wr(sb);
    doc.Accept(wr);

    const std::string jsonPath = basePath + ".json";
    std::ofstream     jout(jsonPath);
    if (!jout) {
        NeonException exp("mField::ioToCheckpoint");
        exp << "Unable to open " << jsonPath;
        NEON_THROW(exp);
    }
    jout << sb.GetString();
}

template <typename T, int C, typename SBlock>
auto mField<T, C, SBlock>::ioFromCheckpoint(const std::string& basePath) -> void
{
    const std::string jsonPath = basePath + ".json";
    std::ifstream     jin(jsonPath);
    if (!jin) {
        NeonException exp("mField::ioFromCheckpoint");
        exp << "Unable to open " << jsonPath;
        NEON_THROW(exp);
    }
    std::string json((std::istreambuf_iterator<char>(jin)), std::istreambuf_iterator<char>());

    rapidjson::Document doc;
    doc.Parse(json.c_str());
    if (doc.HasParseError()) {
        NeonException exp("mField::ioFromCheckpoint");
        exp << "Invalid JSON in " << jsonPath;
        NEON_THROW(exp);
    }

    if (!parseDtype<T>(doc["dtype"].GetString())) {
        NeonException exp("mField::ioFromCheckpoint");
        exp << "dtype mismatch";
        NEON_THROW(exp);
    }

    const int card = (*this)(0).getCardinality();
    if (doc["cardinality"].GetInt() != card) {
        NeonException exp("mField::ioFromCheckpoint");
        exp << "cardinality mismatch";
        NEON_THROW(exp);
    }

    auto& grid = *mData->grid;
    if (topologyIdFromJson(doc["topology_id"]) != grid.getTopologyId()) {
        NeonException exp("mField::ioFromCheckpoint");
        exp << "topology_id mismatch";
        NEON_THROW(exp);
    }

    const uint64_t numElements = doc["num_elements"].GetUint64();
    if (numElements != grid.getCheckpointFieldNumElementsPerCard() * static_cast<uint64_t>(card)) {
        NeonException exp("mField::ioFromCheckpoint");
        exp << "num_elements mismatch";
        NEON_THROW(exp);
    }

    std::vector<T> buffer(numElements);
    const std::string binPath = basePath + ".bin";
    std::ifstream     bin(binPath, std::ios::binary);
    if (!bin) {
        NeonException exp("mField::ioFromCheckpoint");
        exp << "Unable to open " << binPath;
        NEON_THROW(exp);
    }
    if constexpr (std::is_same_v<T, bool>) {
        std::vector<uint8_t> bytes(numElements);
        bin.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        for (uint64_t i = 0; i < numElements; ++i) {
            buffer[i] = bytes[i] != 0;
        }
    } else {
        bin.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(T)));
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
                                part(idx, c) = buffer[readIdx];
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
    template auto mField<T, 0, SB>::ioToCheckpoint(const std::string&, const std::string&) const -> void;                      \
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
