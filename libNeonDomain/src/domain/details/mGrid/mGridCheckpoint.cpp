#include "Neon/domain/details/mGrid/mGrid.h"

#include "Neon/core/tools/git_sha1.h"

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace Neon::domain::details::mGrid {
namespace {

constexpr char     kMagic[8] = {'N', 'E', 'O', 'N', 'M', 'G', 'R', 'D'};
constexpr uint32_t kFormatVersion = 1;

auto fnv1a64(const uint8_t* data, size_t len, uint64_t hash = 14695981039346656037ULL) -> uint64_t
{
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

auto writeU64LE(std::ostream& out, uint64_t v) -> void
{
    for (int i = 0; i < 8; ++i) {
        out.put(static_cast<char>((v >> (8 * i)) & 0xFF));
    }
}

auto readU64LE(std::istream& in, uint64_t& v) -> void
{
    v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(static_cast<unsigned char>(in.get())) << (8 * i);
    }
}

auto levelShiftForSpacing(int voxelSpacing) -> int
{
    int s = 0;
    while ((1 << s) < voxelSpacing) {
        ++s;
    }
    return s;
}

template <typename SBlock>
auto collectBlockOriginsAndMasks(const mGrid<SBlock>& grid, int level)
    -> std::pair<std::vector<Neon::index_3d>, std::vector<typename SBlock::BitMask>>
{
    using BitMask = typename SBlock::BitMask;
    std::vector<Neon::index_3d> origins;
    std::vector<BitMask>          masks;

    const int voxelSpacing = grid.getDescriptor().getSpacing(level - 1);
    auto&     bgrid = const_cast<mGrid<SBlock>&>(grid)(level);
    SetIdx    devID(0);

    bgrid.helpGetPartitioner1D().forEachSeq(devID, [&](const uint32_t /*blockIdx*/, const Neon::int32_3d memBlockOrigin, auto /*byPartition*/) {
        Neon::index_3d blockOrigin = memBlockOrigin;
        blockOrigin.x *= SBlock::memBlockSizeX * voxelSpacing;
        blockOrigin.y *= SBlock::memBlockSizeY * voxelSpacing;
        blockOrigin.z *= SBlock::memBlockSizeZ * voxelSpacing;
        origins.push_back(blockOrigin);

        BitMask bm;
        bm.reset();
        for (int z = 0; z < SBlock::memBlockSizeZ; ++z) {
            for (int y = 0; y < SBlock::memBlockSizeY; ++y) {
                for (int x = 0; x < SBlock::memBlockSizeX; ++x) {
                    const Neon::index_3d voxelGlobalID(x * voxelSpacing + blockOrigin.x,
                                                       y * voxelSpacing + blockOrigin.y,
                                                       z * voxelSpacing + blockOrigin.z);
                    if (bgrid.isInsideDomain(voxelGlobalID)) {
                        bm.setActive(x, y, z);
                    }
                }
            }
        }
        masks.push_back(bm);
    });

    return {origins, masks};
}

template <typename SBlock>
auto fieldElementsPerLevel(const mGrid<SBlock>& grid, int level) -> uint64_t
{
    auto [origins, /*masks*/] = collectBlockOriginsAndMasks(grid, level);
    (void)origins;
    return static_cast<uint64_t>(origins.size()) * SBlock::memBlockCountElements;
}

}  // namespace

template <typename SBlock>
auto mGrid<SBlock>::getCheckpointFieldNumElementsPerCard() const -> uint64_t
{
    uint64_t total = 0;
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        total += fieldElementsPerLevel(*this, l);
    }
    return total;
}

template <typename SBlock>
auto mGrid<SBlock>::getTopologyId() const -> uint64_t
{
    if (mData->mTopologyId != 0) {
        return mData->mTopologyId;
    }
    std::vector<uint8_t> payload;
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        auto [origins, masks] = collectBlockOriginsAndMasks(*this, l);
        for (auto const& o : origins) {
            payload.insert(payload.end(), reinterpret_cast<const uint8_t*>(&o.x), reinterpret_cast<const uint8_t*>(&o.x) + 12);
        }
        for (auto const& m : masks) {
            payload.insert(payload.end(), reinterpret_cast<const uint8_t*>(m.bits), reinterpret_cast<const uint8_t*>(m.bits) + sizeof(m.bits));
        }
    }
    mData->mTopologyId = fnv1a64(payload.data(), payload.size());
    return mData->mTopologyId;
}

template <typename SBlock>
auto mGrid<SBlock>::ioToTopologyFile(const std::string& path) const -> void
{
    std::vector<std::vector<Neon::index_3d>>              originsPerLevel;
    std::vector<std::vector<typename SBlock::BitMask>> activeMasksPerLevel;
    std::vector<uint64_t>                                 fieldOffsetPerCard;

    uint64_t cumulative = 0;
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        auto [origins, masks] = collectBlockOriginsAndMasks(*this, l);
        originsPerLevel.push_back(std::move(origins));
        activeMasksPerLevel.push_back(std::move(masks));
        fieldOffsetPerCard.push_back(cumulative);
        cumulative += fieldElementsPerLevel(*this, l);
    }

    std::vector<uint8_t> binary;
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        for (auto const& o : originsPerLevel[l]) {
            binary.insert(binary.end(), reinterpret_cast<const uint8_t*>(&o.x), reinterpret_cast<const uint8_t*>(&o.x) + 12);
        }
        for (auto const& m : activeMasksPerLevel[l]) {
            binary.insert(binary.end(), reinterpret_cast<const uint8_t*>(m.bits), reinterpret_cast<const uint8_t*>(m.bits) + sizeof(m.bits));
        }
    }
    const uint64_t topologyId = fnv1a64(binary.data(), binary.size());

    rapidjson::Document doc;
    doc.SetObject();
    auto& alloc = doc.GetAllocator();

    doc.AddMember("format_version", kFormatVersion, alloc);
    doc.AddMember("neon_git_sha1", rapidjson::Value(g_GIT_SHA1, alloc), alloc);
    doc.AddMember("topology_id", rapidjson::Value().SetUint64(topologyId), alloc);

    {
        rapidjson::Value ds(rapidjson::kArrayType);
        ds.PushBack(mData->domainSize.x, alloc).PushBack(mData->domainSize.y, alloc).PushBack(mData->domainSize.z, alloc);
        doc.AddMember("domain_size", ds, alloc);
    }
    doc.AddMember("num_levels", mData->mDescriptor.getDepth(), alloc);
    doc.AddMember("strong_balanced", mData->mStrongBalanced, alloc);
    doc.AddMember("cull_overlaps", mData->mCullOverlaps, alloc);

    {
        rapidjson::Value spacing(rapidjson::kArrayType);
        spacing.PushBack(mData->mSpacingData.x, alloc).PushBack(mData->mSpacingData.y, alloc).PushBack(mData->mSpacingData.z, alloc);
        doc.AddMember("spacing_data", spacing, alloc);
    }
    {
        rapidjson::Value origin(rapidjson::kArrayType);
        origin.PushBack(mData->mOrigin.x, alloc).PushBack(mData->mOrigin.y, alloc).PushBack(mData->mOrigin.z, alloc);
        doc.AddMember("origin", origin, alloc);
    }

    {
        rapidjson::Value ref(rapidjson::kArrayType);
        rapidjson::Value sp(rapidjson::kArrayType);
        for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
            ref.PushBack(mData->mDescriptor.getRefFactor(l), alloc);
            sp.PushBack(mData->mDescriptor.getSpacing(l), alloc);
        }
        doc.AddMember("ref_factors", ref, alloc);
        doc.AddMember("spacings", sp, alloc);
    }

    {
        rapidjson::Value stencilPts(rapidjson::kArrayType);
        for (auto const& p : mData->mStencil.points()) {
            rapidjson::Value pt(rapidjson::kArrayType);
            pt.PushBack(p.x, alloc).PushBack(p.y, alloc).PushBack(p.z, alloc);
            stencilPts.PushBack(pt, alloc);
        }
        doc.AddMember("stencil", stencilPts, alloc);
    }

    {
        rapidjson::Value mbs(rapidjson::kArrayType);
        mbs.PushBack(SBlock::memBlockSizeX, alloc).PushBack(SBlock::memBlockSizeY, alloc).PushBack(SBlock::memBlockSizeZ, alloc);
        doc.AddMember("mem_block_size", mbs, alloc);
    }
    {
        rapidjson::Value ubs(rapidjson::kArrayType);
        ubs.PushBack(SBlock::userBlockSizeX, alloc).PushBack(SBlock::userBlockSizeY, alloc).PushBack(SBlock::userBlockSizeZ, alloc);
        doc.AddMember("user_block_size", ubs, alloc);
    }
    doc.AddMember(
        "field_layout",
        rapidjson::Value(
            "SoA per level: block order = Partitioner1D forEachSeq; pitch = (blockIdx * cardinality + card) * 64 + x + 4*y + 16*z; inactive bitmask slots zeroed in .bin",
            alloc),
        alloc);

    rapidjson::Value levels(rapidjson::kArrayType);
    size_t           binaryOffset = 0;
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        rapidjson::Value lvl(rapidjson::kObjectType);
        const auto       ld = getDimension(l);
        rapidjson::Value lds(rapidjson::kArrayType);
        lds.PushBack(ld.x, alloc).PushBack(ld.y, alloc).PushBack(ld.z, alloc);
        lvl.AddMember("level_domain_size", lds, alloc);
        lvl.AddMember("voxel_spacing", mData->mDescriptor.getSpacing(l - 1), alloc);
        lvl.AddMember("num_blocks", static_cast<uint64_t>(originsPerLevel[l].size()), alloc);
        lvl.AddMember("field_offset_per_card", fieldOffsetPerCard[l], alloc);

        const size_t originsBytes = originsPerLevel[l].size() * 12;
        const size_t masksBytes = activeMasksPerLevel[l].size() * sizeof(typename SBlock::BitMask);

        rapidjson::Value bo(rapidjson::kObjectType);
        bo.AddMember("offset", binaryOffset, alloc);
        bo.AddMember("bytes", originsBytes, alloc);
        bo.AddMember("dtype", "int32", alloc);
        rapidjson::Value boShape(rapidjson::kArrayType);
        boShape.PushBack(static_cast<uint64_t>(originsPerLevel[l].size()), alloc).PushBack(3, alloc);
        bo.AddMember("shape", boShape, alloc);
        lvl.AddMember("block_origins", bo, alloc);
        binaryOffset += originsBytes;

        rapidjson::Value am(rapidjson::kObjectType);
        am.AddMember("offset", binaryOffset, alloc);
        am.AddMember("bytes", masksBytes, alloc);
        am.AddMember("dtype", "uint32", alloc);
        rapidjson::Value amShape(rapidjson::kArrayType);
        amShape.PushBack(static_cast<uint64_t>(activeMasksPerLevel[l].size()), alloc)
            .PushBack(static_cast<uint64_t>(SBlock::BitMask::nWords), alloc);
        am.AddMember("shape", amShape, alloc);
        lvl.AddMember("active_masks", am, alloc);
        binaryOffset += masksBytes;

        levels.PushBack(lvl, alloc);
    }
    doc.AddMember("levels", levels, alloc);

    rapidjson::StringBuffer                         buffer;
    rapidjson::Writer<rapidjson::StringBuffer>      writer(buffer);
    doc.Accept(writer);
    const std::string headerJson = buffer.GetString();

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        NeonException exp("mGrid::ioToTopologyFile");
        exp << "Unable to open " << path;
        NEON_THROW(exp);
    }
    out.write(kMagic, 8);
    writeU64LE(out, headerJson.size());
    out.write(headerJson.data(), static_cast<std::streamsize>(headerJson.size()));
    out.write(reinterpret_cast<const char*>(binary.data()), static_cast<std::streamsize>(binary.size()));

    const_cast<mGrid<SBlock>*>(this)->mData->mTopologyId = topologyId;
}

template <typename SBlock>
auto mGrid<SBlock>::helpRestoreBitmasksFromTopology(
    const std::vector<std::vector<Neon::index_3d>>&              blockOriginsPerLevel,
    const std::vector<std::vector<typename SBlock::BitMask>>& activeMasksPerLevel) -> void
{
    const Neon::int32_3d& domainSize = mData->domainSize;
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        const int voxelSpacing = mData->mDescriptor.getSpacing(l - 1);
        const int levelShift = levelShiftForSpacing(voxelSpacing);

        for (size_t b = 0; b < blockOriginsPerLevel[l].size(); ++b) {
            const Neon::index_3d& blockOrigin = blockOriginsPerLevel[l][b];
            const auto&           mask = activeMasksPerLevel[l][b];
            for (int z = 0; z < SBlock::memBlockSizeZ; ++z) {
                for (int y = 0; y < SBlock::memBlockSizeY; ++y) {
                    for (int x = 0; x < SBlock::memBlockSizeX; ++x) {
                        if (!mask.isActive(x, y, z)) {
                            continue;
                        }
                        const Neon::index_3d voxelGlobalID(x * voxelSpacing + blockOrigin.x,
                                                           y * voxelSpacing + blockOrigin.y,
                                                           z * voxelSpacing + blockOrigin.z);
                        if (voxelGlobalID.x < 0 || voxelGlobalID.y < 0 || voxelGlobalID.z < 0) {
                            continue;
                        }
                        if (!(voxelGlobalID < domainSize)) {
                            continue;
                        }
                        const Neon::index_3d levelCoord(voxelGlobalID.x / voxelSpacing,
                                                      voxelGlobalID.y / voxelSpacing,
                                                      voxelGlobalID.z / voxelSpacing);
                        const Neon::index_3d bxyz(levelCoord.x >> levelShift,
                                                  levelCoord.y >> levelShift,
                                                  levelCoord.z >> levelShift);
                        mData->sparseLevelsBitmask.at(l).template activatePoint<true>(bxyz);
                    }
                }
            }
        }
    }
}

template <typename SBlock>
auto mGrid<SBlock>::helpVerifyTopologyBlockLayout(
    const std::vector<std::vector<Neon::index_3d>>&              blockOriginsPerLevel,
    const std::vector<std::vector<typename SBlock::BitMask>>& activeMasksPerLevel) const -> void
{
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        auto [origins, masks] = collectBlockOriginsAndMasks(*this, l);
        if (origins.size() != blockOriginsPerLevel[l].size()) {
            NeonException exp("mGrid::fromTopologyFile");
            exp << "Level " << l << " num_blocks mismatch";
            NEON_THROW(exp);
        }
        for (size_t i = 0; i < origins.size(); ++i) {
            if (origins[i].x != blockOriginsPerLevel[l][i].x || origins[i].y != blockOriginsPerLevel[l][i].y ||
                origins[i].z != blockOriginsPerLevel[l][i].z) {
                NeonException exp("mGrid::fromTopologyFile");
                exp << "Level " << l << " block origin mismatch at index " << i;
                NEON_THROW(exp);
            }
            for (uint32_t w = 0; w < SBlock::BitMask::nWords; ++w) {
                if (masks[i].bits[w] != activeMasksPerLevel[l][i].bits[w]) {
                    NeonException exp("mGrid::fromTopologyFile");
                    exp << "Level " << l << " active mask mismatch at block " << i;
                    NEON_THROW(exp);
                }
            }
        }
    }
}

template <typename SBlock>
auto mGrid<SBlock>::fromTopologyFile(const Neon::Backend& backend, const std::string& path) -> mGrid
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        NeonException exp("mGrid::fromTopologyFile");
        exp << "Unable to open " << path;
        NEON_THROW(exp);
    }

    char magic[8];
    in.read(magic, 8);
    if (std::memcmp(magic, kMagic, 8) != 0) {
        NeonException exp("mGrid::fromTopologyFile");
        exp << "Invalid topology magic";
        NEON_THROW(exp);
    }

    uint64_t headerLen = 0;
    readU64LE(in, headerLen);
    std::string headerJson(headerLen, '\0');
    in.read(headerJson.data(), static_cast<std::streamsize>(headerLen));

    rapidjson::Document doc;
    doc.Parse(headerJson.c_str());
    if (doc.HasParseError()) {
        NeonException exp("mGrid::fromTopologyFile");
        exp << "Invalid JSON header";
        NEON_THROW(exp);
    }

    if (doc["format_version"].GetUint() != kFormatVersion) {
        NeonException exp("mGrid::fromTopologyFile");
        exp << "Unsupported format version";
        NEON_THROW(exp);
    }

    const int depth = doc["num_levels"].GetInt();
    Descriptor descriptor(depth);

    Neon::int32_3d domainSize;
    domainSize.x = doc["domain_size"][0].GetInt();
    domainSize.y = doc["domain_size"][1].GetInt();
    domainSize.z = doc["domain_size"][2].GetInt();

    const bool strongBalanced = doc["strong_balanced"].GetBool();
    const bool cullOverlaps = doc["cull_overlaps"].GetBool();

    double_3d spacingData(1, 1, 1);
    double_3d origin(0, 0, 0);
    if (doc.HasMember("spacing_data")) {
        spacingData.x = doc["spacing_data"][0].GetDouble();
        spacingData.y = doc["spacing_data"][1].GetDouble();
        spacingData.z = doc["spacing_data"][2].GetDouble();
    }
    if (doc.HasMember("origin")) {
        origin.x = doc["origin"][0].GetDouble();
        origin.y = doc["origin"][1].GetDouble();
        origin.z = doc["origin"][2].GetDouble();
    }

    std::vector<Neon::index_3d> stencilPts;
    for (auto& pt : doc["stencil"].GetArray()) {
        stencilPts.emplace_back(pt[0].GetInt(), pt[1].GetInt(), pt[2].GetInt());
    }
    Neon::domain::Stencil stencil(stencilPts);

    const auto& mbs = doc["mem_block_size"];
    if (mbs[0].GetInt() != static_cast<int>(SBlock::memBlockSizeX) || mbs[1].GetInt() != static_cast<int>(SBlock::memBlockSizeY) ||
        mbs[2].GetInt() != static_cast<int>(SBlock::memBlockSizeZ)) {
        NeonException exp("mGrid::fromTopologyFile");
        exp << "mem_block_size mismatch";
        NEON_THROW(exp);
    }

    std::vector<std::vector<Neon::index_3d>>              blockOriginsPerLevel(depth);
    std::vector<std::vector<typename SBlock::BitMask>> activeMasksPerLevel(depth);

    for (int l = 0; l < depth; ++l) {
        const auto& lvl = doc["levels"][l];
        const auto  numBlocks = lvl["num_blocks"].GetUint64();
        const auto  bo = lvl["block_origins"];
        const auto  am = lvl["active_masks"];

        blockOriginsPerLevel[l].resize(numBlocks);
        in.seekg(static_cast<std::streamoff>(bo["offset"].GetUint64()) + static_cast<std::streamoff>(8 + headerLen), std::ios::beg);
        for (uint64_t i = 0; i < numBlocks; ++i) {
            in.read(reinterpret_cast<char*>(&blockOriginsPerLevel[l][i].x), 12);
        }

        activeMasksPerLevel[l].resize(numBlocks);
        in.seekg(static_cast<std::streamoff>(am["offset"].GetUint64()) + static_cast<std::streamoff>(8 + headerLen), std::ios::beg);
        for (uint64_t i = 0; i < numBlocks; ++i) {
            in.read(reinterpret_cast<char*>(activeMasksPerLevel[l][i].bits), sizeof(typename SBlock::BitMask));
        }
    }

    std::vector<uint8_t> binary;
    in.seekg(0, std::ios::end);
    const auto fileEnd = in.tellg();
    in.seekg(static_cast<std::streamoff>(8 + headerLen), std::ios::beg);
    const auto binSize = static_cast<size_t>(fileEnd - in.tellg());
    binary.resize(binSize);
    in.read(reinterpret_cast<char*>(binary.data()), static_cast<std::streamsize>(binSize));
    const uint64_t topologyId = fnv1a64(binary.data(), binary.size());
    if (topologyId != doc["topology_id"].GetUint64()) {
        NeonException exp("mGrid::fromTopologyFile");
        exp << "topology_id mismatch";
        NEON_THROW(exp);
    }

    mGrid grid;
    grid.helpInit(backend, domainSize, descriptor, strongBalanced, cullOverlaps);
    grid.helpRestoreBitmasksFromTopology(blockOriginsPerLevel, activeMasksPerLevel);
    grid.helpBuildGrids(stencil, spacingData, origin);
    grid.helpVerifyTopologyBlockLayout(blockOriginsPerLevel, activeMasksPerLevel);
    grid.mData->mTopologyId = topologyId;
    return grid;
}

using SB8 = Neon::domain::details::StaticBlock<8, 8, 8, 2, 2, 2, true>;
using SB4 = Neon::domain::details::StaticBlock<4, 4, 4, 2, 2, 2, true>;
using SB2 = Neon::domain::details::StaticBlock<2, 2, 2, 2, 2, 2, true>;

template auto mGrid<SB8>::ioToTopologyFile(const std::string&) const -> void;
template auto mGrid<SB4>::ioToTopologyFile(const std::string&) const -> void;
template auto mGrid<SB2>::ioToTopologyFile(const std::string&) const -> void;
template auto mGrid<SB8>::fromTopologyFile(const Neon::Backend&, const std::string&) -> mGrid<SB8>;
template auto mGrid<SB4>::fromTopologyFile(const Neon::Backend&, const std::string&) -> mGrid<SB4>;
template auto mGrid<SB2>::fromTopologyFile(const Neon::Backend&, const std::string&) -> mGrid<SB2>;
template auto mGrid<SB8>::getTopologyId() const -> uint64_t;
template auto mGrid<SB4>::getTopologyId() const -> uint64_t;
template auto mGrid<SB2>::getTopologyId() const -> uint64_t;
template auto mGrid<SB8>::getCheckpointFieldNumElementsPerCard() const -> uint64_t;
template auto mGrid<SB4>::getCheckpointFieldNumElementsPerCard() const -> uint64_t;
template auto mGrid<SB2>::getCheckpointFieldNumElementsPerCard() const -> uint64_t;

}  // namespace Neon::domain::details::mGrid
