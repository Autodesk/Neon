#include "gtest/gtest.h"

#include <omp.h>

#include <atomic>
#include <cstdint>
#include <set>
#include <tuple>
#include <vector>

#include "Neon/core/core.h"
#include "Neon/domain/details/mGrid/sparseBitMask.h"

using Neon::domain::details::mGrid::SparseBitBlocks;

namespace {

// Block space of the 2 mm F1 grid's level 0: 1612 x 1184 x 444 = 847M block slots, far
// beyond the 64M slots at which the previous implementation switched to an unordered_map.
const Neon::index_3d largeBBox(12896, 9472, 3552);

auto blockTuple(const Neon::index_3d& p) -> std::tuple<int, int, int>
{
    return {p.x / 8, p.y / 8, p.z / 8};
}

// Deterministic, irregular activation pattern
auto isSelected(const Neon::index_3d& p) -> bool
{
    uint32_t h = uint32_t(p.x) * 73856093u ^ uint32_t(p.y) * 19349663u ^ uint32_t(p.z) * 83492791u;
    return (h % 7u) == 0u;
}

}  // namespace

TEST(sparseBitMask, parallelActivateLargeBBox)
{
    SparseBitBlocks<300> bits(largeBBox);

    const Neon::index_3d origin(4000, 4000, 0);
    const Neon::index_3d slab(400, 300, 160);

#pragma omp parallel for collapse(3) schedule(static)
    for (int z = 0; z < slab.z; z++) {
        for (int y = 0; y < slab.y; y++) {
            for (int x = 0; x < slab.x; x++) {
                bits.activatePoint<true>(Neon::index_3d(origin.x + x, origin.y + y, origin.z + z));
            }
        }
    }

    int64_t missing = 0;
#pragma omp parallel for collapse(3) schedule(static) reduction(+ : missing)
    for (int z = 0; z < slab.z; z++) {
        for (int y = 0; y < slab.y; y++) {
            for (int x = 0; x < slab.x; x++) {
                missing += bits.isActivePoint(Neon::index_3d(origin.x + x, origin.y + y, origin.z + z)) ? 0 : 1;
            }
        }
    }
    ASSERT_EQ(missing, 0);

    // Points just outside the slab, in partially covered and in untouched blocks
    ASSERT_FALSE(bits.isActivePoint(Neon::index_3d(origin.x - 1, origin.y, origin.z)));
    ASSERT_FALSE(bits.isActivePoint(Neon::index_3d(origin.x + slab.x, origin.y, origin.z)));
    ASSERT_FALSE(bits.isActivePoint(Neon::index_3d(origin.x, origin.y + slab.y, origin.z)));
    ASSERT_FALSE(bits.isActivePoint(Neon::index_3d(origin.x, origin.y, origin.z + slab.z)));
    ASSERT_FALSE(bits.isActivePoint(Neon::index_3d(0, 0, 0)));
    ASSERT_FALSE(bits.isActivePoint(Neon::index_3d(largeBBox.x - 1, largeBBox.y - 1, largeBBox.z - 1)));

    // Every block was allocated exactly once
    auto const     blocks = bits.getAllocatedBlockCoords();
    std::set<std::tuple<int, int, int>> unique;
    for (auto const& b : blocks) {
        unique.insert({b.x, b.y, b.z});
    }
    auto const firstBlock = blockTuple(origin);
    auto const lastBlock = blockTuple(Neon::index_3d(origin.x + slab.x - 1, origin.y + slab.y - 1, origin.z + slab.z - 1));
    size_t const expectedBlocks = size_t(std::get<0>(lastBlock) - std::get<0>(firstBlock) + 1) *
                                  size_t(std::get<1>(lastBlock) - std::get<1>(firstBlock) + 1) *
                                  size_t(std::get<2>(lastBlock) - std::get<2>(firstBlock) + 1);
    ASSERT_EQ(blocks.size(), unique.size());
    ASSERT_EQ(blocks.size(), expectedBlocks);
}

TEST(sparseBitMask, concurrentReadersAndWriters)
{
    SparseBitBlocks<300> bits(largeBBox);

    const Neon::index_3d origin(8000, 2000, 1000);
    const Neon::index_3d slab(256, 128, 64);
    const int64_t        numPoints = int64_t(slab.x) * slab.y * slab.z;
    auto                 pointAt = [&](int64_t i) {
        return Neon::index_3d(origin.x + int(i % slab.x),
                                              origin.y + int((i / slab.x) % slab.y),
                                              origin.z + int(i / (int64_t(slab.x) * slab.y)));
    };

    std::atomic<int> writersDone{0};
    std::atomic<int> regressions{0};
    std::atomic<int> numThreads{0};

#pragma omp parallel num_threads(8)
    {
        // The runtime may grant fewer threads than requested: split whatever we got
        const int nThreads = omp_get_num_threads();
        const int numWriters = nThreads > 1 ? nThreads / 2 : 1;
        const int numReaders = nThreads - numWriters;
        const int tid = omp_get_thread_num();
        if (tid == 0) {
            numThreads.store(nThreads);
        }
        if (tid < numWriters) {
            // Interleaved partition: writers keep colliding on the same blocks and tiles
            for (int64_t i = tid; i < numPoints; i += numWriters) {
                bits.activatePoint<true>(pointAt(i));
            }
            writersDone.fetch_add(1, std::memory_order_release);
        } else {
            // Readers poll a strided subset and check that activation is never undone
            const int            r = tid - numWriters;
            std::vector<int64_t> sample;
            for (int64_t i = r; i < numPoints; i += 97 * numReaders) {
                sample.push_back(i);
            }
            std::vector<uint8_t> seen(sample.size(), 0);
            bool                 lastPass = false;
            while (!lastPass) {
                lastPass = writersDone.load(std::memory_order_acquire) == numWriters;
                for (size_t s = 0; s < sample.size(); s++) {
                    bool const active = bits.isActivePoint(pointAt(sample[s]));
                    if (seen[s] != 0 && !active) {
                        regressions.fetch_add(1);
                    }
                    seen[s] = active ? 1 : 0;
                    if (lastPass && !active) {
                        regressions.fetch_add(1);
                    }
                }
            }
        }
    }
    ASSERT_EQ(regressions.load(), 0);

    int64_t missing = 0;
    for (int64_t i = 0; i < numPoints; i++) {
        missing += bits.isActivePoint(pointAt(i)) ? 0 : 1;
    }
    ASSERT_EQ(missing, 0);
    if (numThreads.load() < 2) {
        GTEST_SKIP() << "OpenMP gave a single thread, so readers never ran concurrently with writers";
    }
}

TEST(sparseBitMask, smallBBoxMatchesSerialReference)
{
    // Not a multiple of the 8^3 block or the 16^3-block tile in any dimension
    const Neon::index_3d bbox(203, 141, 77);
    auto const           pitch = [&](const Neon::index_3d& p) { return p.mPitch(bbox); };

    std::vector<uint8_t> reference(size_t(bbox.x) * size_t(bbox.y) * size_t(bbox.z), 0);
    for (int z = 0; z < bbox.z; z++) {
        for (int y = 0; y < bbox.y; y++) {
            for (int x = 0; x < bbox.x; x++) {
                Neon::index_3d const p(x, y, z);
                reference[pitch(p)] = isSelected(p) ? 1 : 0;
            }
        }
    }

    SparseBitBlocks<100> bits(bbox);
#pragma omp parallel for collapse(3) schedule(dynamic, 64)
    for (int z = 0; z < bbox.z; z++) {
        for (int y = 0; y < bbox.y; y++) {
            for (int x = 0; x < bbox.x; x++) {
                Neon::index_3d const p(x, y, z);
                if (isSelected(p)) {
                    bits.activatePoint<true>(p);
                }
            }
        }
    }

    // Clear every point with x divisible by 5, in parallel
#pragma omp parallel for collapse(3) schedule(static)
    for (int z = 0; z < bbox.z; z++) {
        for (int y = 0; y < bbox.y; y++) {
            for (int x = 0; x < bbox.x; x += 5) {
                bits.removePoint<true>(Neon::index_3d(x, y, z));
            }
        }
    }
    for (int z = 0; z < bbox.z; z++) {
        for (int y = 0; y < bbox.y; y++) {
            for (int x = 0; x < bbox.x; x += 5) {
                reference[pitch(Neon::index_3d(x, y, z))] = 0;
            }
        }
    }

    int64_t mismatches = 0;
    for (int z = 0; z < bbox.z; z++) {
        for (int y = 0; y < bbox.y; y++) {
            for (int x = 0; x < bbox.x; x++) {
                Neon::index_3d const p(x, y, z);
                mismatches += (bits.isActivePoint(p) == (reference[pitch(p)] != 0)) ? 0 : 1;
            }
        }
    }
    ASSERT_EQ(mismatches, 0);

    // forEachActivePointInBlock over the allocated blocks enumerates exactly the active set
    std::vector<uint8_t> enumerated(reference.size(), 0);
    int64_t              duplicates = 0;
    for (auto const& b : bits.getAllocatedBlockCoords()) {
        bits.forEachActivePointInBlock(b, [&](const Neon::index_3d& p) {
            duplicates += enumerated[pitch(p)];
            enumerated[pitch(p)] = 1;
        });
    }
    ASSERT_EQ(duplicates, 0);
    ASSERT_EQ(enumerated, reference);
}

TEST(sparseBitMask, rejectsUnsupportedCoordinateSpace)
{
    // 2^23 tiles per axis: the directory size does not even fit in size_t
    const Neon::index_3d huge(1 << 30, 1 << 30, 1 << 30);
    ASSERT_THROW(SparseBitBlocks<100>{huge}, Neon::NeonException);
}
