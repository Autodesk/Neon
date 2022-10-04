
#include <Neon/core/types/vec.h>

#include <cstring>
#include <iostream>

#include "Neon/core/core.h"
#include "Neon/core/tools/IO.h"
#include "Neon/core/tools/io/exportVTI.h"
#include "Neon/core/tools/io/ioToVTK.h"
#include "Neon/core/tools/io/ioToVti.h"
#include "gtest/gtest.h"

TEST(coreUt_io, flushAndClear)
{
    Neon::index_3d space(2, 3, 4);
    auto           A = Neon::IODense<int, int>::makeMasked(1, space, 2);
    A.ioVtk("coreUt_io.flushAndClear_t0", "A", Neon::ioToVTKns::VtiDataType_e::node);
    Neon::IODenseVTK<int, int> toVtk("coreUt_io.flushAndClear_t1");
    toVtk.addField(A, "A", true);
    toVtk.setFormat(Neon::IoFileType::ASCII);
    toVtk.flushAndClear();
}

TEST(coreUt_io, denseDiff)
{
    int dim1 = 400;

    auto A = Neon::IODense<int, int>::makeLinear(1, dim1, 3);
    auto B = Neon::IODense<int, int>::makeLinear(3, dim1, 3);

    int t0maxDiff = 40;
    int targetDiff = t0maxDiff * 2;
    for (int i = 0; i < dim1 / 5; i++) {
        const Neon::index_3d maxDiff(i, i / 2, i / 4);
        const int            maxCard = i % 2;
        targetDiff += 20;
        A.getReference(maxDiff, maxCard) = targetDiff + B(maxDiff, maxCard);
        auto [diff, id, card] = A.maxDiff(A, B);

        ASSERT_EQ(diff, targetDiff);
        ASSERT_EQ(maxDiff.x, id.x);
        ASSERT_EQ(maxDiff.y, id.y);
        ASSERT_EQ(maxDiff.z, id.z);
        ASSERT_EQ(maxCard, card);
    }
    //    Neon::IoDenseToVTK toVtk(denseG, "coreUt_io.denseDiff");
    //    toVtk.addField(A, "A");
    //    toVtk.setFormat(Neon::IoFileType::ASCII);
    //    toVtk.flushAndClear();
}


TEST(coreUt_io, denseDiffRandom)
{
    int dim1 = 400;

    auto A = Neon::IODense<int, int>::makeRandom(-10, 10, dim1, 3);
    auto B = Neon::IODense<int, int>::makeRandom(-20, 20, dim1, 3);

    int t0maxDiff = 40;
    int targetDiff = t0maxDiff * 2;
    for (int i = 0; i < dim1 / 5; i++) {
        const Neon::index_3d maxDiff(i, i / 2, i / 4);
        const int            maxCard = i % 2;
        targetDiff += 20;
        A.getReference(maxDiff, maxCard) = targetDiff + B(maxDiff, maxCard);
        auto [diff, id, card] = A.maxDiff(A, B);

        ASSERT_EQ(diff, targetDiff);
        ASSERT_EQ(maxDiff.x, id.x);
        ASSERT_EQ(maxDiff.y, id.y);
        ASSERT_EQ(maxDiff.z, id.z);
        ASSERT_EQ(maxCard, card);
    }
}