#include "Neon/domain/aGrid.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"

#include "Neon/skeleton/Skeleton.h"

#include "gtest/gtest.h"
#include "sUt.runHelper.h"
#include "sUt_common.h"
#include "sUt_skeleton.sequenceMapKernels.h"


template <typename Grid_ta, typename T_ta>
void AXPY(Neon::index64_3d                    dim,
          int                                 nGPU,
          int                                 cardinality,
          const Neon::Runtime& backendType)
{
    storage_t<Grid_ta, T_ta> storage(dim, nGPU, cardinality, backendType);
    storage.initLinearly();

    Neon::skeleton::Skeleton          skl(storage.m_backend);
    Neon::skeleton::Options           opt;
    std::vector<Neon::set::Container> sVec;
    sVec.push_back(UserTools::xpy(storage.Xf, storage.Yf));
    skl.sequence(sVec,
                 "AXPY", opt);

    for (int i = 0; i < 10; i++) {
        skl.run();
    }

    for (int i = 0; i < 10; i++) {
        storage.sum(storage.Xd, storage.Yd);
    }
    storage.m_backend.syncAll();
    bool isOk = storage.compare(storage.Yd, storage.Yf);
    isOk = isOk && storage.compare(storage.Xd, storage.Xf);
    isOk = isOk && storage.compare(storage.Zd, storage.Zf);
    ASSERT_TRUE(isOk);
}

template <typename Grid_ta, typename T_ta>
void AXPY_struct(Neon::index64_3d                    dim,
                 int                                 nGPU,
                 int                                 cardinality,
                 const Neon::Runtime& backendType)
{
    storage_t<Grid_ta, T_ta> storage(dim, nGPU, cardinality, backendType);
    storage.initLinearly();

    Neon::skeleton::Skeleton          skl(storage.m_backend);
    Neon::skeleton::Options           opt;
    std::vector<Neon::set::Container> sVec;

    UserTools::blas_t<decltype(storage.Xf)> blas(storage.Xf, storage.Yf, storage.Zf);

    sVec.push_back(blas.xpy());
    skl.sequence(sVec,
                 "AXPY_struct", opt);

    for (int i = 0; i < 10; i++) {
        skl.run();
    }

    for (int i = 0; i < 10; i++) {
        storage.sum(storage.Xd, storage.Yd);
    }
    storage.m_backend.syncAll();

    bool isOk = storage.compare(storage.Yd, storage.Yf);
    isOk = isOk && storage.compare(storage.Xd, storage.Xf);
    isOk = isOk && storage.compare(storage.Zd, storage.Zf);
    ASSERT_TRUE(isOk);
}


template <typename Grid_ta, typename T_ta>
void AXPY_2(Neon::index64_3d                    dim,
            int                                 nGPU,
            int                                 cardinality,
            const Neon::Runtime& backendType)
{
    storage_t<Grid_ta, T_ta> storage(dim, nGPU, cardinality, backendType);
    storage.initLinearly();

    Neon::skeleton::Skeleton          skl(storage.m_backend);
    Neon::skeleton::Options           opt;
    std::vector<Neon::set::Container> sVec;
    sVec.push_back(UserTools::xpy(storage.Xf, storage.Yf));
    sVec.push_back(UserTools::xpy(storage.Yf, storage.Zf));
    skl.sequence(sVec,
                 "AXPY_2", opt);

    for (int i = 0; i < 10; i++) {
        skl.run();
    }

    for (int i = 0; i < 10; i++) {
        storage.sum(storage.Xd, storage.Yd);
        storage.sum(storage.Yd, storage.Zd);
    }
    storage.m_backend.syncAll();

    bool isOk = storage.compare(storage.Yd, storage.Yf);
    isOk = isOk && storage.compare(storage.Xd, storage.Xf);
    isOk = isOk && storage.compare(storage.Zd, storage.Zf);
    ASSERT_TRUE(isOk);
}

template <typename Grid_ta, typename T_ta>
void AXPY_2_struct(Neon::index64_3d                    dim,
                   int                                 nGPU,
                   int                                 cardinality,
                   const Neon::Runtime& backendType)
{
    storage_t<Grid_ta, T_ta> storage(dim, nGPU, cardinality, backendType);
    storage.initLinearly();

    Neon::skeleton::Skeleton          skl(storage.m_backend);
    Neon::skeleton::Options           opt;
    std::vector<Neon::set::Container> sVec;

    UserTools::blas_t<decltype(storage.Xf)> blas(storage.Xf, storage.Yf, storage.Zf);

    sVec.push_back(blas.xpy());
    sVec.push_back(blas.xpy());
    skl.sequence(sVec,
                 "AXPY_2_struct", opt);

    for (int i = 0; i < 10; i++) {
        skl.run();
    }

    for (int i = 0; i < 10; i++) {
        storage.sum(storage.Xd, storage.Yd);
        storage.sum(storage.Xd, storage.Yd);
    }
    storage.m_backend.syncAll();

    bool isOk = storage.compare(storage.Yd, storage.Yf);
    isOk = isOk && storage.compare(storage.Xd, storage.Xf);
    isOk = isOk && storage.compare(storage.Zd, storage.Zf);
    ASSERT_TRUE(isOk);
}

template <typename Grid_ta, typename T_ta>
void AXPY_3(Neon::index64_3d                    dim,
            int                                 nGPU,
            int                                 cardinality,
            const Neon::Runtime& backendType)
{
    storage_t<Grid_ta, T_ta> storage(dim, nGPU, cardinality, backendType);
    storage.initLinearly();

    Neon::skeleton::Skeleton          skl(storage.m_backend);
    Neon::skeleton::Options           opt;
    std::vector<Neon::set::Container> sVec;
    sVec.push_back(UserTools::xpy(storage.Xf, storage.Xf));
    sVec.push_back(UserTools::xpy(storage.Yf, storage.Yf));
    sVec.push_back(UserTools::xpy(storage.Zf, storage.Zf));
    sVec.push_back(UserTools::xpy(storage.Zf, storage.Xf));
    sVec.push_back(UserTools::xpy(storage.Zf, storage.Yf));
    sVec.push_back(UserTools::xpy(storage.Xf, storage.Yf));

    skl.sequence(sVec,
                 "AXPY_3", opt);
    for (int i = 0; i < 10; i++) {
        skl.run();
    }

    for (int i = 0; i < 10; i++) {
        storage.sum(storage.Xd, storage.Xd);
        storage.sum(storage.Yd, storage.Yd);
        storage.sum(storage.Zd, storage.Zd);
        storage.sum(storage.Zd, storage.Xd);
        storage.sum(storage.Zd, storage.Yd);
        storage.sum(storage.Xd, storage.Yd);
    }
    storage.m_backend.syncAll();

    bool isOk = storage.compare(storage.Yd, storage.Yf);
    isOk = isOk && storage.compare(storage.Xd, storage.Xf);
    isOk = isOk && storage.compare(storage.Zd, storage.Zf);

    ASSERT_TRUE(isOk);
}

TEST(sUt, AXPY)
{
    NEON_INFO("AXPY");
    int nGpus = 3;
    // runAllTestConfiguration(AXPY_struct<eGrid_t, int64_t>, nGpus);
    runOneTestConfiguration(AXPY<eGrid_t, int64_t>, nGpus);
}

TEST(sUt, AXPY_2)
{
    NEON_INFO("AXPY_2");
    int nGpus = 3;
    // runAllTestConfiguration(AXPY_2_struct<eGrid_t, int64_t>, nGpus);
    runOneTestConfiguration(AXPY_2<eGrid_t, int64_t>, nGpus);
}

TEST(sUt, AXPY_3)
{
    NEON_INFO("AXPY_3");
    int nGpus = 3;
    runAllTestConfiguration(AXPY_3<eGrid_t, int64_t>, nGpus);
}
