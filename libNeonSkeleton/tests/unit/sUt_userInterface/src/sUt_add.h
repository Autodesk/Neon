#pragma once
#include <functional>
#include "Neon/core/tools/metaprogramming/debugHelp.h"

#include "Neon/domain/aGrid.h"
#include "Neon/domain/bGrid.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"

#include "gtest/gtest.h"
#include "sUt.runHelper.h"
#include "sUt_common.h"
template <typename Field_ta>
auto add(const Field_ta& X,
         const Field_ta& Y,
         Field_ta&       Z) -> Neon::set::Container
{
    auto Kontainer = X.getGrid().getContainer(
        "add", [&](Neon::set::Loader & L) -> auto {
            auto& x = L.load(X);
            auto& y = L.load(Y);
            auto& z = L.load(Z);
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field_ta::Cell& e) mutable {
                for (int i = 0; i < z.cardinality(); i++) {
                    z(e, i) = x(e, i) + y(e, i);
                }
            };
        });
    return Kontainer;
}

template <typename Grid_ta, typename T_ta>
void dataViewAddTest(Neon::index64_3d     dim,
                     int                  nGPU,
                     int                  cardinality,
                     const Neon::Runtime& backendType)
{
    storage_t<Grid_ta, T_ta> storage(dim, nGPU, cardinality, backendType);
    storage.initLinearly();

    auto Kontainer = add(storage.Xf, storage.Yf, storage.Zf);
    Kontainer.run(0);

    storage.sum(storage.Xd, storage.Yd, storage.Zd);

    storage.m_backend.syncAll();

    //storage.Zf.template ioToVtk<T_ta>("sUt_add.vti", "Z");

    bool isOk = storage.compare(storage.Zd, storage.Zf);
    ASSERT_TRUE(isOk);
}

TEST(eGrid, Add)
{
    NEON_INFO("eGrid_t");
    int nGpus = 3;
    runAllTestConfiguration(dataViewAddTest<eGrid_t, int64_t>, nGpus);
}

TEST(bGrid, Add)
{
    std::cout << "bGrid_t" << std::endl;
    int nGpus = 1;
    runAllTestConfiguration(dataViewAddTest<bGrid_t, int64_t>, nGpus);
}

TEST(dGrid, Add)
{
    std::cout << "dGrid_t" << std::endl;
    int nGpus = 3;
    runAllTestConfiguration(dataViewAddTest<dGrid_t, int64_t>, nGpus);
}
