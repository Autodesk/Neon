#pragma once
#include <functional>
#include "Neon/domain/eGrid.h"
#include "Neon/set/ContainerTools/Loader.h"
#include "gUt_map.run.h"
#include "gUt_map.storage.h"
#include "gtest/gtest.h"
namespace basicOp_kContainer_ns {

template <typename Grid, typename T, int Cardinality>
auto runMyKernel(int                                                  streamIdx,
                 T&                                                   val,
                 const typename Grid::template Field<T, Cardinality>& fA,
                 typename Grid::template Field<T, Cardinality>&       fC)
    ->Neon::set::Container
{
    const auto&                 grid = fC.getGrid();
    return grid.getContainer("MyKernel",
                             [&, val](Neon::set::Loader& loader) {
                                 const auto la = loader.load(fA);
                                 auto       lc = loader.load(fC);

                                 return [=] NEON_CUDA_HOST_DEVICE(const typename Grid::template Field<T, Cardinality>::Cell& e) mutable {
                                     for (int i = 0; i < lc.cardinality(); i++) {
                                         //printf("GPU %ld <- %ld + %ld\n", lc(e, i) , la(e, i) , val);
                                         lc(e, i) = la(e, i) + val;
                                     }
                                 };
                             });
}

/**
 *
 * @tparam T
 * @param length
 * @param cardinality
 * @param backendConfig
 */
template <typename Grid_ta, typename T_ta, int card_ta>
void runKernel(Neon::index64_3d                    dim,
               int                                 nGPU,
               int                                 cardinality,
               const Neon::Runtime& backendType,
               Neon::Timer_ms&                     timer,
               const Neon::MemSetOptions_t&        memSetOptions)
{
    //    if(backendType == Neon::Backend_t::runtime_et::openmp){
    //        return;
    //    }
    storage_t<Grid_ta, T_ta, card_ta> s(dim, nGPU, cardinality, backendType, memSetOptions);

    s.initLinearly();
    s.ioToVti("t0_gUt_basicOperations_kernel");

    T_ta val = T_ta(33);
    runMyKernel<Grid_ta, T_ta, card_ta>(Neon::Backend::mainStreamIdx,
                                        val, s.Xf, s.Zf).run(0);
    s.backend.sync();

    s.lamdaTest(s.Xd, s.Zd);

    s.ioToVti("t1_gUt_basicOperations_kernel");
    bool isOk = s.compare(s.Zd, s.Zf);
    ASSERT_TRUE(isOk);
}
}  // namespace basicOp_kContainer_ns
