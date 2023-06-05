#include "CellType.h"
#include "D3Q19.h"
#include "LbmTools.h"
#include "Neon/Neon.h"
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"
#include "Neon/skeleton/Skeleton.h"

template <typename DKQW,
          typename PopulationField,
          typename LbmComputeType>
struct LbmSkeleton
{
};


template <typename PopulationField,
          typename LbmComputeType>
struct LbmIterationD3Q19
{
    using LbmStoreType = typename PopulationField::Type;
    using CellTypeField = typename PopulationField::Grid::template Field<CellType, 1>;
    using D3Q19 = D3Q19Template<LbmStoreType, LbmComputeType>;
    using LbmTools = LbmContainers<D3Q19, PopulationField, LbmComputeType>;


    LbmIterationD3Q19(Neon::set::StencilSemantic stencilSemantic,
                      Neon::skeleton::Occ        occ,
                      Neon::set::TransferMode    transfer,
                      PopulationField&           fIn /*!   inpout population field */,
                      PopulationField&           fOut,
                      CellTypeField&             cellTypeField /*!       Cell type field     */,
                      LbmComputeType             omega /*! LBM omega parameter */)
    {
        pop[0] = fIn;
        pop[1] = fOut;

        setupSkeletons(0, stencilSemantic, occ, transfer, pop[0], pop[1], cellTypeField, omega);
        setupSkeletons(1, stencilSemantic, occ, transfer, pop[1], pop[0], cellTypeField, omega);

        parity = 0;
    }
    auto getInput()
        -> PopulationField&
    {
        return pop[parity];
    }

    auto getOutput()
        -> PopulationField&
    {
        int other = parity == 0 ? 1 : 0;
        return pop[other];
    }

    auto run()
        -> void
    {
        lbmTwoPop[parity].run();
        updateParity();
    }

    auto sync()
        -> void
    {
        pop[0].getBackend().syncAll();
    }

   private:
    auto updateParity()
        -> void
    {
        parity = parity == 0 ? 1 : 0;
    }

    auto setupSkeletons(int                        target,
                        Neon::set::StencilSemantic stencilSemantic,
                        Neon::skeleton::Occ        occ,
                        Neon::set::TransferMode    transfer,
                        PopulationField&           inField /*!   inpout population field */,
                        PopulationField&           outField,
                        CellTypeField&             cellTypeField /*!       Cell type field     */,
                        LbmComputeType             omega /*! LBM omega parameter */)
    {
        std::vector<Neon::set::Container> ops;
        lbmTwoPop[target] = Neon::skeleton::Skeleton(inField.getBackend());
        Neon::skeleton::Options opt(occ, transfer);
        ops.push_back(LbmTools::iteration(stencilSemantic,
                                          inField,
                                          cellTypeField,
                                          omega,
                                          outField));
        std::stringstream appName;
        appName << "LBM_iteration_" << std::to_string(target);
        lbmTwoPop[target].sequence(ops, appName.str(), opt);
    }

    Neon::skeleton::Skeleton lbmTwoPop[2];
    PopulationField          pop[2];
    int                      parity;
};