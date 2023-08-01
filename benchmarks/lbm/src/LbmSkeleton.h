#include "CellType.h"
#include "ContainerFactory.h"
#include "ContainersD3Q19.h"
#include "D3Q19.h"
#include "Neon/Neon.h"
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"
#include "Neon/skeleton/Skeleton.h"

template <typename Method_,
          typename Precision_,
          typename Lattice,
          typename Grid>
struct LbmSkeleton
{
};


template <typename Method_,
          typename Precision_,
          typename Grid_>
struct LbmSkeleton<Method_,
                   Precision_,
                   D3Q19<Precision_>,
                   Grid_>
{
    using Lattice = D3Q19<Precision_>;
    using Precision = Precision_;
    using Compute = typename Precision::Compute;
    using Storage = typename Precision::Storage;
    using Grid = Grid_;

    using PopField = typename Grid::template Field<Storage, Lattice::Q>;
    using CellTypeField = typename Grid::template Field<CellType, 1>;

    using Idx = typename PopField::Idx;
    using Rho = typename Grid::template Field<Storage, 1>;
    using U = typename Grid::template Field<Storage, 3>;

    using ContainerFactory = common::ContainerFactory<Precision, Lattice, Grid>;

    LbmSkeleton(Neon::set::StencilSemantic stencilSemantic,
                Neon::skeleton::Occ        occ,
                Neon::set::TransferMode    transfer,
                PopField&                  fIn /*!   inpout population field */,
                PopField&                  fOut,
                CellTypeField&             cellTypeField /*!       Cell type field     */,
                Compute                    omega /*! LBM omega parameter */)
    {
        pop[0] = fIn;
        pop[1] = fOut;

        setupSkeletons(0, stencilSemantic, occ, transfer, pop[0], pop[1], cellTypeField, omega);
        setupSkeletons(1, stencilSemantic, occ, transfer, pop[1], pop[0], cellTypeField, omega);

        parity = 0;
    }

    auto getInput()
        -> PopField&
    {
        return pop[parity];
    }

    auto getOutput()
        -> PopField&
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
                        PopField&                  inField /*!   inpout population field */,
                        PopField&                  outField,
                        CellTypeField&             cellTypeField /*!       Cell type field     */,
                        Compute                    omega /*! LBM omega parameter */)
    {
        std::vector<Neon::set::Container> ops;
        lbmTwoPop[target] = Neon::skeleton::Skeleton(inField.getBackend());
        Neon::skeleton::Options opt(occ, transfer);
        ops.push_back(ContainerFactory::template iteration<Method_>(stencilSemantic,
                                                  inField,
                                                  cellTypeField,
                                                  omega,
                                                  outField));
        std::stringstream appName;
        appName << "LBM_iteration_" << std::to_string(target);
        lbmTwoPop[target].sequence(ops, appName.str(), opt);
    }

    Neon::skeleton::Skeleton lbmTwoPop[2];
    PopField                 pop[2];
    int                      parity;
};