#pragma once

#include "CellType.h"
#include "D3Q19.h"
#include "DeviceD3Q19.h"
#include "Neon/Neon.h"
#include "Neon/set/Containter.h"

template <typename Precision_,
          typename Lattice,
          typename Grid>
struct ContainerFactory
{
};

/**
 * Specialization for D3Q19
 */
template <typename Precision_, typename Grid_>
struct ContainerFactory<Precision_,
                        D3Q19<Precision_>,
                        Grid_>
{
    using Lattice = D3Q19<Precision_>;
    using Precision = Precision_;
    using Compute = typename Precision::Compute;
    using Storage = typename Precision::Storage;
    using Grid = Grid_;

    using PopField = typename Grid::template Field<Precision::Storage, Lattice::Q>;
    using CellTypeField = typename Grid::template Field<CellType, 1>;

    using Idx = typename PopField::Idx;
    using Rho = typename Grid::template Field<Precision::Storage, 1>;
    using U = typename Grid::template Field<Precision::Storage, 3>;

    using Functions = DeviceD3Q19<Precision, Grid>;

    static auto
    iteration(Neon::set::StencilSemantic stencilSemantic,
              const PopField&            fInField /*!      Input population field */,
              const CellTypeField&       cellTypeField /*! Cell type field     */,
              const Compute              omega /*!         LBM omega parameter */,
              PopField&                  fOutField /*!     Output Population field */)
        -> Neon::set::Container
    {
        Neon::set::Container container = fInField.getGrid().newContainer(
            "D3Q19_TwoPop",
            [&, omega](Neon::set::Loader& L) -> auto {
                auto&       fIn = L.load(fInField,
                                         Neon::Pattern::STENCIL, stencilSemantic);
                auto&       fOut = L.load(fOutField);
                const auto& cellInfoPartition = L.load(cellTypeField);

                return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                    CellType cellInfo = cellInfoPartition(gidx, 0);
                    if (cellInfo.classification == CellType::bulk) {

                        Storage popIn[Lattice::Q];
                        Functions::pullStream(gidx, cellInfo.wallNghBitflag, fIn, NEON_OUT popIn);

                        Compute                rho;
                        std::array<Compute, 3> u{.0, .0, .0};
                        Functions::macroscopic(popIn, NEON_OUT rho, NEON_OUT u);

                        Compute usqr = 1.5 * (u[0] * u[0] +
                                              u[1] * u[1] +
                                              u[2] * u[2]);

                        Functions::collideBgkUnrolled(gidx,
                                                      popIn,
                                                      rho, u,
                                                      usqr, omega,
                                                      NEON_OUT fOut);
                    }
                };
            });
        return container;
    }


    static auto
    computeWallNghMask(const CellTypeField& infoInField,
                       CellTypeField&       infoOutpeField)

        -> Neon::set::Container
    {
        Neon::set::Container container = infoInField.getGrid().newContainer(
            "LBM_iteration",
            [&](Neon::set::Loader& L) -> auto {
                auto& infoIn = L.load(infoInField, Neon::Pattern::STENCIL);
                auto& infoOut = L.load(infoOutpeField);

                return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                    CellType cellType = infoIn(gidx, 0);
                    cellType.wallNghBitflag = 0;

                    if (cellType.classification == CellType::bulk) {
                        Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto GOid) {
                            if constexpr (GOid != Lattice::center) {
                                constexpr int BKid = Lattice::oppositeDirection[GOid];
                                constexpr int BKx = Lattice::stencil[BKid].x;
                                constexpr int BKy = Lattice::stencil[BKid].y;
                                constexpr int BKz = Lattice::stencil[BKid].z;

                                CellType nghCellType = infoIn.template getNghData<BKx, BKy, BKz>(gidx, 0, CellType::undefined)();
                                if (nghCellType.classification != CellType::bulk) {
                                    cellType.wallNghBitflag = cellType.wallNghBitflag | ((uint32_t(1) << GOid));
                                }
                            }
                        });

                        infoOut(gidx, 0) = cellType;
                    }
                };
            });
        return container;
    }


    static auto
    computeRhoAndU([[maybe_unused]] const PopField& fInField /*!   inpout population field */,
                   const CellTypeField&             cellTypeField /*!       Cell type field     */,
                   Rho&                             rhoField /*!  output Population field */,
                   U&                               uField /*!  output Population field */)

        -> Neon::set::Container
    {
        Neon::set::Container container = fInField.getGrid().newContainer(
            "LBM_iteration",
            [&](Neon::set::Loader& L) -> auto {
                auto& fIn = L.load(fInField,
                                   Neon::Pattern::STENCIL);
                auto& rhoXpu = L.load(rhoField);
                auto& uXpu = L.load(uField);

                const auto& cellInfoPartition = L.load(cellTypeField);

                return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                    CellType               cellInfo = cellInfoPartition(gidx, 0);
                    Compute                rho = 0;
                    std::array<Compute, 3> u{.0, .0, .0};
                    Storage                popIn[Lattice::Q];

                    if (cellInfo.classification == CellType::bulk) {

                        Functions::pullStream(gidx, cellInfo.wallNghBitflag, fIn, NEON_OUT popIn);
                        Functions::macroscopic(popIn, NEON_OUT rho, NEON_OUT u);

                    } else {
                        if (cellInfo.classification == CellType::movingWall) {
                            Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto GOid) {
                                if constexpr (GOid == Lattice::center) {
                                    popIn[Lattice::center] = fIn(gidx, Lattice::center);
                                } else {
                                    popIn[GOid] = fIn(gidx, GOid);
                                }
                            });

                            rho = 1.0;
                            u = std::array<Compute, 3>{static_cast<Compute>(popIn[0]) / static_cast<Compute>(6. * 1. / 18.),
                                                       static_cast<Compute>(popIn[1]) / static_cast<Compute>(6. * 1. / 18.),
                                                       static_cast<Compute>(popIn[2]) / static_cast<Compute>(6. * 1. / 18.)};
                        }
                    }

                    rhoXpu(gidx, 0) = static_cast<Storage>(rho);
                    uXpu(gidx, 0) = static_cast<Storage>(u[0]);
                    uXpu(gidx, 1) = static_cast<Storage>(u[1]);
                    uXpu(gidx, 2) = static_cast<Storage>(u[2]);
                };
            });
        return container;
    }
};
