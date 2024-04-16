#pragma once

#include "CellType.h"
#include "D3Q19.h"
#include "DeviceD3Q19.h"
#include "Methods.h"
#include "Neon/Neon.h"
#include "Neon/set/Containter.h"

namespace pull {
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

    using PopField = typename Grid::template Field<Storage, Lattice::Q>;
    using CellTypeField = typename Grid::template Field<CellType, 1>;

    using Idx = typename PopField::Idx;
    using Rho = typename Grid::template Field<Storage, 1>;
    using U = typename Grid::template Field<Storage, 3>;

    using PullFunctions = pull::DeviceD3Q19<Precision, Grid>;
    using CommonFunctions = common::DeviceD3Q19<Precision, Grid>;

    static auto
    iteration(Neon::set::StencilSemantic stencilSemantic,
              const PopField&            fInField /*!      Input population field */,
              const CellTypeField&       cellTypeField /*! Cell type field     */,
              const Compute              omega /*!         LBM omega parameter */,
              PopField&                  fOutField /*!     Output Population field */)
        -> Neon::set::Container
    {
        Neon::set::Container container = fInField.getGrid().newContainer(
            "D3Q19_TwoPop_Pull",
            [&, omega](Neon::set::Loader& L) -> auto {
                auto&       fIn = L.load(fInField,
                                         Neon::Pattern::STENCIL, stencilSemantic);
                auto&       fOut = L.load(fOutField);
                const auto& cellInfoPartition = L.load(cellTypeField);

                return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                    CellType cellInfo = cellInfoPartition(gidx, 0);
                    if (cellInfo.classification == CellType::bulk) {

                        Storage popIn[Lattice::Q];
                        PullFunctions::pullStream(gidx, cellInfo.wallNghBitflag, fIn, NEON_OUT popIn);

                        Compute                rho;
                        std::array<Compute, 3> u{.0, .0, .0};
                        CommonFunctions::macroscopic(popIn, NEON_OUT rho, NEON_OUT u);

                        Compute usqr = 1.5 * (u[0] * u[0] +
                                              u[1] * u[1] +
                                              u[2] * u[2]);

                        PullFunctions::collideBgkUnrolled(gidx,
                                                          popIn,
                                                          rho, u,
                                                          usqr, omega,
                                                          NEON_OUT fOut);
                    }
                };
            });
        return container;
    }

};
}  // namespace pull
namespace push {
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

    using PopField = typename Grid::template Field<Storage, Lattice::Q>;
    using CellTypeField = typename Grid::template Field<CellType, 1>;

    using Idx = typename PopField::Idx;
    using Rho = typename Grid::template Field<Storage, 1>;
    using U = typename Grid::template Field<Storage, 3>;

    using PushFunctions = push::DeviceD3Q19<Precision, Grid>;
    using CommonFunctions = common::DeviceD3Q19<Precision, Grid>;

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
                        PushFunctions::localLoad(gidx, fIn, NEON_OUT popIn);

                        Compute                rho;
                        std::array<Compute, 3> u{.0, .0, .0};
                        CommonFunctions::macroscopic(popIn,
                                                     NEON_OUT rho, NEON_OUT u);

                        Compute usqr = 1.5 * (u[0] * u[0] +
                                              u[1] * u[1] +
                                              u[2] * u[2]);

                        CommonFunctions::collideBgkUnrolled(gidx,
                                                            rho, u,
                                                            usqr, omega,
                                                            NEON_IO popIn);

                        PushFunctions::pushStream(gidx, cellInfo.wallNghBitflag, popIn, NEON_OUT fOut);
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
                        Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto GOMemoryId) {
                            if constexpr (GOMemoryId != Lattice::Memory::center) {
                                constexpr int BKMemoryId = Lattice::Memory::opposite[GOMemoryId];
                                constexpr int BKx = Lattice::Memory::stencil[BKMemoryId].x;
                                constexpr int BKy = Lattice::Memory::stencil[BKMemoryId].y;
                                constexpr int BKz = Lattice::Memory::stencil[BKMemoryId].z;

                                CellType nghCellType = infoIn.template getNghData<BKx, BKy, BKz>(gidx, 0, CellType::undefined)();
                                if (nghCellType.classification != CellType::bulk) {
                                    cellType.wallNghBitflag = cellType.wallNghBitflag | ((uint32_t(1) << GOMemoryId));
                                }
                            }
                        });

                        infoOut(gidx, 0) = cellType;
                    }
                };
            });
        return container;
    }
};
}  // namespace push
namespace common {
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

    using PopField = typename Grid::template Field<Storage, Lattice::Q>;
    using CellTypeField = typename Grid::template Field<CellType, 1>;

    using Idx = typename PopField::Idx;
    using Rho = typename Grid::template Field<Storage, 1>;
    using U = typename Grid::template Field<Storage, 3>;

    using PullFunctions = pull::DeviceD3Q19<Precision, Grid>;
    using PushFunctions = push::DeviceD3Q19<Precision, Grid>;
    using CommonFunctions = common::DeviceD3Q19<Precision, Grid>;

    template <int method>
    static auto
    iteration(Neon::set::StencilSemantic stencilSemantic,
              const PopField&            fInField /*!      Input population field */,
              const CellTypeField&       cellTypeField /*! Cell type field     */,
              const Compute              omega /*!         LBM omega parameter */,
              PopField&                  fOutField /*!     Output Population field */)
        -> Neon::set::Container
    {
        if constexpr (method == int(Method::push)) {
            using Factory = push::ContainerFactory<Precision_,
                                                   D3Q19<Precision_>,
                                                   Grid_>;
            return Factory::iteration(stencilSemantic,
                                      fInField,
                                      fOutField,
                                      omega,
                                      fOutField);
        }
        if constexpr (method == int(Method::pull)) {
            using Factory = pull::ContainerFactory<Precision_,
                                                   D3Q19<Precision_>,
                                                   Grid_>;
            return Factory::iteration(stencilSemantic,
                                      fInField,
                                      fOutField,
                                      omega,
                                      fOutField);
        }
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
                        Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto GOMemoryId) {
                            if constexpr (GOMemoryId != Lattice::Memory::center) {
                                constexpr int BKMemoryId = Lattice::Memory::opposite[GOMemoryId];
                                constexpr int BKx = Lattice::Memory::stencil[BKMemoryId].x;
                                constexpr int BKy = Lattice::Memory::stencil[BKMemoryId].y;
                                constexpr int BKz = Lattice::Memory::stencil[BKMemoryId].z;

                                CellType nghCellType = infoIn.template getNghData<BKx, BKy, BKz>(gidx, 0, CellType::undefined)();
                                if (nghCellType.classification != CellType::bulk) {
                                    cellType.wallNghBitflag = cellType.wallNghBitflag | ((uint32_t(1) << GOMemoryId));
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
        Neon::set::Container container =
            fInField.getGrid().newContainer(
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
                        CommonFunctions::localLoad(gidx, fIn, NEON_OUT popIn);

                        if (cellInfo.classification == CellType::bulk) {
                            CommonFunctions::macroscopic(popIn, NEON_OUT rho, NEON_OUT u);
                        } else {
                            if (cellInfo.classification == CellType::movingWall) {
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

    static auto
    problemSetup(PopField&       fInField /*!   inpout population field */,
                 PopField&       fOutField,
                 CellTypeField&  cellTypeField,
                 Neon::double_3d ulid,
                 double          ulb)

        -> Neon::set::Container
    {
        Neon::set::Container container = fInField.getGrid().newContainer(
            "LBM_iteration",
            [&, ulid, ulb](Neon::set::Loader& L) -> auto {
                auto& fIn = L.load(fInField, Neon::Pattern::MAP);
                auto& fOut = L.load(fOutField, Neon::Pattern::MAP);
                auto& cellInfoPartition = L.load(cellTypeField, Neon::Pattern::MAP);

                return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                    const auto globalIdx = fIn.getGlobalIndex(gidx);
                    const auto domainDim = fIn.getDomainSize();

                    CellType flagVal;
                    flagVal.classification = CellType::bulk;
                    flagVal.wallNghBitflag = 0;

                    typename Lattice::Precision::Storage popVal = 0;

                    if (globalIdx.x == 0 || globalIdx.x == domainDim.x - 1 ||
                        globalIdx.y == 0 || globalIdx.y == domainDim.y - 1 ||
                        globalIdx.z == 0 || globalIdx.z == domainDim.z - 1) {
                        flagVal.classification = CellType::bounceBack;

                        if (globalIdx.y == domainDim.y - 1) {
                            flagVal.classification = CellType::movingWall;
                        }

                        Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                            if (globalIdx.y == domainDim.y - 1) {
                                popVal = -6. * Lattice::Memory::template getT<q>() * ulb *
                                         (Lattice::Memory::template getDirection<q>().v[0] * ulid.v[0] +
                                          Lattice::Memory::template getDirection<q>().v[1] * ulid.v[1] +
                                          Lattice::Memory::template getDirection<q>().v[2] * ulid.v[2]);
                            } else {
                                popVal = 0;
                            }
                            fIn(gidx, q) = popVal;
                            fOut(gidx, q) = popVal;
                        });
                    } else {
                        flagVal.classification = CellType::bulk;
                        Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                            fIn(gidx, q) = Lattice::Memory::template getT<q>();
                            fOut(gidx, q) = Lattice::Memory::template getT<q>();
                        });
                    }
                    cellInfoPartition(gidx, 0) = flagVal;
                };
            });
        return container;
    }
};
}  // namespace common