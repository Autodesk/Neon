#include "CellType.h"
#include "D3Q19.h"
#include "Neon/Neon.h"
#include "Neon/set/Containter.h"

#define COMPUTE_CAST(VAR) static_cast<LbmComputeType>((VAR))

template <typename Lattice,
          typename PopulationField,
          typename LbmComputeType>
struct LbmContainers
{
};

/**
 * Specialization for Lattice
 * @tparam PopulationField
 * @tparam LbmComputeType
 */
template <typename PopulationField,
          typename LbmComputeType>
struct LbmContainers<D3Q19Template<typename PopulationField::Type, LbmComputeType>,
                     PopulationField,
                     LbmComputeType>
{
    using LbmStoreType = typename PopulationField::Type;
    using CellTypeField = typename PopulationField::Grid::template Field<CellType, 1>;
    using Lattice = D3Q19Template<LbmStoreType, LbmComputeType>;
    using Idx = typename PopulationField::Idx;
    using Grid = typename PopulationField::Grid;
    using Rho = typename Grid::template Field<LbmStoreType, 1>;
    using U = typename Grid::template Field<LbmStoreType, 3>;

#define LOADPOP(GOx, GOy, GOz, GOid, BKx, BKy, BKz, BKid)                                                               \
    {                                                                                                                   \
        { /*GO*/                                                                                                        \
            if (wallBitFlag & (uint32_t(1) << GOid)) {                                                                  \
                /*std::cout << "cell " << i.mLocation << " direction " << GOid << " opposite " << BKid << std::endl; */ \
                popIn[GOid] = fin(i, BKid);                                                                             \
            } else {                                                                                                    \
                popIn[GOid] = fin.template nghVal<BKx, BKy, BKz>(i, GOid, 0.0).value;                                   \
            }                                                                                                           \
        }                                                                                                               \
        { /*BK*/                                                                                                        \
            if (wallBitFlag & (uint32_t(1) << BKid)) {                                                                  \
                popIn[BKid] = fin(i, GOid);                                                                             \
            } else {                                                                                                    \
                popIn[BKid] = fin.template nghVal<GOx, GOy, GOz>(i, BKid, 0.0).value;                                   \
            }                                                                                                           \
        }                                                                                                               \
    }
    static inline NEON_CUDA_HOST_DEVICE auto
    loadPopulation(Idx const&                                 i,
                   const uint32_t&                            wallBitFlag,
                   typename PopulationField::Partition const& fin,
                   NEON_OUT LbmStoreType                      popIn[19])
    {
        // #pragma omp critical
        //        {

        LOADPOP(-1, 0, 0, /*  GOid */ 0, /* --- */ 1, 0, 0, /*  BKid */ 10);
        LOADPOP(0, -1, 0, /*  GOid */ 1, /* --- */ 0, 1, 0, /*  BKid */ 11);
        LOADPOP(0, 0, -1, /*  GOid */ 2, /* --- */ 0, 0, 1, /*  BKid */ 12);
        LOADPOP(-1, -1, 0, /* GOid */ 3, /* --- */ 1, 1, 0, /*  BKid */ 13);
        LOADPOP(-1, 1, 0, /*  GOid */ 4, /* --- */ 1, -1, 0, /* BKid */ 14);
        LOADPOP(-1, 0, -1, /* GOid */ 5, /* --- */ 1, 0, 1, /*  BKid */ 15);
        LOADPOP(-1, 0, 1, /*  GOid */ 6, /* --- */ 1, 0, -1, /* BKid */ 16);
        LOADPOP(0, -1, -1, /* GOid */ 7, /* --- */ 0, 1, 1, /*  BKid */ 17);
        LOADPOP(0, -1, 1, /*  GOid */ 8, /* --- */ 0, 1, -1, /* BKid */ 18);
        //  }
        // Treat the case of the center (c[k] = {0, 0, 0,}).
        {
            popIn[Lattice::centerDirection] = fin(i, Lattice::centerDirection);
        }
    }
#undef LOADPOP

#define PULL_STREAM(GOx, GOy, GOz, GOid, BKx, BKy, BKz, BKid)                                                           \
    {                                                                                                                   \
        { /*GO*/                                                                                                        \
            if (wallBitFlag & (uint32_t(1) << GOid)) {                                                                  \
                /*std::cout << "cell " << i.mLocation << " direction " << GOid << " opposite " << BKid << std::endl; */ \
                popIn[GOid] = fin(gidx, BKid) +                                                                         \
                              fin.template getNghData<BKx, BKy, BKz>(gidx, BKid)();                                     \
            } else {                                                                                                    \
                popIn[GOid] = fin.template getNghData<BKx, BKy, BKz>(gidx, GOid)();                                     \
            }                                                                                                           \
        }                                                                                                               \
        { /*BK*/                                                                                                        \
            if (wallBitFlag & (uint32_t(1) << BKid)) {                                                                  \
                popIn[BKid] = fin(gidx, GOid) + fin.template getNghData<GOx, GOy, GOz>(gidx, GOid)();                   \
            } else {                                                                                                    \
                popIn[BKid] = fin.template getNghData<GOx, GOy, GOz>(gidx, BKid)();                                     \
            }                                                                                                           \
        }                                                                                                               \
    }

    static inline NEON_CUDA_HOST_DEVICE auto
    pullStream(Idx const&                                 gidx,
               const uint32_t&                            wallBitFlag,
               typename PopulationField::Partition const& fin,
               NEON_OUT LbmStoreType                      popIn[19])
    {
        // #pragma omp critical
        //        {
#if 0
        using TopologyByDirection = std::tuple<Neon::int32_3d, int, Neon::int32_3d, int>;
        constexpr std::array<TopologyByDirection, 9> stencil{
            std::make_tuple(Neon::int32_3d(-1, 0, 0), /*  GOid */ 0, /* --- */ Neon::int32_3d(1, 0, 0), /*  BKid */ 10),
            std::make_tuple(Neon::int32_3d(0, -1, 0), /*  GOid */ 1, /* --- */ Neon::int32_3d(0, 1, 0), /*  BKid */ 11),
            std::make_tuple(Neon::int32_3d(0, 0, -1), /*  GOid */ 2, /* --- */ Neon::int32_3d(0, 0, 1), /*  BKid */ 12),
            std::make_tuple(Neon::int32_3d(-1, -1, 0), /* GOid */ 3, /* --- */ Neon::int32_3d(1, 1, 0), /*  BKid */ 13),
            std::make_tuple(Neon::int32_3d(-1, 1, 0), /*  GOid */ 4, /* --- */ Neon::int32_3d(1, -1, 0), /* BKid */ 14),
            std::make_tuple(Neon::int32_3d(-1, 0, -1), /* GOid */ 5, /* --- */ Neon::int32_3d(1, 0, 1), /*  BKid */ 15),
            std::make_tuple(Neon::int32_3d(-1, 0, 1), /*  GOid */ 6, /* --- */ Neon::int32_3d(1, 0, -1), /* BKid */ 16),
            std::make_tuple(Neon::int32_3d(0, -1, -1), /* GOid */ 7, /* --- */ Neon::int32_3d(0, 1, 1), /*  BKid */ 17),
            std::make_tuple(Neon::int32_3d(0, -1, 1), /*  GOid */ 8, /* --- */ Neon::int32_3d(0, 1, -1), /* BKid */ 18)};


        auto pullStream = [&]<int stencilIdx>() {
            static_assert(stencilIdx < 9);
            constexpr int            GOid = std::get<1>(stencil[stencilIdx]);
            constexpr int            BKid = std::get<3>(stencil[stencilIdx]);
            constexpr Neon::int32_3d GoOffset = std::get<0>(stencil[stencilIdx]);
            constexpr Neon::int32_3d BkOffset = std::get<2>(stencil[stencilIdx]);
            {
                if (wallBitFlag & (uint32_t(1) << GOid)) {
                    popIn[GOid] = fin(gidx, BKid) +
                                  fin.template getNghData<BkOffset.x, BkOffset.y, BkOffset.z>(gidx, BKid)();
                } else {
                    popIn[GOid] = fin.template getNghData<BkOffset.x, BkOffset.y, BkOffset.z>(gidx, GOid)();
                }
            }
            { /*BK*/
                if (wallBitFlag & (uint32_t(1) << BKid)) {
                    popIn[BKid] = fin(gidx, GOid) +
                                  fin.template getNghData<GoOffset.x, GoOffset.y, GoOffset.z>(gidx, GOid)();
                } else {
                    popIn[BKid] = fin.template getNghData<GoOffset.x, GoOffset.y, GoOffset.z>(gidx, BKid)();
                }
            }
        };
        pullStream.template operator()<0>();
        pullStream.template operator()<1>();
        pullStream.template operator()<2>();
        pullStream.template operator()<3>();
        pullStream.template operator()<4>();
        pullStream.template operator()<5>();
        pullStream.template operator()<6>();
        pullStream.template operator()<7>();
        pullStream.template operator()<8>();
#endif
        PULL_STREAM(-1, 0, 0, /*  GOid */ 0, /* --- */ 1, 0, 0, /*  BKid */ 10);
        PULL_STREAM(0, -1, 0, /*  GOid */ 1, /* --- */ 0, 1, 0, /*  BKid */ 11);
        PULL_STREAM(0, 0, -1, /*  GOid */ 2, /* --- */ 0, 0, 1, /*  BKid */ 12);
        PULL_STREAM(-1, -1, 0, /* GOid */ 3, /* --- */ 1, 1, 0, /*  BKid */ 13);
        PULL_STREAM(-1, 1, 0, /*  GOid */ 4, /* --- */ 1, -1, 0, /* BKid */ 14);
        PULL_STREAM(-1, 0, -1, /* GOid */ 5, /* --- */ 1, 0, 1, /*  BKid */ 15);
        PULL_STREAM(-1, 0, 1, /*  GOid */ 6, /* --- */ 1, 0, -1, /* BKid */ 16);
        PULL_STREAM(0, -1, -1, /* GOid */ 7, /* --- */ 0, 1, 1, /*  BKid */ 17);
        PULL_STREAM(0, -1, 1, /*  GOid */ 8, /* --- */ 0, 1, -1, /* BKid */ 18);

        //  }
        // Treat the case of the center (c[k] = {0, 0, 0,}).
        {
            popIn[Lattice::centerDirection] = fin(gidx, Lattice::centerDirection);
        }
    }
#undef PULL_STREAM

    static inline NEON_CUDA_HOST_DEVICE auto
    macroscopic(const LbmStoreType       pop[Lattice::Q],
                NEON_OUT LbmComputeType& rho,
                NEON_OUT std::array<LbmComputeType, 3>& u)
        -> void
    {
#define POP(IDX) static_cast<LbmComputeType>(pop[IDX])

        const LbmComputeType X_M1 = POP(0) + POP(3) + POP(4) + POP(5) + POP(6);
        const LbmComputeType X_P1 = POP(10) + POP(13) + POP(14) + POP(15) + POP(16);
        const LbmComputeType X_0 = POP(9) + POP(1) + POP(2) + POP(7) + POP(8) + POP(11) + POP(12) + POP(17) + POP(18);

        const LbmComputeType Y_M1 = POP(1) + POP(3) + POP(7) + POP(8) + POP(14);
        const LbmComputeType Y_P1 = POP(4) + POP(11) + POP(13) + POP(17) + POP(18);

        const LbmComputeType Z_M1 = POP(2) + POP(5) + POP(7) + POP(16) + POP(18);
        const LbmComputeType Z_P1 = POP(6) + POP(8) + POP(12) + POP(15) + POP(17);

#undef POP

        rho = X_M1 + X_P1 + X_0;
        u[0] = (X_P1 - X_M1) / rho;
        u[1] = (Y_P1 - Y_M1) / rho;
        u[2] = (Z_P1 - Z_M1) / rho;
    }


    static inline NEON_CUDA_HOST_DEVICE auto
    collideBgkUnrolled(Idx const&                           i /*!     LbmComputeType iterator   */,
                       const LbmStoreType                   pop[Lattice::Q],
                       LbmComputeType const&                rho /*!   Density            */,
                       std::array<LbmComputeType, 3> const& u /*!     Velocity           */,
                       LbmComputeType const&                usqr /*!  Usqr               */,
                       LbmComputeType const&                omega /*! Omega              */,
                       typename PopulationField::Partition& fOut /*!  Population         */)

        -> void
    {
        const LbmComputeType ck_u03 = u[0] + u[1];
        const LbmComputeType ck_u04 = u[0] - u[1];
        const LbmComputeType ck_u05 = u[0] + u[2];
        const LbmComputeType ck_u06 = u[0] - u[2];
        const LbmComputeType ck_u07 = u[1] + u[2];
        const LbmComputeType ck_u08 = u[1] - u[2];

        const LbmComputeType eq_00 = rho * (1. / 18.) * (1. - 3. * u[0] + 4.5 * u[0] * u[0] - usqr);
        const LbmComputeType eq_01 = rho * (1. / 18.) * (1. - 3. * u[1] + 4.5 * u[1] * u[1] - usqr);
        const LbmComputeType eq_02 = rho * (1. / 18.) * (1. - 3. * u[2] + 4.5 * u[2] * u[2] - usqr);
        const LbmComputeType eq_03 = rho * (1. / 36.) * (1. - 3. * ck_u03 + 4.5 * ck_u03 * ck_u03 - usqr);
        const LbmComputeType eq_04 = rho * (1. / 36.) * (1. - 3. * ck_u04 + 4.5 * ck_u04 * ck_u04 - usqr);
        const LbmComputeType eq_05 = rho * (1. / 36.) * (1. - 3. * ck_u05 + 4.5 * ck_u05 * ck_u05 - usqr);
        const LbmComputeType eq_06 = rho * (1. / 36.) * (1. - 3. * ck_u06 + 4.5 * ck_u06 * ck_u06 - usqr);
        const LbmComputeType eq_07 = rho * (1. / 36.) * (1. - 3. * ck_u07 + 4.5 * ck_u07 * ck_u07 - usqr);
        const LbmComputeType eq_08 = rho * (1. / 36.) * (1. - 3. * ck_u08 + 4.5 * ck_u08 * ck_u08 - usqr);

        const LbmComputeType eqopp_00 = eq_00 + rho * (1. / 18.) * 6. * u[0];
        const LbmComputeType eqopp_01 = eq_01 + rho * (1. / 18.) * 6. * u[1];
        const LbmComputeType eqopp_02 = eq_02 + rho * (1. / 18.) * 6. * u[2];
        const LbmComputeType eqopp_03 = eq_03 + rho * (1. / 36.) * 6. * ck_u03;
        const LbmComputeType eqopp_04 = eq_04 + rho * (1. / 36.) * 6. * ck_u04;
        const LbmComputeType eqopp_05 = eq_05 + rho * (1. / 36.) * 6. * ck_u05;
        const LbmComputeType eqopp_06 = eq_06 + rho * (1. / 36.) * 6. * ck_u06;
        const LbmComputeType eqopp_07 = eq_07 + rho * (1. / 36.) * 6. * ck_u07;
        const LbmComputeType eqopp_08 = eq_08 + rho * (1. / 36.) * 6. * ck_u08;

        const LbmComputeType pop_out_00 = (1. - omega) * static_cast<LbmComputeType>(pop[0]) + omega * eq_00;
        const LbmComputeType pop_out_01 = (1. - omega) * static_cast<LbmComputeType>(pop[1]) + omega * eq_01;
        const LbmComputeType pop_out_02 = (1. - omega) * static_cast<LbmComputeType>(pop[2]) + omega * eq_02;
        const LbmComputeType pop_out_03 = (1. - omega) * static_cast<LbmComputeType>(pop[3]) + omega * eq_03;
        const LbmComputeType pop_out_04 = (1. - omega) * static_cast<LbmComputeType>(pop[4]) + omega * eq_04;
        const LbmComputeType pop_out_05 = (1. - omega) * static_cast<LbmComputeType>(pop[5]) + omega * eq_05;
        const LbmComputeType pop_out_06 = (1. - omega) * static_cast<LbmComputeType>(pop[6]) + omega * eq_06;
        const LbmComputeType pop_out_07 = (1. - omega) * static_cast<LbmComputeType>(pop[7]) + omega * eq_07;
        const LbmComputeType pop_out_08 = (1. - omega) * static_cast<LbmComputeType>(pop[8]) + omega * eq_08;

        const LbmComputeType pop_out_opp_00 = (1. - omega) * static_cast<LbmComputeType>(pop[10]) + omega * eqopp_00;
        const LbmComputeType pop_out_opp_01 = (1. - omega) * static_cast<LbmComputeType>(pop[11]) + omega * eqopp_01;
        const LbmComputeType pop_out_opp_02 = (1. - omega) * static_cast<LbmComputeType>(pop[12]) + omega * eqopp_02;
        const LbmComputeType pop_out_opp_03 = (1. - omega) * static_cast<LbmComputeType>(pop[13]) + omega * eqopp_03;
        const LbmComputeType pop_out_opp_04 = (1. - omega) * static_cast<LbmComputeType>(pop[14]) + omega * eqopp_04;
        const LbmComputeType pop_out_opp_05 = (1. - omega) * static_cast<LbmComputeType>(pop[15]) + omega * eqopp_05;
        const LbmComputeType pop_out_opp_06 = (1. - omega) * static_cast<LbmComputeType>(pop[16]) + omega * eqopp_06;
        const LbmComputeType pop_out_opp_07 = (1. - omega) * static_cast<LbmComputeType>(pop[17]) + omega * eqopp_07;
        const LbmComputeType pop_out_opp_08 = (1. - omega) * static_cast<LbmComputeType>(pop[18]) + omega * eqopp_08;


#define COMPUTE_GO_AND_BACK(GOid, BKid)                                 \
    {                                                                   \
        fOut(i, GOid) = static_cast<LbmStoreType>(pop_out_0##GOid);     \
        fOut(i, BKid) = static_cast<LbmStoreType>(pop_out_opp_0##GOid); \
    }

        COMPUTE_GO_AND_BACK(0, 10)
        COMPUTE_GO_AND_BACK(1, 11)
        COMPUTE_GO_AND_BACK(2, 12)
        COMPUTE_GO_AND_BACK(3, 13)
        COMPUTE_GO_AND_BACK(4, 14)
        COMPUTE_GO_AND_BACK(5, 15)
        COMPUTE_GO_AND_BACK(6, 16)
        COMPUTE_GO_AND_BACK(7, 17)
        COMPUTE_GO_AND_BACK(8, 18)

#undef COMPUTE_GO_AND_BACK

        {
            const LbmComputeType eq_09 = rho * (1. / 3.) * (1. - usqr);
            const LbmComputeType pop_out_09 = (1. - omega) *
                                                  static_cast<LbmComputeType>(pop[Lattice::centerDirection]) +
                                              omega * eq_09;
            fOut(i, Lattice::centerDirection) = static_cast<LbmStoreType>(pop_out_09);
        }
    }

    static auto
    iteration(Neon::set::StencilSemantic stencilSemantic,
              const PopulationField&     fInField /*!   inpout population field */,
              const CellTypeField&       cellTypeField /*!       Cell type field     */,
              const LbmComputeType       omega /*! LBM omega parameter */,
              PopulationField&           fOutField /*!  output Population field */)
        -> Neon::set::Container
    {

        Neon::set::Container container = fInField.getGrid().newContainer(
            "LBM_iteration",
            [&, omega](Neon::set::Loader& L) -> auto {
                auto&       fIn = L.load(fInField,
                                         Neon::Pattern::STENCIL, stencilSemantic);
                auto&       fOut = L.load(fOutField);
                const auto& cellInfoPartition = L.load(cellTypeField);

                return [=] NEON_CUDA_HOST_DEVICE(const typename PopulationField::Idx& gidx) mutable {
                    CellType cellInfo = cellInfoPartition(gidx, 0);
                    if (cellInfo.classification == CellType::bulk) {

                        LbmStoreType popIn[Lattice::Q];
                        pullStream(gidx, cellInfo.wallNghBitflag, fIn, NEON_OUT popIn);

                        LbmComputeType                rho;
                        std::array<LbmComputeType, 3> u{.0, .0, .0};
                        macroscopic(popIn, NEON_OUT rho, NEON_OUT u);

                        LbmComputeType usqr = 1.5 * (u[0] * u[0] +
                                                     u[1] * u[1] +
                                                     u[2] * u[2]);

                        collideBgkUnrolled(gidx,
                                           popIn,
                                           rho, u,
                                           usqr, omega,
                                           NEON_OUT fOut);
                    }
                };
            });
        return container;
    }

#define COMPUTE_MASK_WALL(GOx, GOy, GOz, GOid, BKx, BKy, BKz, BKid)                                           \
    {                                                                                                         \
        { /*GO*/                                                                                              \
            CellType nghCellType = infoIn.template getNghData<BKx, BKy, BKz>(gidx, 0, CellType::undefined)(); \
            if (nghCellType.classification != CellType::bulk) {                                               \
                cellType.wallNghBitflag = cellType.wallNghBitflag | ((uint32_t(1) << GOid));                  \
            }                                                                                                 \
        }                                                                                                     \
        { /*BK*/                                                                                              \
            CellType nghCellType = infoIn.template getNghData<GOx, GOy, GOz>(gidx, 0, CellType::undefined)(); \
            if (nghCellType.classification != CellType::bulk) {                                               \
                cellType.wallNghBitflag = cellType.wallNghBitflag | ((uint32_t(1) << BKid));                  \
            }                                                                                                 \
        }                                                                                                     \
    }

    static auto
    computeWallNghMask(const CellTypeField& infoInField,
                       CellTypeField&       infoOutpeField)

        -> Neon::set::Container
    {
        Neon::set::Container container = infoInField.getGrid().newContainer(
            "LBM_iteration",
            [&](Neon::set::Loader& L) -> auto {
                auto& infoIn = L.load(infoInField,
                                      Neon::Pattern::STENCIL);
                auto& infoOut = L.load(infoOutpeField);

                return [=] NEON_CUDA_HOST_DEVICE(const typename PopulationField::Idx& gidx) mutable {
                    CellType cellType = infoIn(gidx, 0);
                    cellType.wallNghBitflag = 0;

                    if (cellType.classification == CellType::bulk) {
                        COMPUTE_MASK_WALL(-1, 0, 0, /*  GOid */ 0, /* --- */ 1, 0, 0, /*  BKid */ 10)
                        COMPUTE_MASK_WALL(0, -1, 0, /*  GOid */ 1, /* --- */ 0, 1, 0, /*  BKid */ 11)
                        COMPUTE_MASK_WALL(0, 0, -1, /*  GOid */ 2, /* --- */ 0, 0, 1, /*  BKid */ 12)
                        COMPUTE_MASK_WALL(-1, -1, 0, /* GOid */ 3, /* --- */ 1, 1, 0, /*  BKid */ 13)
                        COMPUTE_MASK_WALL(-1, 1, 0, /*  GOid */ 4, /* --- */ 1, -1, 0, /* BKid */ 14)
                        COMPUTE_MASK_WALL(-1, 0, -1, /* GOid */ 5, /* --- */ 1, 0, 1, /*  BKid */ 15)
                        COMPUTE_MASK_WALL(-1, 0, 1, /*  GOid */ 6, /* --- */ 1, 0, -1, /* BKid */ 16)
                        COMPUTE_MASK_WALL(0, -1, -1, /* GOid */ 7, /* --- */ 0, 1, 1, /*  BKid */ 17)
                        COMPUTE_MASK_WALL(0, -1, 1, /*  GOid */ 8, /* --- */ 0, 1, -1, /* BKid */ 18)

                        infoOut(gidx, 0) = cellType;
                    }
                };
            });
        return container;
    }
#undef COMPUTE_MASK_WALL

#define BC_LOAD(GOID, DKID)        \
    popIn[GOID] = fIn(gidx, GOID); \
    popIn[DKID] = fIn(gidx, DKID);

    static auto
    computeRhoAndU([[maybe_unused]] const PopulationField& fInField /*!   inpout population field */,
                   const CellTypeField&                    cellTypeField /*!       Cell type field     */,
                   Rho&                                    rhoField /*!  output Population field */,
                   U&                                      uField /*!  output Population field */)

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

                return [=] NEON_CUDA_HOST_DEVICE(const typename PopulationField::Idx& gidx) mutable {
                    CellType                      cellInfo = cellInfoPartition(gidx, 0);
                    LbmComputeType                rho = 0;
                    std::array<LbmComputeType, 3> u{.0, .0, .0};
                    LbmStoreType                  popIn[Lattice::Q];

                    if (cellInfo.classification == CellType::bulk) {
                        pullStream(gidx, cellInfo.wallNghBitflag, fIn, NEON_OUT popIn);
                        macroscopic(popIn, NEON_OUT rho, NEON_OUT u);
                    } else {
                        if (cellInfo.classification == CellType::movingWall) {
                            BC_LOAD(0, 10)
                            BC_LOAD(1, 11)
                            BC_LOAD(2, 12)
                            BC_LOAD(3, 13)
                            BC_LOAD(4, 14)
                            BC_LOAD(5, 15)
                            BC_LOAD(6, 16)
                            BC_LOAD(7, 17)
                            BC_LOAD(8, 18)
                            popIn[9] = fIn(gidx, 9);

                            rho = 1.0;
                            u = std::array<LbmComputeType, 3>{COMPUTE_CAST(popIn[0]) / COMPUTE_CAST(6. * 1. / 18.),
                                                              COMPUTE_CAST(popIn[1]) / COMPUTE_CAST(6. * 1. / 18.),
                                                              COMPUTE_CAST(popIn[2]) / COMPUTE_CAST(6. * 1. / 18.)};
                        }
                    }

                    rhoXpu(gidx, 0) = static_cast<LbmStoreType>(rho);
                    uXpu(gidx, 0) = static_cast<LbmStoreType>(u[0]);
                    uXpu(gidx, 1) = static_cast<LbmStoreType>(u[1]);
                    uXpu(gidx, 2) = static_cast<LbmStoreType>(u[2]);
                };
            });
        return container;
    }
};

#undef COMPUTE_CAST