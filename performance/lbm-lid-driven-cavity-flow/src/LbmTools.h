#include "CellType.h"
#include "D3Q19.h"
#include "Neon/Neon.h"

template <typename DKQW,
          typename PopulationField,
          typename LbmComputeType>
struct LbmTools
{
};

/**
 * Specialization for D3Q19
 * @tparam PopulationField
 * @tparam LbmComputeType
 */
template <typename PopulationField,
          typename LbmComputeType>
struct LbmTools<D3Q19<typename PopulationField::Type, LbmComputeType>,
                PopulationField,
                LbmComputeType>
{
    using LbmStoreType = typename PopulationField::Type;
    using CellTypeField = typename PopulationField::Grid::template Field<CellType, 1>;
    using D3Q19 = D3Q19<LbmStoreType, LbmComputeType>;
    using Cell = typename PopulationField::Cell;

    static inline NEON_CUDA_HOST_DEVICE auto
    macroscopic(const Cell&        i,
                const LbmStoreType popIn[D3Q19::q])
        -> std::array<LbmComputeType, 3>
    {
        const LbmComputeType X_M1 = popIn[0] + popIn[3] +
                                    popIn[4] + popIn[5] +
                                    popIn[6];

        const LbmComputeType X_P1 = popIn[10] + popIn[13] +
                                    popIn[14] + popIn[15] +
                                    popIn[16];

        const LbmComputeType X_0 = popIn[9] + popIn[1] +
                                   popIn[2] + popIn[7] +
                                   popIn[8] + popIn[11] +
                                   popIn[12] + popIn[17] +
                                   popIn[18];

        const LbmComputeType Y_M1 = popIn[1] + popIn[3] +
                                    popIn[7] + popIn[8] +
                                    popIn[14];

        const LbmComputeType Y_P1 = popIn[4] + popIn[11] +
                                    popIn[13] + popIn[17] +
                                    popIn[18];

        const LbmComputeType Z_M1 = popIn[2] + popIn[5] +
                                    popIn[7] + popIn[16] +
                                    popIn[18];

        const LbmComputeType Z_P1 = popIn[6] + popIn[8] +
                                    popIn[12] + popIn[15] +
                                    popIn[17];

        const LbmComputeType          rho = X_M1 + X_P1 + X_0;
        std::array<LbmComputeType, 3> u{(X_P1 - X_M1) / rho,
                                        (Y_P1 - Y_M1) / rho,
                                        (Z_P1 - Z_M1) / rho};

        return std::make_tuple(rho, u);
    }

    static inline NEON_CUDA_HOST_DEVICE auto
    loadPopulation(Cell const&                                i,
                   uint32_t&                                  bounceBackBitflag,
                   typename PopulationField::Partition const& fin,
                   NEON_OUT LbmStoreType                      popIn[19])
    {

#define LOADPOP(GOx, GOy, GOz, GOid, BKx, BKy, BKz, BKid)                         \
    {                                                                             \
        { /*GO*/                                                                  \
            bool isBounceBack = (bounceBackBitflag & (uint32_t(1) << BKid)) != 0; \
            if (isBounceBack) {                                                   \
                popIn[GOid] = fin(i, BKid);                                       \
            } else {                                                              \
                popIn[GOid] = fin.nghVal<BKx, BKy, BKz>(i, GOid, 0.0).value;      \
            }                                                                     \
        }                                                                         \
        { /*BK*/                                                                  \
            bool isBounceBack = (bounceBackBitflag & (uint32_t(1) << GOid)) != 0; \
            if (isBounceBack) {                                                   \
                popIn[BKid] = fin(i, GOid);                                       \
            } else {                                                              \
                popIn[BKid] = fin.nghVal<GOx, GOy, GOz>(i, BKid, 0.0).value;      \
            }                                                                     \
        }                                                                         \
    }
        LOADPOP(-1, 0, 0, /*  GOid */ 0, /* --- */ 1, 0, 0, /*  BKid */ 10)
        LOADPOP(0, -1, 0, /*  GOid */ 1, /* --- */ 0, 1, 0, /*  BKid */ 11)
        LOADPOP(0, 0, -1, /*  GOid */ 2, /* --- */ 0, 0, 1, /*  BKid */ 12)
        LOADPOP(-1, -1, 0, /* GOid */ 3, /* --- */ 1, 1, 0, /*  BKid */ 13)
        LOADPOP(-1, 1, 0, /*  GOid */ 4, /* --- */ 1, -1, 0, /* BKid */ 14)
        LOADPOP(-1, 0, -1, /* GOid */ 5, /* --- */ 1, 0, 1, /*  BKid */ 15)
        LOADPOP(-1, 0, 1, /*  GOid */ 6, /* --- */ 1, 0, -1, /* BKid */ 16)
        LOADPOP(0, -1, -1, /* GOid */ 7, /* --- */ 0, 1, 1, /*  BKid */ 17)
        LOADPOP(0, -1, 1, /*  GOid */ 8, /* --- */ 0, 1, -1, /* BKid */ 18)

        // Treat the case of the "rest-population" (c[k] = {0, 0, 0,}).
        {
            popIn[D3Q19::centerDirection] = fin(i, D3Q19::centerDirection);
            bounceBackBitflag = bounceBackBitflag | (uint32_t(1) << D3Q19::centerDirection);
        }
    }


    static inline NEON_CUDA_HOST_DEVICE auto
    macroscopic(const Cell&        i,
                const LbmStoreType pop[D3Q19::q])
        -> std::array<LbmComputeType, 3>
    {
#define POP(IDX) static_cast<LbmComputeType>(pop[IDX])
        const LbmComputeType X_M1 = POP(0) + POP(3) + POP(4) + POP(5) + POP(6);
        const LbmComputeType X_P1 = POP(10) + POP(13) + POP(14) + POP(15) + POP(16);

        const LbmComputeType X_0 = POP(9) +
                                   POP(1) + POP(2) + POP(7) + POP(8) +
                                   POP(11) + POP(12) + POP(17) + POP(18);

        const LbmComputeType Y_M1 = POP(1) + POP(3) + POP(7) + POP(8) + POP(14);
        const LbmComputeType Y_P1 = POP(4) + POP(11) + POP(13) + POP(17) + POP(18);

        const LbmComputeType Z_M1 = POP(2) + POP(5) + POP(7) + POP(16) + POP(18);
        const LbmComputeType Z_P1 = POP(6) + POP(8) + POP(12) + POP(15) + POP(17);
#undef POP

        const LbmComputeType          rho = X_M1 + X_P1 + X_0;
        std::array<LbmComputeType, 3> u{(X_P1 - X_M1) / rho, (Y_P1 - Y_M1) / rho, (Z_P1 - Z_M1) / rho};
        return std::make_tuple(rho, u);
    }


    static inline NEON_CUDA_HOST_DEVICE auto
    collideAndStreamBgkUnrolled(Cell const&                                i /*!     LbmComputeType iterator   */,
                                const LbmStoreType                         pop[D3Q19::q],
                                typename PopulationField::Partition const& fin /*!   Population         */,
                                LbmComputeType const&                      rho /*!   Density            */,
                                std::array<double, 3> const&               u /*!     Velocity           */,
                                LbmComputeType const&                      usqr /*!  Usqr               */,
                                LbmComputeType const&                      omega /*! Omega              */,
                                uint32_t const&                            bounceBackBitflag /*!  BC mask */,
                                typename PopulationField::Partition&       fOut /*!  Population         */)

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
        const LbmComputeType eq_09 = rho * (1. / 3.) * (1. - usqr);

        const LbmComputeType eqopp_00 = eq_00 + rho * (1. / 18.) * 6. * u[0];
        const LbmComputeType eqopp_01 = eq_01 + rho * (1. / 18.) * 6. * u[1];
        const LbmComputeType eqopp_02 = eq_02 + rho * (1. / 18.) * 6. * u[2];
        const LbmComputeType eqopp_03 = eq_03 + rho * (1. / 36.) * 6. * ck_u03;
        const LbmComputeType eqopp_04 = eq_04 + rho * (1. / 36.) * 6. * ck_u04;
        const LbmComputeType eqopp_05 = eq_05 + rho * (1. / 36.) * 6. * ck_u05;
        const LbmComputeType eqopp_06 = eq_06 + rho * (1. / 36.) * 6. * ck_u06;
        const LbmComputeType eqopp_07 = eq_07 + rho * (1. / 36.) * 6. * ck_u07;
        const LbmComputeType eqopp_08 = eq_08 + rho * (1. / 36.) * 6. * ck_u08;

        const double pop_out_00 = (1. - omega) * static_cast<LbmComputeType>(pop[0]) + omega * eq_00;
        const double pop_out_01 = (1. - omega) * static_cast<LbmComputeType>(pop[1]) + omega * eq_01;
        const double pop_out_02 = (1. - omega) * static_cast<LbmComputeType>(pop[2]) + omega * eq_02;
        const double pop_out_03 = (1. - omega) * static_cast<LbmComputeType>(pop[3]) + omega * eq_03;
        const double pop_out_04 = (1. - omega) * static_cast<LbmComputeType>(pop[4]) + omega * eq_04;
        const double pop_out_05 = (1. - omega) * static_cast<LbmComputeType>(pop[5]) + omega * eq_05;
        const double pop_out_06 = (1. - omega) * static_cast<LbmComputeType>(pop[6]) + omega * eq_06;
        const double pop_out_07 = (1. - omega) * static_cast<LbmComputeType>(pop[7]) + omega * eq_07;
        const double pop_out_08 = (1. - omega) * static_cast<LbmComputeType>(pop[8]) + omega * eq_08;

        const double pop_out_opp_00 = (1. - omega) * static_cast<LbmComputeType>(pop[10]) + omega * eqopp_00;
        const double pop_out_opp_01 = (1. - omega) * static_cast<LbmComputeType>(pop[11]) + omega * eqopp_01;
        const double pop_out_opp_02 = (1. - omega) * static_cast<LbmComputeType>(pop[12]) + omega * eqopp_02;
        const double pop_out_opp_03 = (1. - omega) * static_cast<LbmComputeType>(pop[13]) + omega * eqopp_03;
        const double pop_out_opp_04 = (1. - omega) * static_cast<LbmComputeType>(pop[14]) + omega * eqopp_04;
        const double pop_out_opp_05 = (1. - omega) * static_cast<LbmComputeType>(pop[15]) + omega * eqopp_05;
        const double pop_out_opp_06 = (1. - omega) * static_cast<LbmComputeType>(pop[16]) + omega * eqopp_06;
        const double pop_out_opp_07 = (1. - omega) * static_cast<LbmComputeType>(pop[17]) + omega * eqopp_07;
        const double pop_out_opp_08 = (1. - omega) * static_cast<LbmComputeType>(pop[18]) + omega * eqopp_08;


#define DIRECTION_AND_OPPOSITE(GOx, GOy, GOz, GOid, BKx, BKy, BKz, BKid)                                                               \
    {                                                                                                                                  \
        /** NOTE the double point operation is always used with GOid */                                                                \
        /** As all data is computed parametrically w.r.t the first part of the symmetry */                                             \
        {                                                                                                                              \
            if (bounceBackBitflag & (uint32_t(1) << GOid)) {                                                                           \
                fOut(i, BKid) = static_cast<LbmStoreType>(pop_out_0##GOid +                                                            \
                                                          static_cast<LbmComputeType>(fin.nghVal<GOx, GOy, GOz>(i, GOid, 0.0).value)); \
            } else {                                                                                                                   \
                fOut(i, GOid) = static_cast<LbmStoreType>(pop_out_0##GOid);                                                            \
            }                                                                                                                          \
        }                                                                                                                              \
                                                                                                                                       \
        {                                                                                                                              \
            if (bounceBackBitflag & (uint32_t(1) << BKid)) {                                                                           \
                fOut(i, GOid) = static_cast<LbmStoreType>(pop_out_opp_0##GOid +                                                        \
                                                          static_cast<LbmComputeType>(fin.nghVal<BKx, BKy, BKz>(i, BKid, 0.0).value)); \
            } else {                                                                                                                   \
                fOut(i, BKid) = static_cast<LbmStoreType>(pop_out_opp_0##GOid);                                                        \
            }                                                                                                                          \
        }                                                                                                                              \
    }

        DIRECTION_AND_OPPOSITE(-1, 0, 0, /*  GOid */ 0, /* --- */ 1, 0, 0, /*  BKid */ 10)
        DIRECTION_AND_OPPOSITE(0, -1, 0, /*  GOid */ 1, /* --- */ 0, 1, 0, /*  BKid */ 11)
        DIRECTION_AND_OPPOSITE(0, 0, -1, /*  GOid */ 2, /* --- */ 0, 0, 1, /*  BKid */ 12)
        DIRECTION_AND_OPPOSITE(-1, -1, 0, /* GOid */ 3, /* --- */ 1, 1, 0, /*  BKid */ 13)
        DIRECTION_AND_OPPOSITE(-1, 1, 0, /*  GOid */ 4, /* --- */ 1, -1, 0, /* BKid */ 14)
        DIRECTION_AND_OPPOSITE(-1, 0, -1, /* GOid */ 5, /* --- */ 1, 0, 1, /*  BKid */ 15)
        DIRECTION_AND_OPPOSITE(-1, 0, 1, /*  GOid */ 6, /* --- */ 1, 0, -1, /* BKid */ 16)
        DIRECTION_AND_OPPOSITE(0, -1, -1, /* GOid */ 7, /* --- */ 0, 1, 1, /*  BKid */ 17)
        DIRECTION_AND_OPPOSITE(0, -1, 1, /*  GOid */ 8, /* --- */ 0, 1, -1, /* BKid */ 18)

#undef DIRECTION_AND_OPPOSITE

        {
            const LbmComputeType pop_out_09 = (1. - omega) * static_cast<LbmComputeType>(pop[D3Q19::centerDirection]) +
                                              omega * eq_09;
            fOut(i, D3Q19::centerDirection) = static_cast<LbmStoreType>(pop_out_09);
        }
    }
};