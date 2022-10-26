template <typename FLocal_ta>
struct kernelsUtils
{
    using Element = typename FLocal_ta::element_t;
    using Eidx = typename FLocal_ta::eIdx_t;
    using LatticeParameters = dGridCoPhase::LatticeParameters<Element>;

    static inline NEON_CUDA_HOST_DEVICE auto macroscopic(const Eidx& i, const FLocal_ta& fin)
    {
        using Element = typename FLocal_ta::element_t;

        const Element X_M1 = fin.eVal(i, 0) + fin.eVal(i, 3) + fin.eVal(i, 4) + fin.eVal(i, 5) + fin.eVal(i, 6);
        const Element X_P1 = fin.eVal(i, 10 - 1) + fin.eVal(i, 13 - 1) + fin.eVal(i, 14 - 1) + fin.eVal(i, 15 - 1) +
                             fin.eVal(i, 16 - 1);
        const Element X_0 = fin.eVal(i, 9 + 9) + fin.eVal(i, 1) + fin.eVal(i, 2) + fin.eVal(i, 7) + fin.eVal(i, 8) +
                            fin.eVal(i, 11 - 1) + fin.eVal(i, 12 - 1) + fin.eVal(i, 17 - 1) + fin.eVal(i, 18 - 1);

        const Element Y_M1 =
            fin.eVal(i, 1) + fin.eVal(i, 3) + fin.eVal(i, 7) + fin.eVal(i, 8) + fin.eVal(i, 14 - 1);
        const Element Y_P1 = fin.eVal(i, 4) + fin.eVal(i, 11 - 1) + fin.eVal(i, 13 - 1) + fin.eVal(i, 17 - 1) +
                             fin.eVal(i, 18 - 1);

        const Element Z_M1 =
            fin.eVal(i, 2) + fin.eVal(i, 5) + fin.eVal(i, 7) + fin.eVal(i, 16 - 1) + fin.eVal(i, 18 - 1);
        const Element Z_P1 =
            fin.eVal(i, 6) + fin.eVal(i, 8) + fin.eVal(i, 12 - 1) + fin.eVal(i, 15 - 1) + fin.eVal(i, 17 - 1);

        const Element          rho = X_M1 + X_P1 + X_0;
        std::array<Element, 3> u{(X_P1 - X_M1) / rho, (Y_P1 - Y_M1) / rho, (Z_P1 - Z_M1) / rho};
        return std::make_tuple(rho, u);
    }

    static inline NEON_CUDA_HOST_DEVICE auto
    macroscopic(const Eidx& i, typename FLocal_ta::element_t const* const pop)
    {
        using Element = typename FLocal_ta::element_t;

        const Element X_M1 = pop[0] + pop[3] + pop[4] + pop[5] + pop[6];
        const Element X_P1 = pop[10 - 1] + pop[13 - 1] + pop[14 - 1] + pop[15 - 1] + pop[16 - 1];
        const Element X_0 =
            pop[9 + 9] + pop[1] + pop[2] + pop[7] + pop[8] + pop[11 - 1] + pop[12 - 1] + pop[17 - 1] +
            pop[18 - 1];

        const Element Y_M1 = pop[1] + pop[3] + pop[7] + pop[8] + pop[14 - 1];
        const Element Y_P1 = pop[4] + pop[11 - 1] + pop[13 - 1] + pop[17 - 1] + pop[18 - 1];

        const Element Z_M1 = pop[2] + pop[5] + pop[7] + pop[16 - 1] + pop[18 - 1];
        const Element Z_P1 = pop[6] + pop[8] + pop[12 - 1] + pop[15 - 1] + pop[17 - 1];

        const Element          rho = X_M1 + X_P1 + X_0;
        std::array<Element, 3> u{(X_P1 - X_M1) / rho, (Y_P1 - Y_M1) / rho, (Z_P1 - Z_M1) / rho};
        return std::make_tuple(rho, u);
    }

    static inline NEON_CUDA_HOST_DEVICE auto stream(const Neon::int8_3d* s,
                                                    const Eidx&          i /*!         Iterator                   */,
                                                    const int&           k /*!         Lattice direction          */,
                                                    const int&           kOpposit /*!  Opposite lattice direction */,
                                                    const FLocal_ta&     fIn /*!       Populations                */,
                                                    const Flag::local_t& flag /*!      Material type              */,
                                                    const Element&       pop_out /*!   Computed population out    */,
                                                    NEON_OUT FLocal_ta&

                                                        fOut /*!      Populations                */)
    {
        // We don't need to check if it is a valid neighbour.
        // stream is called only on bulk cells
        // The bulk cells are all surrounded by boundary cells
        auto flagNBval = flag.nghVal(i, s[k], Flag::local_t::element_t::bulk).value;
        if (flagNBval == CellType::bounce_back) {
            fOut.eRef(i, kOpposit) = pop_out + fIn.nghVal(i, s[k], k, 0.0).value;
        } else {
            fOut.nghRef(i, s[k], k) = pop_out;
        }
    }

    // Execute second-order BGK collision on a cell.
    static inline NEON_CUDA_HOST_DEVICE auto collideBgk(Eidx const&                                i /*!     Element iterator   */,
                                                        FLocal_ta const&                           fin /*!   Population         */,
                                                        typename LatticeParameters::local_t const& lp /*!    Lattice parameters */,
                                                        int const&                                 k /*!     Target cardinality */,
                                                        int const&                                 kOpposite /*!     Target cardinality */,
                                                        Element const&                             rho /*!   Density            */,
                                                        std::array<double, 3> const&               u /*!     Velocity           */,
                                                        Element const&                             usqr /*!  Usqr               */,
                                                        Element const&                             omega /*! Omega              */)
    {
        const Element ck_u = lp.c.eRef(k).x * u[0] + lp.c.eRef(k).y * u[1] + lp.c.eRef(k).z * u[2];
        const Element eq = rho * lp.t.eRef(k) * (1. + 3. * ck_u + 4.5 * ck_u * ck_u - usqr);
        const Element eqOpp = eq - 6. * rho * lp.t.eRef(k) * ck_u;
        const Element popOut = (1. - omega) * fin.eVal(i, k) + omega * eq;
        const Element popOutOpp = (1. - omega) * fin(i, kOpposite) + omega * eqOpp;
        return std::make_pair(popOut, popOutOpp);
    }

    static inline NEON_CUDA_HOST_DEVICE auto
    collideAndStreamBgkUnrolled(const Neon::int8_3d*                       s,
                                Eidx const&                                i /*!     Element iterator   */,
                                typename FLocal_ta::element_t const* const pop,
                                FLocal_ta const&                           fin /*!   Population         */,
                                typename LatticeParameters::local_t const& lp /*!    Lattice parameters */,
                                Element const&                             rho /*!   Density            */,
                                std::array<double, 3> const&               u /*!     Velocity           */,
                                Element const&                             usqr /*!  Usqr               */,
                                Element const&                             omega /*! Omega              */,
                                const Flag::local_t&                       flag /*!  Material type      */,
                                FLocal_ta&                                 fOut /*!  Population         */)

        -> void
    {

        const double ck_u03 = u[0] + u[1];
        const double ck_u04 = u[0] - u[1];
        const double ck_u05 = u[0] + u[2];
        const double ck_u06 = u[0] - u[2];
        const double ck_u07 = u[1] + u[2];
        const double ck_u08 = u[1] - u[2];

        const double eq_00 = rho * (1. / 18.) * (1. - 3. * u[0] + 4.5 * u[0] * u[0] - usqr);
        const double eq_01 = rho * (1. / 18.) * (1. - 3. * u[1] + 4.5 * u[1] * u[1] - usqr);
        const double eq_02 = rho * (1. / 18.) * (1. - 3. * u[2] + 4.5 * u[2] * u[2] - usqr);
        const double eq_03 = rho * (1. / 36.) * (1. - 3. * ck_u03 + 4.5 * ck_u03 * ck_u03 - usqr);
        const double eq_04 = rho * (1. / 36.) * (1. - 3. * ck_u04 + 4.5 * ck_u04 * ck_u04 - usqr);
        const double eq_05 = rho * (1. / 36.) * (1. - 3. * ck_u05 + 4.5 * ck_u05 * ck_u05 - usqr);
        const double eq_06 = rho * (1. / 36.) * (1. - 3. * ck_u06 + 4.5 * ck_u06 * ck_u06 - usqr);
        const double eq_07 = rho * (1. / 36.) * (1. - 3. * ck_u07 + 4.5 * ck_u07 * ck_u07 - usqr);
        const double eq_08 = rho * (1. / 36.) * (1. - 3. * ck_u08 + 4.5 * ck_u08 * ck_u08 - usqr);
        const double eq_09 = rho * (1. / 3.) * (1. - usqr);

        const double eqopp_00 = eq_00 + rho * (1. / 18.) * 6. * u[0];
        const double eqopp_01 = eq_01 + rho * (1. / 18.) * 6. * u[1];
        const double eqopp_02 = eq_02 + rho * (1. / 18.) * 6. * u[2];
        const double eqopp_03 = eq_03 + rho * (1. / 36.) * 6. * ck_u03;
        const double eqopp_04 = eq_04 + rho * (1. / 36.) * 6. * ck_u04;
        const double eqopp_05 = eq_05 + rho * (1. / 36.) * 6. * ck_u05;
        const double eqopp_06 = eq_06 + rho * (1. / 36.) * 6. * ck_u06;
        const double eqopp_07 = eq_07 + rho * (1. / 36.) * 6. * ck_u07;
        const double eqopp_08 = eq_08 + rho * (1. / 36.) * 6. * ck_u08;

        const double pop_out_00 = (1. - omega) * pop[0] + omega * eq_00;
        const double pop_out_01 = (1. - omega) * pop[1] + omega * eq_01;
        const double pop_out_02 = (1. - omega) * pop[2] + omega * eq_02;
        const double pop_out_03 = (1. - omega) * pop[3] + omega * eq_03;
        const double pop_out_04 = (1. - omega) * pop[4] + omega * eq_04;
        const double pop_out_05 = (1. - omega) * pop[5] + omega * eq_05;
        const double pop_out_06 = (1. - omega) * pop[6] + omega * eq_06;
        const double pop_out_07 = (1. - omega) * pop[7] + omega * eq_07;
        const double pop_out_08 = (1. - omega) * pop[8] + omega * eq_08;

        const double pop_out_opp_00 = (1. - omega) * pop[10 - 1] + omega * eqopp_00;
        const double pop_out_opp_01 = (1. - omega) * pop[11 - 1] + omega * eqopp_01;
        const double pop_out_opp_02 = (1. - omega) * pop[12 - 1] + omega * eqopp_02;
        const double pop_out_opp_03 = (1. - omega) * pop[13 - 1] + omega * eqopp_03;
        const double pop_out_opp_04 = (1. - omega) * pop[14 - 1] + omega * eqopp_04;
        const double pop_out_opp_05 = (1. - omega) * pop[15 - 1] + omega * eqopp_05;
        const double pop_out_opp_06 = (1. - omega) * pop[16 - 1] + omega * eqopp_06;
        const double pop_out_opp_07 = (1. - omega) * pop[17 - 1] + omega * eqopp_07;
        const double pop_out_opp_08 = (1. - omega) * pop[18 - 1] + omega * eqopp_08;

        const double pop_out_09 = (1. - omega) * pop[9 + 9] + omega * eq_09;

#define DIRECTION_AND_OPPOSITE(DIRECTION)                                                             \
    {                                                                                                 \
        {                                                                                             \
            constexpr int k = DIRECTION;                                                              \
            constexpr int kOpposite = ((DIRECTION) + 10) - 1;                                         \
            auto          flagNBval = flag.nghVal(i, s[k], 0, dGridCoPhase::CellType::bulk).value;    \
            if (flagNBval == dGridCoPhase::CellType::bounce_back) {                                   \
                fOut.eRef(i, kOpposite) = (pop_out_0##DIRECTION + fin.nghVal(i, s[k], k, 0.0).value); \
            } else {                                                                                  \
                fOut.nghRef(i, s[k], k) = pop_out_0##DIRECTION;                                       \
            }                                                                                         \
        }                                                                                             \
        {                                                                                             \
            constexpr int GO = ((DIRECTION) + 10) - 1;                                                \
            constexpr int BK = DIRECTION;                                                             \
            auto          flagNBval = flag.nghVal(i, s[GO], 0, dGridCoPhase::CellType::bulk).value;   \
            if (flagNBval == dGridCoPhase::CellType::bounce_back) {                                   \
                fOut.eRef(i, BK) = (pop_out_opp_0##DIRECTION + fin.nghVal(i, s[GO], GO, 0.0).value);  \
            } else {                                                                                  \
                fOut.nghRef(i, s[GO], GO) = pop_out_opp_0##DIRECTION;                                 \
            }                                                                                         \
        }                                                                                             \
    }

        DIRECTION_AND_OPPOSITE(0);
        DIRECTION_AND_OPPOSITE(1);
        DIRECTION_AND_OPPOSITE(2);
        DIRECTION_AND_OPPOSITE(3);
        DIRECTION_AND_OPPOSITE(4);
        DIRECTION_AND_OPPOSITE(5);
        DIRECTION_AND_OPPOSITE(6);
        DIRECTION_AND_OPPOSITE(7);
        DIRECTION_AND_OPPOSITE(8);
#undef DIRECTION_AND_OPPOSITE

        {
            constexpr int GO = 9 + 9;
            fOut.eRef(i, GO) = pop_out_09;
        }
    }
};