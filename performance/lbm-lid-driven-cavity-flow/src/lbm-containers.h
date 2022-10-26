template <typename Population_ta>
struct LBMiteration
{
    using Element = typename Population_ta::element_t;
    using Eidx = typename Population_ta::eIdx_t;
    using LatticeParameters = dGridCoPhase::LatticeParameters<Element>;
    using Krn = kernelsUtils<typename Population_ta::local_t>;

    static auto debugSetToIdx(const Flag&    flagG /*!    Cell type field      */,
                              Population_ta& fIn /*!     In population field */,
                              Population_ta& fOut /*!    In population field */)
        -> Neon::set::Container
    {
        auto Kontainer = fIn.grid().container(
            "LBM_iteration", [&](Neon::set::Loader_t & L) -> auto {
                auto& dfIn = L.load(fIn.self());  //;, Neon::Compute::STENCIL);
                auto& dfOut = L.load(fOut);
                auto& flag = L.load(flagG.cself());  // We don't specify stencil for flag as it is alwasy read only

                return [=] NEON_CUDA_HOST_DEVICE(const Eidx& eIdx) mutable -> void {
                    if (flag.eVal(eIdx) == CellType::bulk) {

                        for (int q = 0; q < 19; q++) {
                            dfIn.eRef(eIdx, q) = 770000 + eIdx * 100 + q;
                            dfOut.eRef(eIdx, q) = -(330000 + eIdx * 100 + q);
                        }
                    } else {
                        for (int q = 0; q < 19; q++) {
                            dfOut.eRef(eIdx, q) = -(100000 + eIdx * 100 + q);
                            dfIn.eRef(eIdx, q) = -(100000 + eIdx * 100 + q);
                        }
                    }
                };
            });
        return Kontainer;
    }

    static auto iteration(const Population_ta&                    fIn /*!     In population field */,
                          Population_ta&                          fOut /*!    In population field */,
                          const Flag&                             flag /*!    Cell type field     */,
                          const LatticeParameters&                lp /*!      Lattice parameters  */,
                          const typename Population_ta::element_t omega /*!   LBM omega parameter */)
        -> Neon::set::Container
    {
        auto Kontainer = fIn.grid().container(
            "LBM_iteration", [&, omega ](Neon::set::Loader_t & L) -> auto {
                auto& dfIn = L.load(fIn.cself(), Neon::Compute::STENCIL);
                auto& dfOut = L.load(fOut);

                auto&       dflag = L.load(flag);
                const auto& dLp = L.load(lp);

                return [=] NEON_CUDA_HOST_DEVICE(
                           const Eidx& i) mutable {
                    if (dflag.eVal(i) == CellType::bulk) {


                        const auto [rho, u] = Krn::macroscopic(i, dfIn);
                        const Element usqr = 1.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);

                        for (int k = 0; k < 9; ++k) {
                            const int kOpposite = k + 9;

                            const auto [pop_out, pop_out_opp] = Krn::collideBgk(i, dfIn,
                                                                                dLp,
                                                                                k, kOpposite,
                                                                                rho, u,
                                                                                usqr, omega);

                            Krn::stream(i, k, kOpposite, dfIn, dflag, pop_out, dfOut);
                            Krn::stream(i, kOpposite, k, dfIn, dflag, pop_out_opp, dfOut);
                        }

                        // Treat the case of the "rest-population" (c[k] = {0, 0, 0,}).
                        {
                            const int    k = 18;
                            const double eq = rho * (1. / 3.) * (1. - usqr);
                            dfOut.eRef(i, k) = (1. - omega) * dfIn.eVal(i, k) + omega * eq;
                        }
                    }
                };
            });
        return Kontainer;
    }

    static auto iterationUnrolled(const Stencil19&                        s19Global,
                                  const Population_ta&                    fIn /*!     In population field */,
                                  Population_ta&                          fOut /*!    In population field */,
                                  const Flag&                             flag /*!    Cell type field     */,
                                  const LatticeParameters&                lp /*!      Lattice parameters  */,
                                  const typename Population_ta::element_t omega /*!   LBM omega parameter */)
        -> Neon::set::Container
    {
        auto Kontainer = fIn.grid().container(
            "LBM_iteration", [&, omega ](Neon::set::Loader_t & L) -> auto {
                auto& dfIn = L.load(fIn.cself(), Neon::Compute::STENCIL);
                auto& dfOut = L.load(fOut);

                auto&       dflag = L.load(flag);
                const auto& dLp = L.load(lp);
                const Stencil19::stencil* s19 = L.loadUserContant(s19Global.mem).addr();

                return [=] NEON_CUDA_HOST_DEVICE(
                           const Eidx& i) mutable {
                    if (dflag.eVal(i) == CellType::bulk) {
                        double pop[19];
                        pop[0] = dfIn.eVal(0, 0);
                        pop[1] = dfIn.eVal(1, 0);
                        pop[2] = dfIn.eVal(2, 0);
                        pop[3] = dfIn.eVal(3, 0);
                        pop[4] = dfIn.eVal(4, 0);
                        pop[5] = dfIn.eVal(5, 0);
                        pop[6] = dfIn.eVal(6, 0);
                        pop[7] = dfIn.eVal(7, 0);
                        pop[8] = dfIn.eVal(8, 0);
                        pop[9] = dfIn.eVal(9, 0);
                        pop[10] = dfIn.eVal(10, 0);
                        pop[11] = dfIn.eVal(11, 0);
                        pop[12] = dfIn.eVal(12, 0);
                        pop[13] = dfIn.eVal(13, 0);
                        pop[14] = dfIn.eVal(14, 0);
                        pop[15] = dfIn.eVal(15, 0);
                        pop[16] = dfIn.eVal(16, 0);
                        pop[17] = dfIn.eVal(17, 0);
                        pop[18] = dfIn.eVal(18, 0);


                        const auto [rho, u] = Krn::macroscopic(i, pop);

                        const Element usqr = 1.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
                        Krn::collideAndStreamBgkUnrolled(s19->s, i, pop, dfIn,
                                                         dLp,
                                                         rho, u,
                                                         usqr, omega,
                                                         dflag,
                                                         dfOut);
                    }
                };
            });
        return Kontainer;
    }
};