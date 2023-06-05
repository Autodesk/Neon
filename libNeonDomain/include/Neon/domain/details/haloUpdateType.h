#pragma once
namespace Neon {
namespace domain {
struct HaloUpdateMode_e
{
    enum e
    {
        STANDARD, /** Typical update used by finite difference and finite element methods. A 3 component element is moved moved from one partition to another */
        LATTICE,  /** Typical update used by lattice type of computation (LBM for example). Only certen components of an elements are moved to a neighbour */
        HALOUPDATEMODE_LEN /** Number of option in this enum */
    };

    HaloUpdateMode_e() = delete;
    HaloUpdateMode_e(const HaloUpdateMode_e&) = delete;
    HaloUpdateMode_e(HaloUpdateMode_e&&) = delete;
    static auto toString(e val);
};
}  // namespace grids
}  // namespace Neon