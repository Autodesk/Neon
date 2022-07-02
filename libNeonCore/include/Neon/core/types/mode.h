#pragma once

#include <string>


namespace Neon {
struct run_et
{

    enum et
    {
        async,
        sync
    };

    et m_mode;


    run_et(et mode);

    static const char* name(const et& mode);

    [[nodiscard]] const char* name() const;

    bool operator==(et m);
    bool operator==(run_et m);
};

struct managedMode_t
{

    enum managedMode_e
    {
        system,
        user
    };

    managedMode_t(managedMode_e mode);

    managedMode_e mode;

    static const char* name(const managedMode_e& mode);

    [[nodiscard]] const char* name() const;

    bool operator==(managedMode_e m);
    bool operator==(managedMode_t m);
};

struct computeMode_t
{

    enum computeMode_e
    {
        par,
        seq
    };

    computeMode_t(computeMode_e mode);

    computeMode_e mode;

    static const char* name(const computeMode_e& mode);

    [[nodiscard]] const char* name() const;

    bool operator==(computeMode_e m);
    bool operator==(computeMode_t m);
};

std::ostream& operator<<(std::ostream& os, run_et const& m);
std::ostream& operator<<(std::ostream& os, managedMode_t const& m);

}  // END of namespace Neon
