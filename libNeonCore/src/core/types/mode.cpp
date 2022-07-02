#include "Neon/core/types/mode.h"
#include <iostream>
namespace Neon {

//---- [run_et SECTION] --------------------------------------------------------------------------------------------
//---- [run_et SECTION] --------------------------------------------------------------------------------------------
//---- [run_et SECTION] --------------------------------------------------------------------------------------------

run_et::run_et(run_et::et mode)
{
    m_mode = mode;
}

const char* run_et::name(const et& mode)
{
    switch (mode) {
        case et::sync: {
            return "sync";
        }
        case et::async: {
            return "async";
        }
        default:
            return nullptr;
    }
}

const char* run_et::name() const
{
    return this->name(this->m_mode);
}

bool run_et::operator==(run_et::et m)
{
    return this->m_mode == m;
}
bool run_et::operator==(run_et m)
{
    return this->m_mode == m.m_mode;
}


//---- [managedMode_t SECTION] --------------------------------------------------------------------------------------------
//---- [managedMode_t SECTION] --------------------------------------------------------------------------------------------
//---- [managedMode_t SECTION] --------------------------------------------------------------------------------------------


managedMode_t::managedMode_t(managedMode_e mode)
    : mode(mode){};

const char* managedMode_t::name(const managedMode_e& mode)
{
    switch (mode) {
        case managedMode_e::system: {
            return "system";
        }
        case managedMode_e::user: {
            return "user";
        }
        default:
            return nullptr;
    }
}

const char* managedMode_t::name() const
{
    return this->name(this->mode);
}

bool managedMode_t::operator==(managedMode_t::managedMode_e m)
{
    return this->mode == m;
}
bool managedMode_t::operator==(managedMode_t m)
{
    return this->mode == m.mode;
}


//---- [computeMode_t SECTION] --------------------------------------------------------------------------------------------
//---- [computeMode_t SECTION] --------------------------------------------------------------------------------------------
//---- [computeMode_t SECTION] --------------------------------------------------------------------------------------------


computeMode_t::computeMode_t(computeMode_e mode)
    : mode(mode){};

const char* computeMode_t::name(const computeMode_e& mode)
{
    switch (mode) {
        case computeMode_e::par: {
            return "par";
        }
        case computeMode_e::seq: {
            return "seq";
        }
        default:
            return nullptr;
    }
}

const char* computeMode_t::name() const
{
    return this->name(this->mode);
}

bool computeMode_t::operator==(computeMode_t::computeMode_e m)
{
    return this->mode == m;
}
bool computeMode_t::operator==(computeMode_t m)
{
    return this->mode == m.mode;
}


//---- [operator<< SECTION] --------------------------------------------------------------------------------------------
//---- [operator<< SECTION] --------------------------------------------------------------------------------------------
//---- [operator<< SECTION] --------------------------------------------------------------------------------------------


std::ostream& operator<<(std::ostream& os, run_et const& m)
{
    return os << std::string(m.name());
}
std::ostream& operator<<(std::ostream& os, managedMode_t const& m)
{
    return os << std::string(m.name());
}

std::ostream& operator<<(std::ostream& os, computeMode_t const& m)
{
    return os << std::string(m.name());
}

}  // END of namespace Neon