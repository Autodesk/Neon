#include <vector>
#include "Neon/core//core.h"
#include "Neon/core/types/memOptions.h"


namespace Neon {


auto memPadding_e::toString(int config) -> const char*
{
    switch (config) {
        case memPadding_e::OFF: {
            return "OFF";
        }
        case memPadding_e::ON: {
            return "ON";
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
}

auto memAlignment_e::toString(int config) -> const char*
{
    switch (config) {
        case memAlignment_e::SYSTEM: {
            return "SYSTEM";
        }
        case memAlignment_e::L1: {
            return "L1";
        }
        case memAlignment_e::L2: {
            return "L2";
        }
        case memAlignment_e::PAGE: {
            return "PAGE";
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
}

static std::vector<std::string> mem3dLayoutOrderNames{std::string("structOfArrays"), std::string("arrayOfStructs")};
static std::vector<std::string> mem3dLayoutPaddingNames{std::string("PADDING_ON"), std::string("PADDING_OFF")};


memLayout_et::memLayout_et(order_e order, padding_e padding)
    : m_order(order), m_padding(padding) {}


memLayout_et::memLayout_et(padding_e padding)
    : m_padding(padding) {}


memLayout_et::order_e memLayout_et::order() const
{
    return m_order;
}

memLayout_et::padding_e memLayout_et::padding() const
{
    return m_padding;
}

const char* memLayout_et::order2char(memLayout_et::order_e order)
{
    return mem3dLayoutOrderNames[order].c_str();
}

const char* memLayout_et::padding2char(memLayout_et::padding_e padding)
{
    return mem3dLayoutPaddingNames[padding].c_str();
    ;
}

const char* memLayout_et::order2char() const
{
    return order2char(m_order);
}

const char* memLayout_et::padding2char() const
{
    return padding2char(m_padding);
}

auto memLayout_et::convert(Neon::MemoryLayout  order) -> order_e{
    switch (order) {
        case Neon::MemoryLayout::structOfArrays: {
            return order_e::structOfArrays;
        }
        case Neon::MemoryLayout::arrayOfStructs: {
            return order_e::arrayOfStructs;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
}

auto memLayout_et::convert(Neon::memPadding_e::e padding) -> padding_e {
    switch (padding) {
        case Neon::memPadding_e::OFF: {
            return padding_e::OFF;
        }
        case Neon::memPadding_e::ON: {
            return padding_e::ON;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
}


}  // namespace Neon
