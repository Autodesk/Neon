#include <array>
#include <cstdint>

#include "Neon/core/types/Exceptions.h"
#include "Neon/core/types/SetIdx.h"

namespace Neon {

SetIdx::SetIdx(int32_t idx)
    : m_idx(idx) {}

int32_t SetIdx::idx() const
{
    return m_idx;
}

void SetIdx::idx(int32_t idx)
{
    m_idx = idx;
}

std::ostream& operator<<(std::ostream&  os,
                         SetIdx const& m)
{
    return os << "SetIdx_" << m.idx();
}

void SetIdx::setToNext()
{
    m_idx++;
}

SetIdx SetIdx::getNext() const
{
    return SetIdx(m_idx + 1);
}

SetIdx SetIdx::getPrevious() const
{
    return SetIdx(m_idx - 1);
}

bool SetIdx::validate() const
{
    return m_idx >= 0;
}

bool SetIdx::operator<=(int32_t idx) const
{
    return m_idx <= idx;
}

bool SetIdx::operator==(int32_t idx) const
{
    return m_idx == idx;
}

bool SetIdx::operator>=(int32_t idx) const
{
    return m_idx >= idx;
}

bool SetIdx::operator<(int32_t idx) const
{
    return m_idx < idx;
}

bool SetIdx::operator!=(int32_t idx) const
{
    return m_idx != idx;
}

bool SetIdx::operator>(int32_t idx) const
{
    return m_idx >= idx;
}

SetIdx& SetIdx::operator++()
{
    ++m_idx;
    return *this;
}

SetIdx SetIdx::operator++(int)
{
    m_idx++;
    return *this;
}

SetIdx::operator int32_t() const
{
    return m_idx;
}


}  // End of namespace Neon
