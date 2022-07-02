#pragma once
#include "Neon/core/core.h"

#include <string>
#include <vector>

namespace Neon::domain::internal::eGrid {
namespace internals {


struct partitioning_et
{
   public:
    enum e : int
    {
        FLAT = 0,  //!< uniform distribution of a flat representation of the domain.
        LEN = 1,
        UNDEFINED = LEN
    };

   private:
    const static std::vector<std::string> names;

   public:
    e schema;

    partitioning_et() = default;
    partitioning_et(const partitioning_et&) = default;
    partitioning_et(partitioning_et&&) = default;
    partitioning_et& operator=(const partitioning_et&) = default;
    partitioning_et& operator=(partitioning_et&&) = default;

    partitioning_et(e e);

    const std::string& string() const;
};

std::ostream& operator<<(std::ostream& os, partitioning_et const& m);
}  // namespace internals
}  // namespace Neon::domain::internal::eGrid
