#include "Neon/domain/internal/eGrid/eInternals/Partitioning.h"

namespace Neon::domain::internal::eGrid {
namespace internals {

const std::vector<std::string> partitioning_et::names({"FLAT", "UNDEFINED"});

partitioning_et::partitioning_et(e e)
{
    this->schema = e;
}

const std::string& partitioning_et::string() const
{
    return names[schema];
}


std::ostream& operator<<(std::ostream& os, partitioning_et const& m)
{
    return os << m.string() << std::string("_Partitioning");
}
}  // namespace internals
}  // namespace Neon::domain::internal::eGrid
