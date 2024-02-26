#include "Config.h"
#include "D3Q19.h"
#include "Neon/domain/dGrid.h"

#include "Metrics.h"
#include "Repoert.h"

namespace CavityTwoPop {

auto run(Config& config,
         Report& report,
         std::stringstream&) -> void;
}  // namespace CavityTwoPop