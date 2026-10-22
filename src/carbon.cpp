#include <carbon/carbon.h>
#include <carbon/logging.h>

namespace carbon {

auto
info() -> const char* {
  return CARBON_VERSION_FULL;
}

auto
init() -> void {
  log::init_default();
}

}  // namespace carbon
